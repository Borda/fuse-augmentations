"""Opt-in ``torch.compile`` of the warp core (``compile=True``).

Verifies the compiled warp region matches the eager path within tolerance, that the
compiled region has no graph breaks, and that the default (``compile=False``) path is
unaffected. CPU is a documented no-op, so the compiled function is exercised directly
on the module-level warp core to check graph-break freedom regardless of device.

The numerical-parity and graph-break checks use ``backend="eager"`` and the explicit
``compiling=True`` entry point respectively — neither needs a native C/C++ toolchain,
so they run on any CI runner. A separate smoke test opportunistically exercises the
real (default-backend, typically inductor) cached compiled function and skips itself
when the environment lacks a working codegen toolchain, rather than failing CI on an
unrelated compiler/packaging gap.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.affine.segment import (
    _COMPILED_WARP_CACHE,
    _compiled_warp_fn,
    _compiling_grid_sample_affine_batched,
    _grid_sample_affine_batched,
    _torch_supports_compile,
)

pytestmark = pytest.mark.compile


@pytest.fixture(autouse=True)
def _clear_compile_cache() -> None:
    """Reset dynamo state and the module warp-compile cache before each test."""
    torch._dynamo.reset()
    _COMPILED_WARP_CACHE.clear()


class TestTorchSupportsCompile:
    """Version gate for the compiled warp path."""

    def test_returns_bool(self) -> None:
        """The version helper always returns a plain bool for the installed torch."""
        assert isinstance(_torch_supports_compile(), bool)


@pytest.mark.skipif(not _torch_supports_compile(), reason="torch < 2.2 keeps compile a no-op")
class TestCompiledWarpMatchesEager:
    """The compiled warp core reproduces the eager warp within tolerance."""

    def test_compiled_affine_allclose_eager(self) -> None:
        """Compiled affine warp (traced, eager backend) matches plain eager output within atol 1e-5.

        Uses ``backend="eager"`` rather than the package default (inductor): this test's
        contract is that the compile-safe Cramer-rule branch of ``inv3x3`` is numerically
        correct under dynamo tracing, which the eager backend proves without requiring a
        native C/C++ toolchain (inductor's CPU codegen needs one and is not universally
        available across CI runners/OSes).

        """
        image = torch.rand(2, 3, 24, 24, dtype=torch.float32)
        acc = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        acc[:, 0, 2] = 1.5  # small translation so the warp is non-trivial
        eager_out, _ = _grid_sample_affine_batched(image, acc, "bilinear", "zeros")
        compiled = torch.compile(_compiling_grid_sample_affine_batched, backend="eager", dynamic=True)
        compiled_out, _ = compiled(image, acc, "bilinear", "zeros")
        assert torch.allclose(eager_out, compiled_out, atol=1e-5)

    def test_compiled_region_has_no_graph_breaks(self) -> None:
        """The compiled warp core produces a single graph with zero breaks.

        Exercises ``_compiling_grid_sample_affine_batched`` (``compiling=True`` pinned) —
        the exact entry point ``torch.compile`` wraps in production — rather than the raw
        ambient-detecting function, whose branch selection is not reliably traced as a
        constant across all supported torch/dynamo versions.

        """
        image = torch.rand(1, 3, 16, 16, dtype=torch.float32)
        acc = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        explanation = torch._dynamo.explain(_compiling_grid_sample_affine_batched)(image, acc, "bilinear", "zeros")
        assert explanation.graph_break_count == 0

    def test_cached_compiled_warp_fn_smoke(self) -> None:
        """The real cached compiled function (default backend) round-trips when the toolchain supports it.

        Best-effort: some CI runners lack a working native codegen toolchain for the
        default (typically inductor) backend, which is an environment gap unrelated to
        this library's correctness — skip rather than fail when that happens.

        """
        image = torch.rand(2, 3, 24, 24, dtype=torch.float32)
        acc = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        acc[:, 0, 2] = 1.5
        compiled = _compiled_warp_fn("affine")
        try:
            compiled_out, _ = compiled(image, acc, "bilinear", "zeros")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if "Compiler" in reason or "setuptools" in reason or "InductorError" in reason:
                pytest.skip(f"inductor codegen unavailable on this runner: {reason}")
            raise
        eager_out, _ = _grid_sample_affine_batched(image, acc, "bilinear", "zeros")
        assert torch.allclose(eager_out, compiled_out, atol=1e-5)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
@pytest.mark.skipif(not _torch_supports_compile(), reason="torch < 2.2 keeps compile a no-op")
class TestCompileFlagEndToEnd:
    """``compile=True`` on a real pipeline stays numerically equivalent to eager."""

    def test_compile_flag_matches_default_on_cpu(self) -> None:
        """On CPU the compile flag is a no-op — output equals the default eager path."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        image = torch.rand(2, 3, 32, 32, dtype=torch.float32)

        def _build(compile_flag: bool) -> torch.Tensor:
            torch.manual_seed(0)
            transforms = [
                kornia_aug.RandomRotation(degrees=(20.0, 20.0), p=1.0, align_corners=True),
                kornia_aug.RandomHorizontalFlip(p=1.0),
            ]
            pipe = FusedCompose(transforms, adapter=KorniaAdapter(), compile=compile_flag)
            return pipe(image.clone())

        assert torch.allclose(_build(False), _build(True), atol=1e-6)

    def test_compile_flag_default_off(self) -> None:
        """The default pipeline reports ``compile_warp=False``."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        pipe = FusedCompose([kornia_aug.RandomHorizontalFlip(p=1.0)], adapter=KorniaAdapter())
        assert pipe.compile_warp is False
