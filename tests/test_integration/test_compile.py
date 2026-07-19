"""Opt-in ``torch.compile`` of warp, color, and LUT tensor cores (``compile=True``).

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
    _COMPILED_COLOR_CACHE,
    _COMPILED_LUT_CACHE,
    _COMPILED_WARP_CACHE,
    _apply_color_matrix,
    _apply_interp_lut,
    _compiled_lut_fn,
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
    _COMPILED_COLOR_CACHE.clear()
    _COMPILED_LUT_CACHE.clear()


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
        """The compiled warp core produces a single graph with zero breaks."""
        image = torch.rand(1, 3, 16, 16, dtype=torch.float32)
        acc = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        explanation = torch._dynamo.explain(_compiling_grid_sample_affine_batched)(image, acc, "bilinear", "zeros")
        assert explanation.graph_break_count == 0

    def test_cached_compiled_warp_fn_smoke(self) -> None:
        """The cached default-backend warp function round-trips when codegen is available."""
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


@pytest.mark.skipif(not _torch_supports_compile(), reason="torch < 2.2 keeps compile a no-op")
class TestCompiledColorMatchesEager:
    """The compiled color epilogue reproduces eager matrix application."""

    def test_compiled_color_allclose_eager(self) -> None:
        """Compiled color matrix application matches eager dynamic shapes within atol 1e-5."""
        image = torch.rand(2, 3, 24, 24, dtype=torch.float32)
        matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        matrix[:, 0, 3] = 0.1
        eager_out = _apply_color_matrix(image, matrix)
        compiled = torch.compile(_apply_color_matrix, backend="eager", dynamic=True)
        compiled_out = compiled(image, matrix)
        assert torch.allclose(eager_out, compiled_out, atol=1e-5)

        narrow_image = torch.rand(1, 3, 17, 31, dtype=torch.float32)
        narrow_matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        assert torch.allclose(
            _apply_color_matrix(narrow_image, narrow_matrix),
            compiled(narrow_image, narrow_matrix),
            atol=1e-5,
        )

    def test_compiled_color_region_has_no_graph_breaks(self) -> None:
        """The pure color matrix application produces one graph with zero breaks."""
        image = torch.rand(1, 3, 16, 16, dtype=torch.float32)
        matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        explanation = torch._dynamo.explain(_apply_color_matrix)(image, matrix)
        assert explanation.graph_break_count == 0


@pytest.mark.skipif(not _torch_supports_compile(), reason="torch < 2.2 keeps compile a no-op")
class TestCompiledLUTMatchesEager:
    """The compiled lookup-table gather reproduces eager interpolation."""

    def test_compiled_lut_region_has_no_graph_breaks(self) -> None:
        """The static LUT gather produces one graph with zero breaks (eager, no toolchain)."""
        image = torch.rand(2, 3, 24, 24, dtype=torch.float32)
        table = torch.linspace(0.0, 1.0, 1024).view(1, 1, -1).expand(2, 3, -1).contiguous()
        explanation = torch._dynamo.explain(_apply_interp_lut)(image, table, 1024)
        assert explanation.graph_break_count == 0

    def test_cached_compiled_lut_fn_allclose_eager(self) -> None:
        """The cached default-backend LUT gather round-trips (full and narrow) when codegen is available."""
        image = torch.rand(2, 3, 24, 24, dtype=torch.float32)
        table = torch.linspace(0.0, 1.0, 1024).view(1, 1, -1).expand(2, 3, -1).contiguous()
        narrow_image = torch.rand(1, 3, 17, 31, dtype=torch.float32)
        narrow_table = table[:1]
        eager_out = _apply_interp_lut(image, table, 1024)
        narrow_eager_out = _apply_interp_lut(narrow_image, narrow_table, 1024)
        compiled = _compiled_lut_fn("interp")
        try:
            compiled_out = compiled(image, table, 1024)
            narrow_compiled_out = compiled(narrow_image, narrow_table, 1024)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if "Compiler" in reason or "setuptools" in reason or "InductorError" in reason:
                pytest.skip(f"inductor codegen unavailable on this runner: {reason}")
            raise
        assert torch.allclose(eager_out, compiled_out, atol=1e-5)
        assert torch.allclose(narrow_eager_out, narrow_compiled_out, atol=1e-5)


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
                kornia_aug.RandomBrightness(brightness=(0.9, 0.9), p=1.0),
            ]
            pipe = FusedCompose(transforms, adapter=KorniaAdapter(), compile=compile_flag)
            return pipe(image.clone())

        assert torch.allclose(_build(False), _build(True), atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled pipeline region requires CUDA")
    def test_compiled_geo_color_pipeline_matches_eager(self) -> None:
        """A CUDA geometry-plus-color pipeline matches eager output within atol 1e-5."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        image = torch.rand(2, 3, 32, 32, dtype=torch.float32, device="cuda")

        def _build(compile_flag: bool) -> torch.Tensor:
            torch.manual_seed(0)
            transforms = [
                kornia_aug.RandomRotation(degrees=(20.0, 20.0), p=1.0, align_corners=True),
                kornia_aug.RandomBrightness(brightness=(0.9, 0.9), p=1.0),
            ]
            pipe = FusedCompose(transforms, adapter=KorniaAdapter(), compile=compile_flag)
            return pipe(image.clone())

        assert torch.allclose(_build(False), _build(True), atol=1e-5)

    def test_compile_flag_default_off(self) -> None:
        """The default pipeline reports ``compile_warp=False``."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        pipe = FusedCompose([kornia_aug.RandomHorizontalFlip(p=1.0)], adapter=KorniaAdapter())
        assert pipe.compile_warp is False
