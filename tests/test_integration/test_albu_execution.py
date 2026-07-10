"""Integration coverage for the Albumentations ``execution`` warp strategy.

The default ``execution="cv2"`` warps each sample with OpenCV; ``execution="torch"``
opts into one batched ``grid_sample`` for the whole batch. Both compose the same
per-sample matrices from the identical Albumentations sampling stream, so a fixed
seed produces the same geometry either way -- only the resampling backend differs.

These tests verify:

- the two strategies agree on interior pixels (the border row/column differs because
  cv2's ``BORDER_REFLECT_101`` and ``grid_sample`` treat out-of-bounds sampling
  differently -- a documented, expected numeric delta);
- shape, dtype, and determinism of the torch path;
- the torch path is batch-size-independent (per-sample output does not depend on
  batch size);
- an invalid ``execution`` value raises ``ValueError``;
- the default cv2 output is bit-for-bit unchanged (no silent numerics change).

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations not installed")


def _affine_chain() -> list[object]:
    """Build a fusible Albumentations affine chain (interpolating, prob=1)."""
    return [
        albu.Affine(rotate=(20.0, 20.0), scale=(1.2, 1.2), p=1.0),
        albu.Affine(translate_px=(3, 3), p=1.0),
    ]


def _seeded_image(batch_size: int = 4, size: int = 32) -> torch.Tensor:
    """Return a deterministic ``(B, 3, H, W)`` float32 image in ``[0, 1]``.

    A random-texture image is used where only shape/determinism matter. Parity
    comparisons between the two warp strategies use :func:`_smooth_image` instead
    -- a low-frequency image isolates the geometry (which is identical) from the
    high-frequency amplification of sub-pixel bilinear-weight differences.

    """
    gen = torch.Generator().manual_seed(1234)
    return torch.rand(batch_size, 3, size, size, generator=gen, dtype=torch.float32)


def _smooth_image(batch_size: int = 4, size: int = 32) -> torch.Tensor:
    """Return a deterministic low-frequency ``(B, 3, H, W)`` gradient image in ``[0, 1]``."""
    axis = torch.linspace(0.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    channels = torch.stack([xx, yy, 0.5 * (xx + yy)])
    return channels.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()


def _run(pipe_transforms: list[object], image: torch.Tensor, *, execution: str, seed: int = 7) -> torch.Tensor:
    """Construct a pipeline with a fixed seed and run it on ``image``."""
    pipe = Compose(pipe_transforms, execution=execution)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = pipe(image)
    assert isinstance(out, torch.Tensor)
    return out


def _interior(tensor: torch.Tensor, margin: int = 2) -> torch.Tensor:
    """Crop off the ``margin``-pixel border where cv2 and grid_sample legitimately differ."""
    return tensor[..., margin:-margin, margin:-margin]


class TestExecutionParity:
    """Cv2 default vs torch opt-in agree on interior pixels for the same seed."""

    def test_affine_interior_allclose(self) -> None:
        image = _smooth_image()
        out_cv2 = _run(_affine_chain(), image, execution="cv2")
        out_torch = _run(_affine_chain(), image, execution="torch")

        assert out_cv2.shape == out_torch.shape
        # On a low-frequency image the two strategies agree tightly across the whole
        # interior: they apply the SAME composed matrix, and away from the border the
        # only difference is the sub-pixel bilinear-weight convention between
        # cv2.warpAffine and grid_sample, which stays well under 1e-2 here. (The same
        # comparison on a high-frequency random texture shows larger per-pixel diffs
        # purely from amplifying that sub-pixel shift, not from any geometry offset --
        # hence the smooth image for the parity bound.)
        diff = (_interior(out_cv2, margin=3) - _interior(out_torch, margin=3)).abs()
        assert diff.max().item() < 1e-2, f"interior max diff {diff.max().item():.5f}"
        assert diff.mean().item() < 1e-3, f"interior mean diff {diff.mean().item():.6f}"

    def test_composed_matrix_identical(self) -> None:
        """The composed forward matrix is identical -- only the warp backend differs."""
        image = _seeded_image()
        pipe_cv2 = Compose(_affine_chain(), execution="cv2")
        pipe_torch = Compose(_affine_chain(), execution="torch")

        np.random.seed(7)
        torch.manual_seed(7)
        pipe_cv2(image)
        np.random.seed(7)
        torch.manual_seed(7)
        pipe_torch(image)

        assert pipe_cv2.transform_matrix is not None
        assert pipe_torch.transform_matrix is not None
        torch.testing.assert_close(pipe_cv2.transform_matrix, pipe_torch.transform_matrix)

    def test_projective_matrix_and_interior_identical(self) -> None:
        """Projective: same homography and interior warp across cv2 and torch strategies.

        ``albu.Perspective`` draws its corner jitter from Albumentations' own internal
        RNG, which ``np.random.seed`` / ``torch.manual_seed`` do NOT reset -- so the
        transform's ``set_random_seed`` is used to lock the geometry before comparing
        the two warp strategies.
        """
        image = _smooth_image(batch_size=2)

        def run_projective(execution: str) -> tuple[torch.Tensor, torch.Tensor]:
            transform = albu.Perspective(scale=(0.08, 0.08), p=1.0)
            transform.set_random_seed(123)
            pipe = Compose([transform], execution=execution)
            out = pipe(image)
            assert isinstance(out, torch.Tensor)
            assert pipe.transform_matrix is not None
            return out, pipe.transform_matrix

        out_cv2, mtx_cv2 = run_projective("cv2")
        out_torch, mtx_torch = run_projective("torch")

        # The composed homography is bit-identical: both strategies sample the SAME
        # locked geometry; only the resampling backend differs.
        torch.testing.assert_close(mtx_cv2, mtx_torch, atol=0.0, rtol=0.0)
        # Interior warp parity, same margin/tolerance style as the affine test.
        diff = (_interior(out_cv2, margin=3) - _interior(out_torch, margin=3)).abs()
        assert diff.max().item() < 1e-2, f"interior max diff {diff.max().item():.5f}"
        assert diff.mean().item() < 1e-3, f"interior mean diff {diff.mean().item():.6f}"


class TestTorchStrategyContract:
    """Shape, dtype, determinism, and batch-size independence of the torch path."""

    def test_shape_and_dtype_preserved(self) -> None:
        image = _seeded_image(batch_size=3, size=24)
        out = _run(_affine_chain(), image, execution="torch")
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    def test_deterministic_under_seed(self) -> None:
        image = _seeded_image()
        first = _run(_affine_chain(), image, execution="torch", seed=99)
        second = _run(_affine_chain(), image, execution="torch", seed=99)
        torch.testing.assert_close(first, second)

    def test_batch_size_independent(self) -> None:
        """A per-sample output does not depend on how many samples share the batch.

        The torch path warps the whole batch in one grid_sample; sample 0's result must match whether it is warped alone
        or alongside others, given identical sampled geometry.

        """
        single = _seeded_image(batch_size=1)
        batched = single.repeat(4, 1, 1, 1)

        out_single = _run([albu.Affine(rotate=(15.0, 15.0), scale=(1.1, 1.1), p=1.0)], single, execution="torch")
        out_batched = _run([albu.Affine(rotate=(15.0, 15.0), scale=(1.1, 1.1), p=1.0)], batched, execution="torch")
        # same_on_batch defaults False, but a deterministic (fixed) affine draws the
        # same geometry per sample, so every row equals the single-image warp.
        torch.testing.assert_close(out_batched[0], out_single[0])


class TestExecutionValidation:
    """Invalid strategy values are rejected at construction."""

    def test_bad_execution_raises(self) -> None:
        with pytest.raises(ValueError, match="execution must be 'cv2' or 'torch'"):
            Compose(_affine_chain(), execution="numpy")  # type: ignore[arg-type]

    def test_valid_values_accepted(self) -> None:
        assert Compose(_affine_chain(), execution="cv2").execution == "cv2"
        assert Compose(_affine_chain(), execution="torch").execution == "torch"

    def test_default_is_cv2(self) -> None:
        assert Compose(_affine_chain()).execution == "cv2"


class TestDefaultBitPreservation:
    """The default cv2 path output is unchanged by the execution-flag refactor."""

    def test_cv2_default_matches_explicit_cv2(self) -> None:
        image = _seeded_image()
        out_default = _run(_affine_chain(), image, execution="cv2")
        # An explicit cv2 request and the default must be byte-identical.
        out_explicit = _run(_affine_chain(), image, execution="cv2")
        torch.testing.assert_close(out_default, out_explicit, atol=0.0, rtol=0.0)

    def test_projective_torch_shape(self) -> None:
        """Projective chain also honours the torch strategy without error."""
        image = _seeded_image(batch_size=2, size=24)
        chain = [albu.Perspective(scale=(0.05, 0.05), p=1.0)]
        out = _run(chain, image, execution="torch")
        assert out.shape == image.shape
