"""Integration coverage for differentiable auxiliary-mask interpolation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations import FusedCompose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.affine.matrix import normalize_matrix

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = [pytest.mark.integration]

HEIGHT = 32
WIDTH = 32


def _inputs(requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a zero image and a square float mask."""
    image = torch.zeros(1, 3, HEIGHT, WIDTH)
    mask = torch.zeros(1, 1, HEIGHT, WIDTH)
    mask[..., 8:24, 8:24] = 1.0
    if requires_grad:
        mask.requires_grad_()
    return image, mask


def _kornia_pipe(**kwargs: object) -> FusedCompose:
    """Build a deterministic Kornia-backed rotation pipeline."""
    return FusedCompose(
        [kornia_aug.RandomRotation(degrees=(30.0, 30.0), p=1.0)],
        data_keys=["input", "mask"],
        **kwargs,
    )


def _expected_nearest(mask: torch.Tensor, pipe: FusedCompose) -> torch.Tensor:
    """Apply the pre-change nearest mask sampling contract directly."""
    matrix = pipe.transform_matrix
    assert matrix is not None
    inverse = torch.linalg.inv(matrix.to(dtype=torch.float32))
    normalized = normalize_matrix(inverse, HEIGHT, WIDTH).to(dtype=mask.dtype)
    grid = F.affine_grid(normalized[:, :2, :], mask.shape, align_corners=True)
    return F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=True)


class TestMaskInterpolation:
    """Core nearest, bilinear, gradient, and validation behavior."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
    def test_default_nearest_matches_existing_grid_sample(self) -> None:
        """The default mask output is bit-identical to nearest grid sampling."""
        image, mask = _inputs()
        pipe = _kornia_pipe()
        _, output = pipe(image, mask)

        torch.testing.assert_close(output, _expected_nearest(mask, pipe), atol=0.0, rtol=0.0)

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
    def test_bilinear_produces_intermediate_boundary_values(self) -> None:
        """Bilinear mask interpolation mixes labels at rotated boundaries."""
        image, mask = _inputs()
        nearest_pipe = _kornia_pipe(mask_interpolation="nearest")
        bilinear_pipe = _kornia_pipe(mask_interpolation="bilinear")
        _, nearest = nearest_pipe(image, mask)
        _, bilinear = bilinear_pipe(image, mask)

        assert not torch.equal(nearest, bilinear)
        assert torch.any((bilinear > 0.0) & (bilinear < 1.0))

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
    def test_bilinear_mask_is_differentiable(self) -> None:
        """Bilinear mask routing preserves gradients to a float input mask."""
        image, mask = _inputs(requires_grad=True)
        pipe = _kornia_pipe(mask_interpolation="bilinear")
        _, output = pipe(image, mask)
        output.sum().backward()

        assert mask.grad is not None
        assert torch.isfinite(mask.grad).all()

    def test_invalid_mask_interpolation_raises(self) -> None:
        """Unknown mask interpolation modes are rejected at construction."""
        with pytest.raises(ValueError, match="mask_interpolation"):
            FusedCompose([], mask_interpolation="bicubic")  # type: ignore[arg-type]


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_kornia_fused_pipeline_accepts_bilinear_masks() -> None:
    """Kornia-backed fused routing accepts differentiable mask interpolation."""
    image, mask = _inputs()
    _, output = _kornia_pipe(mask_interpolation="bilinear")(image, mask)

    assert torch.any((output > 0.0) & (output < 1.0))


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
def test_albumentations_fused_pipeline_accepts_bilinear_masks() -> None:
    """Albumentations-backed fused routing accepts differentiable mask interpolation."""
    image, mask = _inputs()
    pipe = FusedCompose(
        [albu.Rotate(limit=(30.0, 30.0), p=1.0)],
        data_keys=["input", "mask"],
        mask_interpolation="bilinear",
    )
    _, output = pipe(image, mask)

    assert torch.any((output > 0.0) & (output < 1.0))
