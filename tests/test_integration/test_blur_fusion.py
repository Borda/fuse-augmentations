"""Gaussian-blur folding and safe affine commutation coverage."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.adapters.kornia import KorniaAdapter
from fuse_augmentations.affine.segment import FusedAffineSegment, _kornia_gaussian_blur
from fuse_augmentations.compose import Compose

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for Gaussian blur fusion")
def test_consecutive_gaussian_blurs_fold_to_one_primitive() -> None:
    """Variance addition stays close to a sequential float64 primitive reference."""
    image = torch.rand(2, 3, 32, 32, dtype=torch.float64)
    pipe = Compose([
        kornia_aug.RandomGaussianBlur((3, 3), (0.8, 0.8), p=1.0),
        kornia_aug.RandomGaussianBlur((3, 3), (1.1, 1.1), p=1.0),
    ])

    output = pipe(image)
    reference = _kornia_gaussian_blur(_kornia_gaussian_blur(image, 0.8, 0.8), 1.1, 1.1)
    assert reference is not None
    assert len(pipe.fusion_plan_descriptors) == 1
    assert pipe.fusion_plan.startswith("gaussian_blur(")
    assert _psnr(output, reference) > 55.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for Gaussian blur fusion")
def test_axis_aligned_blur_scale_is_not_a_passthrough_barrier() -> None:
    """An axis-aligned upscale moves a Gaussian blur to the fused chain end."""
    pipe = Compose([
        kornia_aug.RandomRotation(degrees=(0.0, 0.0), p=1.0),
        kornia_aug.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
        kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(1.25, 1.25), p=1.0),
    ])

    image = torch.rand(2, 3, 24, 24, dtype=torch.float64)
    output = pipe(image)
    reference_blur = _kornia_gaussian_blur(image, 1.0, 1.0)
    assert reference_blur is not None
    reference = FusedAffineSegment(
        [
            kornia_aug.RandomRotation(degrees=(0.0, 0.0), p=1.0),
            kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(1.25, 1.25), p=1.0),
        ],
        KorniaAdapter(),
    )(reference_blur)

    assert "passthrough(RandomGaussianBlur)" not in pipe.fusion_plan
    assert pipe.n_warps_saved == 1
    assert _psnr(output, reference) > 30.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for Gaussian blur fusion")
def test_downscale_prefix_commutes_as_one_high_quality_warp() -> None:
    """A downscaling affine before the blur still fuses to a single warp.

    Commutation is gated on the affine that *follows* the blur (here an upscale), so the surrounding affine run
    collapses to one warp exactly as any affine chain does. The commuted blur then matches a single-warp reference at
    high fidelity, confirming the fused path stays correct for a non-identity (downscaling) prefix and avoids the two-
    warp intermediate detail loss of the previous blur-barrier plan.

    """
    image = torch.rand(2, 3, 24, 24, dtype=torch.float64)
    pipe = Compose([
        kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(0.5, 0.5), p=1.0),
        kornia_aug.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
        kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(2.0, 2.0), p=1.0),
    ])

    output = pipe(image)
    composed_warp = FusedAffineSegment(
        [
            kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(0.5, 0.5), p=1.0),
            kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(2.0, 2.0), p=1.0),
        ],
        KorniaAdapter(),
    )(image)
    # The blur commutes through the upscale suffix, so its sigma scales by the suffix factor.
    reference = _kornia_gaussian_blur(composed_warp, 2.0, 2.0)
    assert reference is not None
    assert len(pipe.fusion_plan_descriptors) == 1
    assert pipe.n_warps_saved >= 1
    assert _psnr(output, reference) > 40.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for Gaussian blur fusion")
@pytest.mark.parametrize(
    "affine",
    [
        kornia_aug.RandomAffine(degrees=(12.0, 12.0), p=1.0),
        kornia_aug.RandomAffine(degrees=(0.0, 0.0), scale=(0.8, 0.8), p=1.0),
    ],
    ids=["rotation", "downscale"],
)
def test_unsafe_affines_keep_the_gaussian_blur_barrier(affine) -> None:
    """Rotation and downscale retain the existing two-segment barrier plan."""
    pipe = Compose([
        kornia_aug.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
        affine,
    ])

    assert len(pipe.fusion_plan_descriptors) == 2
    assert "passthrough(RandomGaussianBlur)" in pipe.fusion_plan


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for Gaussian blur fusion")
def test_folded_gaussian_blur_keeps_per_sample_probability() -> None:
    """Per-sample blur activation remains independent for a batched folded run."""
    torch.manual_seed(4)
    image = torch.rand(16, 3, 24, 24)
    pipe = Compose([
        kornia_aug.RandomGaussianBlur((3, 3), (1.0, 1.0), p=0.5),
        kornia_aug.RandomGaussianBlur((3, 3), (1.0, 1.0), p=0.0),
    ])

    output = pipe(image)
    changed = (output - image).abs().flatten(1).amax(dim=1) > 1e-6
    assert changed.any()
    assert (~changed).any()


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations is required for native NumPy coverage")
def test_albumentations_fold_uses_one_cv2_gaussian_blur() -> None:
    """A native HWC pipeline folds consecutive Gaussian blur transforms."""
    image = torch.rand(18, 19, 3).numpy()
    pipe = Compose([
        albu.GaussianBlur(sigma_limit=(1.0, 1.0), p=1.0),
        albu.GaussianBlur(sigma_limit=(1.0, 1.0), p=1.0),
    ])

    output = pipe(image=image)["image"]
    assert output.shape == image.shape
    assert pipe.fusion_plan.startswith("gaussian_blur(")


def _psnr(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return peak signal-to-noise ratio for unit-range image tensors."""
    mse = torch.mean((actual - reference).square()).item()
    return float("inf") if mse == 0.0 else -10.0 * torch.log10(torch.tensor(mse)).item()
