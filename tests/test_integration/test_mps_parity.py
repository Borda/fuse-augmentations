"""Numeric-parity coverage for the fused pipeline on the Apple MPS device.

The core fusion paths added on top of the plain affine warp are exercised on the
Metal Performance Shaders backend and compared against a trusted CPU float32
reference (or, for the low-precision path, against the same pipeline in float32):

- **Sampled-convolution blur** under a rotation warp (the rotated-Gaussian path
  that lowers to a single ``F.conv2d`` over a sampled kernel).
- **Runtime histogram equalisation** (the per-image lookup-table path).
- **Affine geometry fused with a pointwise colour op** (the ``grid_sample`` warp
  plus the colour matmul).
- **Whole-pipeline compile region**: the compiled forward must stay numerically
  identical to the eager forward on MPS.
- **bfloat16 pipeline dtype**: the low-precision warp/colour path must stay within
  a documented peak-signal-to-noise bound of the float32 result on MPS.

Every pipeline here is built from fixed-parameter transforms so that a CPU
instance and an MPS instance sample identical augmentation parameters without any
in-body RNG seeding; the parity assertion therefore reflects device numerics
only. CUDA remains the standing unverified device (no CUDA runner available).

Requires kornia and an available MPS device; skipped otherwise.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for MPS parity coverage"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required"),
]


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the maximum absolute elementwise difference as a host float."""
    return (actual.float().cpu() - expected.float().cpu()).abs().max().item()


def _psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return peak signal-to-noise ratio in decibels for unit-interval tensors."""
    mse = torch.mean((actual.float().cpu() - expected.float().cpu()) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0) / mse))


def _rotated_blur_transforms() -> list[object]:
    """Return a fixed rotation followed by a fixed Gaussian blur (sampled conv2d path)."""
    return [
        kornia_aug.RandomRotation(degrees=(30.0, 30.0), p=1.0),
        kornia_aug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5), p=1.0),
    ]


def _affine_colour_transforms() -> list[object]:
    """Return a fixed affine warp fused with a fixed brightness op (deterministic)."""
    return [
        kornia_aug.RandomAffine(degrees=(20.0, 20.0), p=1.0),
        kornia_aug.RandomBrightness(brightness=(1.2, 1.2), p=1.0),
    ]


@pytest.fixture
def image() -> torch.Tensor:
    """Return a deterministic (2, 3, 32, 32) float32 image on CPU."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 32, 32, generator=generator)


def test_rotated_blur_matches_cpu_reference(image: torch.Tensor) -> None:
    """The rotated sampled-convolution blur agrees with the CPU float32 result on MPS."""
    reference = Compose(_rotated_blur_transforms())(image.clone())
    on_device = Compose(_rotated_blur_transforms()).to("mps")(image.to("mps"))

    assert on_device.device.type == "mps"
    assert _max_abs_diff(on_device, reference) < 1e-4


def test_equalize_matches_cpu_reference(image: torch.Tensor) -> None:
    """The runtime histogram-equalise lookup table agrees with the CPU result on MPS."""
    reference = Compose([kornia_aug.RandomEqualize(p=1.0)])(image.clone())
    on_device = Compose([kornia_aug.RandomEqualize(p=1.0)]).to("mps")(image.to("mps"))

    assert _max_abs_diff(on_device, reference) < 1e-4


def test_affine_colour_matches_cpu_reference(image: torch.Tensor) -> None:
    """The fused affine warp plus colour matmul agrees with the CPU result on MPS."""
    reference = Compose(_affine_colour_transforms())(image.clone())
    on_device = Compose(_affine_colour_transforms()).to("mps")(image.to("mps"))

    assert _max_abs_diff(on_device, reference) < 1e-4


def test_compile_region_matches_eager_on_mps(image: torch.Tensor) -> None:
    """The whole-pipeline compile region stays numerically identical to eager on MPS."""
    eager = Compose(_affine_colour_transforms()).to("mps")(image.to("mps"))
    compiled = Compose(_affine_colour_transforms(), compile=True).to("mps")(image.to("mps"))

    assert _max_abs_diff(compiled, eager) < 1e-5


def test_bfloat16_pipeline_stays_within_psnr_bound_on_mps(image: torch.Tensor) -> None:
    """The bfloat16 pipeline dtype stays within a documented PSNR bound of float32 on MPS."""
    float32 = Compose(_affine_colour_transforms()).to("mps")(image.to("mps"))
    low_precision = Compose(_affine_colour_transforms(), pipeline_dtype="bfloat16").to("mps")(image.to("mps"))

    assert _psnr(low_precision, float32) > 30.0
