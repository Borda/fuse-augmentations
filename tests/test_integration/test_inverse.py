"""Integration coverage for paired-matrix TTA de-augmentation."""

from __future__ import annotations

import math

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = pytest.mark.integration


def _psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return peak signal-to-noise ratio for tensors in the unit interval."""
    mse = torch.mean((actual.to(torch.float64) - expected.to(torch.float64)) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0, dtype=torch.float64) / mse))


def _smooth_image(size: int = 64) -> torch.Tensor:
    """Build a smooth image that exposes interpolation regressions."""
    axis = torch.linspace(0.0, 1.0, size)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    pattern = 0.5 + 0.2 * torch.sin(2.0 * math.pi * grid_x) + 0.2 * torch.cos(2.0 * math.pi * grid_y)
    return pattern.clamp(0.0, 1.0).repeat(3, 1, 1).unsqueeze(0)


def test_affine_inverse_round_trip_uses_paired_matrix() -> None:
    """A fused affine output de-augments to the original interior image."""
    image = _smooth_image()
    pipe = Compose.from_params(rotation=(12.0, 12.0))

    augmented, matrix = pipe(image, return_matrix=True)
    recovered = pipe.inverse(augmented, matrix=matrix)

    assert matrix is not None
    assert _psnr(recovered[..., 6:-6, 6:-6], image[..., 6:-6, 6:-6]) > 35.0


def test_inverse_requires_paired_matrix() -> None:
    """The mutable compatibility property is never used as an inverse input."""
    pipe = Compose.from_params(translate_x=(4.0, 4.0))

    with pytest.raises(ValueError, match="paired forward matrix"):
        pipe.inverse(_smooth_image())


def test_inverse_routes_aux_targets_through_inverse_matrix() -> None:
    """Masks, boxes, and keypoints use the same inverse geometry as the image."""
    image = _smooth_image()
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 20:40, 20:40] = 1.0
    boxes_xyxy = torch.tensor([[[20.0, 20.0, 40.0, 40.0]]])
    boxes_xywh = torch.tensor([[[20.0, 20.0, 20.0, 20.0]]])
    keypoints = torch.tensor([[[24.0, 28.0]]])
    pipe = Compose.from_params(
        translate_x=(4.0, 4.0),
        data_keys=["input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints"],
    )

    augmented, matrix = pipe(image, mask, boxes_xyxy, boxes_xywh, keypoints, return_matrix=True)
    recovered = pipe.inverse(*augmented, matrix=matrix)

    recovered_image, recovered_mask, recovered_xyxy, recovered_xywh, recovered_keypoints = recovered
    assert matrix is not None
    torch.testing.assert_close(recovered_image[..., :, 8:-8], image[..., :, 8:-8], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered_mask[..., :, 8:-8], mask[..., :, 8:-8], atol=0.0, rtol=0.0)
    torch.testing.assert_close(recovered_xyxy, boxes_xyxy, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(recovered_xywh, boxes_xywh, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(recovered_keypoints, keypoints, atol=1e-4, rtol=1e-4)


def test_inverse_recovers_keypoints_but_inflates_axis_aligned_boxes_under_rotation() -> None:
    """A rotation round-trips keypoints exactly but inflates axis-aligned boxes.

    Bounding boxes are stored as axis-aligned corners, so rotating then inverse-rotating
    encloses a larger axis-aligned box than the original. This pins the documented lossy
    contract: keypoints recover to sampling precision, but boxes do not for a
    non-axis-aligned warp.

    """
    image = _smooth_image()
    boxes_xyxy = torch.tensor([[[20.0, 20.0, 40.0, 40.0]]])
    keypoints = torch.tensor([[[24.0, 28.0]]])
    pipe = Compose.from_params(
        rotation=(23.0, 23.0),
        data_keys=["input", "bbox_xyxy", "keypoints"],
    )

    augmented, matrix = pipe(image, boxes_xyxy, keypoints, return_matrix=True)
    _recovered_image, recovered_xyxy, recovered_keypoints = pipe.inverse(*augmented, matrix=matrix)

    assert matrix is not None
    # Keypoints return to their original coordinates to sampling precision.
    torch.testing.assert_close(recovered_keypoints, keypoints, atol=1e-3, rtol=0.0)
    # The axis-aligned box strictly inflates: round-tripped corners enclose more area.
    assert bool((recovered_xyxy[..., 0] < boxes_xyxy[..., 0] - 1.0).any())
    assert bool((recovered_xyxy[..., 2] > boxes_xyxy[..., 2] + 1.0).any())


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for projective inverse coverage")
def test_projective_inverse_round_trip_uses_homography() -> None:
    """A fused perspective output de-augments through the inverse homography."""
    image = _smooth_image()
    pipe = Compose([kornia_aug.RandomPerspective(distortion_scale=0.15, p=1.0)])

    augmented, matrix = pipe(image, return_matrix=True)
    recovered = pipe.inverse(augmented, matrix=matrix)

    assert matrix is not None
    assert _psnr(recovered[..., 8:-8, 8:-8], image[..., 8:-8, 8:-8]) > 25.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia is required for non-invertible segment coverage")
@pytest.mark.parametrize(
    ("pipe", "message"),
    [
        (
            Compose([kornia_aug.RandomResizedCrop(size=(32, 32), scale=(0.8, 0.8), ratio=(1.0, 1.0), p=1.0)]),
            "crop-resize",
        ),
        (Compose([kornia_aug.RandomBrightness(brightness=(0.8, 0.8), p=1.0)]), "non-geometric color"),
    ],
)
def test_inverse_rejects_non_invertible_pipeline(pipe: Compose, message: str) -> None:
    """Information-losing and non-geometric segments fail with named reasons."""
    with pytest.raises(ValueError, match=message):
        pipe.inverse(_smooth_image())
