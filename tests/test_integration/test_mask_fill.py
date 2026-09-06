"""End-to-end mask border values and exact pixel-edge box flips."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import FusedCompose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("dtype", "mask_fill"),
    [
        pytest.param(torch.uint8, 255, id="uint8-ignore"),
        pytest.param(torch.int64, -1, id="signed-ignore"),
    ],
)
def test_affine_mask_fill_preserves_ignore_label(dtype: torch.dtype, mask_fill: int) -> None:
    """A translated hard mask receives its configured label only outside the source canvas."""
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.full((1, 1, 8, 8), 7, dtype=dtype)
    pipe = FusedCompose.from_params(
        translate_x=(2.0, 2.0),
        data_keys=["input", "mask"],
        mask_fill=mask_fill,
    )

    _, warped_mask = pipe(image, mask)

    assert warped_mask.dtype == dtype
    assert torch.equal(warped_mask[..., :, :2], torch.full_like(warped_mask[..., :, :2], mask_fill))
    assert torch.equal(warped_mask[..., :, 2:], torch.full_like(warped_mask[..., :, 2:], 7))


@pytest.mark.parametrize(
    ("dtype", "mask_fill"),
    [
        pytest.param(torch.uint8, 255, id="uint8-ignore"),
        pytest.param(torch.int64, -1, id="signed-ignore"),
    ],
)
def test_letterbox_mask_fill_paints_pad_only(dtype: torch.dtype, mask_fill: int) -> None:
    """Crop-resize letterbox routing uses the configured value in padded mask rows."""
    image = torch.zeros(1, 3, 4, 8)
    mask = torch.full((1, 1, 4, 8), 7, dtype=dtype)
    pipe = FusedCompose.from_params(
        letterbox=(8, 12),
        data_keys=["input", "mask"],
        mask_fill=mask_fill,
    )

    _, warped_mask = pipe(image, mask)

    assert warped_mask.dtype == dtype
    assert torch.equal(warped_mask[..., 0, :], torch.full_like(warped_mask[..., 0, :], mask_fill))
    assert torch.equal(warped_mask[..., 1:-1, :], torch.full_like(warped_mask[..., 1:-1, :], 7))


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
@pytest.mark.parametrize(
    ("dtype", "mask_fill"),
    [
        pytest.param(torch.uint8, 255, id="uint8-ignore"),
        pytest.param(torch.int64, -1, id="signed-ignore"),
    ],
)
def test_projective_mask_fill_reaches_outside_pixels(dtype: torch.dtype, mask_fill: int) -> None:
    """Projective mask routing retains the configured hard-label border value."""
    torch.manual_seed(0)
    image = torch.zeros(1, 3, 16, 16)
    mask = torch.full((1, 1, 16, 16), 7, dtype=dtype)
    pipe = FusedCompose(
        [kornia_aug.RandomPerspective(distortion_scale=0.8, p=1.0)],
        data_keys=["input", "mask"],
        mask_fill=mask_fill,
    )

    _, warped_mask = pipe(image, mask)

    assert warped_mask.dtype == dtype
    assert torch.any(warped_mask == mask_fill)
    assert torch.any(warped_mask == 7)


def test_default_mask_fill_remains_zero() -> None:
    """An existing pipeline without ``mask_fill`` retains the zero border."""
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.int64)
    pipe = FusedCompose.from_params(translate_x=(2.0, 2.0), data_keys=["input", "mask"])

    _, warped_mask = pipe(image, mask)

    assert torch.equal(warped_mask[..., :, :2], torch.zeros_like(warped_mask[..., :, :2]))


def test_mask_fill_stays_independent_from_image_fill() -> None:
    """Mask ignore labels never inherit the image border colour."""
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.uint8)
    pipe = FusedCompose.from_params(
        translate_x=(2.0, 2.0),
        data_keys=["input", "mask"],
        fill=0.25,
        mask_fill=255,
    )

    warped_image, warped_mask = pipe(image, mask)

    torch.testing.assert_close(warped_image[..., :, :2], torch.full_like(warped_image[..., :, :2], 0.25))
    assert torch.equal(warped_mask[..., :, :2], torch.full_like(warped_mask[..., :, :2], 255))


def test_bilinear_soft_mask_fill_preserves_dtype_and_gradient() -> None:
    """Bilinear float masks mix with the configured border and retain gradients."""
    image = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, requires_grad=True)
    pipe = FusedCompose.from_params(
        translate_x=(1.5, 1.5),
        data_keys=["input", "mask"],
        mask_interpolation="bilinear",
        mask_fill=0.75,
    )

    _, warped_mask = pipe(image, mask)
    warped_mask.sum().backward()

    assert warped_mask.dtype == torch.float32
    assert torch.any((warped_mask > 0.0) & (warped_mask < 0.75))
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()


def test_exact_flips_use_pixel_edge_bbox_extents() -> None:
    """Exact H/V flips preserve full canvas edges and map asymmetric boxes with W/H."""
    image = torch.zeros(1, 3, 8, 16)
    boxes = torch.tensor([[[0.0, 0.0, 16.0, 8.0], [2.0, 1.0, 6.0, 5.0]]])
    pipe = FusedCompose.from_params(
        hflip_p=1.0,
        vflip_p=1.0,
        data_keys=["input", "bbox_xyxy"],
    )

    _, warped_boxes = pipe(image, boxes)

    expected = torch.tensor([[[0.0, 0.0, 16.0, 8.0], [10.0, 3.0, 14.0, 7.0]]])
    torch.testing.assert_close(warped_boxes, expected)
