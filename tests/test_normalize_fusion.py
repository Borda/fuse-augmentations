"""Parity and planning tests for pointwise Normalize fusion."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE
from fuse_augmentations.affine.segment import FusedColorSegment, build_segments
from fuse_augmentations.compose import FusedCompose
from fuse_augmentations.types import ReorderPolicy

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

    from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters.kornia import KorniaAdapter

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms.v2 as tv_v2

    from fuse_augmentations.adapters.torchvision import TorchVisionAdapter


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("torchvision", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")),
        pytest.param("kornia", marks=pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")),
        pytest.param(
            "albumentations",
            marks=pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations"),
        ),
    ],
)
def test_normalize_fused_matches_native(backend: str) -> None:
    """Each installed backend's standard Normalize matches its fused affine matrix."""
    mean = (0.5, 0.4, 0.3)
    std = (0.2, 0.3, 0.4)
    if backend == "torchvision":
        transform = tv_v2.Normalize(mean, std)
        adapter = TorchVisionAdapter()
    elif backend == "kornia":
        transform = kornia_aug.Normalize(mean, std)
        adapter = KorniaAdapter()
    else:
        transform = albu.Normalize(mean=mean, std=std, max_pixel_value=255.0)
        adapter = AlbumentationsAdapter()

    image = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    segment = build_segments([transform], adapter, "bilinear", "zeros")[0]
    assert isinstance(segment, FusedColorSegment)
    actual = segment(image)
    expected = adapter.call_nonfused(transform, image)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert segment.clip_output is False


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
def test_rotate_brightness_normalize_has_fused_color_descriptor() -> None:
    """Rotate plus Brightness plus Normalize leaves one color pass after rotation."""
    transforms = [
        tv_v2.RandomRotation(0.0),
        tv_v2.ColorJitter(brightness=0.2),
        tv_v2.Normalize([0.5, 0.4, 0.3], [0.2, 0.3, 0.4]),
    ]
    pipe = FusedCompose(transforms, reorder=ReorderPolicy.NONE)

    assert pipe.fusion_plan == "fused(RandomRotation) → color(ColorJitter, Normalize)"
    assert pipe.n_warps_saved == 1
    assert [descriptor.kind for descriptor in pipe.fusion_plan_descriptors] == ["fused", "color"]


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
def test_normalize_does_not_touch_auxiliary_targets() -> None:
    """Color fusion leaves masks unchanged and preserves positional data-key order."""
    pipe = FusedCompose(
        [tv_v2.Normalize([0.5, 0.4, 0.3], [0.2, 0.3, 0.4])],
        data_keys=["input", "mask"],
        reorder=ReorderPolicy.NONE,
    )
    image = torch.rand(2, 3, 4, 4)
    mask = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    output_image, output_mask = pipe(image, mask)

    torch.testing.assert_close(output_mask, mask)
    assert output_image.shape == image.shape
