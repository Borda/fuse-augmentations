"""Integration coverage for opt-in per-transform border handling."""

from __future__ import annotations

import copy

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
from fuse_augmentations.adapters.kornia import KorniaAdapter
from fuse_augmentations.adapters.torchvision import TorchVisionAdapter
from fuse_augmentations.types import TransformSpec

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu
    import cv2

try:
    from torchvision.transforms.v2 import RandomAffine as TorchVisionRandomAffine

    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torchvision is absent
    _TORCHVISION_AVAILABLE = False


pytestmark = pytest.mark.integration


def _image() -> torch.Tensor:
    """Return a deterministic image with non-uniform border pixels."""
    return torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 255.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_kornia_adapter_extracts_affine_border_mode() -> None:
    """Kornia's affine flag maps directly to the fused padding vocabulary."""
    transform = kornia_aug.RandomAffine(degrees=10, padding_mode="border", p=1.0)

    assert KorniaAdapter.border_mode(transform) == "border"


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
def test_albumentations_adapter_extracts_compatible_and_opaque_modes() -> None:
    """Albumentations maps reflect-101 exactly and leaves wrap opaque."""
    reflection = albu.Affine(rotate=10, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    wrap = albu.Affine(rotate=10, border_mode=cv2.BORDER_WRAP, p=1.0)

    assert AlbumentationsAdapter.border_mode(reflection) == "reflection"
    assert AlbumentationsAdapter.border_mode(wrap) is None


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required")
def test_torchvision_adapter_leaves_nonzero_fill_opaque() -> None:
    """TorchVision fill is only compatible when it is zero."""
    assert TorchVisionAdapter.border_mode(TorchVisionRandomAffine(10)) == "zeros"
    assert TorchVisionAdapter.border_mode(TorchVisionRandomAffine(10, fill=1)) is None


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_factories_accept_the_opt_in_padding_policy() -> None:
    """Configuration and direct-parameter factories preserve the opt-in value."""
    configured = Compose.from_config(
        [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})],
        backend="kornia",
        padding_mode="per_transform",
    )
    direct = Compose.from_params(rotation=(-10.0, 10.0), padding_mode="per_transform")

    assert configured.padding_mode == direct.padding_mode == "per_transform"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_uniform_border_modes_keep_one_bit_identical_segment() -> None:
    """Per-transform mode keeps a uniform Kornia run fused without changing pixels."""
    transforms = [
        kornia_aug.RandomAffine(degrees=(15, 15), padding_mode="reflection", p=1.0),
        kornia_aug.RandomAffine(degrees=(-10, -10), padding_mode="reflection", p=1.0),
    ]
    override = Compose(copy.deepcopy(transforms), padding_mode="reflection")
    per_transform = Compose(copy.deepcopy(transforms), padding_mode="per_transform")

    torch.manual_seed(17)
    expected = override(_image())
    torch.manual_seed(17)
    actual = per_transform(_image())

    assert len(override.fusion_plan_descriptors) == len(per_transform.fusion_plan_descriptors) == 1
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_mixed_border_modes_split_and_match_native_sequence() -> None:
    """Different Kornia modes split the fused run and preserve native operation order."""
    transforms = [
        kornia_aug.RandomAffine(degrees=(15, 15), padding_mode="reflection", p=1.0),
        kornia_aug.RandomAffine(degrees=(-10, -10), padding_mode="zeros", p=1.0),
    ]
    pipe = Compose(copy.deepcopy(transforms), padding_mode="per_transform")
    native = copy.deepcopy(transforms)

    actual = pipe(_image())
    expected = native[1](native[0](_image()))

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    descriptors = pipe.fusion_plan_descriptors
    assert len(descriptors) == 2
    assert descriptors[1].split_reason == "border_mode_change"


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
def test_opaque_albumentations_border_mode_warns_and_stays_unfused() -> None:
    """Albumentations wrap mode remains a native hard boundary in opt-in mode."""
    with pytest.warns(UserWarning, match="opaque border mode"):
        pipe = Compose(
            [
                albu.Affine(rotate=10, border_mode=cv2.BORDER_WRAP, p=1.0),
                albu.Affine(rotate=-10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            ],
            padding_mode="per_transform",
        )

    descriptors = pipe.fusion_plan_descriptors
    assert len(descriptors) == 2
    assert descriptors[0].kind == "passthrough"
    assert descriptors[0].split_reason == "opaque_border_mode"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_default_padding_path_still_uses_the_single_override() -> None:
    """Omitting the opt-in preserves the historical one-segment override path."""
    transforms = [
        kornia_aug.RandomAffine(degrees=(15, 15), padding_mode="reflection", p=1.0),
        kornia_aug.RandomAffine(degrees=(-10, -10), padding_mode="zeros", p=1.0),
    ]
    implicit = Compose(copy.deepcopy(transforms))
    explicit = Compose(copy.deepcopy(transforms), padding_mode="zeros")

    torch.manual_seed(29)
    expected = explicit(_image())
    torch.manual_seed(29)
    actual = implicit(_image())

    assert len(implicit.fusion_plan_descriptors) == len(explicit.fusion_plan_descriptors) == 1
    assert torch.equal(actual, expected)


@pytest.mark.skip(reason="In-bounds affine border-mode elision is intentionally deferred; splitting is always safe.")
def test_zoomed_in_mixed_border_modes_can_stay_one_segment() -> None:
    """A future in-bounds proof may elide an otherwise required border-mode split."""
