"""Unit tests for Albumentations backend detection and transform classification.

No albumentations installation required — uses module-path mock objects that mimic the class hierarchy without importing
the real package.

"""

from __future__ import annotations

import warnings

import pytest

from fuse_augmentations._backend import Backend, detect_backend
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations._types import TransformCategory

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter


def _make_mock(module_path: str) -> object:
    """Create a mock transform instance with a given __module__."""
    cls = type("MockTransform", (), {"__module__": module_path, "__qualname__": "MockTransform"})
    return cls()


@pytest.mark.parametrize(
    "module_path",
    [
        "albumentations.augmentations.geometric.transforms",
        "albumentations.augmentations.geometric.rotate",
        "albumentations.augmentations.geometric.flip",
        "albumentations.core.transforms_interface",
    ],
)
def test_detect_backend_albumentations(module_path: str) -> None:
    """Any albumentations.* module path detects ALBUMENTATIONS backend."""
    mock = _make_mock(module_path)
    assert detect_backend([mock]) == Backend.ALBUMENTATIONS


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbumentationsAdapterCategory:
    """Category lookup for the 5 registered transforms."""

    @pytest.fixture
    def adapter(self) -> AlbumentationsAdapter:
        return AlbumentationsAdapter()

    def test_unknown_transform_returns_spatial_kernel_with_warning(self, adapter):
        """An unknown albumentations transform returns SPATIAL_KERNEL and warns."""
        mock = _make_mock("albumentations.augmentations.transforms")
        with pytest.warns(UserWarning, match="Unknown Albumentations transform"):
            cat = adapter.category(mock)
        assert cat == TransformCategory.SPATIAL_KERNEL

    def test_category_real_affine(self, adapter):
        """A.Affine → GEOMETRIC_INTERP."""
        assert adapter.category(albu.Affine()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_rotate(self, adapter):
        """A.Rotate → GEOMETRIC_INTERP."""
        assert adapter.category(albu.Rotate()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_shift_scale_rotate(self, adapter):
        """A.ShiftScaleRotate → GEOMETRIC_INTERP."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert adapter.category(albu.ShiftScaleRotate()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_hflip(self, adapter):
        """A.HorizontalFlip → GEOMETRIC_EXACT."""
        assert adapter.category(albu.HorizontalFlip()) == TransformCategory.GEOMETRIC_EXACT

    def test_category_real_vflip(self, adapter):
        """A.VerticalFlip → GEOMETRIC_EXACT."""
        assert adapter.category(albu.VerticalFlip()) == TransformCategory.GEOMETRIC_EXACT

    def test_perspective_is_projective(self, adapter):
        """A.Perspective → PROJECTIVE."""
        cat = adapter.category(albu.Perspective())
        assert cat == TransformCategory.PROJECTIVE
