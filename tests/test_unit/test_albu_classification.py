"""Unit tests for Albumentations backend detection and transform classification.

No albumentations installation required — uses module-path mock objects that mimic the class hierarchy without importing
the real package.

"""

from __future__ import annotations

import pytest

from fuse_augmentations._backend import Backend, detect_backend
from fuse_augmentations._types import TransformCategory


def _make_mock(module_path: str) -> object:
    """Create a mock transform instance with a given __module__."""
    cls = type("MockTransform", (), {"__module__": module_path, "__qualname__": "MockTransform"})
    return cls()


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AlbumentationsAdapter.category() — module-path mock approach
# ---------------------------------------------------------------------------


class TestAlbumentationsAdapterCategory:
    """Category lookup for the 5 registered transforms."""

    @pytest.fixture
    def adapter(self):
        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

        return AlbumentationsAdapter()

    def test_unknown_transform_returns_spatial_kernel_with_warning(self, adapter):
        mock = _make_mock("albumentations.augmentations.transforms")
        with pytest.warns(UserWarning, match="Unknown Albumentations transform"):
            cat = adapter.category(mock)
        assert cat == TransformCategory.SPATIAL_KERNEL

    def test_category_real_affine(self, adapter):
        """A.Affine → GEOMETRIC_INTERP (requires albumentations)."""
        pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")
        import albumentations as A

        assert adapter.category(A.Affine()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_rotate(self, adapter):
        pytest.importorskip("albumentations")
        import albumentations as A

        assert adapter.category(A.Rotate()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_shift_scale_rotate(self, adapter):
        pytest.importorskip("albumentations")
        import warnings

        import albumentations as A

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # deprecation warning from SSR
            assert adapter.category(A.ShiftScaleRotate()) == TransformCategory.GEOMETRIC_INTERP

    def test_category_real_hflip(self, adapter):
        pytest.importorskip("albumentations")
        import albumentations as A

        assert adapter.category(A.HorizontalFlip()) == TransformCategory.GEOMETRIC_EXACT

    def test_category_real_vflip(self, adapter):
        pytest.importorskip("albumentations")
        import albumentations as A

        assert adapter.category(A.VerticalFlip()) == TransformCategory.GEOMETRIC_EXACT
