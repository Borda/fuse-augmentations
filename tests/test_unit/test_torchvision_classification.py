"""Unit tests for TorchVision backend detection and transform classification.

Backend-detection tests use module-path mock objects that mimic the class hierarchy without importing the real
torchvision package. Transform-category tests use real TorchVision transforms and are skipped via pytest.importorskip
when torchvision is not installed.
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
        "torchvision.transforms",
        "torchvision.transforms.functional",
        "torchvision.transforms.autoaugment",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2._geometry",
    ],
)
def test_detect_backend_torchvision(module_path: str) -> None:
    """Any torchvision.* module path detects TORCHVISION backend."""
    mock = _make_mock(module_path)
    assert detect_backend([mock]) == Backend.TORCHVISION


# ---------------------------------------------------------------------------
# TorchVisionAdapter.category() -- real transforms (requires torchvision)
# ---------------------------------------------------------------------------


class TestTorchVisionAdapterCategory:
    """Category lookup for the 4 registered transforms."""

    @pytest.fixture
    def adapter(self):
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        return TorchVisionAdapter()

    @pytest.mark.parametrize(
        "transform_factory, expected_cat",
        [
            (
                lambda T: T.RandomRotation(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
            ),
            (
                lambda T: T.RandomAffine(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
            ),
            (
                lambda T: T.RandomHorizontalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
            ),
            (
                lambda T: T.RandomVerticalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
            ),
        ],
        ids=["RandomRotation", "RandomAffine", "RandomHorizontalFlip", "RandomVerticalFlip"],
    )
    def test_category_real_v1(self, adapter, transform_factory, expected_cat):
        """v1 transforms map to expected categories (requires torchvision)."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms as T

        assert adapter.category(transform_factory(T)) == expected_cat

    @pytest.mark.parametrize(
        "transform_factory, expected_cat",
        [
            (
                lambda T: T.RandomRotation(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
            ),
            (
                lambda T: T.RandomAffine(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
            ),
            (
                lambda T: T.RandomHorizontalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
            ),
            (
                lambda T: T.RandomVerticalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
            ),
        ],
        ids=["v2.RandomRotation", "v2.RandomAffine", "v2.RandomHorizontalFlip", "v2.RandomVerticalFlip"],
    )
    def test_category_real_v2(self, adapter, transform_factory, expected_cat):
        """v2 transforms map to expected categories (requires torchvision)."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms.v2 as T

        assert adapter.category(transform_factory(T)) == expected_cat

    def test_unknown_transform_returns_spatial_kernel_with_warning(self, adapter):
        """An unknown transform returns SPATIAL_KERNEL and emits UserWarning."""
        mock = _make_mock("torchvision.transforms")
        with pytest.warns(UserWarning, match="Unknown TorchVision transform"):
            cat = adapter.category(mock)
        assert cat == TransformCategory.SPATIAL_KERNEL

    def test_expand_true_raises_value_error(self, adapter):
        """expand=True on a transform raises ValueError."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms as T

        t = T.RandomRotation(degrees=30, expand=True)
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.category(t)
