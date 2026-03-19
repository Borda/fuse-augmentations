"""Unit tests for TorchVision backend detection and transform classification.

Backend-detection tests use module-path mock objects that mimic the class hierarchy without importing the real
torchvision package. Transform-category tests use real TorchVision transforms and are skipped via pytest.importorskip
when torchvision is not installed.

"""

from __future__ import annotations

import pytest

from fuse_augmentations._backend import Backend, detect_backend, detect_backends_per_transform
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
        """V1 transforms map to expected categories (requires torchvision)."""
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
        """V2 transforms map to expected categories (requires torchvision)."""
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
        """Expand=True on a transform raises ValueError."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms as T

        t = T.RandomRotation(degrees=30, expand=True)
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.category(t)


# ---------------------------------------------------------------------------
# Bug #3 regression: Backend.UNKNOWN raises ValueError (not opaque NotImplementedError)
# ---------------------------------------------------------------------------


class TestUnknownBackendError:
    """Regression for Bug #3: all-unknown pipeline raises actionable ValueError."""

    def test_all_unknown_transforms_raises_value_error(self):
        """Compose([unknown]) raises ValueError with actionable message."""
        from fuse_augmentations import Compose

        unknown = _make_mock("completely_unknown.lib")
        with pytest.raises(ValueError, match="No recognised backend"):
            Compose([unknown])

    def test_all_unknown_transforms_error_mentions_supported_backends(self):
        """The error message mentions supported backends by name."""
        from fuse_augmentations import Compose

        unknown = _make_mock("my_custom.module")
        with pytest.raises(ValueError, match=r"Kornia.*TorchVision.*Albumentations"):
            Compose([unknown])


# ---------------------------------------------------------------------------
# Bug #4 regression: detect_backends_per_transform MRO fallback
# ---------------------------------------------------------------------------


class TestDetectBackendMROFallback:
    """Regression for Bug #4: subclasses defined outside backend package are detected."""

    def test_torchvision_subclass_detected_via_mro(self):
        """A subclass of T.RandomRotation defined in __main__ detects as TORCHVISION."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms as T

        class MyRot(T.RandomRotation):
            pass

        result = detect_backends_per_transform([MyRot(30)])
        assert result == [Backend.TORCHVISION]

    def test_torchvision_v2_subclass_detected_via_mro(self):
        """A v2 subclass defined outside torchvision.* is detected as TORCHVISION."""
        pytest.importorskip("torchvision", reason="torchvision required")
        import torchvision.transforms.v2 as T

        class MyFlip(T.RandomHorizontalFlip):
            pass

        result = detect_backends_per_transform([MyFlip(p=0.5)])
        assert result == [Backend.TORCHVISION]

    def test_plain_unknown_still_returns_none(self):
        """A transform with no backend ancestor still returns None."""
        unknown = _make_mock("my_custom.module")
        result = detect_backends_per_transform([unknown])
        assert result == [None]


# ---------------------------------------------------------------------------
# Bug #5 regression: _build_mixed_segments all-None fallback is dead code
# ---------------------------------------------------------------------------


class TestBuildMixedSegmentsGuard:
    """Regression for Bug #5: __init__ guards prevent _build_mixed_segments with all-None backends."""

    def test_all_unknown_never_reaches_build_mixed_segments(self):
        """All-unknown transforms raise ValueError in __init__ before _build_mixed_segments."""
        from fuse_augmentations import Compose

        unknown = _make_mock("fake.lib")
        # This should raise ValueError in the single-backend fast path,
        # never reaching _build_mixed_segments.
        with pytest.raises(ValueError, match="No recognised backend"):
            Compose([unknown])

    def test_build_mixed_segments_raises_on_all_none(self):
        """Direct call to _build_mixed_segments with all-None backends raises ValueError."""
        from fuse_augmentations._compose import _build_mixed_segments
        from fuse_augmentations._types import ReorderPolicy

        unknown = _make_mock("fake.lib")
        with pytest.raises(ValueError, match="_build_mixed_segments called with no recognised"):
            _build_mixed_segments(
                [unknown],
                [None],
                ReorderPolicy.NONE,
                None,
                None,
            )
