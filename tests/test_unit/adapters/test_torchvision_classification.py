"""Unit tests for TorchVision backend detection and transform classification.

Backend-detection tests use module-path mock objects that mimic the class hierarchy without importing the real
torchvision package. Transform-category tests use real TorchVision transforms and are skipped via
@pytest.mark.skipif when torchvision is not installed.

"""

from __future__ import annotations

import pytest

from fuse_augmentations import Compose
from fuse_augmentations._backend import Backend, detect_backend, detect_backends_per_transform
from fuse_augmentations._compat import _TORCHVISION_AVAILABLE, _TORCHVISION_V2_AVAILABLE
from fuse_augmentations._compose import _build_mixed_segments
from fuse_augmentations._types import ReorderPolicy, TransformCategory

if _TORCHVISION_AVAILABLE:
    import torch
    import torchvision.transforms as tv_trans

    from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

if _TORCHVISION_V2_AVAILABLE:
    import torchvision.transforms.v2 as Tv2


def _make_mock(module_path: str) -> object:
    """Create a mock transform instance with a given __module__."""
    cls = type("MockTransform", (), {"__module__": module_path, "__qualname__": "MockTransform"})
    return cls()


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


class TestTorchVisionAdapterCategory:
    """Category lookup for the 4 registered transforms."""

    @pytest.fixture
    def adapter(self) -> TorchVisionAdapter:
        """Return a fresh TorchVisionAdapter."""
        return TorchVisionAdapter()

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    @pytest.mark.parametrize(
        "transform_factory, expected_cat",
        [
            pytest.param(
                lambda T: T.RandomRotation(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
                id="RandomRotation",
            ),
            pytest.param(
                lambda T: T.RandomAffine(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
                id="RandomAffine",
            ),
            pytest.param(
                lambda T: T.RandomHorizontalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
                id="RandomHorizontalFlip",
            ),
            pytest.param(
                lambda T: T.RandomVerticalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
                id="RandomVerticalFlip",
            ),
            pytest.param(
                lambda T: T.RandomPerspective(distortion_scale=0.5, p=0.5),
                TransformCategory.PROJECTIVE,
                id="RandomPerspective",
            ),
        ],
    )
    def test_category_real_v1(self, adapter: TorchVisionAdapter, transform_factory, expected_cat):
        """V1 transforms map to expected categories (requires torchvision)"""
        assert adapter.category(transform_factory(tv_trans)) == expected_cat

    @pytest.mark.skipif(not _TORCHVISION_V2_AVAILABLE, reason="missing torchvision.transforms.v2")
    @pytest.mark.parametrize(
        "transform_factory, expected_cat",
        [
            pytest.param(
                lambda Tv2: Tv2.RandomRotation(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
                id="v2.RandomRotation",
            ),
            pytest.param(
                lambda Tv2: Tv2.RandomAffine(degrees=30),
                TransformCategory.GEOMETRIC_INTERP,
                id="v2.RandomAffine",
            ),
            pytest.param(
                lambda Tv2: Tv2.RandomHorizontalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
                id="v2.RandomHorizontalFlip",
            ),
            pytest.param(
                lambda Tv2: Tv2.RandomVerticalFlip(p=0.5),
                TransformCategory.GEOMETRIC_EXACT,
                id="v2.RandomVerticalFlip",
            ),
            pytest.param(
                lambda Tv2: Tv2.RandomPerspective(distortion_scale=0.5, p=0.5),
                TransformCategory.PROJECTIVE,
                id="v2.RandomPerspective",
            ),
        ],
    )
    def test_category_real_v2(self, adapter: TorchVisionAdapter, transform_factory, expected_cat):
        """V2 transforms map to expected categories (requires torchvision.transforms.v2)"""
        assert adapter.category(transform_factory(Tv2)) == expected_cat

    def test_unknown_transform_returns_spatial_kernel_with_warning(self, adapter: TorchVisionAdapter):
        """An unknown transform returns SPATIAL_KERNEL and emits UserWarning."""
        mock = _make_mock("torchvision.transforms")
        with pytest.warns(UserWarning, match="Unknown TorchVision transform"):
            cat = adapter.category(mock)
        assert cat == TransformCategory.SPATIAL_KERNEL

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_expand_true_raises_value_error(self, adapter: TorchVisionAdapter):
        """Expand=True on a transform raises ValueError."""
        transform = tv_trans.RandomRotation(degrees=30, expand=True)
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.category(transform)

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_sample_params_expand_true_raises_value_error(self, adapter: TorchVisionAdapter):
        """sample_params with expand=True raises ValueError."""
        transform = tv_trans.RandomRotation(degrees=30, expand=True)
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.sample_params(transform, (2, 3, 64, 64), torch.device("cpu"))


@pytest.mark.skipif(not _TORCHVISION_V2_AVAILABLE, reason="missing torchvision.transforms.v2")
@pytest.mark.xfail(
    strict=False,
    reason=(
        "_v1_transform_cls is an undocumented TorchVision internal, stable 0.15-0.20; "
        "may be removed in future TorchVision versions"
    ),
)
def test_v1_transform_cls_attribute_exists_on_v2_transform():
    """V2 transforms expose _v1_transform_cls, used by the adapter for v1/v2 detection."""
    transform = Tv2.RandomHorizontalFlip(p=1.0)
    assert hasattr(transform, "_v1_transform_cls"), (
        "torchvision.transforms.v2.RandomHorizontalFlip no longer has _v1_transform_cls; "
        "TorchVisionAdapter._is_torchvision_v2_transform may need updating"
    )


class TestUnknownBackendError:
    """Regression for Bug #3: all-unknown pipeline raises actionable ValueError."""

    def test_all_unknown_transforms_raises_value_error(self):
        """Compose([unknown]) raises ValueError with actionable message."""
        unknown = _make_mock("completely_unknown.lib")
        with pytest.raises(ValueError, match="No recognised backend"):
            Compose([unknown])

    def test_all_unknown_transforms_error_mentions_supported_backends(self):
        """The error message mentions supported backends by name."""
        unknown = _make_mock("my_custom.module")
        with pytest.raises(ValueError, match=r"Kornia.*TorchVision.*Albumentations"):
            Compose([unknown])


class TestDetectBackendMROFallback:
    """Regression for Bug #4: subclasses defined outside backend package are detected."""

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_torchvision_subclass_detected_via_mro(self):
        """A subclass of tv_trans.RandomRotation defined in __main__ detects as TORCHVISION."""

        class MyRot(tv_trans.RandomRotation):
            pass

        result = detect_backends_per_transform([MyRot(30)])
        assert result == [Backend.TORCHVISION]

    @pytest.mark.skipif(not _TORCHVISION_V2_AVAILABLE, reason="missing torchvision.transforms.v2")
    def test_torchvision_v2_subclass_detected_via_mro(self):
        """A v2 subclass defined outside torchvision.* is detected as TORCHVISION."""

        class MyFlip(Tv2.RandomHorizontalFlip):
            pass

        result = detect_backends_per_transform([MyFlip(p=0.5)])
        assert result == [Backend.TORCHVISION]

    def test_plain_unknown_still_returns_none(self):
        """A transform with no backend ancestor still returns None."""
        unknown = _make_mock("my_custom.module")
        result = detect_backends_per_transform([unknown])
        assert result == [None]


class TestBuildMixedSegmentsGuard:
    """Regression for Bug #5: __init__ guards prevent _build_mixed_segments with all-None backends."""

    def test_all_unknown_never_reaches_build_mixed_segments(self):
        """All-unknown transforms raise ValueError in __init__ before _build_mixed_segments."""
        unknown = _make_mock("fake.lib")
        with pytest.raises(ValueError, match="No recognised backend"):
            Compose([unknown])

    def test_build_mixed_segments_raises_on_all_none(self):
        """Direct call to _build_mixed_segments with all-None backends raises ValueError."""
        unknown = _make_mock("fake.lib")
        with pytest.raises(ValueError, match="_build_mixed_segments called with no recognised"):
            _build_mixed_segments(
                [unknown],
                [None],
                ReorderPolicy.NONE,
                None,
                None,
            )
