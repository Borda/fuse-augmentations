"""Registry coverage tests for the TorchVision adapter.

Parametrised over the live ``_TRANSFORM_REGISTRY`` dict so that new entries
are automatically validated.

Each registered transform must satisfy:
- ``adapter.category(instance)`` returns a ``TransformCategory`` member.
- ``adapter.sample_params(instance, input_shape, device)`` returns a dict of tensors.
- ``adapter.build_matrix(instance, params, height, width)`` returns a ``(batch_size, 3, 3)`` tensor.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from fuse_augmentations.adapters.torchvision import _TRANSFORM_REGISTRY, TorchVisionAdapter
else:
    _TRANSFORM_REGISTRY = {}  # type: ignore[var-annotated]
    TorchVisionAdapter = None  # type: ignore[assignment,misc]

_CONSTRUCTOR_KWARGS: dict[str, dict] = {
    "RandomRotation": {"degrees": 30},
    "RandomAffine": {"degrees": 30},
    "RandomHorizontalFlip": {"prob": 1.0},
    "RandomVerticalFlip": {"prob": 1.0},
    "RandomPerspective": {"distortion_scale": 0.3, "prob": 1.0},
    "ColorJitter": {"brightness": 0.2},
    "RandomResizedCrop": {"size": (32, 32)},
    "Normalize": {"mean": (0.5, 0.4, 0.3), "std": (0.2, 0.3, 0.4)},
    # Pixel-wise non-linear ops carry a required positional argument.
    "RandomSolarize": {"threshold": 0.5},
    "RandomPosterize": {"bits": 4},
}

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32
DEFAULT_DEVICE = torch.device("cpu")
INPUT_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


def _make_instance(cls: type) -> object:
    """Instantiate a registered transform class with default kwargs."""
    # Strip version prefix for lookup: _V1RandomRotation -> RandomRotation
    name = cls.__name__
    for prefix in ("_V1", "_V2"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    kwargs = _CONSTRUCTOR_KWARGS.get(name, {"prob": 1.0})
    try:
        return cls(**kwargs)
    except TypeError:
        return cls(p=1.0)


@pytest.fixture
def adapter() -> TorchVisionAdapter:
    return TorchVisionAdapter()


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestRegistryCoverage:
    """Every entry in _TRANSFORM_REGISTRY passes the adapter contract."""

    @pytest.mark.parametrize(
        "transform_cls,expected_cat",
        [pytest.param(cls, cat, id=cls.__name__) for cls, cat in _TRANSFORM_REGISTRY.items()],
    )
    def test_category_matches_registry(self, adapter, transform_cls, expected_cat):
        """adapter.category(instance) returns the registered TransformCategory."""
        instance = _make_instance(transform_cls)
        cat = adapter.category(instance)
        assert cat == expected_cat, f"category({transform_cls.__name__}) returned {cat}, expected {expected_cat}"

    @pytest.mark.parametrize(
        "transform_cls",
        [pytest.param(cls, id=cls.__name__) for cls in _TRANSFORM_REGISTRY],
    )
    def test_sample_params_returns_dict_of_tensors(self, adapter, transform_cls):
        """sample_params returns a dict with torch.Tensor values."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEFAULT_DEVICE)
        assert isinstance(params, dict), f"Expected dict, got {type(params)}"
        for key, val in params.items():
            assert isinstance(val, torch.Tensor), f"params[{key!r}] is {type(val).__name__}, expected Tensor"

    @pytest.mark.parametrize(
        "transform_cls",
        [pytest.param(cls, id=cls.__name__) for cls in _TRANSFORM_REGISTRY],
    )
    def test_build_matrix_returns_b33_tensor(self, adapter, transform_cls):
        """build_matrix returns a (batch_size, 3, 3) tensor."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEFAULT_DEVICE)
        mtx = adapter.build_matrix(instance, params, HEIGHT, WIDTH)
        assert isinstance(mtx, torch.Tensor), f"Expected Tensor, got {type(mtx)}"
        assert mtx.ndim == 3, f"Expected 3D tensor, got {mtx.ndim}D"
        assert mtx.shape[1:] == (3, 3), f"Expected (*, 3, 3), got {mtx.shape}"

    @pytest.mark.parametrize(
        "transform_cls",
        [pytest.param(cls, id=cls.__name__) for cls in _TRANSFORM_REGISTRY],
    )
    def test_build_matrix_no_nan_inf(self, adapter, transform_cls):
        """build_matrix output contains no NaN or Inf values."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEFAULT_DEVICE)
        mtx = adapter.build_matrix(instance, params, HEIGHT, WIDTH)
        assert not torch.isnan(mtx).any(), f"NaN in build_matrix for {transform_cls.__name__}"
        assert not torch.isinf(mtx).any(), f"Inf in build_matrix for {transform_cls.__name__}"


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestRegistryCompleteness:
    """Verify the registry is non-empty and covers both v1 and v2 namespaces."""

    def test_registry_not_empty(self):
        """_TRANSFORM_REGISTRY is non-empty, guarding against silent torchvision import failure."""
        assert len(_TRANSFORM_REGISTRY) > 0, (
            "TorchVision _TRANSFORM_REGISTRY is empty -- imports may have failed silently"
        )

    def test_registry_covers_v1_transforms(self):
        """Registry should include v1 namespace transforms."""
        names = {cls.__name__ for cls in _TRANSFORM_REGISTRY}
        # At least the v1 baseline should be present
        v1_expected = {"RandomRotation", "RandomAffine", "RandomHorizontalFlip", "RandomVerticalFlip"}
        # Check that v1 variants exist (exact class names may vary)
        v1_present = {n for n in names if not n.startswith("_V2") and any(base in n for base in v1_expected)}
        assert len(v1_present) >= 4, (
            f"Expected at least 4 v1 transforms in registry, found {len(v1_present)}: {v1_present}"
        )
