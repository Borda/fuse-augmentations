"""Registry coverage tests for the Albumentations adapter.

Parametrised over the live ``TRANSFORM_REGISTRY`` dict so that new entries
are automatically validated.

Each registered transform must satisfy:
- ``adapter.category(instance)`` returns a ``TransformCategory`` member
  matching the registry value.
- ``adapter.sample_params(instance, input_shape, device)`` returns a dict
  with ``torch.Tensor`` values.
- ``adapter.build_matrix(instance, params, height, width)`` returns a ``(batch_size, 3, 3)``
  ``torch.Tensor``.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    from fuse_augmentations.adapters.albumentations import TRANSFORM_REGISTRY, AlbumentationsAdapter
else:
    TRANSFORM_REGISTRY = {}  # type: ignore[var-annotated]
    AlbumentationsAdapter = None  # type: ignore[assignment,misc]

_CONSTRUCTOR_KWARGS: dict[str, dict] = {
    "Affine": {"rotate": (-30, 30), "prob": 1.0},
    "Rotate": {"limit": (-30, 30), "prob": 1.0},
    "ShiftScaleRotate": {"prob": 1.0},
    "HorizontalFlip": {"prob": 1.0},
    "VerticalFlip": {"prob": 1.0},
    "Perspective": {"scale": (0.05, 0.1), "prob": 1.0},
    # Future entries:
    "SafeRotate": {"limit": (-30, 30), "prob": 1.0},
    "RandomRotate90": {"prob": 1.0},
    "D4": {"prob": 1.0},
    "Transpose": {"prob": 1.0},
    "RandomResizedCrop": {"size": (32, 32), "prob": 1.0},
}

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32
DEFAULT_DEVICE = torch.device("cpu")
INPUT_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


def _make_instance(cls: type) -> object:
    """Instantiate a registered transform class with default kwargs."""
    name = cls.__name__
    kwargs = _CONSTRUCTOR_KWARGS.get(name, {"prob": 1.0})
    try:
        return cls(**kwargs)
    except TypeError:
        return cls(p=1.0)


@pytest.fixture
def adapter() -> AlbumentationsAdapter:
    return AlbumentationsAdapter()


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestRegistryCoverage:
    """Every entry in TRANSFORM_REGISTRY passes the adapter contract."""

    @pytest.mark.parametrize(
        "transform_cls,expected_cat",
        [pytest.param(cls, cat, id=cls.__name__) for cls, cat in TRANSFORM_REGISTRY.items()],
    )
    def test_category_matches_registry(self, adapter, transform_cls, expected_cat):
        """adapter.category(instance) returns the registered TransformCategory."""
        instance = _make_instance(transform_cls)
        cat = adapter.category(instance)
        assert cat == expected_cat, f"category({transform_cls.__name__}) returned {cat}, expected {expected_cat}"

    @pytest.mark.parametrize(
        "transform_cls",
        [pytest.param(cls, id=cls.__name__) for cls in TRANSFORM_REGISTRY],
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
        [pytest.param(cls, id=cls.__name__) for cls in TRANSFORM_REGISTRY],
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
        [pytest.param(cls, id=cls.__name__) for cls in TRANSFORM_REGISTRY],
    )
    def test_build_matrix_no_nan_inf(self, adapter, transform_cls):
        """build_matrix output contains no NaN or Inf values."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEFAULT_DEVICE)
        mtx = adapter.build_matrix(instance, params, HEIGHT, WIDTH)
        assert not torch.isnan(mtx).any(), f"NaN in build_matrix for {transform_cls.__name__}"
        assert not torch.isinf(mtx).any(), f"Inf in build_matrix for {transform_cls.__name__}"


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestRegistryCompleteness:
    """Verify the registry is non-empty and covers the baseline transforms."""

    def test_registry_not_empty(self):
        """TRANSFORM_REGISTRY is non-empty, guarding against silent import failure."""
        assert len(TRANSFORM_REGISTRY) > 0, (
            "Albumentations TRANSFORM_REGISTRY is empty -- imports may have failed silently"
        )

    def test_registry_has_expected_minimum(self):
        """Registry should have at least the v0.2 baseline transforms."""
        names = {cls.__name__ for cls in TRANSFORM_REGISTRY}
        expected = {"Affine", "Rotate", "ShiftScaleRotate", "HorizontalFlip", "VerticalFlip"}
        missing = expected - names
        assert not missing, f"Registry missing baseline transforms: {missing}"
