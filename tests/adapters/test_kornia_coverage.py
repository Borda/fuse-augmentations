"""Registry coverage tests for the Kornia adapter.

Parametrised over the live ``TRANSFORM_REGISTRY`` dict so that new entries
added by sw-engineer are automatically validated.

Each registered transform must satisfy:
- ``adapter.category(instance)`` returns a ``TransformCategory`` member
  matching the registry value.
- ``adapter.sample_params(instance, input_shape, device)`` returns a dict
  (possibly empty) with ``torch.Tensor`` values (or sentinel ``_batch_size``).
- ``adapter.build_matrix(instance, params, H, W)`` returns a ``(B, 3, 3)``
  ``torch.Tensor``.

"""

from __future__ import annotations

import pytest
import torch

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")

from fuse_augmentations.adapters._kornia import (  # noqa: E402
    TRANSFORM_REGISTRY,
    KorniaAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default constructor kwargs per transform class for instantiation.
# Extend this dict when new transforms are added to the registry.
_CONSTRUCTOR_KWARGS: dict[str, dict] = {
    "RandomRotation": {"degrees": 30, "p": 1.0},
    "RandomAffine": {"degrees": 30, "p": 1.0},
    "RandomHorizontalFlip": {"p": 1.0},
    "RandomVerticalFlip": {"p": 1.0},
    "RandomPerspective": {"distortion_scale": 0.3, "p": 1.0},
    # Future entries (Phase A.2):
    "RandomShear": {"shear": (10.0, 10.0), "p": 1.0},
    "RandomTranslate": {"translate": (0.3, 0.3), "p": 1.0},
    "RandomRotation90": {"times": (0, 3), "p": 1.0},
}

BSZ, C, H, W = 2, 3, 32, 32
DEVICE = torch.device("cpu")
INPUT_SHAPE = (BSZ, C, H, W)


def _make_instance(cls: type) -> object:
    """Instantiate a registered transform class with default kwargs."""
    name = cls.__name__
    kwargs = _CONSTRUCTOR_KWARGS.get(name, {"p": 1.0})
    try:
        return cls(**kwargs)
    except TypeError:
        # Fallback: try with just p=1.0
        return cls(p=1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> KorniaAdapter:
    return KorniaAdapter()


class TestRegistryCoverage:
    """Every entry in TRANSFORM_REGISTRY must pass category/sample_params/build_matrix contracts."""

    @pytest.mark.parametrize(
        "transform_cls,expected_cat",
        list(TRANSFORM_REGISTRY.items()),
        ids=[cls.__name__ for cls in TRANSFORM_REGISTRY],
    )
    def test_category_matches_registry(self, adapter, transform_cls, expected_cat):
        """adapter.category(instance) returns the registered TransformCategory."""
        instance = _make_instance(transform_cls)
        cat = adapter.category(instance)
        assert cat == expected_cat, f"category({transform_cls.__name__}) returned {cat}, expected {expected_cat}"

    @pytest.mark.parametrize(
        "transform_cls",
        list(TRANSFORM_REGISTRY),
        ids=[cls.__name__ for cls in TRANSFORM_REGISTRY],
    )
    def test_sample_params_returns_dict_of_tensors(self, adapter, transform_cls):
        """sample_params returns a dict with torch.Tensor values."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEVICE)
        assert isinstance(params, dict), f"Expected dict, got {type(params)}"
        for key, val in params.items():
            assert isinstance(val, torch.Tensor), f"params[{key!r}] is {type(val).__name__}, expected Tensor"

    @pytest.mark.parametrize(
        "transform_cls",
        list(TRANSFORM_REGISTRY),
        ids=[cls.__name__ for cls in TRANSFORM_REGISTRY],
    )
    def test_build_matrix_returns_b33_tensor(self, adapter, transform_cls):
        """build_matrix returns a (B, 3, 3) tensor."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEVICE)
        mtx = adapter.build_matrix(instance, params, H, W)
        assert isinstance(mtx, torch.Tensor), f"Expected Tensor, got {type(mtx)}"
        assert mtx.ndim == 3, f"Expected 3D tensor, got {mtx.ndim}D"
        assert mtx.shape[1:] == (3, 3), f"Expected (*, 3, 3), got {mtx.shape}"

    @pytest.mark.parametrize(
        "transform_cls",
        list(TRANSFORM_REGISTRY),
        ids=[cls.__name__ for cls in TRANSFORM_REGISTRY],
    )
    def test_build_matrix_no_nan_inf(self, adapter, transform_cls):
        """build_matrix output contains no NaN or Inf values."""
        instance = _make_instance(transform_cls)
        params = adapter.sample_params(instance, INPUT_SHAPE, DEVICE)
        mtx = adapter.build_matrix(instance, params, H, W)
        assert not torch.isnan(mtx).any(), f"NaN in build_matrix for {transform_cls.__name__}"
        assert not torch.isinf(mtx).any(), f"Inf in build_matrix for {transform_cls.__name__}"


class TestRegistryCompleteness:
    """Verify the registry is non-empty (guards against broken lazy import)."""

    def test_registry_not_empty(self):
        assert len(TRANSFORM_REGISTRY) > 0, "TRANSFORM_REGISTRY is empty -- kornia imports may have failed silently"

    def test_registry_has_expected_minimum(self):
        """Registry should have at least the v0.1 baseline transforms."""
        names = {cls.__name__ for cls in TRANSFORM_REGISTRY}
        expected = {"RandomRotation", "RandomAffine", "RandomHorizontalFlip", "RandomVerticalFlip"}
        missing = expected - names
        assert not missing, f"Registry missing baseline transforms: {missing}"
