"""Unit tests for AlbumentationsAdapter — no albumentations installation required.

Stub transforms are used to test protocol compliance, matrix shape, and flip dims
without importing the albumentations package.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations._types import TransformAdapter, TransformCategory


# ---------------------------------------------------------------------------
# TDD entry point — fails until adapters/_albumentations.py is created
# ---------------------------------------------------------------------------


def test_albumentations_adapter_exists_and_satisfies_protocol():
    """AlbumentationsAdapter can be imported and satisfies TransformAdapter protocol."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()
    assert isinstance(adapter, TransformAdapter)


def test_albumentations_adapter_exported_from_adapters_package():
    """AlbumentationsAdapter is importable from the adapters package."""
    from fuse_augmentations.adapters import AlbumentationsAdapter

    assert AlbumentationsAdapter is not None


# ---------------------------------------------------------------------------
# Stub helpers — no albumentations dependency
# ---------------------------------------------------------------------------


class _StubInterpTransform:
    """Stub geometric transform returning a known 3x3 matrix from get_params_dependent_on_data."""

    p = 1.0
    same_on_batch = False

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def get_params(self) -> dict:
        return {}

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        return {"matrix": self._matrix.copy()}


class _StubFlipTransform:
    """Stub flip transform with no sampled parameters."""

    p = 1.0
    same_on_batch = False

    def get_params(self) -> dict:
        return {}

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        return {}


class _StubAdapter:
    """Minimal adapter wrapping AlbumentationsAdapter with stub-friendly registry."""

    def __init__(self, registry: dict) -> None:
        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

        self._inner = AlbumentationsAdapter()
        self._registry = registry

    def category(self, transform: object) -> TransformCategory:
        return self._registry.get(type(transform), TransformCategory.SPATIAL_KERNEL)

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return self._inner.sample_params(transform, input_shape, device)

    def build_matrix(
        self,
        transform: object,
        params: dict[str, torch.Tensor],
        H: int,
        W: int,
    ) -> torch.Tensor:
        return self._inner.build_matrix(transform, params, H, W)

    def exact_flip_dims(self, transform: object) -> list[int]:
        return self._inner.exact_flip_dims(transform)


# ---------------------------------------------------------------------------
# build_matrix shape and value tests
# ---------------------------------------------------------------------------


def test_build_matrix_shape_from_known_matrix():
    """build_matrix returns (B, 3, 3) when params contains a 'matrix' key."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()
    B, H, W = 3, 64, 64

    # build_matrix reads params["matrix"] directly — construct it as sample_params would
    matrix_tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
    stub = _StubInterpTransform(np.eye(3))
    mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, H, W)

    assert mtx.shape == (B, 3, 3)


def test_build_matrix_identity_on_identity_matrix():
    """build_matrix with identity params['matrix'] produces identity (B, 3, 3)."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()
    B, H, W = 2, 32, 32

    matrix_tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
    stub = _StubInterpTransform(np.eye(3))
    mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, H, W)

    expected = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(mtx, expected, atol=1e-6)


def test_build_matrix_non_identity_matrix_preserved():
    """build_matrix propagates non-identity matrix values exactly."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()
    B, H, W = 1, 64, 64

    angle = np.deg2rad(30.0)
    known = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    matrix_tensor = torch.tensor(known, dtype=torch.float32).unsqueeze(0)
    stub = _StubInterpTransform(known)
    mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, H, W)

    assert torch.allclose(mtx[0], torch.tensor(known, dtype=torch.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# Flip dims tests
# ---------------------------------------------------------------------------


def test_exact_flip_dims_hflip():
    """exact_flip_dims returns [3] for horizontal flip."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()

    class _HFlipStub(_StubFlipTransform):
        pass

    registry = {_HFlipStub: TransformCategory.GEOMETRIC_EXACT}
    # Inject stub type as hflip
    from fuse_augmentations.adapters import _albumentations as _mod
    original = _mod._HFLIP_TYPES
    try:
        _mod._HFLIP_TYPES = frozenset({_HFlipStub})
        assert adapter.exact_flip_dims(_HFlipStub()) == [3]
    finally:
        _mod._HFLIP_TYPES = original


def test_exact_flip_dims_vflip():
    """exact_flip_dims returns [2] for vertical flip."""
    from fuse_augmentations.adapters import _albumentations as _mod
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()

    class _VFlipStub(_StubFlipTransform):
        pass

    original = _mod._VFLIP_TYPES
    try:
        _mod._VFLIP_TYPES = frozenset({_VFlipStub})
        assert adapter.exact_flip_dims(_VFlipStub()) == [2]
    finally:
        _mod._VFLIP_TYPES = original


def test_exact_flip_dims_unknown_raises():
    """exact_flip_dims raises TypeError for non-flip transforms."""
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()

    class _NotAFlip:
        pass

    with pytest.raises(TypeError):
        adapter.exact_flip_dims(_NotAFlip())


# ---------------------------------------------------------------------------
# sample_params batch size
# ---------------------------------------------------------------------------


def test_sample_params_matrix_batch_size():
    """sample_params stacks B matrices into (B, 3, 3) via the 'matrix' key."""
    from fuse_augmentations.adapters import _albumentations as _mod
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()
    B, H, W = 5, 48, 48

    class _InterpStub(_StubInterpTransform):
        pass

    original = _mod._INTERP_TYPES
    try:
        _mod._INTERP_TYPES = frozenset({_InterpStub})
        stub = _InterpStub(np.eye(3, dtype=np.float64))
        params = adapter.sample_params(stub, (B, 3, H, W), torch.device("cpu"))
        assert "matrix" in params
        assert params["matrix"].shape == (B, 3, 3)
    finally:
        _mod._INTERP_TYPES = original


def test_sample_params_flip_returns_batch_size_key():
    """sample_params for a flip stub returns _batch_size tensor."""
    from fuse_augmentations.adapters import _albumentations as _mod
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

    adapter = AlbumentationsAdapter()

    class _HFlipStub(_StubFlipTransform):
        pass

    original = _mod._HFLIP_TYPES
    try:
        _mod._HFLIP_TYPES = frozenset({_HFlipStub})
        params = adapter.sample_params(_HFlipStub(), (4, 3, 32, 32), torch.device("cpu"))
        assert "_batch_size" in params
        assert int(params["_batch_size"].item()) == 4
    finally:
        _mod._HFLIP_TYPES = original
