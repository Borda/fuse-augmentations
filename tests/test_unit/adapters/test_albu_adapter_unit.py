"""Unit tests for AlbumentationsAdapter internals.

All tests require albumentations to be installed and are skipped automatically when it is absent. Stub transforms stand
in for real albumentations classes to isolate adapter logic from albumentations version changes.

"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.affine.matrix import hflip_matrix, vflip_matrix
from fuse_augmentations.types import TransformAdapter, TransformCategory

if _ALBUMENTATIONS_AVAILABLE:
    from fuse_augmentations.adapters import AlbumentationsAdapter
    from fuse_augmentations.adapters import albumentations as _mod
    from fuse_augmentations.adapters.albumentations import (
        _D4_ELEM_TO_CODE,
        _d4_matrix,
        hflip_matrix_np,
        vflip_matrix_np,
    )


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
def test_albumentations_adapter_exists_and_satisfies_protocol():
    """AlbumentationsAdapter can be imported and satisfies TransformAdapter protocol."""
    adapter = AlbumentationsAdapter()
    assert isinstance(adapter, TransformAdapter)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
def test_albumentations_adapter_exported_from_adapters_package():
    """AlbumentationsAdapter is importable from the adapters package."""
    assert AlbumentationsAdapter is not None


class _StubInterpTransform:
    """Stub geometric transform returning a known 3x3 matrix from get_params_dependent_on_data."""

    p = 1.0
    same_on_batch = False

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def get_params(self) -> dict:
        return {}

    def update_transform_params(self, params: dict, data: dict) -> dict:
        return params

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
        height: int,
        width: int,
    ) -> torch.Tensor:
        return self._inner.build_matrix(transform, params, height, width)

    def exact_flip_dims(self, transform: object) -> list[int]:
        return self._inner.exact_flip_dims(transform)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestBuildMatrix:
    """AlbumentationsAdapter.build_matrix shape, identity, value propagation, and Transpose."""

    def test_shape_from_known_matrix(self):
        """build_matrix returns (B, 3, 3) when params contains a 'matrix' key."""
        adapter = AlbumentationsAdapter()
        batch, height, width = 3, 64, 64

        # build_matrix reads params["matrix"] directly — construct it as sample_params would
        matrix_tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).clone()
        stub = _StubInterpTransform(np.eye(3))
        mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, height, width)

        assert mtx.shape == (batch, 3, 3)

    def test_identity_on_identity_matrix(self):
        """build_matrix with identity params['matrix'] produces identity (B, 3, 3)"""
        adapter = AlbumentationsAdapter()
        batch, height, width = 2, 32, 32

        matrix_tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).clone()
        stub = _StubInterpTransform(np.eye(3))
        mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, height, width)

        expected = torch.eye(3).unsqueeze(0).expand(batch, -1, -1)
        assert torch.allclose(mtx, expected, atol=1e-6)

    def test_non_identity_matrix_preserved(self):
        """build_matrix propagates non-identity matrix values exactly."""
        adapter = AlbumentationsAdapter()
        height, width = 64, 64

        angle = np.deg2rad(30.0)
        known = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        matrix_tensor = torch.tensor(known, dtype=torch.float32).unsqueeze(0)
        stub = _StubInterpTransform(known)
        mtx = adapter.build_matrix(stub, {"matrix": matrix_tensor}, height, width)

        assert torch.allclose(mtx[0], torch.tensor(known, dtype=torch.float32), atol=1e-5)

    def test_transpose_returns_swap_matrix_on_square_images(self):
        """Transpose produces the (x, y) -> (y, x) swap matrix on square inputs.

        Transpose is an exact-discrete D4 element that mixed fused segments rely on; verified here by patching the
        module-level type registries so the test does not depend on the installed Albumentations version exposing the
        Transpose class.

        """
        adapter = AlbumentationsAdapter()

        class _TransposeStub:
            pass

        params = {"_batch_size": torch.tensor([2], dtype=torch.int64)}
        with (
            patch.object(_mod, "_RandomRotate90", type("_RandomRotate90Stub", (), {}), create=True),
            patch.object(_mod, "_D4", type("_D4Stub", (), {}), create=True),
            patch.object(_mod, "_Transpose", _TransposeStub, create=True),
            patch.object(_mod, "_EXACT_DISCRETE_TYPES", frozenset({_TransposeStub})),
        ):
            mtx = adapter.build_matrix(_TransposeStub(), params, height=32, width=32)

        expected = (
            torch
            .tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(2, -1, -1)
        )
        assert torch.allclose(mtx, expected)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestExactFlipDims:
    """exact_flip_dims dispatch for hflip, vflip, and unknown transforms."""

    def test_hflip_returns_3(self):
        """exact_flip_dims returns [3] for horizontal flip."""
        adapter = AlbumentationsAdapter()

        class _HFlipStub(_StubFlipTransform):
            pass

        with patch.object(_mod, "_HFLIP_TYPES", frozenset({_HFlipStub})):
            assert adapter.exact_flip_dims(_HFlipStub()) == [3]

    def test_vflip_returns_2(self):
        """exact_flip_dims returns [2] for vertical flip."""
        adapter = AlbumentationsAdapter()

        class _VFlipStub(_StubFlipTransform):
            pass

        with patch.object(_mod, "_VFLIP_TYPES", frozenset({_VFlipStub})):
            assert adapter.exact_flip_dims(_VFlipStub()) == [2]

    def test_unknown_raises_type_error(self):
        """exact_flip_dims raises TypeError for non-flip transforms."""
        adapter = AlbumentationsAdapter()

        class _NotAFlip:
            pass

        with pytest.raises(TypeError):
            adapter.exact_flip_dims(_NotAFlip())


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestSampleParams:
    """AlbumentationsAdapter.sample_params for interp and flip transform types."""

    def test_matrix_batch_size(self):
        """sample_params stacks B matrices into (B, 3, 3) via the 'matrix' key."""
        adapter = AlbumentationsAdapter()
        batch, height, width = 5, 48, 48

        class _InterpStub(_StubInterpTransform):
            pass

        with patch.object(_mod, "_INTERP_TYPES", frozenset({_InterpStub})):
            stub = _InterpStub(np.eye(3, dtype=np.float64))
            params = adapter.sample_params(stub, (batch, 3, height, width), torch.device("cpu"))
            assert "matrix" in params
            assert params["matrix"].shape == (batch, 3, 3)

    def test_flip_returns_batch_size_key(self):
        """sample_params for a flip stub returns _batch_size tensor."""
        adapter = AlbumentationsAdapter()

        class _HFlipStub(_StubFlipTransform):
            pass

        with patch.object(_mod, "_HFLIP_TYPES", frozenset({_HFlipStub})):
            params = adapter.sample_params(_HFlipStub(), (4, 3, 32, 32), torch.device("cpu"))
            assert "_batch_size" in params
            assert int(params["_batch_size"].item()) == 4


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestHflipMatrixNp:
    """hflip_matrix_np shape, dtype, coordinate mapping, involutory, and torch consistency."""

    def test_shape(self):
        """hflip_matrix_np returns a (3, 3) matrix."""
        assert hflip_matrix_np(width=64).shape == (3, 3)

    def test_dtype(self):
        """hflip_matrix_np returns a float64 matrix."""
        assert hflip_matrix_np(width=64).dtype == np.float64

    def test_maps_x_correctly(self):
        """First row encodes x mirror: [-1, 0, W-1]"""
        width = 10
        mtx = hflip_matrix_np(width=width)
        assert mtx[0, 0] == -1.0
        assert mtx[0, 1] == 0.0
        assert mtx[0, 2] == float(width - 1)

    def test_maps_y_unchanged(self):
        """Second row leaves y unchanged: [0, 1, 0]"""
        mtx = hflip_matrix_np(width=10)
        assert mtx[1, 0] == 0.0
        assert mtx[1, 1] == 1.0
        assert mtx[1, 2] == 0.0

    def test_homogeneous_row(self):
        """Third row equals [0, 0, 1] for a homogeneous affine matrix."""
        mtx = hflip_matrix_np(width=10)
        np.testing.assert_array_equal(mtx[2], [0.0, 0.0, 1.0])

    @pytest.mark.parametrize("width", [1, 4, 64, 256])
    def test_involutory(self, width):
        """Hflip applied twice should be identity."""
        mtx = hflip_matrix_np(width=width)
        assert np.allclose(mtx @ mtx, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("width", [4, 64])
    def test_pixel_transform(self, width):
        """Pixel (0, y, 1) maps to (W-1, y, 1)"""
        mtx = hflip_matrix_np(width=width)
        point = np.array([0.0, 3.0, 1.0])
        result = mtx @ point
        assert result[0] == pytest.approx(width - 1)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(1.0)

    def test_consistent_with_torch_hflip_matrix(self):
        """Matches the torch hflip_matrix from affine._matrix."""
        width = 32
        torch_mat = hflip_matrix(width=width, batch_size=1, device=torch.device("cpu"), dtype=torch.float64)
        np_mat = hflip_matrix_np(width=width)
        torch.testing.assert_close(torch.as_tensor(np_mat.copy()), torch_mat[0], rtol=1e-4, atol=1e-10)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestVflipMatrixNp:
    """vflip_matrix_np shape, dtype, coordinate mapping, involutory, and torch consistency."""

    def test_shape(self):
        """vflip_matrix_np returns a (3, 3) matrix."""
        assert vflip_matrix_np(height=64).shape == (3, 3)

    def test_dtype(self):
        """vflip_matrix_np returns a float64 matrix."""
        assert vflip_matrix_np(height=64).dtype == np.float64

    def test_maps_y_correctly(self):
        """Second row encodes y mirror: [0, -1, H-1]"""
        height = 10
        mtx = vflip_matrix_np(height=height)
        assert mtx[1, 0] == 0.0
        assert mtx[1, 1] == -1.0
        assert mtx[1, 2] == float(height - 1)

    def test_maps_x_unchanged(self):
        """First row leaves x unchanged: [1, 0, 0]"""
        mtx = vflip_matrix_np(height=10)
        assert mtx[0, 0] == 1.0
        assert mtx[0, 1] == 0.0
        assert mtx[0, 2] == 0.0

    def test_homogeneous_row(self):
        """Third row equals [0, 0, 1] for a homogeneous affine matrix."""
        mtx = vflip_matrix_np(height=10)
        np.testing.assert_array_equal(mtx[2], [0.0, 0.0, 1.0])

    @pytest.mark.parametrize("height", [1, 4, 64, 256])
    def test_involutory(self, height):
        """Vflip applied twice should be identity."""
        mtx = vflip_matrix_np(height=height)
        assert np.allclose(mtx @ mtx, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("height", [4, 64])
    def test_pixel_transform(self, height):
        """Pixel (x, 0, 1) maps to (x, H-1, 1)"""
        mtx = vflip_matrix_np(height=height)
        point = np.array([5.0, 0.0, 1.0])
        result = mtx @ point
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(height - 1)
        assert result[2] == pytest.approx(1.0)

    def test_consistent_with_torch_vflip_matrix(self):
        """Matches the torch vflip_matrix from affine._matrix."""
        height = 32
        torch_mat = vflip_matrix(height=height, batch_size=1, device=torch.device("cpu"), dtype=torch.float64)
        np_mat = vflip_matrix_np(height=height)
        torch.testing.assert_close(torch.as_tensor(np_mat.copy()), torch_mat[0], rtol=1e-4, atol=1e-10)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
@pytest.mark.parametrize(
    ("elem", "inv_elem"),
    [
        ("e", "e"),
        ("r90", "r270"),
        ("r180", "r180"),
        ("r270", "r90"),
        ("h", "h"),
        ("v", "v"),
        ("t", "t"),
        ("hvt", "hvt"),
    ],
)
def test_d4_matrix_composition_with_inverse_is_identity(elem: str, inv_elem: str) -> None:
    """M[elem] @ M[inv_elem] == I for all D4 elements on square images.

    Confirms that the D4 group inverses encoded in ``_D4_ELEM_TO_CODE`` are correctly composed into identity by
    ``_d4_matrix``; covers self-inverse elements (e, r180, h, v, t, hvt) and rotation pairs (r90/r270).

    """
    height = width = 8
    device = torch.device("cpu")
    dtype = torch.float32
    code = torch.tensor([_D4_ELEM_TO_CODE[elem]], dtype=torch.int64)
    inv_code = torch.tensor([_D4_ELEM_TO_CODE[inv_elem]], dtype=torch.int64)
    mtx = _d4_matrix(code, height=height, width=width, device=device, dtype=dtype)[0]
    mtx_inv = _d4_matrix(inv_code, height=height, width=width, device=device, dtype=dtype)[0]
    product = mtx @ mtx_inv
    assert torch.allclose(product, torch.eye(3, dtype=dtype), atol=1e-5), (
        f"D4: {elem!r} @ {inv_elem!r} != I (max diff {(product - torch.eye(3)).abs().max():.2e})"
    )


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
@pytest.mark.parametrize("elem", ["r90", "r270", "hvt"])
def test_d4_matrix_raises_on_nonsquare_for_shape_changing_elements(elem: str) -> None:
    """_d4_matrix raises RuntimeError for shape-changing D4 elements on non-square images."""
    code = torch.tensor([_D4_ELEM_TO_CODE[elem]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="changes spatial dimensions"):
        _d4_matrix(code, height=8, width=12, device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestIsAlbuInstanceSubclassDispatch:
    """Verify that isinstance-based dispatch correctly routes subclasses of registered base types.

    The core semantic guarantee: category/sample_params/build_matrix/exact_flip_dims all
    use _is_albu_instance which calls isinstance(), so a subclass of a registered base type
    is routed to the same path as the base type itself.

    """

    def test_subclass_of_hflip_is_matched(self):
        """A subclass of a registered HFlip base type routes to the hflip path."""

        class _BaseHFlip(_StubFlipTransform):
            pass

        class _SubHFlip(_BaseHFlip):
            pass

        with patch.object(_mod, "_HFLIP_TYPES", frozenset({_BaseHFlip})):
            assert _mod._is_albu_instance(_SubHFlip(), _mod._HFLIP_TYPES) is True

    def test_sibling_class_is_not_matched(self):
        """A sibling (unrelated) class does not match the flip frozenset."""

        class _BaseHFlip(_StubFlipTransform):
            pass

        class _SiblingFlip(_StubFlipTransform):
            pass

        with patch.object(_mod, "_HFLIP_TYPES", frozenset({_BaseHFlip})):
            assert _mod._is_albu_instance(_SiblingFlip(), _mod._HFLIP_TYPES) is False

    def test_exact_flip_dims_with_subclass(self):
        """exact_flip_dims returns correct dims for a subclass of a registered flip type."""
        adapter = AlbumentationsAdapter()

        class _BaseHFlip(_StubFlipTransform):
            pass

        class _SubHFlip(_BaseHFlip):
            pass

        with patch.object(_mod, "_HFLIP_TYPES", frozenset({_BaseHFlip})):
            assert adapter.exact_flip_dims(_SubHFlip()) == [3]
