"""Unit tests for TorchVisionAdapter -- no torchvision installation required.

Stub transforms are used to test protocol compliance, matrix shape, flip dims, and parameter sampling without importing
the torchvision package.

"""

from __future__ import annotations

import math

import pytest
import torch

from fuse_augmentations._types import TransformAdapter, TransformCategory

# ---------------------------------------------------------------------------
# Stub transforms -- no torchvision dependency
# ---------------------------------------------------------------------------


class _StubRotation:
    degrees = (-30.0, 30.0)
    expand = False

    @staticmethod
    def get_params(degrees):
        return 15.0  # fixed for determinism


class _StubAffine:
    degrees = (-30.0, 30.0)
    translate = (0.1, 0.1)
    scale = (0.8, 1.2)
    shear = (-10.0, 10.0, -10.0, 10.0)
    expand = False

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        return 10.0, (5, -3), 0.9, (5.0, 0.0)


class _StubHFlip:
    p = 0.5
    expand = False


class _StubVFlip:
    p = 0.5
    expand = False


class _StubV2Rotation:
    __module__ = "torchvision.transforms.v2._geometry"

    degrees = (-30.0, 30.0)
    expand = False
    _v1_transform_cls = _StubRotation

    @staticmethod
    def get_params(degrees):
        return 15.0


class _StubV2Affine:
    __module__ = "torchvision.transforms.v2._geometry"

    degrees = (-30.0, 30.0)
    translate = (0.1, 0.1)
    scale = (0.8, 1.2)
    shear = (-10.0, 10.0, -10.0, 10.0)
    expand = False
    _v1_transform_cls = _StubAffine

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        return 10.0, (5, -3), 0.9, (5.0, 0.0)


class _StubBatchTransform:
    __module__ = "torchvision.transforms.v2._color"

    _v1_transform_cls = object

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(image.shape))
        return image + 1


class _StubPerSampleTransform:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(image.shape))
        return image + 1


class _StubExpandTrue:
    expand = True
    degrees = (-30.0, 30.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _register_stubs(monkeypatch):
    """Register stub transforms in TorchVisionAdapter's type sets via monkeypatch."""
    from fuse_augmentations.adapters import _torchvision as _mod

    monkeypatch.setattr(
        _mod,
        "_TRANSFORM_REGISTRY",
        {
            _StubRotation: TransformCategory.GEOMETRIC_INTERP,
            _StubAffine: TransformCategory.GEOMETRIC_INTERP,
            _StubV2Rotation: TransformCategory.GEOMETRIC_INTERP,
            _StubV2Affine: TransformCategory.GEOMETRIC_INTERP,
            _StubHFlip: TransformCategory.GEOMETRIC_EXACT,
            _StubVFlip: TransformCategory.GEOMETRIC_EXACT,
        },
    )
    monkeypatch.setattr(_mod, "_HFLIP_TYPES_FS", frozenset({_StubHFlip}))
    monkeypatch.setattr(_mod, "_VFLIP_TYPES_FS", frozenset({_StubVFlip}))
    monkeypatch.setattr(_mod, "_ROTATION_TYPES_FS", frozenset({_StubRotation, _StubV2Rotation}))
    monkeypatch.setattr(_mod, "_AFFINE_TYPES_FS", frozenset({_StubAffine, _StubV2Affine}))


@pytest.fixture
def adapter():
    from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

    return TorchVisionAdapter()


# ---------------------------------------------------------------------------
# TDD entry point -- exists and satisfies protocol
# ---------------------------------------------------------------------------


def test_torchvision_adapter_exists_and_satisfies_protocol():
    """TorchVisionAdapter can be imported and satisfies TransformAdapter protocol."""
    from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

    adapter = TorchVisionAdapter()
    assert isinstance(adapter, TransformAdapter)


def test_torchvision_adapter_exported_from_adapters_package():
    """TorchVisionAdapter is importable from the adapters package."""
    from fuse_augmentations.adapters import TorchVisionAdapter

    assert TorchVisionAdapter is not None


# ---------------------------------------------------------------------------
# sample_params
# ---------------------------------------------------------------------------


class TestSampleParamsRotation:
    """sample_params for rotation stub."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_returns_angle_rad_key(self, adapter):
        params = adapter.sample_params(_StubRotation(), (2, 3, 64, 64), torch.device("cpu"))
        assert "angle_rad" in params

    @pytest.mark.usefixtures("_register_stubs")
    def test_angle_rad_shape(self, adapter):
        B = 4
        params = adapter.sample_params(_StubRotation(), (B, 3, 64, 64), torch.device("cpu"))
        assert params["angle_rad"].shape == (B,)

    @pytest.mark.usefixtures("_register_stubs")
    def test_angle_rad_dtype(self, adapter):
        params = adapter.sample_params(_StubRotation(), (2, 3, 64, 64), torch.device("cpu"))
        assert params["angle_rad"].dtype == torch.float32

    @pytest.mark.usefixtures("_register_stubs")
    def test_angle_rad_value(self, adapter):
        """Stub returns 15 degrees -> radians."""
        params = adapter.sample_params(_StubRotation(), (1, 3, 64, 64), torch.device("cpu"))
        expected = math.radians(15.0)
        assert params["angle_rad"].item() == pytest.approx(expected, abs=1e-6)


class TestSampleParamsAffine:
    """sample_params for affine stub."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_returns_all_six_keys(self, adapter):
        params = adapter.sample_params(_StubAffine(), (2, 3, 64, 64), torch.device("cpu"))
        expected_keys = {"angle_rad", "translate_x", "translate_y", "scale", "shear_x_rad", "shear_y_rad"}
        assert expected_keys == set(params.keys())

    @pytest.mark.usefixtures("_register_stubs")
    @pytest.mark.parametrize("key", ["angle_rad", "translate_x", "translate_y", "scale", "shear_x_rad", "shear_y_rad"])
    def test_affine_param_shapes(self, adapter, key):
        B = 3
        params = adapter.sample_params(_StubAffine(), (B, 3, 64, 64), torch.device("cpu"))
        assert params[key].shape == (B,)

    @pytest.mark.usefixtures("_register_stubs")
    def test_affine_values_from_stub(self, adapter):
        """Stub returns angle=10, translate=(5,-3), scale=0.9, shear=(5.0, 0.0)."""
        params = adapter.sample_params(_StubAffine(), (1, 3, 64, 64), torch.device("cpu"))
        assert params["angle_rad"].item() == pytest.approx(math.radians(10.0), abs=1e-6)
        assert params["translate_x"].item() == pytest.approx(5.0, abs=1e-6)
        assert params["translate_y"].item() == pytest.approx(-3.0, abs=1e-6)
        assert params["scale"].item() == pytest.approx(0.9, abs=1e-6)
        assert params["shear_x_rad"].item() == pytest.approx(math.radians(5.0), abs=1e-6)
        assert params["shear_y_rad"].item() == pytest.approx(math.radians(0.0), abs=1e-6)

    @pytest.mark.usefixtures("_register_stubs")
    def test_v2_affine_samples_one_param_set_per_batch_tensor(self, adapter):
        params = adapter.sample_params(_StubV2Affine(), (3, 3, 64, 64), torch.device("cpu"))
        for key in ["angle_rad", "translate_x", "translate_y", "scale", "shear_x_rad", "shear_y_rad"]:
            assert params[key].shape == (1,)


class TestSampleParamsFlips:
    """sample_params for flip stubs."""

    @pytest.mark.usefixtures("_register_stubs")
    @pytest.mark.parametrize("stub_cls", [_StubHFlip, _StubVFlip], ids=["hflip", "vflip"])
    def test_flip_returns_batch_size_key(self, adapter, stub_cls):
        params = adapter.sample_params(stub_cls(), (4, 3, 32, 32), torch.device("cpu"))
        assert "_batch_size" in params
        assert int(params["_batch_size"].item()) == 4


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------


class TestBuildMatrixRotation:
    """build_matrix for rotation stub."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_shape(self, adapter):
        B, H, W = 3, 64, 64
        params = adapter.sample_params(_StubRotation(), (B, 3, H, W), torch.device("cpu"))
        mtx = adapter.build_matrix(_StubRotation(), params, H, W)
        assert mtx.shape == (B, 3, 3)


class TestBuildMatrixAffine:
    """build_matrix for affine stub."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_shape(self, adapter):
        B, H, W = 2, 64, 64
        params = adapter.sample_params(_StubAffine(), (B, 3, H, W), torch.device("cpu"))
        mtx = adapter.build_matrix(_StubAffine(), params, H, W)
        assert mtx.shape == (B, 3, 3)


class TestBuildMatrixIdentity:
    """build_matrix with identity params produces identity matrix."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_identity_affine_params(self, adapter):
        """Angle=0, scale=1, shear=0, translate=0 -> identity."""
        B, H, W = 2, 64, 64
        params = {
            "angle_rad": torch.zeros(B),
            "translate_x": torch.zeros(B),
            "translate_y": torch.zeros(B),
            "scale": torch.ones(B),
            "shear_x_rad": torch.zeros(B),
            "shear_y_rad": torch.zeros(B),
        }
        mtx = adapter.build_matrix(_StubAffine(), params, H, W)
        expected = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(mtx, expected, atol=1e-5), (
            f"Expected identity, max diff: {(mtx - expected).abs().max().item():.2e}"
        )


class TestBuildMatrixHFlip:
    """build_matrix for horizontal flip."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_hflip_first_row(self, adapter):
        """Hflip matrix first row is [-1, 0, W-1]."""
        B, H, W = 2, 32, 64
        params = {"_batch_size": torch.tensor([B], dtype=torch.int64)}
        mtx = adapter.build_matrix(_StubHFlip(), params, H, W)
        for i in range(B):
            assert mtx[i, 0, 0].item() == pytest.approx(-1.0)
            assert mtx[i, 0, 1].item() == pytest.approx(0.0)
            assert mtx[i, 0, 2].item() == pytest.approx(float(W - 1))


class TestBuildMatrixVFlip:
    """build_matrix for vertical flip."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_vflip_second_row(self, adapter):
        """Vflip matrix second row is [0, -1, H-1]."""
        B, H, W = 2, 64, 32
        params = {"_batch_size": torch.tensor([B], dtype=torch.int64)}
        mtx = adapter.build_matrix(_StubVFlip(), params, H, W)
        for i in range(B):
            assert mtx[i, 1, 0].item() == pytest.approx(0.0)
            assert mtx[i, 1, 1].item() == pytest.approx(-1.0)
            assert mtx[i, 1, 2].item() == pytest.approx(float(H - 1))


# ---------------------------------------------------------------------------
# exact_flip_dims
# ---------------------------------------------------------------------------


class TestExactFlipDims:
    """exact_flip_dims for flip stubs."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_hflip_returns_3(self, adapter):
        assert adapter.exact_flip_dims(_StubHFlip()) == [3]

    @pytest.mark.usefixtures("_register_stubs")
    def test_vflip_returns_2(self, adapter):
        assert adapter.exact_flip_dims(_StubVFlip()) == [2]

    @pytest.mark.usefixtures("_register_stubs")
    def test_unknown_raises_type_error(self, adapter):
        with pytest.raises(TypeError, match="Cannot determine flip dims"):
            adapter.exact_flip_dims(_StubRotation())


# ---------------------------------------------------------------------------
# expand=True guard
# ---------------------------------------------------------------------------


class TestExpandGuard:
    """Expand=True raises ValueError in sample_params and category."""

    @pytest.mark.usefixtures("_register_stubs")
    def test_expand_true_in_category(self, adapter):
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.category(_StubExpandTrue())

    @pytest.mark.usefixtures("_register_stubs")
    def test_expand_true_in_sample_params(self, adapter):
        with pytest.raises(ValueError, match="expand=True is not supported"):
            adapter.sample_params(_StubExpandTrue(), (2, 3, 64, 64), torch.device("cpu"))


class TestV2BatchSemantics:
    @pytest.mark.usefixtures("_register_stubs")
    def test_v2_rotation_samples_one_angle_per_batch_tensor(self, adapter):
        params = adapter.sample_params(_StubV2Rotation(), (4, 3, 64, 64), torch.device("cpu"))
        assert params["angle_rad"].shape == (1,)

    @pytest.mark.usefixtures("_register_stubs")
    def test_same_on_batch_true_for_v2_transform(self, adapter):
        assert adapter.same_on_batch(_StubV2Rotation()) is True


class TestCallNonfused:
    def test_v1_passthrough_loops_per_sample(self, adapter):
        transform = _StubPerSampleTransform()
        image = torch.zeros(3, 2, 4, 5)

        out = adapter.call_nonfused(transform, image)

        torch.testing.assert_close(out, image + 1)
        assert transform.calls == [(2, 4, 5)] * 3

    def test_v2_passthrough_runs_once_on_whole_batch(self, adapter):
        transform = _StubBatchTransform()
        image = torch.zeros(3, 2, 4, 5)

        out = adapter.call_nonfused(transform, image)

        torch.testing.assert_close(out, image + 1)
        assert transform.calls == [(3, 2, 4, 5)]

    def test_empty_batch_returns_input_without_calling_transform(self, adapter):
        transform = _StubBatchTransform()
        image = torch.zeros(0, 2, 4, 5)

        out = adapter.call_nonfused(transform, image)

        assert out is image
        assert transform.calls == []
