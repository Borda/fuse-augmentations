"""Comprehensive tests for from_params(specs=...) keyword-only overload.

Tests cover the specs path, mutual exclusivity with keyword args, per-transform probability, regression for existing
callers, and edge cases.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose, FusedCompose, TransformSpec


class TestFromParamsSpecsBasic:
    """from_params(specs=[...]) basic functionality."""

    def test_backend_free_single_spec(self, image64x64_batch2):
        """Backend-free rotation spec produces valid (2, 3, 64, 64) output."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image64x64_batch2)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_multiple_specs_different_ops(self, image64x64_batch2):
        """Rotation + hflip specs produce valid output shape."""
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0),
            TransformSpec(operation="hflip", params={}, prob=0.5),
        ]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image64x64_batch2)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_specs_with_hflip_and_vflip(self, image32x32_batch2):
        """Hflip + vflip specs preserve input shape."""
        specs = [
            TransformSpec(operation="hflip", params={}, prob=0.5),
            TransformSpec(operation="vflip", params={}, prob=0.5),
        ]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_specs_with_scale(self, image32x32_batch2):
        """Scale spec preserves input shape."""
        specs = [TransformSpec(operation="scale", params={"factor": (0.8, 1.2)}, prob=1.0)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_specs_with_shear(self, image32x32_batch2):
        """Shear spec preserves input shape."""
        specs = [TransformSpec(operation="shear_x", params={"degrees": (-10.0, 10.0)}, prob=1.0)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_returns_fused_compose(self):
        """from_params(specs=[...]) returns a FusedCompose instance."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_params(specs=specs)
        assert isinstance(pipe, FusedCompose), f"Expected FusedCompose, got {type(pipe).__name__}"

    def test_rotation_spec_with_degrees_param(self, image32x32_batch2):
        """Verify that op='rotation' with params={'degrees': (-30, 30)} is accepted."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape


class TestFromParamsSpecsEmpty:
    """from_params(specs=[]) empty list produces identity pipeline."""

    def test_empty_specs_identity_output(self, image32x32_batch2):
        """Empty specs list produces exact identity output."""
        pipe = Compose.from_params(specs=[])
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape
        torch.testing.assert_close(out, image32x32_batch2, atol=1e-5, rtol=1e-5)

    def test_empty_specs_returns_fused_compose(self):
        """Empty specs list returns FusedCompose instance."""
        pipe = Compose.from_params(specs=[])
        assert isinstance(pipe, FusedCompose)


class TestFromParamsSpecsNone:
    """from_params(specs=None) falls back to keyword-only behavior."""

    def test_specs_none_with_rotation(self, image32x32_batch2):
        """Specs=None with rotation keyword preserves shape."""
        pipe = Compose.from_params(specs=None, rotation=(-30.0, 30.0))
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_specs_none_with_hflip(self, image16x16_batch1):
        """Specs=None with hflip_p=1.0 produces horizontal flip."""
        pipe = Compose.from_params(specs=None, hflip_p=1.0)
        out = pipe(image16x16_batch1)
        expected = image16x16_batch1.flip(dims=[-1])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_specs_none_no_kwargs_identity(self, image16x16_batch2):
        """Specs=None with no other kwargs produces identity."""
        pipe = Compose.from_params(specs=None)
        out = pipe(image16x16_batch2)
        torch.testing.assert_close(out, image16x16_batch2, atol=1e-5, rtol=1e-5)


class TestFromParamsSpecsMutualExclusion:
    """Specs and keyword geometric args are mutually exclusive."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"rotation": (-30.0, 30.0)}, id="rotation"),
            pytest.param({"hflip_p": 0.5}, id="hflip_p"),
            pytest.param({"vflip_p": 0.3}, id="vflip_p"),
            pytest.param({"scale": (0.8, 1.2)}, id="scale"),
            pytest.param({"shear_x": (-5.0, 5.0)}, id="shear_x"),
            pytest.param({"translate_x": (-10.0, 10.0)}, id="translate_x"),
        ],
    )
    def test_specs_with_kwarg_raises_value_error(self, kwargs):
        """Passing specs= together with any geometric keyword raises ValueError."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError, match="mutually exclusive"):
            Compose.from_params(specs=specs, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"hflip_p": -0.1}, id="hflip_p_negative"),
            pytest.param({"vflip_p": -0.1}, id="vflip_p_negative"),
        ],
    )
    def test_specs_with_non_default_flip_kwarg_still_raises_value_error(self, kwargs):
        """Non-default flip_p kwargs with specs= still raise ValueError."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError, match="mutually exclusive"):
            Compose.from_params(specs=specs, **kwargs)


class TestFromParamsSpecsProbability:
    """Per-transform probability in specs path."""

    def test_p_zero_never_applied(self, image64x64_batch2):
        """p=0.0 transform is never applied; output matches input."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-90.0, 90.0)}, prob=0.0)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image64x64_batch2)
        assert torch.allclose(out, image64x64_batch2, atol=1e-5), "p=0.0 transform should never be applied"

    def test_p_one_always_applied(self, image32x32_batch2):
        """p=1.0 hflip always flips; output matches torch.flip reference."""
        specs = [TransformSpec(operation="hflip", params={}, prob=1.0)]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        expected = image32x32_batch2.flip(dims=[3])
        assert torch.allclose(out, expected, atol=1e-5), "p=1.0 hflip must always flip"

    def test_mixed_probabilities_shape_preserved(self, image32x32_batch4):
        """Mixed-probability pipeline preserves (4, 3, 32, 32) output shape."""
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.5),
            TransformSpec(operation="hflip", params={}, prob=0.3),
            TransformSpec(operation="vflip", params={}, prob=0.7),
        ]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch4)
        assert out.shape == image32x32_batch4.shape

    def test_all_p_zero_is_identity(self, image32x32_batch2):
        """All-p=0.0 pipeline produces exact identity output."""
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-90.0, 90.0)}, prob=0.0),
            TransformSpec(operation="hflip", params={}, prob=0.0),
        ]
        pipe = Compose.from_params(specs=specs)
        out = pipe(image32x32_batch2)
        assert torch.allclose(out, image32x32_batch2, atol=1e-5), "All p=0.0 should produce identity"


class TestFromParamsSpecsRegression:
    """Existing from_params keyword-only API still works (regression guard)."""

    def test_rotation_keyword(self, image32x32_batch2):
        """Rotation= keyword still works after specs= overload was added."""
        pipe = Compose.from_params(rotation=(-30.0, 30.0))
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_hflip_keyword(self, image16x16_batch1):
        """hflip_p= keyword still works; p=1.0 flips correctly."""
        pipe = Compose.from_params(hflip_p=1.0)
        out = pipe(image16x16_batch1)
        expected = image16x16_batch1.flip(dims=[-1])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_scale_keyword(self, image16x16_batch2):
        """Scale= keyword still works; output shape preserved."""
        pipe = Compose.from_params(scale=(0.8, 1.2))
        out = pipe(image16x16_batch2)
        assert out.shape == image16x16_batch2.shape

    def test_combined_keywords(self, image32x32_batch2):
        """Rotation + hflip_p + scale keywords combined preserve shape."""
        pipe = Compose.from_params(rotation=(-15.0, 15.0), hflip_p=0.5, scale=(0.9, 1.1))
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape

    def test_all_none_identity(self, image16x16_batch2):
        """from_params() with no args produces identity output."""
        pipe = Compose.from_params()
        out = pipe(image16x16_batch2)
        torch.testing.assert_close(out, image16x16_batch2, atol=1e-5, rtol=1e-5)

    def test_data_keys_still_work(self):
        """data_keys= kwarg still works after specs= overload was added."""
        pipe = Compose.from_params(
            rotation=(-15.0, 15.0),
            data_keys=["input", "mask"],
        )
        img = torch.rand(2, 3, 16, 16)
        mask = torch.randint(0, 3, (2, 1, 16, 16)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple)
        assert len(out) == 2


class TestFromParamsSpecsShapeEquivalence:
    """Specs pipeline produces same shape as equivalent keyword call."""

    def test_rotation_shape_matches(self, image32x32_batch2):
        """Rotation spec and rotation keyword produce identical output shape."""
        pipe_kw = Compose.from_params(rotation=(-30.0, 30.0))
        pipe_spec = Compose.from_params(
            specs=[TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0)]
        )
        out_kw = pipe_kw(image32x32_batch2)
        out_spec = pipe_spec(image32x32_batch2)
        assert out_kw.shape == out_spec.shape, f"Shape mismatch: keyword={out_kw.shape}, specs={out_spec.shape}"

    def test_hflip_shape_matches(self, image32x32_batch2):
        """Hflip spec and hflip_p keyword produce identical output shape."""
        pipe_kw = Compose.from_params(hflip_p=0.5)
        pipe_spec = Compose.from_params(specs=[TransformSpec(operation="hflip", params={}, prob=0.5)])
        out_kw = pipe_kw(image32x32_batch2)
        out_spec = pipe_spec(image32x32_batch2)
        assert out_kw.shape == out_spec.shape


class TestFromParamsSpecsValidation:
    """Input validation for specs-based parameter extraction."""

    def test_missing_rotation_range_raises_value_error(self):
        """Missing 'degrees' key in rotation spec raises ValueError."""
        with pytest.raises(ValueError, match="Missing required range"):
            Compose.from_params(specs=[TransformSpec(operation="rotation", params={})])

    def test_invalid_rotation_range_type_raises_value_error(self):
        """List instead of tuple for 'degrees' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range"):
            Compose.from_params(specs=[TransformSpec(operation="rotation", params={"degrees": [-30.0, 30.0]})])


class TestFromParamsSpecsUnsupportedOp:
    """from_params(specs=...) with ops not in the backend-free op set."""

    @pytest.mark.parametrize(
        "op, params",
        [
            pytest.param("perspective", {"distortion_scale": 0.5}, id="perspective"),
            pytest.param("affine", {}, id="affine"),
        ],
    )
    def test_unsupported_op_raises(self, op, params) -> None:
        """Ops valid in from_config but unsupported in backend-free from_params raise ValueError."""
        specs = [TransformSpec(operation=op, params=params, prob=1.0)]
        with pytest.raises(ValueError, match="Unsupported op for from_params"):
            Compose.from_params(specs=specs)


class TestFromParamsSpecsOrderingWithReservedParams:
    """Brightness/contrast keyword args remain mutually exclusive with specs."""

    @pytest.mark.parametrize(
        "reserved_param, value",
        [
            pytest.param("brightness", 0.5, id="brightness"),
            pytest.param("contrast", 0.3, id="contrast"),
        ],
    )
    def test_reserved_param_with_specs_raises(self, reserved_param, value) -> None:
        """A color keyword and declarative specs cannot be combined."""
        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError, match="mutually exclusive"):
            Compose.from_params(specs=specs, **{reserved_param: value})
