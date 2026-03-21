"""Comprehensive tests for from_params(specs=...) keyword-only overload (Phase C.4 + C.5).

Tests cover the specs path, mutual exclusivity with keyword args,
per-transform probability, regression for existing callers, and edge cases.
"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose, FusedCompose


class TestFromParamsSpecsBasic:
    """from_params(specs=[...]) basic functionality."""

    def test_backend_free_single_spec(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=1.0)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_multiple_specs_different_ops(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=1.0),
            TransformSpec(op="hflip", params={}, p=0.5),
        ]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_specs_with_hflip_and_vflip(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="hflip", params={}, p=0.5),
            TransformSpec(op="vflip", params={}, p=0.5),
        ]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_specs_with_scale(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="scale", params={"factor": (0.8, 1.2)}, p=1.0)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_specs_with_shear(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="shear_x", params={"degrees": (-10.0, 10.0)}, p=1.0)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_returns_fused_compose(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_params(specs=specs)
        assert isinstance(pipe, FusedCompose), f"Expected FusedCompose, got {type(pipe).__name__}"

    def test_rotation_spec_with_degrees_param(self):
        """Verify that op='rotation' with params={'degrees': (-30, 30)} is accepted."""
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape


class TestFromParamsSpecsEmpty:
    """from_params(specs=[]) empty list produces identity pipeline."""

    def test_empty_specs_identity_output(self):
        pipe = Compose.from_params(specs=[])
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_empty_specs_returns_fused_compose(self):
        pipe = Compose.from_params(specs=[])
        assert isinstance(pipe, FusedCompose)


class TestFromParamsSpecsNone:
    """from_params(specs=None) falls back to keyword-only behavior."""

    def test_specs_none_with_rotation(self):
        pipe = Compose.from_params(specs=None, rotation=(-30.0, 30.0))
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_specs_none_with_hflip(self):
        pipe = Compose.from_params(specs=None, hflip_p=1.0)
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        expected = x.flip(dims=[-1])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_specs_none_no_kwargs_identity(self):
        pipe = Compose.from_params(specs=None)
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


class TestFromParamsSpecsMutualExclusion:
    """specs and keyword geometric args are mutually exclusive."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rotation": (-30.0, 30.0)},
            {"hflip_p": 0.5},
            {"vflip_p": 0.3},
            {"scale": (0.8, 1.2)},
            {"shear_x": (-5.0, 5.0)},
            {"translate_x": (-10.0, 10.0)},
        ],
        ids=["rotation", "hflip_p", "vflip_p", "scale", "shear_x", "translate_x"],
    )
    def test_specs_with_kwarg_raises_value_error(self, kwargs):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError, match="mutually exclusive"):
            Compose.from_params(specs=specs, **kwargs)


class TestFromParamsSpecsProbability:
    """Per-transform probability in specs path (Phase C.5)."""

    def test_p_zero_never_applied(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-90.0, 90.0)}, p=0.0)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert torch.allclose(out, x, atol=1e-5), "p=0.0 transform should never be applied"

    def test_p_one_always_applied(self):
        from fuse_augmentations import TransformSpec

        torch.manual_seed(42)
        specs = [TransformSpec(op="hflip", params={}, p=1.0)]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        expected = x.flip(dims=[3])
        assert torch.allclose(out, expected, atol=1e-5), "p=1.0 hflip must always flip"

    def test_mixed_probabilities_shape_preserved(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.5),
            TransformSpec(op="hflip", params={}, p=0.3),
            TransformSpec(op="vflip", params={}, p=0.7),
        ]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(4, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_all_p_zero_is_identity(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-90.0, 90.0)}, p=0.0),
            TransformSpec(op="hflip", params={}, p=0.0),
        ]
        pipe = Compose.from_params(specs=specs)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert torch.allclose(out, x, atol=1e-5), "All p=0.0 should produce identity"


class TestFromParamsSpecsRegression:
    """Existing from_params keyword-only API still works after C.4 (regression guard)."""

    def test_rotation_keyword(self):
        pipe = Compose.from_params(rotation=(-30.0, 30.0))
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_hflip_keyword(self):
        pipe = Compose.from_params(hflip_p=1.0)
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        expected = x.flip(dims=[-1])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_scale_keyword(self):
        pipe = Compose.from_params(scale=(0.8, 1.2))
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape

    def test_combined_keywords(self):
        pipe = Compose.from_params(rotation=(-15.0, 15.0), hflip_p=0.5, scale=(0.9, 1.1))
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    def test_all_none_identity(self):
        pipe = Compose.from_params()
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_data_keys_still_work(self):
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

    def test_rotation_shape_matches(self):
        from fuse_augmentations import TransformSpec

        pipe_kw = Compose.from_params(rotation=(-30.0, 30.0))
        pipe_spec = Compose.from_params(
            specs=[TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=1.0)]
        )
        x = torch.rand(2, 3, 32, 32)
        out_kw = pipe_kw(x)
        out_spec = pipe_spec(x)
        assert out_kw.shape == out_spec.shape, (
            f"Shape mismatch: keyword={out_kw.shape}, specs={out_spec.shape}"
        )

    def test_hflip_shape_matches(self):
        from fuse_augmentations import TransformSpec

        pipe_kw = Compose.from_params(hflip_p=0.5)
        pipe_spec = Compose.from_params(
            specs=[TransformSpec(op="hflip", params={}, p=0.5)]
        )
        x = torch.rand(2, 3, 32, 32)
        out_kw = pipe_kw(x)
        out_spec = pipe_spec(x)
        assert out_kw.shape == out_spec.shape
