"""Comprehensive integration tests for Compose.from_config() classmethod (Phase C.3).

Tests cover pipeline construction from TransformSpec lists, backend dispatch,
error handling, probability masking, data_keys forwarding, and fusion verification.
"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose, FusedCompose


class TestFromConfigBasicKornia:
    """from_config with backend='kornia' — basic pipeline construction."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    def test_single_rotation_produces_output(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_two_geometric_specs_fuse(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}),
            TransformSpec(op="hflip", params={}),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(2, 3, 64, 64)
        pipe(x)
        assert pipe.n_warps_saved >= 1, f"Expected fusion, got plan: {pipe.fusion_plan}"

    def test_returns_fused_compose_instance(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="hflip", params={}, p=0.5)]
        pipe = Compose.from_config(specs, backend="kornia")
        assert isinstance(pipe, FusedCompose), f"Expected FusedCompose, got {type(pipe).__name__}"

    def test_fusion_plan_descriptors_populated(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-15.0, 15.0)}),
            TransformSpec(op="hflip", params={}, p=0.5),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        descriptors = pipe.fusion_plan_descriptors
        assert isinstance(descriptors, list)
        assert len(descriptors) >= 1, "Expected at least one segment descriptor"

    def test_output_no_nan(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-45.0, 45.0)}, p=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(4, 3, 32, 32)
        out = pipe(x)
        assert not torch.isnan(out).any(), "from_config pipeline produced NaN values"


class TestFromConfigEmptySpecs:
    """from_config with empty specs list returns identity pipeline."""

    def test_empty_specs_identity(self):
        pipe = Compose.from_config([], backend="kornia")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x), "Empty specs should produce identity pipeline"

    def test_empty_specs_returns_fused_compose(self):
        pipe = Compose.from_config([], backend="kornia")
        assert isinstance(pipe, FusedCompose)


class TestFromConfigErrors:
    """Error paths for from_config."""

    def test_invalid_backend_raises_value_error(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError):
            Compose.from_config(specs, backend="nonexistent_backend")

    def test_invalid_op_raises_at_construction(self):
        """Invalid op name should raise ValueError at construction, not at forward time."""
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="definitely_not_a_real_op", params={})]
        with pytest.raises(ValueError):
            Compose.from_config(specs, backend="kornia")


class TestFromConfigProbability:
    """Per-transform probability via TransformSpec.p."""

    def test_p_zero_never_applied(self):
        """p=0.0 large rotation should produce output identical to input."""
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-90.0, 90.0)}, p=0.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert torch.allclose(out, x, atol=1e-5), "p=0.0 transform should never be applied"

    def test_p_one_hflip_always_applied(self):
        """p=1.0 hflip should always flip the image."""
        from fuse_augmentations import TransformSpec

        pytest.importorskip("kornia")
        torch.manual_seed(42)
        specs = [TransformSpec(op="hflip", params={}, p=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        expected = x.flip(dims=[3])
        assert torch.allclose(out, expected, atol=1e-5), "p=1.0 hflip must always flip"

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0], ids=["p=0.0", "p=0.5", "p=1.0"])
    def test_p_values_produce_valid_shape(self, p):
        from fuse_augmentations import TransformSpec

        pytest.importorskip("kornia")
        specs = [TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=p)]
        pipe = Compose.from_config(specs, backend="kornia")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape


class TestFromConfigTorchVision:
    """from_config with backend='torchvision'."""

    @pytest.fixture(autouse=True)
    def _skip_without_torchvision(self):
        pytest.importorskip("torchvision")

    def test_hflip_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="hflip", params={}, p=0.5)]
        pipe = Compose.from_config(specs, backend="torchvision")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-15.0, 15.0)})]
        pipe = Compose.from_config(specs, backend="torchvision")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])


class TestFromConfigAlbumentations:
    """from_config with backend='albumentations'."""

    @pytest.fixture(autouse=True)
    def _skip_without_albumentations(self):
        pytest.importorskip("albumentations")

    def test_hflip_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="hflip", params={}, p=0.5)]
        pipe = Compose.from_config(specs, backend="albumentations")
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])


class TestFromConfigDataKeys:
    """from_config with data_keys parameter."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    def test_data_keys_input_mask(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-15.0, 15.0)})]
        pipe = Compose.from_config(specs, backend="kornia", data_keys=["input", "mask"])
        img = torch.rand(2, 3, 32, 32)
        mask = torch.randint(0, 3, (2, 1, 32, 32)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple), f"Expected tuple with data_keys, got {type(out)}"
        assert len(out) == 2
        out_img, out_mask = out
        assert out_img.shape == img.shape
        assert out_mask.shape == mask.shape


class TestFromConfigInterpolationPadding:
    """from_config forwards interpolation and padding_mode."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    @pytest.mark.parametrize("interpolation", ["bilinear", "nearest", "bicubic"])
    def test_interpolation_modes(self, interpolation):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", interpolation=interpolation)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_padding_modes(self, padding_mode):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(op="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", padding_mode=padding_mode)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
