"""Comprehensive integration tests for Compose.from_config() classmethod.

Tests cover pipeline construction from TransformSpec lists, backend dispatch, error handling, probability masking,
data_keys forwarding, and fusion verification.

"""

from __future__ import annotations

import contextlib

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

        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_two_geometric_specs_fuse(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}),
            TransformSpec(operation="hflip", params={}),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        pipe(image)
        assert pipe.n_warps_saved >= 1, f"Expected fusion, got plan: {pipe.fusion_plan}"

    def test_returns_fused_compose_instance(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="kornia")
        assert isinstance(pipe, FusedCompose), f"Expected FusedCompose, got {type(pipe).__name__}"

    def test_fusion_plan_descriptors_populated(self):
        from fuse_augmentations import TransformSpec

        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)}),
            TransformSpec(operation="hflip", params={}, prob=0.5),
        ]
        pipe = Compose.from_config(specs, backend="kornia")
        descriptors = pipe.fusion_plan_descriptors
        assert isinstance(descriptors, list)
        assert len(descriptors) >= 1, "Expected at least one segment descriptor"

    def test_output_no_nan(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-45.0, 45.0)}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(4, 3, 32, 32)
        out = pipe(image)
        assert not torch.isnan(out).any(), "from_config pipeline produced NaN values"


class TestFromConfigEmptySpecs:
    """from_config with empty specs list returns identity pipeline."""

    def test_empty_specs_identity(self):
        pipe = Compose.from_config([], backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape
        assert torch.allclose(out, image), "Empty specs should produce identity pipeline"

    def test_empty_specs_returns_fused_compose(self):
        pipe = Compose.from_config([], backend="kornia")
        assert isinstance(pipe, FusedCompose)

    def test_empty_specs_forwards_output_backend(self) -> None:
        import numpy as np

        pipe = Compose.from_config([], backend="kornia", output_backend="numpy")
        image = torch.rand(1, 3, 32, 32)
        out = pipe(image)
        assert isinstance(out, np.ndarray)
        assert out.shape == (32, 32, 3)


class TestFromConfigErrors:
    """Error paths for from_config."""

    def test_invalid_backend_raises_value_error(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        with pytest.raises(ValueError, match="unknown backend"):
            Compose.from_config(specs, backend="nonexistent_backend")

    def test_invalid_backend_raises_value_error_for_empty_specs(self):
        with pytest.raises(ValueError, match="unknown backend"):
            Compose.from_config([], backend="nonexistent_backend")

    def test_invalid_op_raises_at_construction(self):
        """Invalid operation name should raise ValueError at construction, not at forward time."""
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="definitely_not_a_real_op", params={})]
        with pytest.raises(ValueError, match="unknown operation"):
            Compose.from_config(specs, backend="kornia")

    def test_params_cannot_shadow_transform_probability(self) -> None:
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="hflip", params={"prob": 0.1}, prob=0.5)]
        with pytest.raises(ValueError, match="must not include 'prob'"):
            Compose.from_config(specs, backend="kornia")


class TestFromConfigProbability:
    """Per-transform probability via TransformSpec.prob."""

    def test_p_zero_never_applied(self):
        """prob=0.0 large rotation should produce output identical to input."""
        pytest.importorskip("kornia")
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-90.0, 90.0)}, prob=0.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 64, 64)
        out = pipe(image)
        assert torch.allclose(out, image, atol=1e-5), "prob=0.0 transform should never be applied"

    def test_p_one_hflip_always_applied(self):
        """prob=1.0 hflip should always flip the image."""
        from fuse_augmentations import TransformSpec

        pytest.importorskip("kornia")
        torch.manual_seed(42)
        specs = [TransformSpec(operation="hflip", params={}, prob=1.0)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        expected = image.flip(dims=[3])
        assert torch.allclose(out, expected, atol=1e-5), "prob=1.0 hflip must always flip"

    @pytest.mark.parametrize("p_value", [0.0, 0.5, 1.0], ids=["prob=0.0", "prob=0.5", "prob=1.0"])
    def test_p_values_produce_valid_shape(self, p_value):
        from fuse_augmentations import TransformSpec

        pytest.importorskip("kornia")
        specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=p_value)]
        pipe = Compose.from_config(specs, backend="kornia")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


class TestFromConfigTorchVision:
    """from_config with backend='torchvision'."""

    @pytest.fixture(autouse=True)
    def _skip_without_torchvision(self):
        pytest.importorskip("torchvision")

    def test_hflip_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="torchvision")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)})]
        pipe = Compose.from_config(specs, backend="torchvision")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_scale_uses_zero_degree_affine_default(self) -> None:
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="scale", params={"scale": (0.8, 1.2)})]
        pipe = Compose.from_config(specs, backend="torchvision")
        transform = pipe.original_transforms[0]
        assert transform.degrees == [0.0, 0.0]
        assert transform.scale == (0.8, 1.2)
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == image.shape


class TestFromConfigAlbumentations:
    """from_config with backend='albumentations'."""

    @pytest.fixture(autouse=True)
    def _skip_without_albumentations(self):
        pytest.importorskip("albumentations")

    def test_hflip_works(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = Compose.from_config(specs, backend="albumentations")
        image = torch.rand(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_translates_canonical_degrees_to_limit(self) -> None:
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="albumentations")
        transform = pipe.original_transforms[0]
        assert transform.limit == (-10.0, 10.0)

    def test_affine_translates_canonical_degrees_to_rotate(self) -> None:
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="affine", params={"degrees": (-5.0, 5.0), "scale": (0.9, 1.1)})]
        pipe = Compose.from_config(specs, backend="albumentations")
        transform = pipe.original_transforms[0]
        assert transform.rotate == (-5.0, 5.0)
        assert transform.scale == {"x": (0.9, 1.1), "y": (0.9, 1.1)}


class TestFromConfigScaleKornia:
    """Canonical scale config remains constructible on Kornia."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    def test_scale_uses_zero_degree_affine_default(self) -> None:
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="scale", params={"scale": (0.8, 1.2)})]
        pipe = Compose.from_config(specs, backend="kornia")
        transform = pipe.original_transforms[0]
        assert "degrees=0.0" in repr(transform)
        assert "scale=(0.8, 1.2)" in repr(transform)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape


class TestFromConfigDataKeys:
    """from_config with data_keys parameter."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    def test_data_keys_input_mask(self):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-15.0, 15.0)})]
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

        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", interpolation=interpolation)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_padding_modes(self, padding_mode):
        from fuse_augmentations import TransformSpec

        specs = [TransformSpec(operation="rotation", params={"degrees": (-10.0, 10.0)})]
        pipe = Compose.from_config(specs, backend="kornia", padding_mode=padding_mode)
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape


class TestFromConfigUserWarning:
    """UserWarning emitted when spec.prob != 1.0 and the backend cannot set p= on the transform."""

    def test_p_not_applied_emits_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UserWarning fires when prob≠1.0 and backend transform has no settable prob attribute."""
        import warnings

        import fuse_augmentations._resolver as resolver_mod
        from fuse_augmentations import Compose, TransformSpec

        class _NoPTransform:
            """Mock: explicitly rejects p= kwarg, has no settable prob attribute (slots)."""

            __slots__ = ()

            def __init__(self, **_kwargs: object) -> None:
                if "p" in _kwargs:
                    raise TypeError("_NoPTransform does not accept 'p' keyword argument")

        def _mock_resolve(op: str, backend: str) -> type:
            return _NoPTransform

        monkeypatch.setattr(resolver_mod, "resolve_op", _mock_resolve)
        monkeypatch.setattr(resolver_mod, "SUPPORTED_BACKENDS", frozenset({"mock_backend"}))

        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with contextlib.suppress(ValueError, Exception):
                # FusedCompose.__init__ rejects unknown transforms after the warning fires
                Compose.from_config(specs, backend="mock_backend")

        matching = [
            w for w in recorded if issubclass(w.category, UserWarning) and "could not be applied" in str(w.message)
        ]
        assert matching, f"Expected UserWarning about p= not applied. Got: {[str(w.message) for w in recorded]}"


class TestFromConfigBackendGap:
    """Ops that are valid globally but unsupported by a specific backend."""

    def test_rotation90_unsupported_by_torchvision_raises(self) -> None:
        """Rotation90 is in SUPPORTED_OPS but TorchVision has no such transform."""
        pytest.importorskip("torchvision")
        from fuse_augmentations import Compose, TransformSpec

        specs = [TransformSpec(operation="rotation90", params={}, prob=1.0)]
        with pytest.raises(ValueError, match="does not support op_name"):
            Compose.from_config(specs, backend="torchvision")
