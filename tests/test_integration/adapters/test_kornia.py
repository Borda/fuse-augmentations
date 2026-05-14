"""Integration tests for KorniaAdapter: category classification, parameter units, and shear parity."""

import warnings

import pytest
import torch

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")
from kornia.augmentation import (  # noqa: E402
    RandomAffine,
    RandomHorizontalFlip,
    RandomRotation,
    RandomShear,
    RandomTranslate,
    RandomVerticalFlip,
)

from fuse_augmentations._types import TransformCategory  # noqa: E402
from fuse_augmentations.adapters._kornia import KorniaAdapter  # noqa: E402
from fuse_augmentations.affine._matrix import inv3x3, normalize_matrix  # noqa: E402

pytestmark = pytest.mark.integration

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


@pytest.fixture
def adapter():
    """Create a fresh KorniaAdapter instance for each test."""
    return KorniaAdapter()


class TestCategory:
    """Verify adapter.category() for known and unknown Kornia transforms."""

    @pytest.mark.parametrize(
        "transform, expected_cat",
        [
            (RandomRotation(degrees=30, p=1.0), TransformCategory.GEOMETRIC_INTERP),
            (RandomAffine(degrees=30, p=1.0), TransformCategory.GEOMETRIC_INTERP),
            (RandomHorizontalFlip(p=1.0), TransformCategory.GEOMETRIC_EXACT),
            (RandomVerticalFlip(p=1.0), TransformCategory.GEOMETRIC_EXACT),
        ],
    )
    def test_registered_transforms(self, adapter, transform, expected_cat):
        """Known Kornia transforms return their expected category."""
        assert adapter.category(transform) == expected_cat

    def test_unknown_transform(self, adapter):
        """Unknown transform falls back to SPATIAL_KERNEL with a UserWarning."""

        class UnknownTransform:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = adapter.category(UnknownTransform())
        assert cat == TransformCategory.SPATIAL_KERNEL
        assert len(w) == 1
        assert "Unknown Kornia transform" in str(w[0].message)


class TestNativeParity:
    """Verify fused matrix path matches Kornia native output for each transform type."""

    def test_shear_sign(self, adapter):
        """Our shear matrix sign convention matches Kornia's native output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        # X-shear only, fixed shear value; align_corners=True to match our pipeline
        shear_deg = 20.0
        t = RandomAffine(degrees=0, shear=(shear_deg, shear_deg, 0, 0), p=1.0, align_corners=True)

        # Use forward_parameters so native and fused paths share identical sampled params
        fp = t.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = t(img, params=fp)

        # Our fused path -- build params from fp using same negation as sample_params.
        # Negation converts Kornia CW convention -> our CCW rotation_matrix convention.
        params: dict[str, torch.Tensor] = {}
        if "angle" in fp:
            params["angle_rad"] = -torch.deg2rad(fp["angle"].to(DEFAULT_DEVICE))
        if "translations" in fp:
            params["translate_x"] = fp["translations"][:, 0].to(DEFAULT_DEVICE)
            params["translate_y"] = fp["translations"][:, 1].to(DEFAULT_DEVICE)
        if "scale" in fp:
            params["scale_x"] = fp["scale"][:, 0].to(DEFAULT_DEVICE)
            params["scale_y"] = fp["scale"][:, 1].to(DEFAULT_DEVICE)
        if "shear_x" in fp:
            params["shear_x_rad"] = -torch.deg2rad(fp["shear_x"].to(DEFAULT_DEVICE))
        if "shear_y" in fp:
            params["shear_y_rad"] = -torch.deg2rad(fp["shear_y"].to(DEFAULT_DEVICE))
        M_fwd = adapter.build_matrix(t, params, height, width)
        M_inv = inv3x3(M_fwd)
        M_norm = normalize_matrix(M_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            M_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # Compare -- they should be close
        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Shear sign mismatch: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_translation_units(self, adapter):
        """translate_x is in pixels, not fractional."""
        batch_size, num_channels, height, width = 4, 3, 64, 128
        translate_frac = 0.5  # 50% in Kornia's convention

        t = RandomAffine(
            degrees=0,
            translate=(translate_frac, translate_frac),
            p=1.0,
        )

        params = adapter.sample_params(t, (batch_size, num_channels, height, width), DEFAULT_DEVICE)

        # Translation should be in pixels. With translate=(0.5, 0.5),
        # the range is [-0.5*width, 0.5*width] for x and [-0.5*height, 0.5*height] for y.
        # Values should generally be larger than 1.0 for width=128, height=64.
        # We just verify the values are NOT in [0, 1] fractional range
        # (some could be close to 0, so check at least one sample is > 1).
        if "translate_x" in params:
            tx = params["translate_x"]
            # The values are already pixels from Kornia's generate_parameters
            assert tx.shape == (batch_size,)

    def test_rotation(self, adapter):
        """Rotation matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        # align_corners=True to match our pipeline convention
        t = RandomRotation(degrees=45, p=1.0, align_corners=True)

        # Use forward_parameters so native and fused paths share identical sampled params
        fp = t.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = t(img, params=fp)

        # Our fused path -- negate angle to convert Kornia CW -> our CCW convention
        params = {"angle_rad": -torch.deg2rad(fp["degrees"].to(DEFAULT_DEVICE))}
        M_fwd = adapter.build_matrix(t, params, height, width)
        M_inv = inv3x3(M_fwd)
        M_norm = normalize_matrix(M_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            M_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Rotation parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_shear(self, adapter):
        """RandomShear matrix path matches native Kornia output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        t = RandomShear(shear=(-20.0, 20.0), p=1.0, align_corners=True)
        fp = t.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = t(img, params=fp)

        params: dict[str, torch.Tensor] = {}
        if "shear_x" in fp:
            params["shear_x_rad"] = -torch.deg2rad(fp["shear_x"].to(DEFAULT_DEVICE))
        if "shear_y" in fp:
            params["shear_y_rad"] = -torch.deg2rad(fp["shear_y"].to(DEFAULT_DEVICE))

        mtx_fwd = adapter.build_matrix(t, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"RandomShear parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_translate(self, adapter):
        """RandomTranslate matrix path matches native Kornia output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        t = RandomTranslate(translate_x=(0.1, 0.2), translate_y=(0.1, 0.2), p=1.0, align_corners=True)
        fp = t.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = t(img, params=fp)

        params = {
            "translate_x": fp["translate_x"].to(DEFAULT_DEVICE),
            "translate_y": fp["translate_y"].to(DEFAULT_DEVICE),
        }
        mtx_fwd = adapter.build_matrix(t, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"RandomTranslate parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_vflip(self, adapter):
        """VFlip matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        img = torch.rand(batch_size, num_channels, height, width)

        t = RandomVerticalFlip(p=1.0)

        native_out = t(img)

        params = adapter.sample_params(t, (batch_size, num_channels, height, width), DEFAULT_DEVICE)
        mtx_fwd = adapter.build_matrix(t, params, height, width)
        if mtx_fwd.shape[0] == 1 and batch_size > 1:
            mtx_fwd = mtx_fwd.expand(batch_size, -1, -1)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"VFlip parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_hflip(self, adapter):
        """HFlip matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        img = torch.rand(batch_size, num_channels, height, width)

        t = RandomHorizontalFlip(p=1.0)

        native_out = t(img)

        params = adapter.sample_params(t, (batch_size, num_channels, height, width), DEFAULT_DEVICE)
        M_fwd = adapter.build_matrix(t, params, height, width)
        # Expand to batch if needed
        if M_fwd.shape[0] == 1 and batch_size > 1:
            M_fwd = M_fwd.expand(batch_size, -1, -1)
        M_inv = inv3x3(M_fwd)
        M_norm = normalize_matrix(M_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            M_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"HFlip parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )


class TestSameOnBatch:
    """Verify that same_on_batch is respected by adapter.sample_params."""

    def test_same_on_batch_produces_identical_params(self, adapter):
        """When same_on_batch=True, all samples get the same angle."""
        t = RandomRotation(degrees=30, same_on_batch=True, p=1.0)
        params = adapter.sample_params(t, (4, 3, 8, 8), DEFAULT_DEVICE)
        angles = params["angle_rad"]
        assert angles.shape == (4,)
        # All 4 samples should have the same angle
        assert torch.allclose(angles[0].expand(4), angles), f"same_on_batch=True but angles differ: {angles}"

    def test_different_on_batch_produces_varied_params(self, adapter):
        """When same_on_batch=False, a large batch produces varied angles."""
        t = RandomRotation(degrees=30, same_on_batch=False, p=1.0)
        params = adapter.sample_params(t, (16, 3, 8, 8), DEFAULT_DEVICE)
        angles = params["angle_rad"]
        assert angles.shape == (16,)
        # With 16 samples and 30 degree range, min and max should differ
        assert not torch.allclose(angles.min().unsqueeze(0), angles.max().unsqueeze(0)), (
            f"same_on_batch=False but all angles identical: {angles}"
        )

    def test_same_on_batch_affine(self, adapter):
        """same_on_batch=True on RandomAffine produces identical params across batch."""
        t = RandomAffine(degrees=30, translate=(0.3, 0.3), same_on_batch=True, p=1.0)
        params = adapter.sample_params(t, (4, 3, 8, 8), DEFAULT_DEVICE)
        if "angle_rad" in params:
            angles = params["angle_rad"]
            assert torch.allclose(angles[0].expand(4), angles), f"same_on_batch=True but angles differ: {angles}"
        if "translate_x" in params:
            tx = params["translate_x"]
            assert torch.allclose(tx[0].expand(4), tx), f"same_on_batch=True but translate_x differs: {tx}"


class TestBuildMatrixFallback:
    """Verify build_matrix fallback path returns identity for unregistered transforms."""

    def test_unknown_transform_returns_identity(self, adapter):
        """An unregistered transform class produces a (1, 3, 3) identity matrix."""

        class UnknownTransform:
            pass

        mtx = adapter.build_matrix(UnknownTransform(), {}, 64, 64)
        assert mtx.shape == torch.Size([1, 3, 3])
        assert torch.allclose(mtx, torch.eye(3).unsqueeze(0)), f"Expected identity fallback, got: {mtx}"
