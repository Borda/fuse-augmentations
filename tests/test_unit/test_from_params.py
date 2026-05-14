"""Unit tests for FusedCompose.from_params() classmethod.

Tests cover the v0.3 from_params() factory contract:
- All-None -> returns FusedCompose that acts as identity
- rotation / scale -> forward pass produces correct shape, no crash
- hflip_p=1.0 -> output is horizontally flipped
- brightness -> raises NotImplementedError
- from_params(data_keys=["input","mask"]) with rotation -> returns tuple

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compose import FusedCompose as Compose


class TestFromParamsIdentity:
    """All-None params produce an identity pipeline."""

    def test_all_none_returns_fused_compose(self):
        """from_params() with no arguments returns a FusedCompose instance."""
        pipe = Compose.from_params()
        assert isinstance(pipe, Compose)

    def test_all_none_acts_as_identity(self):
        """from_params() with all defaults returns input unchanged."""
        pipe = Compose.from_params()
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_identity_shape_preserved(self):
        """Identity pipeline preserves input shape exactly."""
        pipe = Compose.from_params()
        x = torch.rand(1, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape


class TestFromParamsRotation:
    """Rotation parameter produces valid output."""

    def test_rotation_shape_preserved(self):
        """from_params(rotation=(-30, 30)) preserves (batch_size, num_channels, height, width) shape."""
        pipe = Compose.from_params(rotation=(-30.0, 30.0))
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_rotation_no_nan(self):
        """from_params(rotation=(-30, 30)) produces no NaN values."""
        pipe = Compose.from_params(rotation=(-30.0, 30.0))
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        assert not torch.isnan(out).any(), "Rotation produced NaN values"

    def test_rotation_returns_tensor(self):
        """from_params(rotation=...) returns a Tensor, not a tuple."""
        pipe = Compose.from_params(rotation=(-10.0, 10.0))
        x = torch.rand(1, 3, 8, 8)
        out = pipe(x)
        assert isinstance(out, torch.Tensor)


class TestFromParamsScale:
    """Scale parameter produces valid output."""

    def test_scale_shape_preserved(self):
        """from_params(scale=(0.8, 1.2)) preserves (batch_size, num_channels, height, width) shape."""
        pipe = Compose.from_params(scale=(0.8, 1.2))
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_scale_no_nan(self):
        """from_params(scale=(0.8, 1.2)) produces no NaN values."""
        pipe = Compose.from_params(scale=(0.8, 1.2))
        x = torch.rand(2, 3, 16, 16)
        out = pipe(x)
        assert not torch.isnan(out).any(), "Scale produced NaN values"


class TestFromParamsHFlip:
    """hflip_p=1.0 produces a horizontal flip."""

    def test_hflip_p1_flips_image(self):
        """from_params(hflip_p=1.0) produces a horizontally flipped image."""
        pipe = Compose.from_params(hflip_p=1.0)
        x = torch.rand(1, 3, 8, 8)
        out = pipe(x)
        expected = x.flip(dims=[-1])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_hflip_p0_is_identity(self):
        """from_params(hflip_p=0.0) is identity (no flip)."""
        pipe = Compose.from_params(hflip_p=0.0)
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


class TestFromParamsBrightnessContrast:
    """Brightness and contrast raise NotImplementedError."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"brightness": 0.2}, "brightness"),
            ({"contrast": 0.3}, "contrast"),
        ],
        ids=["brightness", "contrast"],
    )
    def test_raises_not_implemented(self, kwargs, match):
        """Brightness/contrast params raise NotImplementedError with descriptive message."""
        with pytest.raises(NotImplementedError, match=match):
            Compose.from_params(**kwargs)


class TestFromParamsWithDataKeys:
    """from_params() with data_keys routes aux targets correctly."""

    def test_rotation_with_mask_returns_tuple(self):
        """from_params(rotation=..., data_keys=["input","mask"]) returns (img, mask) tuple."""
        pipe = Compose.from_params(
            rotation=(-15.0, 15.0),
            data_keys=["input", "mask"],
        )
        img = torch.rand(2, 3, 16, 16)
        mask = torch.randint(0, 3, (2, 1, 16, 16)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        assert len(out) == 2
        out_img, out_mask = out
        assert out_img.shape == img.shape
        assert out_mask.shape == mask.shape

    def test_hflip_with_mask_shapes(self):
        """from_params(hflip_p=1.0, data_keys=["input","mask"]) preserves shapes."""
        pipe = Compose.from_params(
            hflip_p=1.0,
            data_keys=["input", "mask"],
        )
        img = torch.rand(1, 3, 8, 8)
        mask = torch.randint(0, 2, (1, 1, 8, 8)).float()
        out_img, out_mask = pipe(img, mask)
        assert out_img.shape == img.shape
        assert out_mask.shape == mask.shape


class TestFromParamsDegenerateScale:
    """Degenerate scale=(0,0) produces a singular matrix and raises ValueError."""

    def test_from_params_degenerate_scale(self):
        """from_params(scale=(0,0)) raises ValueError at forward time.

        Zero scale makes the affine matrix singular (det=0). The inv3x3 function in _matrix.py detects this and raises
        ValueError with a descriptive message about near-singular matrices.

        """
        pipe = Compose.from_params(scale=(0.0, 0.0))
        x = torch.rand(2, 3, 16, 16)
        with pytest.raises(ValueError, match="Near-singular matrix"):
            pipe(x)
