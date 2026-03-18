"""Unit tests for data_keys routing in FusedCompose.

Tests cover the v0.3 data_keys contract:
- data_keys=None -> forward(img) returns single tensor (backward compat)
- data_keys=["input"] single key -> returns single tensor (unwrapped)
- data_keys=["input", "mask"] -> forward(img, mask) returns (img_out, mask_out) tuple
- Unknown key in data_keys emits UserWarning and passes value through
- Wrong number of args vs data_keys raises ValueError or TypeError
"""

from __future__ import annotations

import warnings

import pytest
import torch

from fuse_augmentations._compose import Compose


class TestDataKeysNone:
    """data_keys=None preserves backward compatibility (v0.1/v0.2 behavior)."""

    def test_returns_single_tensor(self):
        """forward(img) with data_keys=None returns a single Tensor, not a tuple."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"

    def test_shape_preserved(self):
        """Output shape matches input shape for data_keys=None."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape

    def test_values_unchanged_empty_pipeline(self):
        """Empty pipeline with data_keys=None returns input unchanged."""
        pipe = Compose([], data_keys=None)
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        torch.testing.assert_close(out, x)


class TestDataKeysSingleKey:
    """data_keys=["input"] with single key returns unwrapped tensor."""

    def test_single_key_returns_tensor_not_tuple(self):
        """Single data_key returns an unwrapped Tensor, not a 1-tuple."""
        pipe = Compose([], data_keys=["input"])
        x = torch.rand(2, 3, 8, 8)
        out = pipe(x)
        assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
        assert not isinstance(out, tuple), "Single key should not return a tuple"

    def test_single_key_shape_preserved(self):
        """Single data_key preserves input shape."""
        pipe = Compose([], data_keys=["input"])
        x = torch.rand(1, 3, 16, 16)
        out = pipe(x)
        assert out.shape == x.shape


class TestDataKeysMultipleKeys:
    """data_keys=["input", "mask"] returns tuple in data_keys order."""

    def test_two_keys_returns_tuple(self):
        """forward(img, mask) with two data_keys returns a 2-tuple."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out = pipe(img, mask)
        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        assert len(out) == 2, f"Expected 2-tuple, got length {len(out)}"

    def test_two_keys_shapes_preserved(self):
        """Both image and mask preserve their shapes."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out_img, out_mask = pipe(img, mask)
        assert out_img.shape == img.shape, f"Image shape mismatch: {out_img.shape} vs {img.shape}"
        assert out_mask.shape == mask.shape, f"Mask shape mismatch: {out_mask.shape} vs {mask.shape}"

    def test_values_unchanged_empty_pipeline(self):
        """Empty pipeline with data_keys returns inputs unchanged."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 3, (2, 1, 8, 8)).float()
        out_img, out_mask = pipe(img, mask)
        torch.testing.assert_close(out_img, img)
        torch.testing.assert_close(out_mask, mask)

    def test_tuple_order_matches_data_keys(self):
        """Return tuple order matches data_keys declaration order."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.ones(1, 3, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        out_img, out_mask = pipe(img, mask)
        # img was all ones, mask was all zeros — order matters
        assert out_img.mean() > 0.5, "First element should be the image (ones)"
        assert out_mask.mean() < 0.5, "Second element should be the mask (zeros)"


class TestDataKeysUnknownKey:
    """Unknown keys in data_keys emit UserWarning and pass through unchanged."""

    def test_unknown_key_warns(self):
        """Unknown data_key emits a UserWarning at construction time."""
        img = torch.rand(1, 3, 4, 4)
        custom = torch.rand(1, 1, 4, 4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe = Compose([], data_keys=["input", "custom_field"])
            pipe(img, custom)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected at least one UserWarning for unknown key"
        assert any("custom_field" in str(x.message) for x in user_warnings), (
            "Warning should mention the unknown key name"
        )

    def test_unknown_key_passes_through_unchanged(self):
        """Unknown key value is returned unchanged (passthrough)."""
        img = torch.rand(1, 3, 4, 4)
        custom = torch.rand(1, 2, 4, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipe = Compose([], data_keys=["input", "custom_field"])
            out_img, out_custom = pipe(img, custom)
        torch.testing.assert_close(out_custom, custom)


class TestDataKeysArgCountMismatch:
    """Wrong number of positional args vs data_keys raises an error."""

    def test_too_few_args(self):
        """Fewer positional args than data_keys raises ValueError or TypeError."""
        pipe = Compose([], data_keys=["input", "mask"])
        img = torch.rand(1, 3, 4, 4)
        with pytest.raises((ValueError, TypeError)):
            pipe(img)  # missing mask

    def test_too_many_args(self):
        """More positional args than data_keys raises ValueError or TypeError."""
        pipe = Compose([], data_keys=["input"])
        img = torch.rand(1, 3, 4, 4)
        extra = torch.rand(1, 1, 4, 4)
        with pytest.raises((ValueError, TypeError)):
            pipe(img, extra)

    def test_zero_args(self):
        """Zero positional args with non-empty data_keys raises ValueError or TypeError."""
        pipe = Compose([], data_keys=["input", "mask"])
        with pytest.raises((ValueError, TypeError)):
            pipe()
