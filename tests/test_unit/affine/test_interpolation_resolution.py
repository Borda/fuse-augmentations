"""Tests for _interpolation.py."""

from fuse_augmentations._interpolation import resolve_interpolation, resolve_padding
from fuse_augmentations._types import InterpolationMode, PaddingMode


class TestResolveInterpolation:
    """resolve_interpolation picks the highest-quality mode or honours an override."""

    def test_bilinear_bicubic_resolves_to_bicubic(self):
        """[BILINEAR, BICUBIC] resolves to BICUBIC."""
        result = resolve_interpolation([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
        assert result == InterpolationMode.BICUBIC

    def test_nearest_nearest_resolves_to_nearest(self):
        """[NEAREST, NEAREST] resolves to NEAREST."""
        result = resolve_interpolation([InterpolationMode.NEAREST, InterpolationMode.NEAREST])
        assert result == InterpolationMode.NEAREST

    def test_user_override(self):
        """User override NEAREST overrides chain's BICUBIC."""
        result = resolve_interpolation(
            [InterpolationMode.BICUBIC, InterpolationMode.BILINEAR],
            override=InterpolationMode.NEAREST,
        )
        assert result == InterpolationMode.NEAREST

    def test_empty_defaults_bilinear(self):
        """Empty mode list defaults to BILINEAR."""
        result = resolve_interpolation([])
        assert result == InterpolationMode.BILINEAR

    def test_single_mode(self):
        """Single-element list returns that mode."""
        result = resolve_interpolation([InterpolationMode.BICUBIC])
        assert result == InterpolationMode.BICUBIC


class TestResolvePadding:
    """resolve_padding picks the highest-quality mode or honours an override."""

    def test_zeros_reflection_resolves_to_reflection(self):
        """[ZEROS, REFLECTION] resolves to REFLECTION."""
        result = resolve_padding([PaddingMode.ZEROS, PaddingMode.REFLECTION])
        assert result == PaddingMode.REFLECTION

    def test_user_override(self):
        """User override ZEROS overrides chain's REFLECTION."""
        result = resolve_padding(
            [PaddingMode.REFLECTION, PaddingMode.BORDER],
            override=PaddingMode.ZEROS,
        )
        assert result == PaddingMode.ZEROS

    def test_empty_defaults_zeros(self):
        """Empty mode list defaults to ZEROS."""
        result = resolve_padding([])
        assert result == PaddingMode.ZEROS

    def test_single_mode(self):
        """Single-element list returns that mode."""
        result = resolve_padding([PaddingMode.BORDER])
        assert result == PaddingMode.BORDER
