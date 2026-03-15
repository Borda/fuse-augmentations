"""Tests for _interpolation.py — spec tests #53-57."""

from fuse_augmentations._interpolation import resolve_interpolation, resolve_padding
from fuse_augmentations._types import InterpolationMode, PaddingMode

# --- Test #53: [BILINEAR, BICUBIC] -> BICUBIC ---


def test_bilinear_bicubic_resolves_to_bicubic():
    result = resolve_interpolation([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
    assert result == InterpolationMode.BICUBIC


# --- Test #54: [NEAREST, NEAREST] -> NEAREST ---


def test_nearest_nearest_resolves_to_nearest():
    result = resolve_interpolation([InterpolationMode.NEAREST, InterpolationMode.NEAREST])
    assert result == InterpolationMode.NEAREST


# --- Test #55: User override NEAREST overrides chain's BICUBIC ---


def test_user_override_interpolation():
    result = resolve_interpolation(
        [InterpolationMode.BICUBIC, InterpolationMode.BILINEAR],
        override=InterpolationMode.NEAREST,
    )
    assert result == InterpolationMode.NEAREST


# --- Test #56: [ZEROS, REFLECTION] -> REFLECTION ---


def test_zeros_reflection_resolves_to_reflection():
    result = resolve_padding([PaddingMode.ZEROS, PaddingMode.REFLECTION])
    assert result == PaddingMode.REFLECTION


# --- Test #57: User override ZEROS overrides chain's REFLECTION ---


def test_user_override_padding():
    result = resolve_padding(
        [PaddingMode.REFLECTION, PaddingMode.BORDER],
        override=PaddingMode.ZEROS,
    )
    assert result == PaddingMode.ZEROS


# --- Additional edge cases ---


def test_empty_modes_interpolation_defaults_bilinear():
    result = resolve_interpolation([])
    assert result == InterpolationMode.BILINEAR


def test_empty_modes_padding_defaults_zeros():
    result = resolve_padding([])
    assert result == PaddingMode.ZEROS


def test_single_mode_interpolation():
    result = resolve_interpolation([InterpolationMode.BICUBIC])
    assert result == InterpolationMode.BICUBIC


def test_single_mode_padding():
    result = resolve_padding([PaddingMode.BORDER])
    assert result == PaddingMode.BORDER
