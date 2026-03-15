"""Interpolation and padding mode resolution for fused segments.

When multiple transforms are fused into a single grid_sample call,
a single interpolation mode and padding mode must be chosen. These
functions resolve the modes by taking the finest/highest-quality option.

Example:
    >>> from fuse_augmentations._interpolation import resolve_interpolation, resolve_padding
    >>> from fuse_augmentations._types import InterpolationMode, PaddingMode
    >>> resolve_interpolation([InterpolationMode.BILINEAR, InterpolationMode.NEAREST])
    <InterpolationMode.BILINEAR: 1>
    >>> resolve_padding([PaddingMode.ZEROS, PaddingMode.REFLECTION])
    <PaddingMode.REFLECTION: 2>
"""

from __future__ import annotations

from fuse_augmentations._types import InterpolationMode, PaddingMode


def resolve_interpolation(
    modes: list[InterpolationMode],
    override: InterpolationMode | None = None,
) -> InterpolationMode:
    """Resolve the interpolation mode for a fused segment.

    Takes the finest mode in the chain. User override supersedes auto-resolution.
    Priority: NEAREST(0) < BILINEAR(1) < BICUBIC(2).

    Args:
        modes: List of interpolation modes from transforms in the segment.
        override: Optional user-specified override; supersedes auto-resolution.

    Returns:
        Resolved InterpolationMode.

    Example:
        >>> from fuse_augmentations._interpolation import resolve_interpolation
        >>> from fuse_augmentations._types import InterpolationMode
        >>> resolve_interpolation([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
        <InterpolationMode.BICUBIC: 2>
    """
    if override is not None:
        return override
    if not modes:
        return InterpolationMode.BILINEAR
    return max(modes)


def resolve_padding(
    modes: list[PaddingMode],
    override: PaddingMode | None = None,
) -> PaddingMode:
    """Resolve the padding mode for a fused segment.

    Takes the highest-quality mode. User override supersedes auto-resolution.
    Priority: ZEROS(0) < BORDER(1) < REFLECTION(2).

    Args:
        modes: List of padding modes from transforms in the segment.
        override: Optional user-specified override; supersedes auto-resolution.

    Returns:
        Resolved PaddingMode.

    Example:
        >>> from fuse_augmentations._interpolation import resolve_padding
        >>> from fuse_augmentations._types import PaddingMode
        >>> resolve_padding([PaddingMode.ZEROS, PaddingMode.REFLECTION])
        <PaddingMode.REFLECTION: 2>
    """
    if override is not None:
        return override
    if not modes:
        return PaddingMode.ZEROS
    return max(modes)
