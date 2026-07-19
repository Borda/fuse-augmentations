"""Shared validation and construction configuration helpers.

This leaf module keeps configuration normalization independent of pipeline runtime, factory construction, and inspection
so those layers do not import one another.

"""

from __future__ import annotations

from typing import cast

import torch

from fuse_augmentations.types import ClipPolicyStr, MaskInterpolationStr, PipelineDtypeStr, RandomnessPolicy

_COORD_DATA_KEYS = {"bbox_xyxy", "bbox_xywh", "keypoints"}
_PIPELINE_TORCH_DTYPES: dict[PipelineDtypeStr, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _has_coord_aux(data_keys: list[str] | None) -> bool:
    """Return whether ``data_keys`` carries any box/keypoint auxiliary target.

    Args:
        data_keys: The pipeline's positional data-key names, or ``None``.

    Returns:
        ``True`` if any coordinate aux key (``bbox_xyxy``/``bbox_xywh``/``keypoints``)
        is present, else ``False``.

    Examples:
        >>> _has_coord_aux(["input", "keypoints"])
        True
        >>> _has_coord_aux(["input", "mask"])
        False
        >>> _has_coord_aux(None)
        False

    """
    if data_keys is None:
        return False
    return any(key in _COORD_DATA_KEYS for key in data_keys)


def _has_aux_target(data_keys: list[str] | None) -> bool:
    """Return whether ``data_keys`` carries any auxiliary target (mask/box/keypoint).

    Unlike :func:`_has_coord_aux` (box/keypoint only), this also counts ``mask``: a
    ``CROP_RESIZE_FIXED`` op resizes the image, so *every* auxiliary target — masks
    included — must be routed to the new output size or it silently desyncs.

    Args:
        data_keys: The pipeline's positional data-key names, or ``None``.

    Returns:
        ``True`` if any auxiliary key (any entry beyond ``data_keys[0]``) is present.

    Examples:
        >>> _has_aux_target(["input", "mask"])
        True
        >>> _has_aux_target(["input"])
        False
        >>> _has_aux_target(None)
        False

    """
    return data_keys is not None and len(data_keys) > 1


# Plan-time segment dispatch tags. Computed once per pipeline at construction and
# stored on the instance so ``forward`` loops over integer tags instead of running
# an isinstance chain per segment per call. Kept as plain ints (not bound methods)
# so the dispatch plan survives the default nn.Module pickle round-trip.
_TAG_MATRIX = 0  # sets last_matrix: FusedAffine/AlbuFusedAffine/Projective/AlbuProjective
_TAG_PLAIN = 1  # no matrix: ExactAffine/FusedColor/CropResize
_TAG_PASSTHROUGH = 2  # _PassthroughSegment: adapter.call_nonfused
_TAG_LEGACY = 3  # pre-wrapper pickles / unknown segment — resolve adapter by index


def _apply_passthrough_substitution(transforms: list[object]) -> list[object]:
    """Replace registered non-fusible passthrough ops with torch-native equivalents.

    Each transform whose class name is registered in
    :mod:`fuse_augmentations.substitution` is swapped for an already-installed
    backend's torch-native equivalent (e.g. Albumentations ``GaussianBlur`` ->
    Kornia ``RandomGaussianBlur``), keeping the pipeline on-device. Substitution is
    behaviour-changing, so each swap emits a :class:`UserWarning`. Transforms without
    a registered substitution — or whose target backend is not importable — are kept
    unchanged.

    Args:
        transforms: The original transform list.

    Returns:
        A new list with substitutable transforms replaced; a fresh list is always
        returned so the caller's input is not mutated.

    """
    from fuse_augmentations.substitution import try_substitute_passthrough

    result: list[object] = []
    for transform in transforms:
        substitute = try_substitute_passthrough(transform)
        result.append(substitute if substitute is not None else transform)
    return result


def _coerce_randomness_policy(randomness: RandomnessPolicy | str) -> RandomnessPolicy:
    """Normalize a randomness policy argument."""
    if isinstance(randomness, RandomnessPolicy):
        return randomness
    try:
        return RandomnessPolicy(randomness)
    except ValueError as exc:
        msg = f"unknown randomness policy {randomness!r}; expected one of {[item.value for item in RandomnessPolicy]}"
        raise ValueError(msg) from exc


def _validate_clip_policy(clip_policy: ClipPolicyStr | str) -> ClipPolicyStr:
    """Validate and normalize the fused color-segment clipping policy.

    Args:
        clip_policy: ``"final"`` for one final clamp or ``"per_op_parity"`` for
            range-aware native-style clamp boundaries.

    Returns:
        The validated policy string.

    Raises:
        ValueError: If *clip_policy* is not a supported policy.

    """
    if clip_policy in ("final", "per_op_parity"):
        return cast("ClipPolicyStr", clip_policy)
    msg = "unknown clip policy {!r}; expected 'final' or 'per_op_parity'"
    raise ValueError(msg.format(clip_policy))


def _validate_mask_interpolation(mask_interpolation: str) -> MaskInterpolationStr:
    """Validate the auxiliary mask sampling mode and return its typed value."""
    if mask_interpolation not in ("nearest", "bilinear"):
        raise ValueError(f"invalid mask_interpolation {mask_interpolation!r}; expected 'nearest' or 'bilinear'")
    return cast(MaskInterpolationStr, mask_interpolation)


def _validate_pipeline_dtype(pipeline_dtype: PipelineDtypeStr | None) -> PipelineDtypeStr | None:
    """Validate the optional GPU image-operation dtype."""
    if pipeline_dtype is None or pipeline_dtype in _PIPELINE_TORCH_DTYPES:
        return pipeline_dtype
    msg = "pipeline_dtype must be 'bfloat16', 'float16', or None"
    raise ValueError(msg)
