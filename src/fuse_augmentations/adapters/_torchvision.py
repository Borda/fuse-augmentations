"""TorchVision backend adapter for the fused affine engine.

Bridges TorchVision augmentation transforms to the canonical parameter
representation used by ``FusedAffineSegment``.

Supports both ``torchvision.transforms`` (v1) and ``torchvision.transforms.v2``
namespaces. Each geometric transform samples parameters via TorchVision's
``get_params()`` static method and reconstructs the affine matrix from
``fuse_augmentations.affine._matrix`` primitives.

Flip transforms (``RandomHorizontalFlip``, ``RandomVerticalFlip``) return a
minimal parameter dict containing a ``"_batch_size"`` sentinel; ``build_matrix()``
uses this sentinel to construct their matrices from the shared matrix primitives.

Optional: ``torchvision`` must be installed at runtime for transform dispatch
to function; the module is importable without it.

Example:
    >>> from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
    >>> adapter = TorchVisionAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...TorchVisionAdapter...>

"""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import numpy as np
import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import (
    crop_resize_matrix,
    hflip_matrix,
    perspective_from_points,
    rotation_matrix,
    vflip_matrix,
)

# ---------------------------------------------------------------------------
# Transform registry -- lazy import guards (torchvision is optional)
# ---------------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[type, TransformCategory] = {}
_HFLIP_TYPES: set[type] = set()
_VFLIP_TYPES: set[type] = set()
_ROTATION_TYPES: set[type] = set()
_AFFINE_TYPES: set[type] = set()
_PERSPECTIVE_TYPES: set[type] = set()
_COLOR_JITTER_TYPES: set[type] = set()
_CROP_RESIZE_TYPES: set[type] = set()

# v1: torchvision.transforms
try:
    from torchvision.transforms import ColorJitter as _V1ColorJitter
    from torchvision.transforms import RandomAffine as _V1RandomAffine
    from torchvision.transforms import RandomHorizontalFlip as _V1RandomHorizontalFlip
    from torchvision.transforms import RandomPerspective as _V1RandomPerspective
    from torchvision.transforms import RandomResizedCrop as _V1RandomResizedCrop
    from torchvision.transforms import RandomRotation as _V1RandomRotation
    from torchvision.transforms import RandomVerticalFlip as _V1RandomVerticalFlip

    _TRANSFORM_REGISTRY[_V1RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V1RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V1RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V1RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V1RandomPerspective] = TransformCategory.PROJECTIVE
    _TRANSFORM_REGISTRY[_V1ColorJitter] = TransformCategory.POINTWISE_LINEAR
    _TRANSFORM_REGISTRY[_V1RandomResizedCrop] = TransformCategory.CROP_RESIZE_FIXED

    _HFLIP_TYPES.add(_V1RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V1RandomVerticalFlip)
    _ROTATION_TYPES.add(_V1RandomRotation)
    _AFFINE_TYPES.add(_V1RandomAffine)
    _PERSPECTIVE_TYPES.add(_V1RandomPerspective)
    _COLOR_JITTER_TYPES.add(_V1ColorJitter)
    _CROP_RESIZE_TYPES.add(_V1RandomResizedCrop)
except ImportError:
    pass

# v2: torchvision.transforms.v2
try:
    from torchvision.transforms.v2 import ColorJitter as _V2ColorJitter
    from torchvision.transforms.v2 import RandomAffine as _V2RandomAffine
    from torchvision.transforms.v2 import RandomHorizontalFlip as _V2RandomHorizontalFlip
    from torchvision.transforms.v2 import RandomPerspective as _V2RandomPerspective
    from torchvision.transforms.v2 import RandomResizedCrop as _V2RandomResizedCrop
    from torchvision.transforms.v2 import RandomRotation as _V2RandomRotation
    from torchvision.transforms.v2 import RandomVerticalFlip as _V2RandomVerticalFlip

    _TRANSFORM_REGISTRY[_V2RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V2RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V2RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V2RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V2RandomPerspective] = TransformCategory.PROJECTIVE
    _TRANSFORM_REGISTRY[_V2ColorJitter] = TransformCategory.POINTWISE_LINEAR
    _TRANSFORM_REGISTRY[_V2RandomResizedCrop] = TransformCategory.CROP_RESIZE_FIXED

    _HFLIP_TYPES.add(_V2RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V2RandomVerticalFlip)
    _ROTATION_TYPES.add(_V2RandomRotation)
    _AFFINE_TYPES.add(_V2RandomAffine)
    _PERSPECTIVE_TYPES.add(_V2RandomPerspective)
    _COLOR_JITTER_TYPES.add(_V2ColorJitter)
    _CROP_RESIZE_TYPES.add(_V2RandomResizedCrop)
except ImportError:
    pass

# Freeze into frozensets for immutability after init
_HFLIP_TYPES_FS: frozenset[type] = frozenset(_HFLIP_TYPES)
_VFLIP_TYPES_FS: frozenset[type] = frozenset(_VFLIP_TYPES)
_ROTATION_TYPES_FS: frozenset[type] = frozenset(_ROTATION_TYPES)
_AFFINE_TYPES_FS: frozenset[type] = frozenset(_AFFINE_TYPES)
_PERSPECTIVE_TYPES_FS: frozenset[type] = frozenset(_PERSPECTIVE_TYPES)
_COLOR_JITTER_TYPES_FS: frozenset[type] = frozenset(_COLOR_JITTER_TYPES)
_CROP_RESIZE_TYPES_FS: frozenset[type] = frozenset(_CROP_RESIZE_TYPES)


def _check_expand(transform: object) -> None:
    """Raise if transform has expand=True (unsupported by fused engine)."""
    if getattr(transform, "expand", False):
        msg = "TorchVision RandomRotation with expand=True is not supported by the fused engine"
        raise ValueError(msg)


def _is_torchvision_v2_transform(transform: object) -> bool:
    """Return whether the transform comes from ``torchvision.transforms.v2``."""
    transform_type = type(transform)
    # _v1_transform_cls is an undocumented TorchVision internal, stable from 0.15-0.20;
    # revisit on major TorchVision bumps.
    return transform_type.__module__.startswith("torchvision.transforms.v2") or hasattr(
        transform,
        "_v1_transform_cls",
    )


class TorchVisionAdapter:
    """Adapter between TorchVision transforms and the fused affine engine.

    Implements the ``TransformAdapter`` protocol for the TorchVision backend.
    Supports ``RandomRotation``, ``RandomAffine``, ``RandomHorizontalFlip``,
    and ``RandomVerticalFlip`` from both ``torchvision.transforms`` (v1) and
    ``torchvision.transforms.v2``.

    Example:
        >>> adapter = TorchVisionAdapter()
        >>> isinstance(adapter, TorchVisionAdapter)
        True

    """

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory for the given TorchVision transform.

        Args:
            transform: A TorchVision transform instance.

        Returns:
            The category for the transform. Unknown transforms default to
            ``SPATIAL_KERNEL`` with a ``UserWarning``.

        """
        _check_expand(transform)
        for base_type, cat in _TRANSFORM_REGISTRY.items():
            if isinstance(transform, base_type):
                return cat
        warnings.warn(
            f"Unknown TorchVision transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
            UserWarning,
            stacklevel=2,
        )
        return TransformCategory.SPATIAL_KERNEL

    @staticmethod
    def sample_params(
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample random parameters for a batch of images.

        TorchVision v1 samples one parameter set per image, while TorchVision v2 samples one parameter set per
        input tensor. For batched v2 inputs, the returned tensors therefore have shape ``(1,)`` and are expanded
        later by the fused segment.

        Flip transforms return a ``{"_batch_size": tensor([batch_size])}`` sentinel.

        Args:
            transform: A TorchVision transform instance.
            input_shape: ``(batch_size, channels, height, width)`` shape tuple.
            device: Target device for the returned tensors.

        Returns:
            Dict of parameter tensors keyed by canonical names.

        """
        _check_expand(transform)
        batch_size, _num_channels, height, width = input_shape

        # Flip transforms -- no sampled params, only need batch size
        if isinstance(transform, tuple(_HFLIP_TYPES_FS | _VFLIP_TYPES_FS)):
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        # RandomRotation
        if isinstance(transform, tuple(_ROTATION_TYPES_FS)):
            if _is_torchvision_v2_transform(transform):
                angle_deg = _sample_rotation_angle(transform)
                return {
                    "angle_rad": torch.tensor([math.radians(angle_deg)], dtype=torch.float32, device=device),
                }
            angles = []
            for _ in range(batch_size):
                angle_deg = _sample_rotation_angle(transform)
                angles.append(math.radians(angle_deg))
            return {
                "angle_rad": torch.tensor(angles, dtype=torch.float32, device=device),
            }

        # RandomAffine
        if isinstance(transform, tuple(_AFFINE_TYPES_FS)):
            return _sample_affine_params(
                transform,
                batch_size,
                height,
                width,
                device,
                shared_across_batch=_is_torchvision_v2_transform(transform),
            )

        # RandomPerspective
        if isinstance(transform, tuple(_PERSPECTIVE_TYPES_FS)):
            is_v2 = _is_torchvision_v2_transform(transform)
            sample_count = 1 if is_v2 else batch_size
            starts, ends = [], []
            for _ in range(sample_count):
                start_points, end_points = type(transform).get_params(width, height, transform.distortion_scale)  # type: ignore[attr-defined]
                starts.append(start_points)  # list of 4 [x,y] pairs
                ends.append(end_points)
            # Convert to (sample_count, 4, 2) tensors
            start_t = torch.tensor(starts, dtype=torch.float32, device=device)  # (count, 4, 2)
            end_t = torch.tensor(ends, dtype=torch.float32, device=device)
            return {"start_points": start_t.clone(), "end_points": end_t.clone()}

        # ColorJitter
        if isinstance(transform, tuple(_COLOR_JITTER_TYPES_FS)):
            return _sample_color_jitter_params(transform, batch_size, device, _is_torchvision_v2_transform(transform))

        # RandomResizedCrop
        if isinstance(transform, tuple(_CROP_RESIZE_TYPES_FS)):
            return _sample_crop_resize_params(
                transform, batch_size, height, width, device, _is_torchvision_v2_transform(transform)
            )

        # Unknown -- return empty
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build a (batch_size, 3, 3) pixel-space forward affine matrix.

        For ``RandomRotation``, composes center-rotate-uncenter via
        ``rotation_matrix``. For ``RandomAffine``, composes in the order:
        center-rotate-uncenter, then scale about center, then shear, then
        translate. Note: uses ``(width-1)/2`` rotation centre (align_corners=True),
        not TorchVision's native ``width/2`` — see ``TestAlignCornersOffset``.

        Flip transforms use ``hflip_matrix`` / ``vflip_matrix`` expanded to
        batch size.

        Args:
            transform: A TorchVision transform instance.
            params: Parameter dict from ``sample_params()``.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            ``(batch_size, 3, 3)`` forward affine matrix in pixel coordinates.

        """
        if isinstance(transform, tuple(_HFLIP_TYPES_FS)):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return hflip_matrix(width=width, batch_size=batch_size, device=device, dtype=torch.float32)

        if isinstance(transform, tuple(_VFLIP_TYPES_FS)):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return vflip_matrix(height=height, batch_size=batch_size, device=device, dtype=torch.float32)

        if isinstance(transform, tuple(_ROTATION_TYPES_FS)):
            return rotation_matrix(params["angle_rad"], height=height, width=width)

        if isinstance(transform, tuple(_AFFINE_TYPES_FS)):
            return _build_affine_matrix(params, height, width)

        if isinstance(transform, tuple(_PERSPECTIVE_TYPES_FS)):
            return perspective_from_points(params["start_points"], params["end_points"])

        if isinstance(transform, tuple(_CROP_RESIZE_TYPES_FS)):
            return crop_resize_matrix(
                top=params["crop_top"],
                left=params["crop_left"],
                crop_h=params["crop_h"],
                crop_w=params["crop_w"],
                target_h=params["target_h"],
                target_w=params["target_w"],
            )

        # Fallback: identity (unreachable for registered transforms)
        return torch.eye(3).unsqueeze(0)

    @staticmethod
    def exact_flip_dims(transform: object) -> list[int]:
        """Return the spatial dims to flip for GEOMETRIC_EXACT transforms.

        Args:
            transform: A TorchVision flip transform.

        Returns:
            ``[3]`` for ``RandomHorizontalFlip`` (width axis) or ``[2]`` for
            ``RandomVerticalFlip`` (height axis).

        Raises:
            TypeError: If the transform is not a recognised flip type.

        """
        if isinstance(transform, tuple(_HFLIP_TYPES_FS)):
            return [3]
        if isinstance(transform, tuple(_VFLIP_TYPES_FS)):
            return [2]
        raise TypeError(f"Cannot determine flip dims for {type(transform).__name__!r}")

    @staticmethod
    def exact_apply(transform: object, image: torch.Tensor) -> torch.Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly.

        TorchVision currently only supports flip transforms as GEOMETRIC_EXACT.

        Args:
            transform: A TorchVision GEOMETRIC_EXACT transform.
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor.

        """
        if isinstance(transform, tuple(_HFLIP_TYPES_FS)):
            return image.flip(dims=[3])
        if isinstance(transform, tuple(_VFLIP_TYPES_FS)):
            return image.flip(dims=[2])
        msg = f"Cannot apply exact op for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def same_on_batch(transform: object) -> bool:
        """Return whether randomness should be shared across the input batch."""
        return _is_torchvision_v2_transform(transform) or bool(getattr(transform, "same_on_batch", False))

    @staticmethod
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build a ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Maps the linear color transform ``c' = M @ c + b`` to the 4x4
        homogeneous form ``[[M, b], [0^T, 1]]``.

        Supported transforms:

        - ``ColorJitter``: composes brightness (multiplicative ``bf * c``)
          and contrast (midpoint-0.5 approximation ``cf * c + (1-cf)*0.5``)
          in the sampled ``order``. Saturation and hue are treated as identity.

        Args:
            transform: A TorchVision color transform instance.
            params: Parameter dict from ``sample_params()``.

        Returns:
            ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Raises:
            NotImplementedError: If the transform type is not supported.

        """
        if isinstance(transform, tuple(_COLOR_JITTER_TYPES_FS)):
            # Saturation and hue are not exactly representable as 4x4 linear.
            # Fall back to passthrough for ColorJitter instances that use them.
            sat = getattr(transform, "saturation", None)
            hue = getattr(transform, "hue", None)
            _has_sat = sat is not None and sat not in (0.0, (1.0, 1.0))
            _has_hue = hue is not None and hue not in (0.0, (0.0, 0.0))
            if _has_sat or _has_hue:
                msg = (
                    "ColorJitter with non-trivial saturation or hue is not "
                    "exactly representable as a 4x4 linear color matrix; "
                    "use brightness/contrast only for FusedColorSegment support."
                )
                raise NotImplementedError(msg)
            return _build_color_jitter_matrix(params)

        msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
        raise NotImplementedError(msg)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply a TorchVision transform directly via its native forward method.

        TorchVision v1 transforms are applied per sample because they accept
        ``(channels, height, width)`` inputs. TorchVision v2 transforms are applied to the
        whole ``(batch_size, channels, height, width)`` tensor to preserve batch-wide randomness.

        Note:
            The v1 path loops ``batch_size`` times and calls ``torch.stack``, giving
            O(batch_size) allocations. For large batches prefer v2 transforms or use
            :meth:`~fuse_augmentations.Compose.from_params` to stay in the
            fused path and avoid passthrough entirely.

        Args:
            transform: A TorchVision transform instance.
            image: ``(batch_size, channels, height, width)`` float32 image tensor.
            **kwargs: Unused; accepted for protocol compatibility.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor on the same device as input.

        """
        if image.shape[0] == 0:
            return image

        device = image.device
        dtype = image.dtype
        batch_size = image.shape[0]

        if _is_torchvision_v2_transform(transform):
            image_output = transform(image)  # type: ignore[operator]
            return cast(torch.Tensor, image_output.to(device=device, dtype=dtype))

        # v1 per-sample path: TV v1 transforms accept (channels, height, width) input.
        # For batch_size=1, unsqueeze(0) creates a view — no data copy vs torch.stack.
        if batch_size == 1:
            image_output = transform(image[0])  # type: ignore[operator]
            return cast(torch.Tensor, image_output.unsqueeze(0).to(device=device, dtype=dtype))

        ndarray_results = []
        for idx_sample in range(batch_size):
            image_output = transform(image[idx_sample])  # type: ignore[operator]
            ndarray_results.append(image_output)

        return torch.stack(ndarray_results).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_affine_params(
    transform: object,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    shared_across_batch: bool = False,
) -> dict[str, torch.Tensor]:
    """Call ``RandomAffine.get_params()`` B times, collect into canonical dict.

    TorchVision's ``get_params`` returns ``(angle, translations, scale, shear)``
    where ``shear`` is a tuple of ``(shear_x_deg, shear_y_deg)``.

    Args:
        transform: A TorchVision ``RandomAffine`` instance.
        batch_size: Batch size.
        height: Image height.
        width: Image width.
        device: Target device for returned tensors.
        shared_across_batch: Whether TorchVision should sample one parameter
            set for the entire batch.

    Returns:
        Dict with keys ``angle_rad``, ``translate_x``, ``translate_y``,
        ``scale``, ``shear_x_rad``, ``shear_y_rad`` as ``(batch_size,)`` tensors.

    """
    angles = []
    translate_xs = []
    translate_ys = []
    scales = []
    shear_xs = []
    shear_ys = []

    sample_count = 1 if shared_across_batch else batch_size
    for _ in range(sample_count):
        angle, translations, scale_val, shear = type(transform).get_params(  # type: ignore[attr-defined]
            transform.degrees,  # type: ignore[attr-defined]
            transform.translate,  # type: ignore[attr-defined]
            transform.scale,  # type: ignore[attr-defined]
            transform.shear,  # type: ignore[attr-defined]
            img_size=(width, height),
        )
        angles.append(math.radians(angle))
        translate_xs.append(float(translations[0]))
        translate_ys.append(float(translations[1]))
        scales.append(float(scale_val))
        shear_xs.append(math.radians(shear[0]))
        shear_ys.append(math.radians(shear[1]))

    return {
        "angle_rad": torch.tensor(angles, dtype=torch.float32, device=device),
        "translate_x": torch.tensor(translate_xs, dtype=torch.float32, device=device),
        "translate_y": torch.tensor(translate_ys, dtype=torch.float32, device=device),
        "scale": torch.tensor(scales, dtype=torch.float32, device=device),
        "shear_x_rad": torch.tensor(shear_xs, dtype=torch.float32, device=device),
        "shear_y_rad": torch.tensor(shear_ys, dtype=torch.float32, device=device),
    }


def _sample_rotation_angle(transform: object) -> float:
    """Sample one rotation angle in degrees using TorchVision's native helper."""
    return float(type(transform).get_params(transform.degrees))  # type: ignore[attr-defined]


def _build_affine_matrix(
    params: dict[str, torch.Tensor],
    height: int,
    width: int,
) -> torch.Tensor:
    """Compose full RandomAffine matrix matching TorchVision's semantics.

    TorchVision builds the forward affine as ``T * C * RSS * C^-1`` where
    ``RSS = R(rot) * S(scale) * SHy(scale_y) * SHx(scale_x)`` is a single 2x2 block
    centered once.  This function reproduces that composition in pixel
    coordinates with center ``((width-1)/2, (height-1)/2)`` (align_corners=True).

    The rotation centre intentionally differs from native TorchVision's
    ``width/2`` centre — see ``TestAlignCornersOffset``.

    Args:
        params: Canonical-unit parameter dict from ``_sample_affine_params``.
        height: Image height.
        width: Image width.

    Returns:
        ``(batch_size, 3, 3)`` composed forward matrix.

    """
    # Determine batch size from any param
    batch_size = None
    device = torch.device("cpu")
    dtype = torch.float32
    for param_value in params.values():
        if isinstance(param_value, torch.Tensor):
            batch_size = param_value.shape[0]
            device = param_value.device
            dtype = param_value.dtype
            break
    if batch_size is None:
        return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    rot = params.get("angle_rad", torch.zeros(batch_size, device=device, dtype=dtype))
    scale = params.get("scale", torch.ones(batch_size, device=device, dtype=dtype))
    shear_x = params.get("shear_x_rad", torch.zeros(batch_size, device=device, dtype=dtype))
    shear_y = params.get("shear_y_rad", torch.zeros(batch_size, device=device, dtype=dtype))
    translation_x = params.get("translate_x", torch.zeros(batch_size, device=device, dtype=dtype))
    translation_y = params.get("translate_y", torch.zeros(batch_size, device=device, dtype=dtype))

    # RSS 2x2 block: R(rot) * S(scale) * SHy(shear_y) * SHx(shear_x)
    # Matches TorchVision's _get_inverse_affine_matrix (functional.py L1037-1040)
    cos_sy = torch.cos(shear_y)
    tan_sx = torch.tan(shear_x)
    cos_rot_sy = torch.cos(rot - shear_y)
    sin_rot_sy = torch.sin(rot - shear_y)
    sin_rot = torch.sin(rot)
    cos_rot = torch.cos(rot)

    m00 = scale * cos_rot_sy / cos_sy
    m01 = scale * (-cos_rot_sy * tan_sx / cos_sy - sin_rot)
    m10 = scale * sin_rot_sy / cos_sy
    m11 = scale * (-sin_rot_sy * tan_sx / cos_sy + cos_rot)

    # Forward matrix: M = T_translate * C * [[m00,m01],[m10,m11]] * C^-1
    zeros = torch.zeros(batch_size, device=device, dtype=dtype)
    ones = torch.ones(batch_size, device=device, dtype=dtype)

    row0 = torch.stack([m00, m01, center_x * (1 - m00) - center_y * m01 + translation_x], dim=-1)
    row1 = torch.stack([m10, m11, center_y * (1 - m11) - center_x * m10 + translation_y], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _sample_crop_resize_params(
    transform: object,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    shared_across_batch: bool = False,
) -> dict[str, torch.Tensor]:
    """Call ``RandomResizedCrop.get_params()`` and collect into canonical dict.

    TorchVision's ``get_params(img, scale, ratio)`` returns
    ``(top, left, height, width)`` where height/width are the crop dimensions.

    Args:
        transform: A TorchVision ``RandomResizedCrop`` instance.
        batch_size: Batch size.
        height: Image height.
        width: Image width.
        device: Target device for returned tensors.
        shared_across_batch: Whether to sample one parameter set for the batch
            (v2 behaviour).

    Returns:
        Dict with keys ``crop_top``, ``crop_left``, ``crop_h``, ``crop_w``,
        ``target_h``, ``target_w`` as ``(batch_size,)`` or ``(1,)`` float32 tensors.

    """
    target_h_val, target_w_val = transform.size  # type: ignore[attr-defined]
    sample_count = 1 if shared_across_batch else batch_size

    # get_params needs an image-like tensor for dimension extraction.
    # v2 accepts (batch_size, channels, height, width); v1 accepts (channels, height, width).
    # Use (1, height, width) for both.
    dummy = torch.zeros(1, height, width)

    tops: list[float] = []
    lefts: list[float] = []
    crop_hs: list[float] = []
    crop_ws: list[float] = []

    for _ in range(sample_count):
        top, left, crop_h, crop_w = type(transform).get_params(  # type: ignore[attr-defined]
            dummy,
            scale=transform.scale,  # type: ignore[attr-defined]
            ratio=transform.ratio,  # type: ignore[attr-defined]
        )
        tops.append(float(top))
        lefts.append(float(left))
        crop_hs.append(float(crop_h))
        crop_ws.append(float(crop_w))

    return {
        "crop_top": torch.tensor(tops, dtype=torch.float32, device=device),
        "crop_left": torch.tensor(lefts, dtype=torch.float32, device=device),
        "crop_h": torch.tensor(crop_hs, dtype=torch.float32, device=device),
        "crop_w": torch.tensor(crop_ws, dtype=torch.float32, device=device),
        "target_h": torch.full((sample_count,), float(target_h_val), dtype=torch.float32, device=device),
        "target_w": torch.full((sample_count,), float(target_w_val), dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Color matrix helpers
# ---------------------------------------------------------------------------

_MIDPOINT = 0.5  # Fixed midpoint for contrast approximation


def _sample_color_jitter_params(
    transform: object,
    batch_size: int,
    device: torch.device,
    shared_across_batch: bool = False,
) -> dict[str, torch.Tensor]:
    """Sample parameters for ``ColorJitter`` and return as canonical dict.

    Calls ``get_params()`` to obtain brightness, contrast, saturation, hue
    factors and the application order. Only brightness and contrast factors
    are used for the 4x4 matrix; saturation and hue are stored for
    completeness.

    Args:
        transform: A TorchVision ``ColorJitter`` instance.
        batch_size: Batch size.
        device: Target device for returned tensors.
        shared_across_batch: Whether to sample a single parameter set for the
            entire batch (v2 behaviour).

    Returns:
        Dict with keys ``brightness_factor``, ``contrast_factor``, and ``order``.

    """
    sample_count = 1 if shared_across_batch else batch_size
    brightness_list: list[float] = []
    contrast_list: list[float] = []
    order_tensors: list[torch.Tensor] = []

    for _ in range(sample_count):
        function_indices, brightness_factor, contrast_factor, _saturation_factor, _hue_factor = type(
            transform
        ).get_params(  # type: ignore[attr-defined]
            transform.brightness,  # type: ignore[attr-defined]
            transform.contrast,  # type: ignore[attr-defined]
            transform.saturation,  # type: ignore[attr-defined]
            transform.hue,  # type: ignore[attr-defined]
        )
        brightness_list.append(float(brightness_factor) if brightness_factor is not None else 1.0)
        contrast_list.append(float(contrast_factor) if contrast_factor is not None else 1.0)
        order_tensors.append(function_indices.to(device=device, dtype=torch.int64))

    return {
        "brightness_factor": torch.tensor(brightness_list, dtype=torch.float32, device=device),
        "contrast_factor": torch.tensor(contrast_list, dtype=torch.float32, device=device),
        # shared_across_batch=True  → single order (num_ops,)
        # shared_across_batch=False → per-sample order (batch_size, num_ops)
        "order": order_tensors[0] if shared_across_batch else torch.stack(order_tensors),
    }


def _make_eye4(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return ``(batch_size, 4, 4)`` identity matrices."""
    return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()


def _build_color_jitter_matrix(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build ``(batch_size, 4, 4)`` homogeneous matrix for TorchVision ColorJitter.

    Composes brightness (multiplicative: ``brightness_factors * c``) and contrast (midpoint-0.5
    approximation: ``contrast_factors * c + (1-contrast_factors)*0.5``) in the sampled ``order``.
    Saturation and hue steps are treated as identity.

    Args:
        params: Parameter dict containing ``brightness_factor``, ``contrast_factor``,
            and ``order`` tensors.

    Returns:
        ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

    """
    brightness_factors = params["brightness_factor"]  # (batch_size,)
    contrast_factors = params["contrast_factor"]  # (batch_size,)
    order = params["order"]  # (num_ops,) shared or (batch_size, num_ops) per-sample
    batch_size = brightness_factors.shape[0]
    device = brightness_factors.device
    dtype = brightness_factors.dtype

    mtx_acc = _make_eye4(batch_size, device, dtype)

    if order.dim() == 1:
        # Shared order across the batch — original fast path.
        for op_index in order.tolist():
            operation_id = int(op_index)
            if operation_id == 0:
                # Brightness: multiplicative c' = brightness_factors * c
                mtx_step = _make_eye4(batch_size, device, dtype)
                mtx_step[:, 0, 0] = brightness_factors
                mtx_step[:, 1, 1] = brightness_factors
                mtx_step[:, 2, 2] = brightness_factors
            elif operation_id == 1:
                # Contrast: midpoint approximation c' = contrast_factors * c + (1-contrast_factors)*0.5
                mtx_step = _make_eye4(batch_size, device, dtype)
                mtx_step[:, 0, 0] = contrast_factors
                mtx_step[:, 1, 1] = contrast_factors
                mtx_step[:, 2, 2] = contrast_factors
                bias = (1.0 - contrast_factors) * _MIDPOINT
                mtx_step[:, 0, 3] = bias
                mtx_step[:, 1, 3] = bias
                mtx_step[:, 2, 3] = bias
            else:
                # Saturation (operation_id=2) and hue (operation_id=3) treated as identity
                continue
            # Compose: mtx_acc = mtx_step @ mtx_acc
            mtx_acc = torch.bmm(mtx_step, mtx_acc)
    else:
        # Per-sample order (batch_size, num_ops) — v1 ColorJitter samples order per image.
        for b_idx in range(batch_size):
            mtx_b = _make_eye4(1, device, dtype)
            for op_index in order[b_idx].tolist():
                operation_id = int(op_index)
                if operation_id == 0:
                    mtx_step_b = _make_eye4(1, device, dtype)
                    mtx_step_b[:, 0, 0] = brightness_factors[b_idx]
                    mtx_step_b[:, 1, 1] = brightness_factors[b_idx]
                    mtx_step_b[:, 2, 2] = brightness_factors[b_idx]
                elif operation_id == 1:
                    mtx_step_b = _make_eye4(1, device, dtype)
                    mtx_step_b[:, 0, 0] = contrast_factors[b_idx]
                    mtx_step_b[:, 1, 1] = contrast_factors[b_idx]
                    mtx_step_b[:, 2, 2] = contrast_factors[b_idx]
                    bias_b = (1.0 - contrast_factors[b_idx]) * _MIDPOINT
                    mtx_step_b[:, 0, 3] = bias_b
                    mtx_step_b[:, 1, 3] = bias_b
                    mtx_step_b[:, 2, 3] = bias_b
                else:
                    continue
                mtx_b = torch.bmm(mtx_step_b, mtx_b)
            mtx_acc[b_idx] = mtx_b[0]

    return mtx_acc


# ---------------------------------------------------------------------------
# NumPy-native matrix builders for batch_size=1 cv2 warp fast path
# ---------------------------------------------------------------------------


def _affine_matrix_np_b1_tv(
    params: dict[str, torch.Tensor],
    center_x: float,
    center_y: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compose TorchVision-style affine matrix from canonical params (batch_size=1, NumPy only).

    Implements the ``T * C * RSS * C^-1`` composition used by TorchVision's
    ``_get_inverse_affine_matrix``.  Extracts scalar values from ``params`` via
    ``.item()`` and computes the (3, 3) float64 matrix entirely in NumPy.

    Args:
        params: Canonical parameter dict from ``TorchVisionAdapter.sample_params``.
        center_x: Horizontal centre in pixels ``(width - 1) / 2``.
        center_y: Vertical centre in pixels ``(height - 1) / 2``.

    Returns:
        ``(3, 3)`` float64 forward affine matrix in pixel coordinates.

    """
    import numpy as np

    rot = float(params["angle_rad"].item()) if "angle_rad" in params else 0.0
    scale = float(params["scale"].item()) if "scale" in params else 1.0
    shear_x = float(params["shear_x_rad"].item()) if "shear_x_rad" in params else 0.0
    shear_y = float(params["shear_y_rad"].item()) if "shear_y_rad" in params else 0.0
    translation_x = float(params["translate_x"].item()) if "translate_x" in params else 0.0
    translation_y = float(params["translate_y"].item()) if "translate_y" in params else 0.0

    cos_sy = np.cos(shear_y)
    tan_sx = np.tan(shear_x)
    cos_rot_sy = np.cos(rot - shear_y)
    sin_rot_sy = np.sin(rot - shear_y)
    sin_rot = np.sin(rot)
    cos_rot = np.cos(rot)

    m00 = scale * cos_rot_sy / cos_sy
    m01 = scale * (-cos_rot_sy * tan_sx / cos_sy - sin_rot)
    m10 = scale * sin_rot_sy / cos_sy
    m11 = scale * (-sin_rot_sy * tan_sx / cos_sy + cos_rot)

    return np.array(
        [
            [m00, m01, center_x * (1.0 - m00) - center_y * m01 + translation_x],
            [m10, m11, center_y * (1.0 - m11) - center_x * m10 + translation_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_matrix_numpy_b1_tv(
    transform: object,
    params: dict[str, torch.Tensor],
    height: int,
    width: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return (3, 3) float64 pixel-space matrix for batch_size=1, bypassing torch tensor creation.

    Drop-in replacement for
    ``TorchVisionAdapter.build_matrix(transform, params, height, width)[0].double().cpu().numpy()``
    in the batch_size=1 CPU cv2 warp path.  Extracts scalar values from ``params`` via
    ``.item()`` and computes the matrix directly in NumPy, avoiding the 8-12
    intermediate ``(1,)`` / ``(1, 3, 3)`` torch tensor allocations produced by
    the torch-based construction path.

    Args:
        transform: A TorchVision augmentation transform (type-dispatched).
        params: Canonical parameter dict from ``TorchVisionAdapter.sample_params``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward affine matrix in pixel coordinates.

    """
    import numpy as np

    center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5
    ttype = type(transform)

    if ttype in _HFLIP_TYPES_FS:
        return np.array([[-1.0, 0.0, float(width - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype in _VFLIP_TYPES_FS:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype in _ROTATION_TYPES_FS:
        angle_rad = float(params["angle_rad"].item())
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return np.array(
            [
                [cos_a, -sin_a, center_x * (1.0 - cos_a) + center_y * sin_a],
                [sin_a, cos_a, center_y * (1.0 - cos_a) - center_x * sin_a],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    if ttype in _AFFINE_TYPES_FS:
        return _affine_matrix_np_b1_tv(params, center_x, center_y)

    # Fallback: torch path for perspective, crop-resize, colour jitter, unknown types.
    return TorchVisionAdapter.build_matrix(transform, params, height, width)[0].double().cpu().numpy()


def sample_and_build_matrix_numpy_b1_tv(
    transform: object,
    input_shape: tuple[int, int, int, int],
    height: int,
    width: int,
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Sample params and build ``(3, 3)`` float64 matrix for batch_size=1, entirely in numpy.

    Fuses :func:`TorchVisionAdapter.sample_params` + :func:`build_matrix_numpy_b1_tv`
    into a single call that invokes ``get_params`` directly, extracts scalar
    floats without creating intermediate canonical torch tensors, and computes
    the affine matrix in NumPy.  Compared to the two-step path this avoids
    4-6 small ``(1,)``/``(1, 3, 3)`` torch tensor allocations per transform per
    forward call.

    Only handles the types common in the benchmark hot path
    (``RandomRotation``, ``RandomAffine``, ``RandomHorizontalFlip``,
    ``RandomVerticalFlip``).  All other types return ``None`` to signal that
    the caller should fall back to the two-step path.

    Args:
        transform: A TorchVision augmentation transform (type-dispatched).
        input_shape: ``(batch_size, channels, height, width)`` shape tuple (unused for TV; kept for
            interface compatibility with the Kornia fused builder).
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward affine matrix, or ``None`` if this type is
        not handled (caller must fall back to the two-step path).

    """
    ttype = type(transform)
    center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5

    if ttype in _HFLIP_TYPES_FS:
        return np.array([[-1.0, 0.0, float(width - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype in _VFLIP_TYPES_FS:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype in _ROTATION_TYPES_FS:
        angle_deg = float(ttype.get_params(transform.degrees))  # type: ignore[attr-defined]
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.cos(angle_rad)
        return np.array(
            [
                [cos_a, -sin_a, center_x * (1.0 - cos_a) + center_y * sin_a],
                [sin_a, cos_a, center_y * (1.0 - cos_a) - center_x * sin_a],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    if ttype in _AFFINE_TYPES_FS:
        angle, translations, scale_val, shear = ttype.get_params(  # type: ignore[attr-defined]
            transform.degrees,  # type: ignore[attr-defined]
            transform.translate,  # type: ignore[attr-defined]
            transform.scale,  # type: ignore[attr-defined]
            transform.shear,  # type: ignore[attr-defined]
            img_size=(width, height),
        )
        rot = math.radians(float(angle))
        translation_x = float(translations[0])
        translation_y = float(translations[1])
        scale = float(scale_val)
        shear_x = math.radians(float(shear[0]))
        shear_y = math.radians(float(shear[1]))

        cos_sy = math.cos(shear_y)
        tan_sx = math.tan(shear_x)
        cos_rot_sy = math.cos(rot - shear_y)
        sin_rot_sy = math.sin(rot - shear_y)
        sin_rot = math.sin(rot)
        cos_rot = math.cos(rot)

        m00 = scale * cos_rot_sy / cos_sy
        m01 = scale * (-cos_rot_sy * tan_sx / cos_sy - sin_rot)
        m10 = scale * sin_rot_sy / cos_sy
        m11 = scale * (-sin_rot_sy * tan_sx / cos_sy + cos_rot)

        return np.array(
            [
                [m00, m01, center_x * (1.0 - m00) - center_y * m01 + translation_x],
                [m10, m11, center_y * (1.0 - m11) - center_x * m10 + translation_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return None
