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
from typing import cast

import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import (
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

# v1: torchvision.transforms
try:
    from torchvision.transforms import RandomAffine as _V1RandomAffine
    from torchvision.transforms import RandomHorizontalFlip as _V1RandomHorizontalFlip
    from torchvision.transforms import RandomPerspective as _V1RandomPerspective
    from torchvision.transforms import RandomRotation as _V1RandomRotation
    from torchvision.transforms import RandomVerticalFlip as _V1RandomVerticalFlip

    _TRANSFORM_REGISTRY[_V1RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V1RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V1RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V1RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V1RandomPerspective] = TransformCategory.PROJECTIVE

    _HFLIP_TYPES.add(_V1RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V1RandomVerticalFlip)
    _ROTATION_TYPES.add(_V1RandomRotation)
    _AFFINE_TYPES.add(_V1RandomAffine)
    _PERSPECTIVE_TYPES.add(_V1RandomPerspective)
except ImportError:
    pass

# v2: torchvision.transforms.v2
try:
    from torchvision.transforms.v2 import RandomAffine as _V2RandomAffine
    from torchvision.transforms.v2 import RandomHorizontalFlip as _V2RandomHorizontalFlip
    from torchvision.transforms.v2 import RandomPerspective as _V2RandomPerspective
    from torchvision.transforms.v2 import RandomRotation as _V2RandomRotation
    from torchvision.transforms.v2 import RandomVerticalFlip as _V2RandomVerticalFlip

    _TRANSFORM_REGISTRY[_V2RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V2RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    _TRANSFORM_REGISTRY[_V2RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V2RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT
    _TRANSFORM_REGISTRY[_V2RandomPerspective] = TransformCategory.PROJECTIVE

    _HFLIP_TYPES.add(_V2RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V2RandomVerticalFlip)
    _ROTATION_TYPES.add(_V2RandomRotation)
    _AFFINE_TYPES.add(_V2RandomAffine)
    _PERSPECTIVE_TYPES.add(_V2RandomPerspective)
except ImportError:
    pass

# Freeze into frozensets for immutability after init
_HFLIP_TYPES_FS: frozenset[type] = frozenset(_HFLIP_TYPES)
_VFLIP_TYPES_FS: frozenset[type] = frozenset(_VFLIP_TYPES)
_ROTATION_TYPES_FS: frozenset[type] = frozenset(_ROTATION_TYPES)
_AFFINE_TYPES_FS: frozenset[type] = frozenset(_AFFINE_TYPES)
_PERSPECTIVE_TYPES_FS: frozenset[type] = frozenset(_PERSPECTIVE_TYPES)


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
        """Sample random parameters for a batch of B images.

        TorchVision v1 samples one parameter set per image, while TorchVision
        v2 samples one parameter set per input tensor. For batched v2 inputs,
        the returned tensors therefore have shape ``(1,)`` and are expanded
        later by the fused segment.

        Flip transforms return a ``{"_batch_size": tensor([B])}`` sentinel.

        Args:
            transform: A TorchVision transform instance.
            input_shape: ``(B, C, H, W)`` shape tuple.
            device: Target device for the returned tensors.

        Returns:
            Dict of parameter tensors keyed by canonical names.

        """
        _check_expand(transform)
        B, _C, H, W = input_shape  # noqa: N806

        # Flip transforms -- no sampled params, only need batch size
        if isinstance(transform, tuple(_HFLIP_TYPES_FS | _VFLIP_TYPES_FS)):
            return {"_batch_size": torch.tensor([B], device=device, dtype=torch.int64)}

        # RandomRotation
        if isinstance(transform, tuple(_ROTATION_TYPES_FS)):
            if _is_torchvision_v2_transform(transform):
                angle_deg = _sample_rotation_angle(transform)
                return {
                    "angle_rad": torch.tensor([math.radians(angle_deg)], dtype=torch.float32, device=device),
                }
            angles = []
            for _ in range(B):
                angle_deg = _sample_rotation_angle(transform)
                angles.append(math.radians(angle_deg))
            return {
                "angle_rad": torch.tensor(angles, dtype=torch.float32, device=device),
            }

        # RandomAffine
        if isinstance(transform, tuple(_AFFINE_TYPES_FS)):
            return _sample_affine_params(
                transform, B, H, W, device, shared_across_batch=_is_torchvision_v2_transform(transform)
            )

        # RandomPerspective
        if isinstance(transform, tuple(_PERSPECTIVE_TYPES_FS)):
            is_v2 = _is_torchvision_v2_transform(transform)
            sample_count = 1 if is_v2 else B
            starts, ends = [], []
            for _ in range(sample_count):
                sp, ep = type(transform).get_params(W, H, transform.distortion_scale)  # type: ignore[attr-defined]
                starts.append(sp)  # list of 4 [x,y] pairs
                ends.append(ep)
            # Convert to (sample_count, 4, 2) tensors
            start_t = torch.tensor(starts, dtype=torch.float32, device=device)  # (count, 4, 2)
            end_t = torch.tensor(ends, dtype=torch.float32, device=device)
            return {"start_points": start_t.clone(), "end_points": end_t.clone()}

        # Unknown -- return empty
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> torch.Tensor:
        """Build a ``(B, 3, 3)`` pixel-space forward affine matrix.

        For ``RandomRotation``, composes center-rotate-uncenter via
        ``rotation_matrix``. For ``RandomAffine``, composes in the order:
        center-rotate-uncenter, then scale about center, then shear, then
        translate. Note: uses ``(W-1)/2`` rotation centre (align_corners=True),
        not TorchVision's native ``W/2`` — see ``TestAlignCornersOffset``.

        Flip transforms use ``hflip_matrix`` / ``vflip_matrix`` expanded to
        batch size B.

        Args:
            transform: A TorchVision transform instance.
            params: Parameter dict from ``sample_params()``.
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            ``(B, 3, 3)`` forward affine matrix in pixel coordinates.

        """
        if isinstance(transform, tuple(_HFLIP_TYPES_FS)):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            return hflip_matrix(W=W, batch_size=B, device=device, dtype=torch.float32)

        if isinstance(transform, tuple(_VFLIP_TYPES_FS)):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            return vflip_matrix(H=H, batch_size=B, device=device, dtype=torch.float32)

        if isinstance(transform, tuple(_ROTATION_TYPES_FS)):
            return rotation_matrix(params["angle_rad"], H=H, W=W)

        if isinstance(transform, tuple(_AFFINE_TYPES_FS)):
            return _build_affine_matrix(params, H, W)

        if isinstance(transform, tuple(_PERSPECTIVE_TYPES_FS)):
            return perspective_from_points(params["start_points"], params["end_points"])

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
    def same_on_batch(transform: object) -> bool:
        """Return whether randomness should be shared across the input batch."""
        return _is_torchvision_v2_transform(transform) or bool(getattr(transform, "same_on_batch", False))

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply a TorchVision transform directly via its native forward method.

        TorchVision v1 transforms are applied per sample because they accept
        ``(C, H, W)`` inputs. TorchVision v2 transforms are applied to the
        whole ``(B, C, H, W)`` tensor to preserve batch-wide randomness.

        Note:
            The v1 path loops ``B`` times and calls ``torch.stack``, giving
            O(B) allocations. For large batches prefer v2 transforms or use
            :meth:`~fuse_augmentations.Compose.from_params` to stay in the
            fused path and avoid passthrough entirely.

        Args:
            transform: A TorchVision transform instance.
            image: ``(B, C, H, W)`` float32 image tensor.
            **kwargs: Unused; accepted for protocol compatibility.

        Returns:
            Transformed ``(B, C, H, W)`` tensor on the same device as input.

        """
        if image.shape[0] == 0:
            return image

        device = image.device
        dtype = image.dtype
        B = image.shape[0]  # noqa: N806

        if _is_torchvision_v2_transform(transform):
            out = transform(image)  # type: ignore[operator]
            return cast(torch.Tensor, out.to(device=device, dtype=dtype))

        results = []
        for i in range(B):
            out = transform(image[i])  # type: ignore[operator]
            results.append(out)

        return torch.stack(results).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_affine_params(
    transform: object,
    B: int,  # noqa: N803
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    device: torch.device,
    shared_across_batch: bool = False,
) -> dict[str, torch.Tensor]:
    """Call ``RandomAffine.get_params()`` B times, collect into canonical dict.

    TorchVision's ``get_params`` returns ``(angle, translations, scale, shear)``
    where ``shear`` is a tuple of ``(shear_x_deg, shear_y_deg)``.

    Args:
        transform: A TorchVision ``RandomAffine`` instance.
        B: Batch size.
        H: Image height.
        W: Image width.
        device: Target device for returned tensors.
        shared_across_batch: Whether TorchVision should sample one parameter
            set for the entire batch.

    Returns:
        Dict with keys ``angle_rad``, ``translate_x``, ``translate_y``,
        ``scale``, ``shear_x_rad``, ``shear_y_rad`` as ``(B,)`` tensors.

    """
    angles = []
    translate_xs = []
    translate_ys = []
    scales = []
    shear_xs = []
    shear_ys = []

    t = transform
    sample_count = 1 if shared_across_batch else B
    for _ in range(sample_count):
        angle, translations, sc, shear = type(t).get_params(  # type: ignore[attr-defined]
            t.degrees,  # type: ignore[attr-defined]
            t.translate,  # type: ignore[attr-defined]
            t.scale,  # type: ignore[attr-defined]
            t.shear,  # type: ignore[attr-defined]
            img_size=(W, H),
        )
        angles.append(math.radians(angle))
        translate_xs.append(float(translations[0]))
        translate_ys.append(float(translations[1]))
        scales.append(float(sc))
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
    H: int,  # noqa: N803
    W: int,  # noqa: N803
) -> torch.Tensor:
    """Compose full RandomAffine matrix matching TorchVision's semantics.

    TorchVision builds the forward affine as ``T * C * RSS * C^-1`` where
    ``RSS = R(rot) * S(scale) * SHy(sy) * SHx(sx)`` is a single 2x2 block
    centered once.  This function reproduces that composition in pixel
    coordinates with center ``((W-1)/2, (H-1)/2)`` (align_corners=True).

    The rotation centre intentionally differs from native TorchVision's
    ``W/2`` centre — see ``TestAlignCornersOffset``.

    Args:
        params: Canonical-unit parameter dict from ``_sample_affine_params``.
        H: Image height.
        W: Image width.

    Returns:
        ``(B, 3, 3)`` composed forward matrix.

    """
    # Determine batch size from any param
    B = None  # noqa: N806
    device = torch.device("cpu")
    dtype = torch.float32
    for v in params.values():
        if isinstance(v, torch.Tensor):
            B = v.shape[0]  # noqa: N806
            device = v.device
            dtype = v.dtype
            break
    if B is None:
        return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    rot = params.get("angle_rad", torch.zeros(B, device=device, dtype=dtype))
    sc = params.get("scale", torch.ones(B, device=device, dtype=dtype))
    sx = params.get("shear_x_rad", torch.zeros(B, device=device, dtype=dtype))
    sy = params.get("shear_y_rad", torch.zeros(B, device=device, dtype=dtype))
    tx = params.get("translate_x", torch.zeros(B, device=device, dtype=dtype))
    ty = params.get("translate_y", torch.zeros(B, device=device, dtype=dtype))

    # RSS 2x2 block: R(rot) * S(scale) * SHy(sy) * SHx(sx)
    # Matches TorchVision's _get_inverse_affine_matrix (functional.py L1037-1040)
    cos_sy = torch.cos(sy)
    tan_sx = torch.tan(sx)
    cos_rot_sy = torch.cos(rot - sy)
    sin_rot_sy = torch.sin(rot - sy)
    sin_rot = torch.sin(rot)
    cos_rot = torch.cos(rot)

    a = sc * cos_rot_sy / cos_sy
    b = sc * (-cos_rot_sy * tan_sx / cos_sy - sin_rot)
    c = sc * sin_rot_sy / cos_sy
    d = sc * (-sin_rot_sy * tan_sx / cos_sy + cos_rot)

    # Forward matrix: M = T_translate * C * [[a,b],[c,d]] * C^-1
    zeros = torch.zeros(B, device=device, dtype=dtype)
    ones = torch.ones(B, device=device, dtype=dtype)

    row0 = torch.stack([a, b, cx * (1 - a) - cy * b + tx], dim=-1)
    row1 = torch.stack([c, d, cy * (1 - d) - cx * c + ty], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
