"""TorchVision backend adapter for the fused affine engine.

Bridges TorchVision augmentation transforms to the canonical parameter
representation used by ``FusedAffineSegment``.

Supports both ``torchvision.transforms`` (v1) and ``torchvision.transforms.v2``
namespaces. Each geometric transform samples parameters via TorchVision's
``get_params()`` static method and reconstructs the affine matrix from
``fuse_augmentations.affine._matrix`` primitives.

Flip transforms (``RandomHorizontalFlip``, ``RandomVerticalFlip``) return an
empty parameter dict; ``build_matrix()`` constructs their matrices from the
shared matrix primitives.

Requires ``torchvision``.

Example:
    >>> from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
    >>> adapter = TorchVisionAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...TorchVisionAdapter...>

"""

from __future__ import annotations

import math
import warnings

import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import (
    hflip_matrix,
    matmul3x3,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)

# ---------------------------------------------------------------------------
# Transform registry -- lazy import guards (torchvision is optional)
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[type, TransformCategory] = {}
_HFLIP_TYPES: set[type] = set()
_VFLIP_TYPES: set[type] = set()
_ROTATION_TYPES: set[type] = set()
_AFFINE_TYPES: set[type] = set()

# v1: torchvision.transforms
try:
    from torchvision.transforms import RandomAffine as _V1RandomAffine
    from torchvision.transforms import RandomHorizontalFlip as _V1RandomHorizontalFlip
    from torchvision.transforms import RandomRotation as _V1RandomRotation
    from torchvision.transforms import RandomVerticalFlip as _V1RandomVerticalFlip

    TRANSFORM_REGISTRY[_V1RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    TRANSFORM_REGISTRY[_V1RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    TRANSFORM_REGISTRY[_V1RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    TRANSFORM_REGISTRY[_V1RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT

    _HFLIP_TYPES.add(_V1RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V1RandomVerticalFlip)
    _ROTATION_TYPES.add(_V1RandomRotation)
    _AFFINE_TYPES.add(_V1RandomAffine)
except ImportError:
    pass

# v2: torchvision.transforms.v2
try:
    from torchvision.transforms.v2 import RandomAffine as _V2RandomAffine
    from torchvision.transforms.v2 import RandomHorizontalFlip as _V2RandomHorizontalFlip
    from torchvision.transforms.v2 import RandomRotation as _V2RandomRotation
    from torchvision.transforms.v2 import RandomVerticalFlip as _V2RandomVerticalFlip

    TRANSFORM_REGISTRY[_V2RandomRotation] = TransformCategory.GEOMETRIC_INTERP
    TRANSFORM_REGISTRY[_V2RandomAffine] = TransformCategory.GEOMETRIC_INTERP
    TRANSFORM_REGISTRY[_V2RandomHorizontalFlip] = TransformCategory.GEOMETRIC_EXACT
    TRANSFORM_REGISTRY[_V2RandomVerticalFlip] = TransformCategory.GEOMETRIC_EXACT

    _HFLIP_TYPES.add(_V2RandomHorizontalFlip)
    _VFLIP_TYPES.add(_V2RandomVerticalFlip)
    _ROTATION_TYPES.add(_V2RandomRotation)
    _AFFINE_TYPES.add(_V2RandomAffine)
except ImportError:
    pass

# Freeze into frozensets for immutability after init
_HFLIP_TYPES_FS: frozenset[type] = frozenset(_HFLIP_TYPES)
_VFLIP_TYPES_FS: frozenset[type] = frozenset(_VFLIP_TYPES)
_ROTATION_TYPES_FS: frozenset[type] = frozenset(_ROTATION_TYPES)
_AFFINE_TYPES_FS: frozenset[type] = frozenset(_AFFINE_TYPES)


def _check_expand(transform: object) -> None:
    """Raise if transform has expand=True (unsupported by fused engine)."""
    if getattr(transform, "expand", False):
        msg = (
            "TorchVision RandomRotation with expand=True is not supported "
            "by the fused engine"
        )
        raise ValueError(msg)


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
        for base_type, cat in TRANSFORM_REGISTRY.items():
            if isinstance(transform, base_type):
                return cat
        warnings.warn(
            f"Unknown TorchVision transform {type(transform).__name__!r}; "
            "treating as SPATIAL_KERNEL barrier.",
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

        For ``RandomRotation``, calls ``get_params(degrees)`` B times and
        converts to radians. For ``RandomAffine``, calls ``get_params(...)``
        B times to obtain angle, translations, scale, and shear values.

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
            angles = []
            for _ in range(B):
                angle_deg = type(transform).get_params(transform.degrees)  # type: ignore[attr-defined]
                angles.append(math.radians(angle_deg))
            return {
                "angle_rad": torch.tensor(angles, dtype=torch.float32, device=device),
            }

        # RandomAffine
        if isinstance(transform, tuple(_AFFINE_TYPES_FS)):
            return _sample_affine_params(transform, B, H, W, device)

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
        ``rotation_matrix``. For ``RandomAffine``, composes in TorchVision's
        documented order: center -> rotate -> scale -> shear -> translate -> uncenter.

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
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply a TorchVision transform directly via its native forward method.

        Loops over the batch dimension, calling the transform on each
        ``(C, H, W)`` tensor individually.

        Args:
            transform: A TorchVision transform instance.
            image: ``(B, C, H, W)`` float32 image tensor.
            **kwargs: Unused; accepted for protocol compatibility.

        Returns:
            Transformed ``(B, C, H, W)`` tensor on the same device as input.

        """
        device = image.device
        dtype = image.dtype
        B = image.shape[0]  # noqa: N806

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
    for _ in range(B):
        angle, translations, sc, shear = type(t).get_params(  # type: ignore[attr-defined]
            t.degrees,  # type: ignore[attr-defined]
            t.translate,  # type: ignore[attr-defined]
            t.scale,  # type: ignore[attr-defined]
            t.shear,  # type: ignore[attr-defined]
            img_size=(H, W),
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


def _build_affine_matrix(
    params: dict[str, torch.Tensor],
    H: int,  # noqa: N803
    W: int,  # noqa: N803
) -> torch.Tensor:
    """Compose full RandomAffine matrix in TorchVision's documented order.

    Order: center -> rotate -> scale -> shear -> translate -> uncenter.

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

    # Start with identity
    acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()

    # Rotation (center-rotate-uncenter is handled by rotation_matrix)
    if "angle_rad" in params:
        R = rotation_matrix(params["angle_rad"], H=H, W=W)  # noqa: N806
        acc = matmul3x3(R, acc)

    # Scale (uniform -- TorchVision RandomAffine returns a single scale factor)
    if "scale" in params:
        S = scale_matrix(params["scale"], params["scale"], H=H, W=W)  # noqa: N806
        acc = matmul3x3(S, acc)

    # X-Shear
    if "shear_x_rad" in params:
        shear_x_tan = torch.tan(params["shear_x_rad"])
        Sh_x = shear_x_matrix(shear_x_tan, H=H, W=W)  # noqa: N806
        acc = matmul3x3(Sh_x, acc)

    # Y-Shear
    if "shear_y_rad" in params:
        shear_y_tan = torch.tan(params["shear_y_rad"])
        Sh_y = shear_y_matrix(shear_y_tan, H=H, W=W)  # noqa: N806
        acc = matmul3x3(Sh_y, acc)

    # Translation
    if "translate_x" in params and "translate_y" in params:
        T = translate_matrix(params["translate_x"], params["translate_y"])  # noqa: N806
        acc = matmul3x3(T, acc)

    return acc
