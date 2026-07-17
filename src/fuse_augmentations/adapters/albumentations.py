"""Albumentations backend adapter for the fused geometric engine.

Bridges Albumentations augmentation transforms to the canonical parameter
representation used by the fused Albumentations geometric segments.

Each geometric transform (``A.Affine``, ``A.Rotate``, ``A.ShiftScaleRotate``,
``A.SafeRotate``) exposes a pre-built ``matrix`` key via
``get_params_dependent_on_data()`` -- a ``(3, 3)`` forward pixel-space affine
matrix in the same convention as ``affine._matrix``. The adapter reads this
matrix directly instead of reconstructing it from raw angle/scale/shear values.

``A.Perspective`` is also supported and mapped to
``TransformCategory.PROJECTIVE``. Its sampled transform matrix is treated as a
forward pixel-space homography and consumed by
``AlbuProjectiveSegment`` rather than the affine fusion path.

Flip transforms (``A.HorizontalFlip``, ``A.VerticalFlip``) return an empty parameter dict; ``build_matrix()``
constructs their matrices from inline NumPy helpers.

Requires ``albumentations >= 2.0``.


Example:
    >>> from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
    >>> adapter = AlbumentationsAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...AlbumentationsAdapter...>

"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.affine.matrix import crop_resize_matrix, hflip_matrix, matmul3x3, rotation_matrix, vflip_matrix
from fuse_augmentations.types import SamplingSemantics, TransformCategory

__doctest_skip__: list[str] = []
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["AlbumentationsAdapter.call_nonfused_numpy"]

# ---------------------------------------------------------------------------
# Inline NumPy matrix helpers (moved from _np_matrix.py)
# ---------------------------------------------------------------------------


def hflip_matrix_np(width: int) -> NDArray[np.float64]:
    """Build a (3, 3) pixel-space forward horizontal flip matrix.

    Maps ``x' = width - 1 - x``, ``y' = y``.

    Args:
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward horizontal flip matrix.

    Example:
        >>> mtx = hflip_matrix_np(width=4)
        >>> mtx[0].tolist()
        [-1.0, 0.0, 3.0]
        >>> mtx[1].tolist()
        [0.0, 1.0, 0.0]

    """
    return np.array(
        [
            [-1.0, 0.0, float(width - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def vflip_matrix_np(height: int) -> NDArray[np.float64]:
    """Build a (3, 3) pixel-space forward vertical flip matrix.

    Maps ``x' = x``, ``y' = height - 1 - y``.

    Args:
        height: Image height in pixels.

    Returns:
        ``(3, 3)`` float64 forward vertical flip matrix.

    Example:
        >>> mtx = vflip_matrix_np(height=4)
        >>> mtx[1].tolist()
        [0.0, -1.0, 3.0]
        >>> mtx[0].tolist()
        [1.0, 0.0, 0.0]

    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, float(height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Transform registry — lazy import guard (albumentations is optional)
# ---------------------------------------------------------------------------

try:
    from albumentations import D4 as _D4
    from albumentations import Affine as _Affine
    from albumentations import HorizontalFlip as _HorizontalFlip
    from albumentations import HueSaturationValue as _HueSaturationValue
    from albumentations import Normalize as _Normalize
    from albumentations import Perspective as _Perspective
    from albumentations import RandomBrightnessContrast as _RandomBrightnessContrast
    from albumentations import RandomResizedCrop as _RandomResizedCrop
    from albumentations import RandomRotate90 as _RandomRotate90
    from albumentations import Rotate as _Rotate
    from albumentations import SafeRotate as _SafeRotate
    from albumentations import ShiftScaleRotate as _ShiftScaleRotate
    from albumentations import Transpose as _Transpose
    from albumentations import VerticalFlip as _VerticalFlip

    TRANSFORM_REGISTRY: dict[type, TransformCategory] = {
        _Affine: TransformCategory.GEOMETRIC_INTERP,
        _Rotate: TransformCategory.GEOMETRIC_INTERP,
        _SafeRotate: TransformCategory.GEOMETRIC_INTERP,
        _ShiftScaleRotate: TransformCategory.GEOMETRIC_INTERP,
        _HorizontalFlip: TransformCategory.GEOMETRIC_EXACT,
        _VerticalFlip: TransformCategory.GEOMETRIC_EXACT,
        _RandomRotate90: TransformCategory.GEOMETRIC_EXACT,
        _D4: TransformCategory.GEOMETRIC_EXACT,
        _Transpose: TransformCategory.GEOMETRIC_EXACT,
        _Perspective: TransformCategory.PROJECTIVE,
        _RandomBrightnessContrast: TransformCategory.POINTWISE_LINEAR,
        _Normalize: TransformCategory.POINTWISE_LINEAR,
        # HueSaturationValue is pixel-wise (non-linear in RGB) — reorderable
        # but not linearly composable into a FusedColorSegment matrix.
        _HueSaturationValue: TransformCategory.POINTWISE,
        _RandomResizedCrop: TransformCategory.CROP_RESIZE_FIXED,
    }

    # Canonical base classes for fast isinstance dispatch in the adapter paths below.
    # NOTE: when adding a new transform to TRANSFORM_REGISTRY, also add it to the
    # appropriate frozenset here — category() iterates TRANSFORM_REGISTRY automatically,
    # but sample_params/build_matrix/exact_flip_dims use these frozensets directly.
    _HFLIP_TYPES: frozenset[type] = frozenset({_HorizontalFlip})
    _VFLIP_TYPES: frozenset[type] = frozenset({_VerticalFlip})
    _EXACT_DISCRETE_TYPES: frozenset[type] = frozenset({_RandomRotate90, _D4, _Transpose})
    _INTERP_TYPES: frozenset[type] = frozenset({_Affine, _Rotate, _SafeRotate, _ShiftScaleRotate})
    _COLOR_TYPES: frozenset[type] = frozenset({_RandomBrightnessContrast})
    _NORMALIZE_TYPES: frozenset[type] = frozenset({_Normalize})
    _CROP_RESIZE_TYPES: frozenset[type] = frozenset({_RandomResizedCrop})
    # POINTWISE (non-linear color): reorderable but no matrix — return identity.
    _POINTWISE_TYPES: frozenset[type] = frozenset({_HueSaturationValue})
    _ALL_REGISTRY_TYPES: frozenset[type] = frozenset(TRANSFORM_REGISTRY)

except ImportError:
    TRANSFORM_REGISTRY = {}
    _HFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _VFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _EXACT_DISCRETE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _INTERP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _COLOR_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _NORMALIZE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _CROP_RESIZE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _POINTWISE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _ALL_REGISTRY_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]


def _is_albu_instance(transform: object, candidates: frozenset[type]) -> bool:
    """Return whether ``transform`` is an instance of any Albumentations base type."""
    return any(isinstance(transform, base_type) for base_type in candidates)


class AlbumentationsAdapter:
    """Adapter between Albumentations transforms and the fused geometric engine.

    Implements the ``TransformAdapter`` protocol for the Albumentations backend.
    Supports ``A.Affine``, ``A.Rotate``, ``A.SafeRotate``, ``A.ShiftScaleRotate``
    (GEOMETRIC_INTERP) and ``A.HorizontalFlip``, ``A.VerticalFlip``
    (GEOMETRIC_EXACT). ``A.Perspective`` is classified as ``PROJECTIVE`` and
    routed through the projective segment path instead of affine fusion.

    Requires ``albumentations >= 2.0``. The adapter reads the pre-built ``matrix`` key returned by affine-style
    transforms and ``Perspective`` rather than reconstructing affine or projective matrices from raw parameters.

    Example:
        >>> adapter = AlbumentationsAdapter()
        >>> isinstance(adapter, AlbumentationsAdapter)
        True

    """

    #: Canonical op names Albumentations can build (mirrors ``resolver._albumentations_registry``; no shear/translate
    #: wrappers).
    capabilities: frozenset[str] = frozenset({
        "rotation",
        "affine",
        "hflip",
        "vflip",
        "scale",
        "perspective",
        "rotation90",
    })

    #: Albumentations draws one parameter set per sample (cv2 per-image loop).
    sampling_semantics: SamplingSemantics = "per_sample"

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory for the given Albumentations transform.

        Args:
            transform: An Albumentations transform instance.

        Returns:
            The category for the transform. Unknown transforms default to
            ``SPATIAL_KERNEL`` with a ``UserWarning``.

        """
        # Use isinstance against registered base types so that subclasses or
        # wrapped Albumentations transforms are classified correctly.
        for base_type, cat in TRANSFORM_REGISTRY.items():
            if isinstance(transform, base_type):
                return cat
        warnings.warn(
            f"Unknown Albumentations transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
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

        For ``GEOMETRIC_INTERP`` and ``PROJECTIVE`` transforms, calls
        ``get_params_dependent_on_data()`` once per sample and stacks the
        resulting ``matrix`` arrays into a ``(batch_size, 3, 3)`` tensor. A dummy
        ``(height, width, 1)`` float32 array is passed so the transform can compute
        center coordinates without reading actual pixel data.

        For flip transforms, returns a ``{"_batch_size": tensor([batch_size])}``
        sentinel so ``build_matrix()`` knows the required output shape.

        Args:
            transform: An Albumentations transform instance.
            input_shape: ``(batch_size, channels, height, width)`` shape tuple.
            device: Target device for the returned tensors.

        Returns:
            Dict of parameter tensors. ``"matrix"`` key holds ``(batch_size, 3, 3)``
            for interpolated affine transforms and projective transforms;
            ``"_batch_size"`` for flips.

        """
        batch_size, _channels, height, width = input_shape

        if _is_albu_instance(transform, _POINTWISE_TYPES):
            # Non-linear color op (e.g. HueSaturationValue): passthrough — no affine matrix.
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}
        if _is_albu_instance(transform, _HFLIP_TYPES | _VFLIP_TYPES):
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}
        if _is_albu_instance(transform, _EXACT_DISCRETE_TYPES):
            result: dict[str, torch.Tensor] = {
                "_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64),
            }
            # NOTE: real Albumentations transforms have no same_on_batch attribute
            # (Kornia concept) — this branch serves duck-typed wrappers/test doubles
            # that opt into batch-shared sampling; genuine albu objects always take
            # the per-sample path.
            same_on_batch = bool(getattr(transform, "same_on_batch", False))

            if TRANSFORM_REGISTRY and isinstance(transform, _RandomRotate90):
                if same_on_batch:
                    factor = int(transform.get_params()["factor"]) % 4
                    result["k90"] = torch.full((batch_size,), factor, device=device, dtype=torch.int64)
                    return result
                result["k90"] = torch.tensor(
                    [int(transform.get_params()["factor"]) % 4 for _ in range(batch_size)],
                    device=device,
                    dtype=torch.int64,
                )
                return result

            if TRANSFORM_REGISTRY and isinstance(transform, _D4):
                if same_on_batch:
                    elem = str(transform.get_params()["group_element"])
                    result["d4_code"] = torch.full(
                        (batch_size,), _D4_ELEM_TO_CODE[elem], device=device, dtype=torch.int64
                    )
                    return result
                result["d4_code"] = torch.tensor(
                    [_D4_ELEM_TO_CODE[str(transform.get_params()["group_element"])] for _ in range(batch_size)],
                    device=device,
                    dtype=torch.int64,
                )
                return result

            return result

        # POINTWISE_LINEAR (color transforms): sample alpha/beta per batch element
        if _is_albu_instance(transform, _COLOR_TYPES):
            return _sample_color_params(transform, batch_size, device)

        if _is_albu_instance(transform, _NORMALIZE_TYPES):
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        # CROP_RESIZE_FIXED: extract crop_coords from get_params_dependent_on_data
        if _is_albu_instance(transform, _CROP_RESIZE_TYPES):
            return _sample_crop_resize_params(transform, batch_size, height, width, device)

        # GEOMETRIC_INTERP: extract the pre-built matrix B times (once per sample)
        if _is_albu_instance(transform, _INTERP_TYPES) or _is_albu_instance(transform, _ALL_REGISTRY_TYPES):
            matrices = _sample_matrices(transform, batch_size, height, width)  # (B, 3, 3) float64 ndarray
            return {
                "matrix": torch.tensor(matrices, dtype=torch.float32, device=device),
            }

        # Unknown transform — return empty, build_matrix will fall back to identity
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build a (batch_size, 3, 3) pixel-space forward geometric matrix.

        For ``GEOMETRIC_INTERP`` and ``PROJECTIVE`` transforms, returns the
        pre-sampled ``params["matrix"]`` tensor directly (it was already
        stacked in ``sample_params()``).

        For flip transforms, constructs the appropriate constant matrix using
        inline NumPy helpers and expands it to batch size.

        Args:
            transform: An Albumentations transform instance.
            params: Parameter dict from ``sample_params()``.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            ``(batch_size, 3, 3)`` forward affine matrix or homography in pixel
            coordinates, depending on the transform category.

        """
        if _is_albu_instance(transform, _POINTWISE_TYPES):
            # Non-linear color op: no spatial change → identity matrix.
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        if _is_albu_instance(transform, _HFLIP_TYPES):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            matrix_np = hflip_matrix_np(width=width)
            return (
                torch
                .tensor(matrix_np, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .clone()
            )

        if _is_albu_instance(transform, _VFLIP_TYPES):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            matrix_np = vflip_matrix_np(height=height)
            return (
                torch
                .tensor(matrix_np, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .clone()
            )

        if _is_albu_instance(transform, _EXACT_DISCRETE_TYPES):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            dtype = torch.float32

            if _is_albu_instance(transform, frozenset({_RandomRotate90})):
                k90 = params.get("k90")
                if k90 is None:
                    k90 = torch.zeros(batch_size, device=device, dtype=torch.int64)
                if height != width and bool(((k90 == 1) | (k90 == 3)).any().item()):
                    msg = (
                        "RandomRotate90 with k in {1, 3} changes spatial dimensions "
                        f"({height}x{width}). Mixed affine fusion requires shape-preserving ops."
                    )
                    raise RuntimeError(msg)
                # rotation_matrix(+θ) warps like torch.rot90(k=-1); negate so the
                # matrix path matches native np.rot90(+k) and exact_apply.
                angles = -k90.to(dtype=dtype) * (torch.pi / 2.0)
                return rotation_matrix(angles, height=height, width=width)

            if _is_albu_instance(transform, frozenset({_Transpose})):
                return _transpose_matrix(height=height, width=width, batch_size=batch_size, device=device, dtype=dtype)

            if _is_albu_instance(transform, frozenset({_D4})):
                d4_code = params.get("d4_code")
                if d4_code is None:
                    d4_code = torch.zeros(batch_size, device=device, dtype=torch.int64)
                return _d4_matrix(d4_code, height=height, width=width, device=device, dtype=dtype)

        if _is_albu_instance(transform, _CROP_RESIZE_TYPES):
            return crop_resize_matrix(
                top=params["crop_top"],
                left=params["crop_left"],
                crop_h=params["crop_h"],
                crop_w=params["crop_w"],
                target_h=params["target_h"],
                target_w=params["target_w"],
            )

        if "matrix" in params:
            # Already (B, 3, 3) float32 from sample_params
            return params["matrix"]

        # Fallback: identity (unreachable for registered transforms)
        return torch.eye(3).unsqueeze(0)

    @staticmethod
    def exact_flip_dims(transform: object) -> list[int]:
        """Return the spatial dims to flip for GEOMETRIC_EXACT transforms.

        Args:
            transform: An Albumentations flip transform.

        Returns:
            ``[3]`` for ``HorizontalFlip`` (width axis) or ``[2]`` for
            ``VerticalFlip`` (height axis).

        Raises:
            TypeError: If the transform is not a recognised flip type.

        """
        if _is_albu_instance(transform, _HFLIP_TYPES):
            return [3]
        if _is_albu_instance(transform, _VFLIP_TYPES):
            return [2]
        raise TypeError(f"Cannot determine flip dims for {type(transform).__name__!r}")

    @staticmethod
    def exact_apply(transform: object, image: torch.Tensor) -> torch.Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly.

        Dispatches flips via ``tensor.flip``, 90-degree rotations via
        ``torch.rot90``, transposes via ``.permute``, and D4 elements via
        the appropriate combination.

        Args:
            transform: An Albumentations GEOMETRIC_EXACT transform.
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor.

        Raises:
            RuntimeError: If a discrete op would change spatial dimensions
                on non-square images.

        """
        if _is_albu_instance(transform, _HFLIP_TYPES):
            return image.flip(dims=[3])
        if _is_albu_instance(transform, _VFLIP_TYPES):
            return image.flip(dims=[2])
        if _is_albu_instance(transform, _EXACT_DISCRETE_TYPES):
            return _apply_discrete_exact(transform, image)
        msg = f"Cannot apply exact op for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

        Maps the linear color transform ``c' = alpha * c + beta`` to the 4x4
        homogeneous form ``[[M, b], [0^T, 1]]``.

        Supported transforms:

        - ``RandomBrightnessContrast``: ``c' = alpha * c + beta``
          Matrix: ``M = alpha * identity_3x3``, ``b = (beta, beta, beta)``.

        Albumentations contrast (``alpha``) is applied around zero (``alpha * c``), not around a mean
        luminance, so ``mean`` is unused here and accepted only for a uniform adapter signature.

        Args:
            transform: An Albumentations color transform instance.
            params: Parameter dict from ``sample_params()``.
            mean: Unused; present for signature parity with mean-relative backends.

        Returns:
            ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Raises:
            NotImplementedError: If the transform type is not supported.

        """
        del mean  # Albumentations color ops have no mean-relative midpoint.
        if _is_albu_instance(transform, _COLOR_TYPES):
            return _build_brightness_contrast_matrix(params)

        if _is_albu_instance(transform, _NORMALIZE_TYPES):
            return _build_normalize_matrix(transform, params)

        msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
        raise NotImplementedError(msg)

    @staticmethod
    def is_normalize(transform: object) -> bool:
        """Return whether *transform* is a standard Albumentations Normalize transform."""
        return _is_albu_instance(transform, _NORMALIZE_TYPES)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply an Albumentations transform directly via its native forward method.

        Performs a single device-to-host transfer for the whole batch, applies the
        transform per sample on the host (Albumentations is inherently per-image),
        then performs a single host-to-device transfer of the stacked result. This
        keeps exactly one D2H copy and one H2D copy per passthrough segment, rather
        than one of each per sample. The per-sample transform calls, their order,
        and their outputs are unchanged, so the returned values are bit-identical to
        a per-sample round-trip.

        Warning:
            The transform receives ``float32`` arrays in ``[0, 1]`` (the pipeline invariant). Albumentations
            transforms that assume ``uint8`` ``[0, 255]`` input (e.g. ``ISONoise``, ``RandomFog``,
            ``ImageCompression`` and several other noise/compression ops) may produce wrong-magnitude or
            no-op results on such input without raising. Use float-safe transforms on the passthrough path,
            or apply uint8-expecting transforms outside the pipeline.

        Args:
            transform: An Albumentations transform instance.
            image: ``(batch_size, channels, height, width)`` float32 image tensor.
            **kwargs: Unused; accepted for protocol compatibility.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor on the same device as input.

        """
        device = image.device
        dtype = image.dtype
        batch_size = image.shape[0]

        # One device→host transfer for the whole batch: permute to channel-last,
        # move to host, materialise NumPy once. Albumentations must still be looped
        # per image (its dict API is single-image), but the loop now runs entirely
        # on the host over views of the single transferred array.
        batch_hwc = image.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C) host
        ndarray_results = []
        for idx_sample in range(batch_size):
            output_np = transform(image=batch_hwc[idx_sample])["image"]  # type: ignore[operator]
            ndarray_results.append(np.ascontiguousarray(output_np))

        # One host→device transfer for the whole batch: stack on the host, convert
        # to a single tensor, permute back to channel-first, move to device once.
        stacked_hwc = np.stack(ndarray_results)  # (B, H, W, C) host
        result_bchw = torch.as_tensor(stacked_hwc).permute(0, 3, 1, 2)
        return result_bchw.to(device=device, dtype=dtype)

    @staticmethod
    def call_nonfused_numpy(transform: object, image_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply a non-fused Albumentations transform to a single HWC NumPy image.

        Calls the transform via its native Albumentations dict API
        (``transform(image=image_hwc)["image"]``) without any tensor conversion.
        Used by :meth:`~fuse_augmentations.compose.FusedCompose._forward_albu_native`
        to apply passthrough transforms in the Albumentations native I/O path.

        Args:
            transform: Any Albumentations transform instance.
            image_hwc: ``(height, width, channels)`` NumPy array.

        Returns:
            Transformed ``(height, width, channels)`` NumPy array.

        Examples:
            >>> import numpy as np
            >>> import albumentations as A
            >>> from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
            >>> image = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = AlbumentationsAdapter.call_nonfused_numpy(A.GaussianBlur(p=1.0), image)
            >>> out.shape
            (8, 8, 3)

        """
        return transform(image=image_hwc)["image"]  # type: ignore[operator,no-any-return]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_square_for_shape_changing_op(
    image: torch.Tensor,
    op_name: str,
) -> None:
    """Raise if a shape-changing discrete op is used on non-square images."""
    if image.shape[2] != image.shape[3]:
        msg = (
            f"{op_name} changes spatial dimensions on non-square images "
            f"({image.shape[2]}x{image.shape[3]}). "
            "ExactAffineSegment requires shape-preserving ops. "
            "Use square images for exact discrete transforms."
        )
        raise RuntimeError(msg)


# D4 group element -> tensor operation mapping
_D4_OPS: dict[str, str] = {
    "e": "identity",
    "r90": "rot90_1",
    "r180": "rot90_2",
    "r270": "rot90_3",
    "h": "hflip",
    "v": "vflip",
    "t": "transpose",
    "hvt": "hvt",
}
_D4_ELEM_TO_CODE: dict[str, int] = {name: idx for idx, name in enumerate(_D4_OPS)}
_D4_CODE_TO_ELEM: dict[int, str] = {idx: name for name, idx in _D4_ELEM_TO_CODE.items()}


def _apply_discrete_exact(
    transform: object,
    image: torch.Tensor,
) -> torch.Tensor:
    """Apply a discrete exact transform (RandomRotate90, D4, Transpose).

    Args:
        transform: An Albumentations discrete exact transform.
        image: ``(batch_size, channels, height, width)`` input tensor.

    Returns:
        Transformed ``(batch_size, channels, height, width)`` tensor.

    """
    ttype = type(transform)
    batch_size = int(image.shape[0])
    # See sample_params: same_on_batch only exists on duck-typed wrappers/test
    # doubles; genuine Albumentations transforms always take the per-sample path.
    same_on_batch = bool(getattr(transform, "same_on_batch", False))

    if TRANSFORM_REGISTRY and ttype is _RandomRotate90:
        if same_on_batch:
            params = transform.get_params()  # type: ignore[attr-defined]
            num_rotations = int(params["factor"]) % 4
            if num_rotations in (1, 3):
                _check_square_for_shape_changing_op(image, "RandomRotate90")
            return torch.rot90(image, k=num_rotations, dims=[2, 3])

        if batch_size == 0:
            return image
        image_output = image.clone()
        for idx_sample in range(batch_size):
            params = transform.get_params()  # type: ignore[attr-defined]
            num_rotations = int(params["factor"]) % 4
            if num_rotations in (1, 3):
                _check_square_for_shape_changing_op(image[idx_sample : idx_sample + 1], "RandomRotate90")
            image_output[idx_sample : idx_sample + 1] = torch.rot90(
                image[idx_sample : idx_sample + 1], k=num_rotations, dims=[2, 3]
            )
        return image_output

    if TRANSFORM_REGISTRY and ttype is _Transpose:
        _check_square_for_shape_changing_op(image, "Transpose")
        return image.permute(0, 1, 3, 2).contiguous()

    if TRANSFORM_REGISTRY and ttype is _D4:
        if same_on_batch:
            params = transform.get_params()  # type: ignore[attr-defined]
            elem = _convert_normalize_d4_elem(params["group_element"])
            return _apply_d4_element(image, elem)

        if batch_size == 0:
            return image
        image_output = image.clone()
        for idx_sample in range(batch_size):
            params = transform.get_params()  # type: ignore[attr-defined]
            elem = _convert_normalize_d4_elem(params["group_element"])
            image_output[idx_sample : idx_sample + 1] = _apply_d4_element(image[idx_sample : idx_sample + 1], elem)
        return image_output

    msg = f"Cannot apply discrete exact op for {ttype.__name__!r}"
    raise TypeError(msg)


def _transpose_matrix(
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the forward pixel-space matrix for transpose."""
    if height != width:
        msg = (
            f"Transpose changes spatial dimensions on non-square images ({height}x{width})."
            f" Mixed affine fusion requires shape-preserving ops."
        )
        raise RuntimeError(msg)
    matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    matrix[:, 0, 1] = 1.0
    matrix[:, 1, 0] = 1.0
    matrix[:, 2, 2] = 1.0
    return matrix


def _d4_matrix(
    d4_code: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build forward pixel-space matrices for D4 group elements."""
    batch_size = int(d4_code.shape[0])
    if height != width:
        _shape_changing = frozenset({_D4_ELEM_TO_CODE["r90"], _D4_ELEM_TO_CODE["r270"], _D4_ELEM_TO_CODE["hvt"]})
        for code in d4_code.tolist():
            elem = _D4_CODE_TO_ELEM[int(code)]
            if int(code) in _shape_changing:
                msg = (
                    f"D4 element {elem!r} changes spatial dimensions on non-square images "
                    f"({height}x{width}). Mixed affine fusion requires shape-preserving ops. "
                    "Use square images for exact discrete transforms."
                )
                raise RuntimeError(msg)
    out = torch.empty(batch_size, 3, 3, device=device, dtype=dtype)
    base_h = hflip_matrix(width=width, batch_size=1, device=device, dtype=dtype)
    base_v = vflip_matrix(height=height, batch_size=1, device=device, dtype=dtype)
    base_t = _transpose_matrix(height=height, width=width, batch_size=1, device=device, dtype=dtype)

    for idx, code in enumerate(d4_code.tolist()):
        elem = _D4_CODE_TO_ELEM[int(code)]
        if elem == "e":
            out[idx] = torch.eye(3, device=device, dtype=dtype)
            continue
        if elem == "r90":
            # rotation_matrix(+θ) warps like torch.rot90(k=-1); negate so r90/r270
            # match native np.rot90 direction and _apply_d4_element.
            out[idx] = rotation_matrix(
                torch.tensor([-torch.pi / 2.0], device=device, dtype=dtype), height=height, width=width
            )[0]
            continue
        if elem == "r180":
            out[idx] = rotation_matrix(
                torch.tensor([torch.pi], device=device, dtype=dtype), height=height, width=width
            )[0]
            continue
        if elem == "r270":
            out[idx] = rotation_matrix(
                torch.tensor([-3.0 * torch.pi / 2.0], device=device, dtype=dtype), height=height, width=width
            )[0]
            continue
        if elem == "h":
            out[idx] = base_h[0]
            continue
        if elem == "v":
            out[idx] = base_v[0]
            continue
        if elem == "t":
            out[idx] = base_t[0]
            continue
        if elem == "hvt":
            out[idx] = matmul3x3(base_t, matmul3x3(base_v, base_h))[0]
            continue
        msg = f"Unknown D4 group element: {elem!r}"
        raise ValueError(msg)
    return out


_D4Elem = Literal["e", "r90", "r180", "r270", "h", "v", "t", "hvt"]


def _convert_normalize_d4_elem(elem: object) -> _D4Elem:
    """Validate and narrow a dynamic D4 group element to the literal type."""
    elem_str = str(elem)
    if elem_str not in _D4_ELEM_TO_CODE:
        msg = f"Unknown D4 group element: {elem_str!r}"
        raise ValueError(msg)
    return cast(_D4Elem, elem_str)


def _apply_d4_element(image: torch.Tensor, elem: _D4Elem) -> torch.Tensor:
    """Apply a single D4 group element to a (batch_size, channels, height, width) tensor.

    Args:
        image: ``(batch_size, channels, height, width)`` input tensor.
        elem: D4 group element name (``"e"``, ``"r90"``, ``"r180"``,
            ``"r270"``, ``"h"``, ``"v"``, ``"t"``, ``"hvt"``).

    Returns:
        Transformed ``(batch_size, channels, height, width)`` tensor.

    """
    if elem == "e":
        return image
    if elem == "r90":
        _check_square_for_shape_changing_op(image, "D4(r90)")
        return torch.rot90(image, k=1, dims=[2, 3])
    if elem == "r180":
        return torch.rot90(image, k=2, dims=[2, 3])
    if elem == "r270":
        _check_square_for_shape_changing_op(image, "D4(r270)")
        return torch.rot90(image, k=3, dims=[2, 3])
    if elem == "h":
        return image.flip(dims=[3])
    if elem == "v":
        return image.flip(dims=[2])
    if elem == "t":
        _check_square_for_shape_changing_op(image, "D4(t)")
        return image.permute(0, 1, 3, 2).contiguous()
    if elem == "hvt":
        _check_square_for_shape_changing_op(image, "D4(hvt)")
        return image.flip(dims=[2, 3]).permute(0, 1, 3, 2).contiguous()
    msg = f"Unknown D4 group element: {elem!r}"
    raise ValueError(msg)


def _sample_crop_resize_params(
    transform: object,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample crop coordinates from ``RandomResizedCrop`` for B images.

    Albumentations ``RandomResizedCrop.get_params_dependent_on_data`` returns
    ``crop_coords = (x_min, y_min, x_max, y_max)`` where the endpoints are
    exclusive (i.e. ``crop_w = x_max - x_min`` pixels).

    Args:
        transform: An Albumentations ``RandomResizedCrop`` instance.
        batch_size: Batch size.
        height: Image height in pixels (used to bound crop coordinates).
        width: Image width in pixels (used to bound crop coordinates).
        device: Target device for returned tensors.

    Returns:
        Dict with canonical keys ``crop_top``, ``crop_left``, ``crop_h``,
        ``crop_w``, ``target_h``, ``target_w`` as ``(B,)`` float32 tensors.

    """
    dummy = np.empty((height, width, 1), dtype=np.float32)  # shape is all that matters; values unused
    data = {"image": dummy}
    tops: list[float] = []
    lefts: list[float] = []
    crop_hs: list[float] = []
    crop_ws: list[float] = []
    for _ in range(batch_size):
        base = transform.get_params()  # type: ignore[attr-defined]
        if hasattr(transform, "update_transform_params"):
            base = transform.update_transform_params(base, data)
        else:
            base = transform.update_params(base, **data)  # type: ignore[attr-defined]
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        x_min, y_min, x_max, y_max = full["crop_coords"]
        tops.append(float(y_min))
        lefts.append(float(x_min))
        crop_hs.append(float(y_max - y_min))
        crop_ws.append(float(x_max - x_min))
    target_h, target_w = transform.size  # type: ignore[attr-defined]
    return {
        "crop_top": torch.tensor(tops, dtype=torch.float32, device=device),
        "crop_left": torch.tensor(lefts, dtype=torch.float32, device=device),
        "crop_h": torch.tensor(crop_hs, dtype=torch.float32, device=device),
        "crop_w": torch.tensor(crop_ws, dtype=torch.float32, device=device),
        "target_h": torch.full((batch_size,), float(target_h), dtype=torch.float32, device=device),
        "target_w": torch.full((batch_size,), float(target_w), dtype=torch.float32, device=device),
    }


# Module-level cache mapping (H, W) -> reusable dummy (H, W, 1) float32 array.
# The dummy is only ever inspected for its .shape by Albumentations' param
# sampling path; pixel values are never read.  Reusing the array avoids a
# fresh ~256 KB allocation on every call, which was ~0.6 µs each at H=W=256.
_DUMMY_IMAGE_CACHE: dict[tuple[int, int], NDArray[np.float32]] = {}


def _sample_matrices(transform: object, batch_size: int, height: int, width: int) -> NDArray[np.float64]:
    """Call ``get_params_dependent_on_data()`` batch_size times, stack into (batch_size, 3, 3).

    A dummy ``(height, width, 1)`` float32 image is passed so the transform can
    compute center coordinates. No actual pixel data is read.

    Args:
        transform: An Albumentations interpolated affine or projective transform.
        batch_size: Batch size.
        height: Image height.
        width: Image width.

    Returns:
        ``(B, 3, 3)`` float64 array of forward pixel-space matrices.

    Raises:
        NotImplementedError: If *transform* is an Albumentations ``Perspective`` with
            ``keep_size=False``. Native ``keep_size=False`` returns a crop of different
            spatial dimensions, which the batched shape-preserving fusion engine cannot
            reproduce in a single warp.
        KeyError: If the transform does not return a ``"matrix"`` from
            ``get_params_dependent_on_data()``.

    """
    if _is_albu_instance(transform, frozenset({_Perspective})) and not bool(getattr(transform, "keep_size", True)):
        msg = (
            "Albumentations Perspective(keep_size=False) returns a crop of different spatial "
            "dimensions, which the batched shape-preserving fusion engine cannot reproduce in a "
            "single warp. Use keep_size=True (the fused path folds its output resize into the "
            "homography) or apply this Perspective outside the fused pipeline."
        )
        raise NotImplementedError(msg)
    _key = (height, width)
    dummy = _DUMMY_IMAGE_CACHE.get(_key)
    if dummy is None:
        dummy = np.empty((height, width, 1), dtype=np.float32)
        _DUMMY_IMAGE_CACHE[_key] = dummy
    data = {"image": dummy}
    matrices = np.empty((batch_size, 3, 3), dtype=np.float64)
    for idx in range(batch_size):
        base = transform.get_params()  # type: ignore[attr-defined]
        # Inject "shape" (and interpolation/fill) so get_params_dependent_on_data can compute
        # center coordinates. API differs between albu 1.x and 2.x.
        if hasattr(transform, "update_transform_params"):
            base = transform.update_transform_params(base, data)
        else:
            base = transform.update_params(base, **data)  # type: ignore[attr-defined]
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        if "matrix" in full:
            matrix = np.asarray(full["matrix"], dtype=np.float64)
            if _is_albu_instance(transform, frozenset({_Perspective})) and bool(getattr(transform, "keep_size", False)):
                # Albumentations applies Perspective into its sampled bounding box,
                # then maps that intermediate image back to the original dimensions.
                # Folding the same output scale into the forward homography lets the
                # fused cv2 path perform the identical geometry in one warp.
                max_width = int(full["max_width"])
                max_height = int(full["max_height"])
                resize = np.array(
                    [[width / max_width, 0.0, 0.0], [0.0, height / max_height, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                matrix = resize @ matrix
            matrices[idx] = matrix
        else:
            msg = f"{type(transform).__name__} did not return 'matrix' from get_params_dependent_on_data()"
            raise KeyError(msg)
    return matrices


# ---------------------------------------------------------------------------
# Color matrix helpers
# ---------------------------------------------------------------------------


def _sample_color_params(
    transform: object,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample alpha/beta from ``RandomBrightnessContrast`` for B images.

    Albumentations ``RandomBrightnessContrast.get_params_dependent_on_data``
    returns ``alpha`` (contrast factor) and ``beta`` (brightness offset)
    such that ``c' = alpha * c + beta``.

    Args:
        transform: An Albumentations ``RandomBrightnessContrast`` instance.
        batch_size: Batch size.
        device: Target device for returned tensors.

    Returns:
        Dict with ``"alpha"`` and ``"beta"`` tensors of shape ``(B,)``.

    """
    dummy = np.zeros((4, 4, 1), dtype=np.float32)
    data = {"image": dummy}
    alphas: list[float] = []
    betas: list[float] = []
    for _idx in range(batch_size):
        base = transform.get_params()  # type: ignore[attr-defined]
        if hasattr(transform, "update_transform_params"):
            base = transform.update_transform_params(base, data)
        else:
            base = transform.update_params(base, **data)  # type: ignore[attr-defined]
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        alphas.append(float(full["alpha"]))
        betas.append(float(full["beta"]))
    return {
        "alpha": torch.tensor(alphas, dtype=torch.float32, device=device),
        "beta": torch.tensor(betas, dtype=torch.float32, device=device),
    }


def _build_brightness_contrast_matrix(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build ``(batch_size, 4, 4)`` for Albumentations ``RandomBrightnessContrast``.

    The transform applies ``c' = alpha * c + beta`` per channel.
    Matrix: ``M = alpha * identity_3x3``, ``b = (beta, beta, beta)``.

    Args:
        params: Dict with ``"alpha"`` (batch_size,) and ``"beta"`` (batch_size,) tensors.

    Returns:
        ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

    """
    alpha = params["alpha"]  # (batch_size,)
    beta = params["beta"]  # (batch_size,)
    batch_size = alpha.shape[0]
    device = alpha.device
    dtype = alpha.dtype
    mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
    mat[:, 0, 0] = alpha
    mat[:, 1, 1] = alpha
    mat[:, 2, 2] = alpha
    mat[:, 0, 3] = beta
    mat[:, 1, 3] = beta
    mat[:, 2, 3] = beta
    return mat


def _build_normalize_matrix(transform: object, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build exact standard Albumentations Normalize scaling as a color matrix.

    Non-standard image-statistics normalization is intentionally excluded because its coefficients depend on the image
    values and cannot be represented by fixed affine parameters.

    """
    normalize = cast(Any, transform)
    if normalize.normalization != "standard":
        raise NotImplementedError("Only standard Albumentations Normalize is affine-fusible")
    max_pixel_value = normalize.max_pixel_value
    if max_pixel_value is None:
        raise NotImplementedError("Albumentations Normalize requires max_pixel_value for fusion")
    batch_size = int(params["_batch_size"].item())
    mean = torch.as_tensor(normalize.mean, dtype=torch.float32).flatten()
    std = torch.as_tensor(normalize.std, dtype=torch.float32).flatten()
    if mean.numel() == 1:
        mean = mean.expand(3)
    if std.numel() == 1:
        std = std.expand(3)
    if mean.numel() != 3 or std.numel() != 3:
        raise NotImplementedError("Albumentations Normalize fusion requires three RGB mean/std values")
    device = params["_batch_size"].device
    mat = torch.eye(4, device=device, dtype=mean.dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
    alpha = std.to(device=device).mul(float(max_pixel_value)).reciprocal()
    beta = -mean.to(device=device).div(std.to(device=device))
    mat[:, 0, 0] = alpha[0]
    mat[:, 1, 1] = alpha[1]
    mat[:, 2, 2] = alpha[2]
    mat[:, :3, 3] = beta
    return mat
