"""Albumentations backend adapter for the fused geometric engine.

Bridges Albumentations augmentation transforms to the canonical parameter
representation used by the fused Albumentations geometric segments.

Each geometric transform (``A.Affine``, ``A.Rotate``, ``A.ShiftScaleRotate``,
``A.SafeRotate``) exposes a pre-built ``matrix`` key via
``get_params_dependent_on_data()`` -- a ``(3, 3)`` forward pixel-space affine
matrix in the same convention as ``affine._matrix``. The adapter reads this
matrix directly instead of reconstructing it from raw angle/scale/shear values.

``A.Perspective`` is also supported and mapped to
``TransformCategory.PROJECTIVE``. Its sampled ``matrix`` key is treated as a
forward pixel-space homography and consumed by
``AlbuProjectiveSegment`` rather than the affine fusion path.

Flip transforms (``A.HorizontalFlip``, ``A.VerticalFlip``) return an empty
parameter dict; ``build_matrix()`` constructs their matrices from inline
NumPy helpers.

Requires ``albumentations >= 2.0``.


Example:
    >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
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

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import crop_resize_matrix, hflip_matrix, matmul3x3, rotation_matrix, vflip_matrix

# ---------------------------------------------------------------------------
# Inline NumPy matrix helpers (moved from _np_matrix.py)
# ---------------------------------------------------------------------------


def hflip_matrix_np(W: int) -> NDArray[np.float64]:  # noqa: N803
    """Build a (3, 3) pixel-space forward horizontal flip matrix.

    Maps ``x' = W - 1 - x``, ``y' = y``.

    Args:
        W: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward horizontal flip matrix.

    Example:
        >>> M = hflip_matrix_np(W=4)
        >>> M[0].tolist()
        [-1.0, 0.0, 3.0]
        >>> M[1].tolist()
        [0.0, 1.0, 0.0]

    """
    return np.array(
        [
            [-1.0, 0.0, float(W - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def vflip_matrix_np(H: int) -> NDArray[np.float64]:  # noqa: N803
    """Build a (3, 3) pixel-space forward vertical flip matrix.

    Maps ``x' = x``, ``y' = H - 1 - y``.

    Args:
        H: Image height in pixels.

    Returns:
        ``(3, 3)`` float64 forward vertical flip matrix.

    Example:
        >>> M = vflip_matrix_np(H=4)
        >>> M[1].tolist()
        [0.0, -1.0, 3.0]
        >>> M[0].tolist()
        [1.0, 0.0, 0.0]

    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, float(H - 1)],
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
    _CROP_RESIZE_TYPES: frozenset[type] = frozenset({_RandomResizedCrop})
    _ALL_REGISTRY_TYPES: frozenset[type] = frozenset(TRANSFORM_REGISTRY)

except ImportError:
    TRANSFORM_REGISTRY = {}
    _HFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _VFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _EXACT_DISCRETE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _INTERP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _COLOR_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _CROP_RESIZE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
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

    Requires ``albumentations >= 2.0``. The adapter reads the pre-built
    ``matrix`` key from ``get_params_dependent_on_data()`` rather than
    reconstructing affine or projective matrices from raw parameters.

    Example:
        >>> adapter = AlbumentationsAdapter()
        >>> isinstance(adapter, AlbumentationsAdapter)
        True

    """

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
        """Sample random parameters for a batch of B images.

        For ``GEOMETRIC_INTERP`` and ``PROJECTIVE`` transforms, calls
        ``get_params_dependent_on_data()`` once per sample and stacks the
        resulting ``matrix`` arrays into a ``(B, 3, 3)`` tensor. A dummy
        ``(H, W, 1)`` float32 array is passed so the transform can compute
        center coordinates without reading actual pixel data.

        For flip transforms, returns a ``{"_batch_size": tensor([B])}``
        sentinel so ``build_matrix()`` knows the required output shape.

        Args:
            transform: An Albumentations transform instance.
            input_shape: ``(B, C, H, W)`` shape tuple.
            device: Target device for the returned tensors.

        Returns:
            Dict of parameter tensors. ``"matrix"`` key holds ``(B, 3, 3)``
            for interpolated affine transforms and projective transforms;
            ``"_batch_size"`` for flips.

        """
        B, _C, H, W = input_shape  # noqa: N806

        if _is_albu_instance(transform, _HFLIP_TYPES | _VFLIP_TYPES):
            return {"_batch_size": torch.tensor([B], device=device, dtype=torch.int64)}
        if _is_albu_instance(transform, _EXACT_DISCRETE_TYPES):
            result: dict[str, torch.Tensor] = {
                "_batch_size": torch.tensor([B], device=device, dtype=torch.int64),
            }
            same_on_batch = bool(getattr(transform, "same_on_batch", False))

            if TRANSFORM_REGISTRY and isinstance(transform, _RandomRotate90):
                if same_on_batch:
                    factor = int(transform.get_params()["factor"]) % 4
                    result["k90"] = torch.full((B,), factor, device=device, dtype=torch.int64)
                    return result
                result["k90"] = torch.tensor(
                    [int(transform.get_params()["factor"]) % 4 for _ in range(B)],
                    device=device,
                    dtype=torch.int64,
                )
                return result

            if TRANSFORM_REGISTRY and isinstance(transform, _D4):
                if same_on_batch:
                    elem = str(transform.get_params()["group_element"])
                    result["d4_code"] = torch.full((B,), _D4_ELEM_TO_CODE[elem], device=device, dtype=torch.int64)
                    return result
                result["d4_code"] = torch.tensor(
                    [_D4_ELEM_TO_CODE[str(transform.get_params()["group_element"])] for _ in range(B)],
                    device=device,
                    dtype=torch.int64,
                )
                return result

            return result

        # POINTWISE_LINEAR (color transforms): sample alpha/beta per batch element
        if _is_albu_instance(transform, _COLOR_TYPES):
            return _sample_color_params(transform, B, device)

        # CROP_RESIZE_FIXED: extract crop_coords from get_params_dependent_on_data
        if _is_albu_instance(transform, _CROP_RESIZE_TYPES):
            return _sample_crop_resize_params(transform, B, H, W, device)

        # GEOMETRIC_INTERP: extract the pre-built matrix B times (once per sample)
        if _is_albu_instance(transform, _INTERP_TYPES) or _is_albu_instance(transform, _ALL_REGISTRY_TYPES):
            matrices = _sample_matrices(transform, B, H, W)  # (B, 3, 3) float64 ndarray
            return {
                "matrix": torch.tensor(matrices, dtype=torch.float32, device=device),
            }

        # Unknown transform — return empty, build_matrix will fall back to identity
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> torch.Tensor:
        """Build a ``(B, 3, 3)`` pixel-space forward geometric matrix.

        For ``GEOMETRIC_INTERP`` and ``PROJECTIVE`` transforms, returns the
        pre-sampled ``params["matrix"]`` tensor directly (it was already
        stacked in ``sample_params()``).

        For flip transforms, constructs the appropriate constant matrix using
        inline NumPy helpers and expands it to batch size B.

        Args:
            transform: An Albumentations transform instance.
            params: Parameter dict from ``sample_params()``.
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            ``(B, 3, 3)`` forward affine matrix or homography in pixel
            coordinates, depending on the transform category.

        """
        if _is_albu_instance(transform, _HFLIP_TYPES):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            M_np = hflip_matrix_np(W=W)  # noqa: N806
            return torch.tensor(M_np, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if _is_albu_instance(transform, _VFLIP_TYPES):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            M_np = vflip_matrix_np(H=H)  # noqa: N806
            return torch.tensor(M_np, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if _is_albu_instance(transform, _EXACT_DISCRETE_TYPES):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            dtype = torch.float32

            if _is_albu_instance(transform, frozenset({_RandomRotate90})):
                k90 = params.get("k90")
                if k90 is None:
                    k90 = torch.zeros(B, device=device, dtype=torch.int64)
                if H != W and bool(((k90 == 1) | (k90 == 3)).any().item()):
                    msg = (
                        "RandomRotate90 with k in {1, 3} changes spatial dimensions "
                        f"({H}x{W}). Mixed affine fusion requires shape-preserving ops."
                    )
                    raise RuntimeError(msg)
                angles = k90.to(dtype=dtype) * (torch.pi / 2.0)
                return rotation_matrix(angles, H=H, W=W)

            if _is_albu_instance(transform, frozenset({_Transpose})):
                return _transpose_matrix(H=H, W=W, batch_size=B, device=device, dtype=dtype)

            if _is_albu_instance(transform, frozenset({_D4})):
                d4_code = params.get("d4_code")
                if d4_code is None:
                    d4_code = torch.zeros(B, device=device, dtype=torch.int64)
                return _d4_matrix(d4_code, H=H, W=W, device=device, dtype=dtype)

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
            image: ``(B, C, H, W)`` input tensor.

        Returns:
            Transformed ``(B, C, H, W)`` tensor.

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
    ) -> torch.Tensor:
        """Build a ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Maps the linear color transform ``c' = alpha * c + beta`` to the 4x4
        homogeneous form ``[[M, b], [0^T, 1]]``.

        Supported transforms:

        - ``RandomBrightnessContrast``: ``c' = alpha * c + beta``
          Matrix: ``M = alpha * I₃``, ``b = (beta, beta, beta)``.

        Args:
            transform: An Albumentations color transform instance.
            params: Parameter dict from ``sample_params()``.

        Returns:
            ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Raises:
            NotImplementedError: If the transform type is not supported.

        """
        if _is_albu_instance(transform, _COLOR_TYPES):
            return _build_brightness_contrast_matrix(params)

        msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
        raise NotImplementedError(msg)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply an Albumentations transform directly via its native forward method.

        Converts the ``(B, C, H, W)`` tensor to ``(H, W, C)`` numpy arrays
        per sample, calls the transform, and converts the results back.

        Args:
            transform: An Albumentations transform instance.
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
            # (C, H, W) → (H, W, C) numpy
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            out_np = transform(image=img_np)["image"]  # type: ignore[operator]
            # (H, W, C) → (C, H, W) tensor
            results.append(torch.as_tensor(np.ascontiguousarray(out_np).copy()).permute(2, 0, 1))

        return torch.stack(results).to(device=device, dtype=dtype)

    @staticmethod
    def call_nonfused_numpy(transform: object, img_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply a non-fused Albumentations transform to a single HWC NumPy image.

        Calls the transform via its native Albumentations dict API
        (``transform(image=img_hwc)["image"]``) without any tensor conversion.
        Used by :meth:`~fuse_augmentations._compose.FusedCompose._forward_albu_native`
        to apply passthrough transforms in the Albumentations native I/O path.

        Args:
            transform: Any Albumentations transform instance.
            img_hwc: ``(H, W, C)`` NumPy array.

        Returns:
            Transformed ``(H, W, C)`` NumPy array.

        Examples:
            >>> import numpy as np
            >>> import albumentations as A
            >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
            >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = AlbumentationsAdapter.call_nonfused_numpy(A.GaussianBlur(p=1.0), img)
            >>> out.shape
            (8, 8, 3)

        """
        return transform(image=img_hwc)["image"]  # type: ignore[operator,no-any-return]


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
        image: ``(B, C, H, W)`` input tensor.

    Returns:
        Transformed ``(B, C, H, W)`` tensor.

    """
    ttype = type(transform)
    bsz = int(image.shape[0])
    same_on_batch = bool(getattr(transform, "same_on_batch", False))

    if TRANSFORM_REGISTRY and ttype is _RandomRotate90:
        if same_on_batch:
            params = transform.get_params()  # type: ignore[attr-defined]
            k = int(params["factor"]) % 4
            if k in (1, 3):
                _check_square_for_shape_changing_op(image, "RandomRotate90")
            return torch.rot90(image, k=k, dims=[2, 3])

        if bsz == 0:
            return image
        out = image.clone()
        for i in range(bsz):
            params = transform.get_params()  # type: ignore[attr-defined]
            k = int(params["factor"]) % 4
            if k in (1, 3):
                _check_square_for_shape_changing_op(image, "RandomRotate90")
            out[i : i + 1] = torch.rot90(image[i : i + 1], k=k, dims=[2, 3])
        return out

    if TRANSFORM_REGISTRY and ttype is _Transpose:
        _check_square_for_shape_changing_op(image, "Transpose")
        return image.permute(0, 1, 3, 2).contiguous()

    if TRANSFORM_REGISTRY and ttype is _D4:
        if same_on_batch:
            params = transform.get_params()  # type: ignore[attr-defined]
            elem = _convert_normalize_d4_elem(params["group_element"])
            return _apply_d4_element(image, elem)

        if bsz == 0:
            return image
        out = image.clone()
        for i in range(bsz):
            params = transform.get_params()  # type: ignore[attr-defined]
            elem = _convert_normalize_d4_elem(params["group_element"])
            out[i : i + 1] = _apply_d4_element(image[i : i + 1], elem)
        return out

    msg = f"Cannot apply discrete exact op for {ttype.__name__!r}"
    raise TypeError(msg)


def _transpose_matrix(
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the forward pixel-space matrix for transpose."""
    if H != W:
        msg = (
            f"Transpose changes spatial dimensions on non-square images ({H}x{W})."
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
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build forward pixel-space matrices for D4 group elements."""
    batch_size = int(d4_code.shape[0])
    if H != W:
        _shape_changing = frozenset({_D4_ELEM_TO_CODE["r90"], _D4_ELEM_TO_CODE["r270"], _D4_ELEM_TO_CODE["hvt"]})
        for code in d4_code.tolist():
            elem = _D4_CODE_TO_ELEM[int(code)]
            if int(code) in _shape_changing:
                msg = (
                    f"D4 element {elem!r} changes spatial dimensions on non-square images "
                    f"({H}x{W}). Mixed affine fusion requires shape-preserving ops. "
                    "Use square images for exact discrete transforms."
                )
                raise RuntimeError(msg)
    out = torch.empty(batch_size, 3, 3, device=device, dtype=dtype)
    base_h = hflip_matrix(W=W, batch_size=1, device=device, dtype=dtype)
    base_v = vflip_matrix(H=H, batch_size=1, device=device, dtype=dtype)
    base_t = _transpose_matrix(H=H, W=W, batch_size=1, device=device, dtype=dtype)

    for idx, code in enumerate(d4_code.tolist()):
        elem = _D4_CODE_TO_ELEM[int(code)]
        if elem == "e":
            out[idx] = torch.eye(3, device=device, dtype=dtype)
            continue
        if elem == "r90":
            out[idx] = rotation_matrix(torch.tensor([torch.pi / 2.0], device=device, dtype=dtype), H=H, W=W)[0]
            continue
        if elem == "r180":
            out[idx] = rotation_matrix(torch.tensor([torch.pi], device=device, dtype=dtype), H=H, W=W)[0]
            continue
        if elem == "r270":
            out[idx] = rotation_matrix(torch.tensor([3.0 * torch.pi / 2.0], device=device, dtype=dtype), H=H, W=W)[0]
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
    """Apply a single D4 group element to a (B, C, H, W) tensor.

    Args:
        image: ``(B, C, H, W)`` input tensor.
        elem: D4 group element name (``"e"``, ``"r90"``, ``"r180"``,
            ``"r270"``, ``"h"``, ``"v"``, ``"t"``, ``"hvt"``).

    Returns:
        Transformed ``(B, C, H, W)`` tensor.

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
    B: int,  # noqa: N803
    H: int,  # noqa: N803
    W: int,  # noqa: N803
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample crop coordinates from ``RandomResizedCrop`` for B images.

    Albumentations ``RandomResizedCrop.get_params_dependent_on_data`` returns
    ``crop_coords = (x_min, y_min, x_max, y_max)`` where the endpoints are
    exclusive (i.e. ``crop_w = x_max - x_min`` pixels).

    Args:
        transform: An Albumentations ``RandomResizedCrop`` instance.
        B: Batch size.
        H: Image height in pixels (used to bound crop coordinates).
        W: Image width in pixels (used to bound crop coordinates).
        device: Target device for returned tensors.

    Returns:
        Dict with canonical keys ``crop_top``, ``crop_left``, ``crop_h``,
        ``crop_w``, ``target_h``, ``target_w`` as ``(B,)`` float32 tensors.

    """
    dummy = np.empty((H, W, 1), dtype=np.float32)  # shape is all that matters; values unused
    data = {"image": dummy}
    tops: list[float] = []
    lefts: list[float] = []
    crop_hs: list[float] = []
    crop_ws: list[float] = []
    for _ in range(B):
        base = transform.get_params()  # type: ignore[attr-defined]
        base = transform.update_transform_params(base, data)  # type: ignore[attr-defined]
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
        "target_h": torch.full((B,), float(target_h), dtype=torch.float32, device=device),
        "target_w": torch.full((B,), float(target_w), dtype=torch.float32, device=device),
    }


def _sample_matrices(transform: object, B: int, H: int, W: int) -> NDArray[np.float64]:  # noqa: N803
    """Call ``get_params_dependent_on_data()`` B times, stack into (B, 3, 3).

    A dummy ``(H, W, 1)`` float32 image is passed so the transform can
    compute center coordinates. No actual pixel data is read.

    Args:
        transform: An Albumentations interpolated affine or projective transform.
        B: Batch size.
        H: Image height.
        W: Image width.

    Returns:
        ``(B, 3, 3)`` float64 array of forward pixel-space matrices.

    """
    dummy = np.empty((H, W, 1), dtype=np.float32)  # shape is all that matters; values unused
    data = {"image": dummy}
    matrices = np.empty((B, 3, 3), dtype=np.float64)
    for i in range(B):
        base = transform.get_params()  # type: ignore[attr-defined]
        # update_transform_params adds "shape" (and interpolation/fill keys) needed
        # by get_params_dependent_on_data to compute center coordinates etc.
        base = transform.update_transform_params(base, data)  # type: ignore[attr-defined]
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        matrices[i] = full["matrix"]
    return matrices


# ---------------------------------------------------------------------------
# Color matrix helpers
# ---------------------------------------------------------------------------


def _sample_color_params(
    transform: object,
    B: int,  # noqa: N803
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample alpha/beta from ``RandomBrightnessContrast`` for B images.

    Albumentations ``RandomBrightnessContrast.get_params_dependent_on_data``
    returns ``alpha`` (contrast factor) and ``beta`` (brightness offset)
    such that ``c' = alpha * c + beta``.

    Args:
        transform: An Albumentations ``RandomBrightnessContrast`` instance.
        B: Batch size.
        device: Target device for returned tensors.

    Returns:
        Dict with ``"alpha"`` and ``"beta"`` tensors of shape ``(B,)``.

    """
    dummy = np.zeros((4, 4, 1), dtype=np.float32)
    data = {"image": dummy}
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(B):
        base = transform.get_params()  # type: ignore[attr-defined]
        base = transform.update_transform_params(base, data)  # type: ignore[attr-defined]
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        alphas.append(float(full["alpha"]))
        betas.append(float(full["beta"]))
    return {
        "alpha": torch.tensor(alphas, dtype=torch.float32, device=device),
        "beta": torch.tensor(betas, dtype=torch.float32, device=device),
    }


def _build_brightness_contrast_matrix(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build ``(B, 4, 4)`` for Albumentations ``RandomBrightnessContrast``.

    The transform applies ``c' = alpha * c + beta`` per channel.
    Matrix: ``M = alpha * I₃``, ``b = (beta, beta, beta)``.

    Args:
        params: Dict with ``"alpha"`` (B,) and ``"beta"`` (B,) tensors.

    Returns:
        ``(B, 4, 4)`` homogeneous color-space affine matrix.

    """
    alpha = params["alpha"]  # (B,)
    beta = params["beta"]  # (B,)
    B = alpha.shape[0]  # noqa: N806
    device = alpha.device
    dtype = alpha.dtype
    mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
    mat[:, 0, 0] = alpha
    mat[:, 1, 1] = alpha
    mat[:, 2, 2] = alpha
    mat[:, 0, 3] = beta
    mat[:, 1, 3] = beta
    mat[:, 2, 3] = beta
    return mat
