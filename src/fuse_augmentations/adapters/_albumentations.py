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

Coverage survey (Phase A.4)
---------------------------

.. list-table:: Albumentations geometric transforms -- adapter coverage
   :header-rows: 1

   * - Transform
     - Category
     - Status
     - Notes
   * - ``Affine``
     - ``GEOMETRIC_INTERP``
     - Covered (v0.2)
     - Pre-built ``matrix`` from ``get_params_dependent_on_data()``.
   * - ``Rotate``
     - ``GEOMETRIC_INTERP``
     - Covered (v0.2)
     - Pre-built ``matrix``; rotation about image center.
   * - ``ShiftScaleRotate``
     - ``GEOMETRIC_INTERP``
     - Covered (v0.2)
     - Pre-built ``matrix``; shift + scale + rotation.
   * - ``SafeRotate``
     - ``GEOMETRIC_INTERP``
     - **NEW (Phase A.4)**
     - Pre-built ``matrix``; rotation with border-safe padding.
   * - ``HorizontalFlip``
     - ``GEOMETRIC_EXACT``
     - Covered (v0.2)
     - Exact pixel flip; no interpolation.
   * - ``VerticalFlip``
     - ``GEOMETRIC_EXACT``
     - Covered (v0.2)
     - Exact pixel flip; no interpolation.
   * - ``Perspective``
     - ``PROJECTIVE``
     - Covered (v0.5)
     - Pre-built ``matrix``; homography.
   * - ``RandomRotate90``
     - (deferred)
     - Not covered
     - No ``matrix`` key; uses discrete 90-degree steps.
   * - ``D4``
     - (deferred)
     - Not covered
     - No ``matrix`` key; dihedral group ops.
   * - ``Transpose``
     - (deferred)
     - Not covered
     - No ``matrix`` key; axes swap.

Example:
    >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
    >>> adapter = AlbumentationsAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...AlbumentationsAdapter...>

"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from fuse_augmentations._types import TransformCategory

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
    from albumentations import Affine as _Affine
    from albumentations import HorizontalFlip as _HorizontalFlip
    from albumentations import Perspective as _Perspective
    from albumentations import Rotate as _Rotate
    from albumentations import SafeRotate as _SafeRotate
    from albumentations import ShiftScaleRotate as _ShiftScaleRotate
    from albumentations import VerticalFlip as _VerticalFlip

    TRANSFORM_REGISTRY: dict[type, TransformCategory] = {
        _Affine: TransformCategory.GEOMETRIC_INTERP,
        _Rotate: TransformCategory.GEOMETRIC_INTERP,
        _SafeRotate: TransformCategory.GEOMETRIC_INTERP,
        _ShiftScaleRotate: TransformCategory.GEOMETRIC_INTERP,
        _HorizontalFlip: TransformCategory.GEOMETRIC_EXACT,
        _VerticalFlip: TransformCategory.GEOMETRIC_EXACT,
        _Perspective: TransformCategory.PROJECTIVE,
    }

    # Frozensets used by exact_flip_dims to identify flip types without
    # enumerating them explicitly (supports subclasses of _ShiftScaleRotate etc.)
    _HFLIP_TYPES: frozenset[type] = frozenset({_HorizontalFlip})
    _VFLIP_TYPES: frozenset[type] = frozenset({_VerticalFlip})
    _INTERP_TYPES: frozenset[type] = frozenset({_Affine, _Rotate, _SafeRotate, _ShiftScaleRotate})
    _PROJECTIVE_TYPES: frozenset[type] = frozenset({_Perspective})

except ImportError:
    TRANSFORM_REGISTRY = {}
    _HFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _VFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _INTERP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _PROJECTIVE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]


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

        ttype = type(transform)

        # Flip transforms have no sampled params — only need batch size.
        if ttype in _HFLIP_TYPES or ttype in _VFLIP_TYPES:
            return {"_batch_size": torch.tensor([B], device=device, dtype=torch.int64)}

        # GEOMETRIC_INTERP: extract the pre-built matrix B times (once per sample)
        if ttype in _INTERP_TYPES or ttype in TRANSFORM_REGISTRY:
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
        ttype = type(transform)

        if ttype in _HFLIP_TYPES:
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            M_np = hflip_matrix_np(W=W)  # noqa: N806
            return torch.tensor(M_np, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if ttype in _VFLIP_TYPES:
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            M_np = vflip_matrix_np(H=H)  # noqa: N806
            return torch.tensor(M_np, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1).clone()

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
        ttype = type(transform)
        if ttype in _HFLIP_TYPES:
            return [3]
        if ttype in _VFLIP_TYPES:
            return [2]
        raise TypeError(f"Cannot determine flip dims for {ttype.__name__!r}")

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    dummy = np.zeros((H, W, 1), dtype=np.float32)
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
