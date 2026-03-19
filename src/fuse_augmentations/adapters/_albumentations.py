"""Albumentations backend adapter for the fused affine engine.

Bridges Albumentations augmentation transforms to the canonical parameter
representation used by ``NumpyFusedAffineSegment``.

Each geometric transform (``A.Affine``, ``A.Rotate``, ``A.ShiftScaleRotate``)
exposes a pre-built ``matrix`` key via ``get_params_dependent_on_data()`` —
a ``(3, 3)`` forward pixel-space affine matrix in the same convention as
``_matrix.py``. The adapter reads this matrix directly instead of
reconstructing it from raw angle/scale/shear values.

Flip transforms (``A.HorizontalFlip``, ``A.VerticalFlip``) return an empty
parameter dict; ``build_matrix()`` constructs their matrices via
``_np_matrix.py``.

Requires ``albumentations >= 2.0``.

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

from fuse_augmentations._np_matrix import hflip_matrix_np, vflip_matrix_np
from fuse_augmentations._types import TransformCategory

# ---------------------------------------------------------------------------
# Transform registry — lazy import guard (albumentations is optional)
# ---------------------------------------------------------------------------

try:
    from albumentations import Affine as _Affine
    from albumentations import HorizontalFlip as _HorizontalFlip
    from albumentations import Rotate as _Rotate
    from albumentations import ShiftScaleRotate as _ShiftScaleRotate
    from albumentations import VerticalFlip as _VerticalFlip

    TRANSFORM_REGISTRY: dict[type, TransformCategory] = {
        _Affine: TransformCategory.GEOMETRIC_INTERP,
        _Rotate: TransformCategory.GEOMETRIC_INTERP,
        _ShiftScaleRotate: TransformCategory.GEOMETRIC_INTERP,
        _HorizontalFlip: TransformCategory.GEOMETRIC_EXACT,
        _VerticalFlip: TransformCategory.GEOMETRIC_EXACT,
    }

    # Frozensets used by exact_flip_dims to identify flip types without
    # enumerating them explicitly (supports subclasses of _ShiftScaleRotate etc.)
    _HFLIP_TYPES: frozenset[type] = frozenset({_HorizontalFlip})
    _VFLIP_TYPES: frozenset[type] = frozenset({_VerticalFlip})
    _INTERP_TYPES: frozenset[type] = frozenset({_Affine, _Rotate, _ShiftScaleRotate})

except ImportError:
    TRANSFORM_REGISTRY = {}
    _HFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _VFLIP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _INTERP_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]


class AlbumentationsAdapter:
    """Adapter between Albumentations transforms and the fused affine engine.

    Implements the ``TransformAdapter`` protocol for the Albumentations backend.
    Supports ``A.Affine``, ``A.Rotate``, ``A.ShiftScaleRotate`` (GEOMETRIC_INTERP)
    and ``A.HorizontalFlip``, ``A.VerticalFlip`` (GEOMETRIC_EXACT).

    Requires ``albumentations >= 2.0``. The adapter reads the pre-built
    ``matrix`` key from ``get_params_dependent_on_data()`` rather than
    reconstructing the matrix from raw parameters.

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
        cat = TRANSFORM_REGISTRY.get(type(transform))
        if cat is not None:
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

        For GEOMETRIC_INTERP transforms, calls
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
            for GEOMETRIC_INTERP transforms; ``"_batch_size"`` for flips.

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
        """Build a ``(B, 3, 3)`` pixel-space forward affine matrix.

        For GEOMETRIC_INTERP transforms, returns the pre-sampled ``params["matrix"]``
        tensor directly (it was already stacked in ``sample_params()``).

        For flip transforms, constructs the appropriate constant matrix using
        ``_np_matrix.py`` and expands it to batch size B.

        Args:
            transform: An Albumentations transform instance.
            params: Parameter dict from ``sample_params()``.
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            ``(B, 3, 3)`` forward affine matrix in pixel coordinates.

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
            results.append(torch.from_numpy(np.ascontiguousarray(out_np)).permute(2, 0, 1))

        return torch.stack(results).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_matrices(transform: object, B: int, H: int, W: int) -> np.ndarray:  # noqa: N803
    """Call ``get_params_dependent_on_data()`` B times, stack into (B, 3, 3).

    A dummy ``(H, W, 1)`` float32 image is passed so the transform can
    compute center coordinates. No actual pixel data is read.

    Args:
        transform: An Albumentations GEOMETRIC_INTERP transform.
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
        full = transform.get_params_dependent_on_data(base, data)  # type: ignore[attr-defined]
        matrices[i] = full["matrix"]
    return matrices
