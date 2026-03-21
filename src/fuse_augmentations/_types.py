"""Type definitions for the fuse-augmentations library."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


class TransformCategory(Enum):
    """Category of an augmentation transform for fusion classification.

    Attributes:
        GEOMETRIC_INTERP: Fusible geometric op requiring interpolation (rotate, scale, shear).
        GEOMETRIC_EXACT: Fusible only when INTERP is present; lossless alone (flip, 90-deg rot).
        POINTWISE: Reorderable per-pixel op; not fusible (color jitter, normalize).
        SPATIAL_KERNEL: Barrier; not fusible and not reorderable (blur, noise, erase).
        PROJECTIVE: Fusible projective (perspective) op requiring full 3x3 homography.

    """

    GEOMETRIC_INTERP = "geometric_interp"
    GEOMETRIC_EXACT = "geometric_exact"
    POINTWISE = "pointwise"
    SPATIAL_KERNEL = "spatial_kernel"
    PROJECTIVE = "projective"


class ReorderPolicy(Enum):
    """Controls whether transforms are reordered before segmentation.

    Attributes:
        NONE: No reordering; fuse only consecutive geometric ops as-is (v0.1 default).
        POINTWISE: Move POINTWISE ops out of geometric chains (v0.2).
        AGGRESSIVE: Reserved; raises NotImplementedError (v0.1-v0.6).

    """

    NONE = "none"
    POINTWISE = "pointwise"
    AGGRESSIVE = "aggressive"


class InterpolationMode(IntEnum):
    """Interpolation modes ordered by quality (higher = finer).

    Example:
        >>> InterpolationMode.BICUBIC > InterpolationMode.BILINEAR
        True

    """

    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class PaddingMode(IntEnum):
    """Padding modes ordered by quality (higher = fewer artifacts).

    Example:
        >>> PaddingMode.REFLECTION > PaddingMode.ZEROS
        True

    """

    ZEROS = 0
    BORDER = 1
    REFLECTION = 2


@runtime_checkable
class TransformAdapter(Protocol):
    """Adapter between a backend transform and the fused affine engine.

    Implementations bridge framework-specific transforms (Kornia, Albumentations, TorchVision) to the canonical
    parameter representation used by FusedAffineSegment.

    """

    def category(self, transform: object) -> TransformCategory:
        """Return the TransformCategory of the given transform.

        Args:
            transform: The backend transform object.

        Returns:
            The category classification for the transform.

        """
        ...

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Sample random parameters for a batch of B images.

        Args:
            transform: The backend transform object.
            input_shape: (B, C, H, W) tuple.
            device: Target device for parameter tensors.

        Returns:
            Dict mapping canonical parameter names to (B,) tensors.

        """
        ...

    def build_matrix(
        self,
        transform: object,
        params: dict[str, Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> Tensor:
        """Build a (B, 3, 3) pixel-space forward affine matrix from sampled params.

        Args:
            transform: The backend transform object.
            params: Canonical-unit parameter dict from sample_params().
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            Tensor of shape (B, 3, 3).

        """
        ...

    def exact_flip_dims(self, transform: object) -> list[int]:
        """Return the tensor dimensions to flip for a GEOMETRIC_EXACT transform.

        Args:
            transform: The backend transform object (must be GEOMETRIC_EXACT category).

        Returns:
            List of dimension indices passed to ``tensor.flip(dims=...)``,
            e.g. ``[3]`` for a horizontal flip, ``[2]`` for a vertical flip.

        Raises:
            NotImplementedError: If the adapter does not support ExactAffineSegment.

        """
        raise NotImplementedError("Adapter does not implement exact_flip_dims; required for ExactAffineSegment support")

    def call_nonfused(
        self,
        transform: object,
        image: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Apply a non-fusible transform directly via its native backend.

        Args:
            transform: The backend transform object.
            image: Input image tensor.
            **kwargs: Additional keyword arguments forwarded to the transform.

        Returns:
            Transformed image tensor.

        """
        ...


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    """Structured description of one segment in the fusion plan.

    Attributes:
        kind: Segment type -- ``"fused"`` | ``"exact"`` | ``"projective"`` | ``"passthrough"``.
        transforms: Tuple of transform class names in this segment.
        n_warps_saved: Number of interpolation passes eliminated by this segment.
        backend: Adapter class name owning this segment (``None`` for ``from_params`` pipelines).

    Example:
        >>> d = SegmentDescriptor(kind="fused", transforms=("RandomRotation",), n_warps_saved=0)
        >>> d.kind
        'fused'
        >>> import json; json.dumps(d.to_dict())  # doctest: +SKIP
        '...'

    """

    kind: str
    transforms: tuple[str, ...]
    n_warps_saved: int
    backend: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict representation."""
        return {
            "kind": self.kind,
            "transforms": list(self.transforms),
            "n_warps_saved": self.n_warps_saved,
            "backend": self.backend,
        }
