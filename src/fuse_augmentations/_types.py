"""Type definitions for the fuse-augmentations library."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

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
        POINTWISE_LINEAR: Reorderable per-pixel *linear* op; self-fusible as 4x4 color-space
            affine matrix (brightness, contrast, channel mix, hue rotation).  Not yet fused by
            the current engine -- treated as a pass-through like ``POINTWISE`` until
            ``FusedColorSegment`` is implemented in a later version.

    """

    GEOMETRIC_INTERP = "geometric_interp"
    GEOMETRIC_EXACT = "geometric_exact"
    POINTWISE = "pointwise"
    SPATIAL_KERNEL = "spatial_kernel"
    PROJECTIVE = "projective"
    POINTWISE_LINEAR = "pointwise_linear"


class ReorderPolicy(Enum):
    """Controls whether transforms are reordered before segmentation.

    Attributes:
        NONE: No reordering; fuse only consecutive geometric ops as-is (v0.1 default).
        POINTWISE: Move POINTWISE ops out of geometric chains (v0.2).
        AGGRESSIVE: Alias of POINTWISE today; reserved for stronger reorder semantics later.

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
class BackendConverter(Protocol):
    """Protocol for output format converters used with ``output_backend=``.

    Implementations convert the pipeline's native ``torch.Tensor`` output to a
    target backend format (e.g. NumPy ``ndarray``).

    """

    def convert(self, tensor: Any) -> Any:  # noqa: ANN401
        """Convert tensor to target backend format.

        Args:
            tensor: The input tensor to convert.

        Returns:
            Converted object in the target backend format.

        """
        ...

    @property
    def target_backend(self) -> str:
        """Target backend identifier (e.g. ``'numpy'``, ``'torch'``)."""
        ...


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

    def exact_apply(self, transform: object, image: Tensor) -> Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly to an image batch.

        Implementers **must** provide this method for adapters that are used
        with :class:`ExactAffineSegment`. A typical implementation flips the
        image along the dims returned by :meth:`exact_flip_dims`.
        Adapters that support non-flip discrete ops (e.g. 90-degree rotations,
        transposes) can instead dispatch via ``torch.rot90``, ``.permute``,
        etc.

        Args:
            transform: The backend transform object (GEOMETRIC_EXACT category).
            image: ``(B, C, H, W)`` input tensor.

        Returns:
            Transformed ``(B, C, H, W)`` tensor.

        """
        return image.flip(dims=self.exact_flip_dims(transform))

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
class TransformSpec:
    """Declarative specification for a single augmentation transform.

    A backend-agnostic, JSON-serialisable description of one augmentation
    operation. Used by :meth:`FusedCompose.from_config
    <fuse_augmentations._compose.FusedCompose.from_config>` and
    :meth:`FusedCompose.from_params
    <fuse_augmentations._compose.FusedCompose.from_params>` to build
    pipelines from configuration data rather than live transform objects.

    Args:
        op: Canonical operation name (e.g. ``"rotation"``, ``"hflip"``).
        params: Operation-specific parameters (e.g. ``{"degrees": (-30, 30)}``).
            Range values (``degrees``, ``scale``, etc.) must be 2-tuples, not
            lists. For JSON/YAML-deserialized configs use :meth:`from_dict`,
            which restores tuple semantics from lists automatically.
        p: Per-sample application probability. Default ``1.0``.

    Example:
        >>> spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        >>> spec.op
        'rotation'
        >>> spec.p
        0.8

    """

    op: str
    params: Mapping[str, object]
    p: float = 1.0

    def __post_init__(self) -> None:
        """Freeze mapping-like params and validate probability bounds."""
        if not (0.0 <= self.p <= 1.0):
            msg = f"TransformSpec.p must be in [0.0, 1.0], got {self.p!r}"
            raise ValueError(msg)
        object.__setattr__(self, "params", _freeze_param_mapping(self.params))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict representation.

        Returns:
            Dict with keys ``"op"``, ``"params"``, and ``"p"``.

        Example:
            >>> spec = TransformSpec(op="hflip", params={}, p=0.5)
            >>> spec.to_dict()
            {'op': 'hflip', 'params': {}, 'p': 0.5}

        """
        params = _to_json_compatible(self.params)
        return {"op": self.op, "params": params, "p": self.p}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "TransformSpec":
        """Construct a ``TransformSpec`` from a dict (e.g. parsed JSON).

        Args:
            d: Dict with at least an ``"op"`` key. ``"params"`` defaults to
                ``{}`` and ``"p"`` defaults to ``1.0`` when absent.

        Returns:
            A new ``TransformSpec`` instance.

        Note:
            Tuple restoration from JSON lists applies only to canonical range-parameter
            keys ('degrees', 'factor', 'scale', 'pixels', etc. — the
            full set is :data:`_RANGE_PARAM_KEYS`). Backend-specific keys not in that
            set (e.g. Albumentations 'limit') are preserved as lists after JSON
            round-trip. This is documented behaviour, not a bug.

        Example:
            >>> import json
            >>> spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
            >>> restored = TransformSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
            >>> restored == spec  # list → tuple restored
            True

        """
        if "params" in d:
            raw_params = d["params"]
            if raw_params is None:
                raw_params = {}
            elif not isinstance(raw_params, Mapping):
                raise TypeError(
                    f"TransformSpec.from_dict expected 'params' to be a mapping, got {type(raw_params).__name__!r}."
                )
        else:
            raw_params = {}
        params = _normalize_loaded_params(raw_params)
        raw_p = d.get("p", 1.0)
        p = float(raw_p)  # type: ignore[arg-type]
        return cls(op=str(d["op"]), params=params, p=p)


_RANGE_PARAM_KEYS: frozenset[str] = frozenset({
    "degrees",
    "factor",
    "pixels",
    "rotation",
    "scale",
    "scale_x",
    "scale_y",
    "shear",
    "shear_x",
    "shear_y",
    "times",
    "translate",
    "translate_x",
    "translate_y",
})


def _freeze_param_mapping(params: Mapping[str, object]) -> MappingProxyType[str, object]:
    """Copy params into an immutable top-level mapping."""
    return MappingProxyType({key: _freeze_param_value(value) for key, value in params.items()})


def _freeze_param_value(value: object) -> object:
    """Recursively freeze nested mappings while preserving list semantics."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_param_value(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return [_freeze_param_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_param_value(item) for item in value)
    return copy.deepcopy(value)


def _to_json_compatible(value: object) -> Any:  # noqa: ANN401
    """Convert params tree into JSON-friendly builtin containers."""
    if isinstance(value, Mapping):
        return {key: _to_json_compatible(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    return value


def _normalize_loaded_params(params: Mapping[str, object]) -> dict[str, object]:
    """Restore range tuples from JSON while preserving non-range lists."""
    return {key: _normalize_loaded_value(key, value) for key, value in params.items()}


def _normalize_loaded_value(key: str | None, value: object) -> object:
    """Normalize one loaded param tree."""
    if isinstance(value, Mapping):
        return {nested_key: _normalize_loaded_value(nested_key, nested) for nested_key, nested in value.items()}
    if isinstance(value, list):
        normalized_items = [_normalize_loaded_value(None, item) for item in value]
        if (
            key in _RANGE_PARAM_KEYS
            and len(normalized_items) == 2
            and all(isinstance(item, (int, float)) for item in normalized_items)
        ):
            return tuple(normalized_items)
        return normalized_items
    return value


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    """Structured description of one segment in a fused augmentation pipeline.

    Returned by :attr:`FusedCompose.fusion_plan_descriptors
    <fuse_augmentations._compose.FusedCompose.fusion_plan_descriptors>`.
    Each instance describes exactly one segment — a fused geometric group,
    a lossless exact segment, a projective segment, or a passthrough barrier —
    and is frozen and JSON-serialisable via :meth:`to_dict`.

    Args:
        kind: Segment type. One of ``"fused"``, ``"exact"``, ``"projective"``,
            or ``"passthrough"``.
        transforms: Class names of the transforms in this segment, in
            execution order.
        n_warps_saved: Number of ``grid_sample`` interpolation passes
            eliminated by fusing this segment. Zero for passthrough and
            single-transform segments.
        backend: Adapter class name used for this segment
            (for example ``"KorniaAdapter"``, ``"AlbumentationsAdapter"``,
            ``"TorchVisionAdapter"``), or ``None`` for backend-free pipelines
            created via
            :meth:`FusedCompose.from_params <fuse_augmentations._compose.FusedCompose.from_params>`.

    Example:
        >>> d = SegmentDescriptor(
        ...     kind="fused",
        ...     transforms=("RandomRotation", "RandomHorizontalFlip"),
        ...     n_warps_saved=1,
        ...     backend="KorniaAdapter",
        ... )
        >>> d.kind
        'fused'
        >>> d.n_warps_saved
        1
        >>> d.to_dict()  # doctest: +SKIP
        {'kind': 'fused', 'transforms': ['RandomRotation', ...], ...}

    """

    kind: str
    transforms: tuple[str, ...]
    n_warps_saved: int
    backend: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict representation of this descriptor.

        Returns:
            Dict with keys ``"kind"``, ``"transforms"``, ``"n_warps_saved"``,
            and ``"backend"``. The ``"transforms"`` value is a ``list`` of
            strings (not a ``tuple``) for JSON compatibility.

        Example:
            >>> d = SegmentDescriptor(
            ...     kind="passthrough",
            ...     transforms=("RandomGaussianBlur",),
            ...     n_warps_saved=0,
            ...     backend="kornia",
            ... )
            >>> d.to_dict()
            {'kind': 'passthrough', 'transforms': ['RandomGaussianBlur'], 'n_warps_saved': 0, 'backend': 'kornia'}

        """
        return {
            "kind": self.kind,
            "transforms": list(self.transforms),
            "n_warps_saved": self.n_warps_saved,
            "backend": self.backend,
        }
