"""Compose -- fused augmentation pipeline replacing the backend's Compose/Sequential.

Wraps a list of augmentation transforms, fuses consecutive geometric ops into a
single ``grid_sample`` pass, and provides the same forward-call interface as the
backend.

v0.3 additions: ``data_keys`` parameter routes auxiliary targets (masks,
bounding boxes, keypoints) alongside the image through every segment.
``from_params()`` classmethod constructs a fused pipeline directly from
numeric parameter ranges, without importing any backend.

v0.5 additions: mixed-backend support allows transforms from multiple
frameworks (e.g. TorchVision geometric + Kornia color) in a single
pipeline. ``_PassthroughSegment`` carries non-fused transforms across
pickle round-trips via an index-keyed adapter map that survives
deserialisation (object ids change; positional indices do not).

Example:
    >>> import torch
    >>> from fuse_augmentations.compose import Compose
    >>> pipe = Compose([])
    >>> image = torch.zeros(1, 3, 8, 8)
    >>> pipe(image).shape
    torch.Size([1, 3, 8, 8])
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from fuse_augmentations._backend import Backend, detect_backends_per_transform
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.affine.matrix import (
    hflip_matrix,
    matmul3x3,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)
from fuse_augmentations.affine.segment import (
    AlbuFusedAffineSegment,
    AlbuProjectiveSegment,
    CropResizeSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    FusedColorSegment,
    ProjectiveSegment,
    _clear_current_call_matrix,
    _current_call_matrix,
    _validate_execution,
    build_segments,
    reorder_aggressive,
    reorder_pointwise,
)
from fuse_augmentations.types import (
    ClipPolicyStr,
    ExecutionStr,
    InterpolationStr,
    MaskInterpolationStr,
    PaddingModeStr,
    RandomnessPolicy,
    ReorderPolicy,
    SegmentDescriptor,
    TransformAdapter,
    TransformCategory,
    TransformSpec,
    is_coordinate_changing_passthrough,
)

__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.from_config"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.__call__"]

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fuse_augmentations.resolver import BackendStr, OpStr

_KNOWN_DATA_KEYS = {"input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints"}
# Albumentations-style keyword aliases accepted by the dict-output call form
# (``pipe(image=..., mask=..., bboxes=...)``). ``image`` maps to the ``"input"``
# data key; ``bboxes`` maps to whichever box key the pipeline declared. Exact
# data-key names are also accepted as their own aliases.
_KWARG_ALIASES: dict[str, str | tuple[str, ...]] = {
    "image": "input",
    "bboxes": ("bbox_xyxy", "bbox_xywh"),
}
# Coordinate aux targets that ExactAffineSegment cannot route through a non-flip exact
# op: their per-sample rotation is not recoverable without re-sampling. When present, an
# all-exact geometric run is routed through the (still-lossless for D4) grid path.
_COORD_DATA_KEYS = {"bbox_xyxy", "bbox_xywh", "keypoints"}


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


@dataclass(frozen=True, slots=True)
class _PassthroughSegment:
    """Serializable passthrough segment that carries its adapter explicitly."""

    transform: object
    adapter: TransformAdapter


class FusedCompose(nn.Module):
    """Fused augmentation pipeline that replaces the backend's native Compose.

    Segments the transform list into fused geometric segments and passthrough transforms, then executes them
    sequentially. Consecutive geometric ops are grouped and executed as either:

    - A :class:`~fuse_augmentations.affine.segment.FusedAffineSegment` - when the run
      contains at least one ``GEOMETRIC_INTERP`` op. Matrices are composed and
      a single ``grid_sample`` call is used, eliminating redundant interpolation
      passes.
    - An :class:`~fuse_augmentations.affine.segment.ExactAffineSegment` - when the run
      contains *only* ``GEOMETRIC_EXACT`` ops (HFlip, VFlip). Transforms are
      applied via ``tensor.flip`` with zero interpolation error.

    ``SPATIAL_KERNEL`` and nonlinear ``POINTWISE`` transforms are passed through
    to the backend adapter unchanged. Supported ``POINTWISE_LINEAR`` color
    transforms, including standard Normalize, are folded into a color segment.

    ``ReorderPolicy.POINTWISE`` is fully implemented: before segmentation,
    ``POINTWISE`` and ``POINTWISE_LINEAR`` ops are bubbled past geometric ops
    within each ``SPATIAL_KERNEL``-bounded stretch, maximising the geometric
    run length available for fusion.

    Args:
        transforms: List of augmentation transform objects.
        reorder: Reorder policy applied before segmentation.
            ``NONE`` (default) preserves the original order. ``POINTWISE`` reorders pointwise ops after geometric
            chains. ``AGGRESSIVE`` currently aliases ``POINTWISE`` and is kept for API compatibility with future
            stronger reorder semantics.
        interpolation: Interpolation mode override for fused segments
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode override for fused segments
            (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        data_keys: List of key names describing positional arguments to
            :meth:`forward`. The first key should be ``"input"`` (the image).
            Auxiliary keys (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
            ``"keypoints"``) are routed through segments and transformed alongside the image. Unknown keys are passed
            through unchanged with a ``UserWarning``. ``None`` preserves backward-compatible single-tensor input/output.
            Albumentations fused segments route auxiliary targets through the composed pixel matrix, matching the
            Kornia/TorchVision coordinate convention, so multi-target ``data_keys`` are supported for every backend.
        output_backend: Target output format. ``"numpy"`` (or its alias
            ``"numpy_hwc"``) converts the primary image output to a NumPy
            ``ndarray`` with channel-last layout: batched inputs of shape
            ``(batch_size, channels, height, width)`` become ``(batch_size, height, width, channels)``,
            while unbatched inputs of shape ``(channels, height, width)`` become ``(height, width, channels)`` (i.e.
            the batch dimension is implicit/squeezed). ``"torch"`` or ``None`` keeps the native ``torch.Tensor``
            output. For multi-target ``data_keys`` the conversion is applied per target: image and mask outputs are
            converted, while coordinate targets (boxes, keypoints) stay tensors.
        randomness: Batch randomness policy. ``"backend"`` preserves native
            backend semantics, including TorchVision v2 batch-shared sampling.
            ``"per_sample"`` asks adapters with canonical samplers to draw
            independent parameters per batch item.
        execution: Warp strategy for the Albumentations fused segments.
            ``"cv2"`` (default) warps each sample with OpenCV -- bit-exact with the
            native cv2 backend and fastest on CPU at small batch sizes.
            ``"torch"`` composes the same per-sample matrices (identical sampling
            and RNG) but applies one batched ``grid_sample`` for the whole batch,
            giving batch-size-independent throughput and a native GPU/MPS warp; its
            border/bilinear numerics differ slightly from cv2. Only affects
            Albumentations pipelines; the Kornia/TorchVision backends already run
            the torch engine.
        compile: Opt-in ``torch.compile`` of the warp core (matrix normalize ->
            ``affine_grid`` -> ``grid_sample``) of the torch-backed fused segments.
            Off by default. When ``True``, the compiled region is used only on
            non-CPU tensors and only when the installed torch is new enough
            (``>= 2.2``); otherwise the eager path runs and the flag is a no-op, so
            the output is unchanged. Probability masking stays outside the compiled
            region, so the graph has no data-dependent breaks. The first non-CPU call
            pays a one-time compilation cost; ``dynamic=True`` avoids per-shape
            recompiles.
        antialias: Opt-in antialiasing for aggressive downscales. Off by default.
            When ``True``, a crop-resize (or fused geo+crop) segment whose worst-axis
            scale drops below ``0.5`` prefilters the input with a Gaussian
            (mipmap-rule sigma) before the single warp, so the downscale no longer
            aliases. A no-op when the scale is safe or the flag is off, keeping the
            default output bit-identical; requires kornia for the blur (falls back to
            the un-filtered warp when kornia is absent).
        substitute_passthrough: When ``True`` (default ``False``), non-fusible
            passthrough ops that have a registered torch-native equivalent in an
            already-installed backend are replaced with that equivalent, so the
            pipeline can stay on-device instead of paying a device-to-host round-trip.
            The initial registry maps Albumentations ``GaussianBlur`` to Kornia
            ``RandomGaussianBlur``. This is **behaviour-changing**: the substitute uses
            a different kernel implementation, border handling, and per-call random
            stream, so outputs and RNG differ from the original op — every substitution
            emits a :class:`UserWarning`. Substitution happens only when the target
            backend is importable; otherwise the original op is kept silently and runs
            on the normal passthrough path. The default (``False``) leaves outputs
            unchanged.
        clip_policy: Clamp policy for fused color segments. ``"final"`` (default)
            applies one clamp after the composed color matmul — most precise.
            ``"per_op_parity"`` splits the fused color run at any op where a
            composed intermediate would leave ``[0, 1]`` and clamps in between,
            approximating the native per-op clamped chain on gamut-escaping
            pipelines. Known gap: when a gamut-escaping op immediately precedes a
            mean-relative contrast op, the contrast midpoint is taken from the
            pre-clamp mean, so output can differ from native by ~1e-2.
        mask_interpolation: Auxiliary mask sampling mode. ``"nearest"`` (default)
            preserves hard labels; ``"bilinear"`` provides differentiable soft-mask
            sampling and requires floating-point mask input.
        **backend_kwargs: Reserved for backend-specific options (currently unused).

    """

    def __init__(
        self,
        transforms: list[object],
        reorder: ReorderPolicy = ReorderPolicy.NONE,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        execution: ExecutionStr = "cv2",
        compile: bool = False,  # noqa: A002 — public flag name mirrors torch.compile; shadowing builtin is intentional
        antialias: bool = False,
        substitute_passthrough: bool = False,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        **backend_kwargs: object,
    ) -> None:
        """Initialize ``FusedCompose``."""
        super().__init__()
        randomness_policy = _coerce_randomness_policy(randomness)
        execution = _validate_execution(execution)
        clip_policy = _validate_clip_policy(clip_policy)
        mask_interpolation = _validate_mask_interpolation(mask_interpolation)

        if reorder not in (ReorderPolicy.NONE, ReorderPolicy.POINTWISE, ReorderPolicy.AGGRESSIVE):
            msg = f"ReorderPolicy.{reorder.name} not yet supported"
            raise NotImplementedError(msg)

        # Opt-in substitution: replace registered non-fusible passthrough ops with an
        # already-installed backend's torch-native equivalent so the pipeline can stay
        # on-device. Applied before backend detection so the substitute (a torch op)
        # flows through the normal segmentation path. Behaviour-changing → each swap
        # warns; default off leaves the transform list untouched.
        if substitute_passthrough and transforms:
            transforms = _apply_passthrough_substitution(transforms)

        adapter: TransformAdapter | None
        segments: list[object]
        tfm_adapters: dict[int, TransformAdapter] | None = None

        if not transforms:
            adapter = None
            segments = []
        else:
            per_backends = detect_backends_per_transform(transforms)
            unique_backends = {backend_name for backend_name in per_backends if backend_name is not None}

            if len(unique_backends) <= 1:
                # Single-backend fast path (backward compatible)
                if not unique_backends:
                    raise ValueError(
                        "No recognised backend transforms found in pipeline. "
                        "Ensure transforms are from a supported backend: "
                        "Kornia, TorchVision, or Albumentations."
                    )
                backend = unique_backends.pop()
                adapter = _adapter_for_backend(backend)
                # Unrecognised transforms fall back to this backend's adapter for
                # passthrough dispatch. Kornia/TorchVision adapters call transforms
                # tensor-style, which is compatible with custom torch callables, but
                # the Albumentations adapter uses the dict convention
                # ``transform(image=<ndarray>)``, which silently misbehaves or breaks
                # on foreign objects — reject that combination up front.
                if backend == Backend.ALBUMENTATIONS and any(bknd is None for bknd in per_backends):
                    unknown_names = [
                        type(tfm).__name__ for tfm, bknd in zip(transforms, per_backends, strict=False) if bknd is None
                    ]
                    raise ValueError(
                        f"Unrecognised transform(s) {unknown_names} cannot be dispatched through the "
                        "Albumentations adapter (dict calling convention transform(image=...)). "
                        "Remove them or use a registered Albumentations transform."
                    )
                if reorder == ReorderPolicy.POINTWISE:
                    transforms = reorder_pointwise(transforms, adapter)
                elif reorder == ReorderPolicy.AGGRESSIVE:
                    transforms = reorder_aggressive(transforms, adapter)
                segments = build_segments(
                    transforms=transforms,
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    randomness=randomness_policy,
                    use_numpy=(backend == Backend.ALBUMENTATIONS),
                    route_coords_via_grid=_has_coord_aux(data_keys),
                    execution=execution,
                    compile_warp=compile,
                    antialias=antialias,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                )
            else:
                # Mixed-backend path: group by backend, build segments per group
                adapter, segments, tfm_adapters = _build_mixed_segments(
                    transforms=transforms,
                    per_backends=per_backends,
                    reorder=reorder,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    randomness=randomness_policy,
                    route_coords_via_grid=_has_coord_aux(data_keys),
                    execution=execution,
                    compile_warp=compile,
                    antialias=antialias,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                )

        self._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            transform_adapters=tfm_adapters,
            output_backend=output_backend,
            randomness=randomness_policy,
            execution=execution,
            compile_warp=compile,
            antialias=antialias,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
        )

    def _setup_instance(
        self,
        transforms: list[object],
        reorder: ReorderPolicy,
        interpolation: InterpolationStr | None,
        padding_mode: PaddingModeStr | None,
        data_keys: list[str] | None,
        adapter: TransformAdapter | None,
        segments: list[object],
        transform_adapters: dict[int, TransformAdapter] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        execution: ExecutionStr = "cv2",
        compile_warp: bool = False,
        antialias: bool = False,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Assign all instance attributes.

        Called by both ``__init__`` and ``from_params``.

        """
        self.original_transforms: list[object] = list(transforms)
        self.reorder: ReorderPolicy = reorder
        self.interpolation: InterpolationStr | None = interpolation
        self.padding_mode: PaddingModeStr | None = padding_mode
        self.data_keys: list[str] | None = data_keys
        self.randomness: RandomnessPolicy = randomness
        self.execution: ExecutionStr = execution
        self.compile_warp: bool = compile_warp
        self.antialias: bool = antialias
        self.clip_policy: ClipPolicyStr = clip_policy
        self.mask_interpolation: MaskInterpolationStr = _validate_mask_interpolation(mask_interpolation)
        self._adapter: TransformAdapter | None = adapter
        # Heterogeneous by design: fused/color/exact/crop segments, passthrough wrappers, or raw
        # legacy transforms — dispatched at runtime via _seg_dispatch_tags, so elements stay Any.
        self._segments: list[Any] = _wrap_passthrough_segments(
            segments=segments,
            default_adapter=adapter,
            transform_adapters=transform_adapters,
            original_transforms=transforms,
        )
        self._last_transform_matrix: Tensor | None = None
        self._last_matrix_segment: object | None = None  # deferred resolution
        self._fusion_plan_cache: tuple[bool, str] | None = None
        self._fusion_plan_descriptors_cache: list[SegmentDescriptor] | None = None
        self._transform_adapters: dict[int, TransformAdapter] = transform_adapters or {}

        # Device tracker: a zero-element buffer whose only purpose is to follow the
        # module across ``.to(device)`` / ``.cuda()`` / ``.mps()`` so ``fusion_plan``
        # can report whether the configured pipeline device is non-CPU (a CPU
        # passthrough on a GPU pipeline is a poison pill worth surfacing). Geometric
        # and passthrough segments hold no buffers of their own, so nothing else
        # tracks the module's device.
        if not hasattr(self, "_device_tracker"):
            self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

        # Plan-time dispatch: precompute one integer tag per segment so forward()
        # loops without an isinstance chain. data_keys is constructor state, so the
        # single-tensor vs multi-target split is resolved once here as well.
        self._seg_dispatch_tags: list[int] = self._build_dispatch_tags()
        self._multi_target: bool = data_keys is not None
        self._aux_keys: list[str] = list(data_keys[1:]) if data_keys is not None else []

        # Pre-compute single-segment fast-path: when the pipeline contains
        # exactly one ExactAffineSegment with a single transform, forward()
        # can bypass the segment's nn.Module.__call__ and probability
        # machinery by calling the native adapter directly via call_nonfused.
        # This eliminates ~6 us/call overhead for operations like single-flip
        # pipelines where the native transform takes <50 us total.
        # Only safe for real backend adapters (Kornia, TorchVision) where
        # call_nonfused delegates to the native transform; _DirectParamAdapter
        # has a no-op call_nonfused and must go through ExactAffineSegment.
        self._single_exact_fast: tuple[TransformAdapter, object] | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], ExactAffineSegment)
            and len(self._segments[0].transforms) == 1
            and adapter is not None
            and not isinstance(adapter, _DirectParamAdapter)
            and randomness is RandomnessPolicy.BACKEND
        ):
            self._single_exact_fast = (adapter, self._segments[0].transforms[0])

        # Single-transform FusedAffineSegment fast path: bypass the segment's
        # nn.Module.__call__ for single-transform GEOMETRIC_INTERP pipelines (a-group
        # rotate/scale/shear).  Saves ~10-15 us vs going through the segment
        # Module machinery.  Guards:
        # - Single segment, single transform (prevents cv2-path regression from
        #   iter 1 where multi-transform segments were incorrectly bypassed)
        # - Kornia or TorchVision adapter only (not Albu — uses _forward_albu_native)
        # - For TorchVision: v2 transforms only (v1 has incompatible interpolation)
        self._single_fused_fast_seg: FusedAffineSegment | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], FusedAffineSegment)
            and len(self._segments[0].transforms) == 1
            and getattr(self._segments[0], "_fast_path", None) in ("kornia", "torchvision")
            and adapter is not None
            and not isinstance(adapter, _DirectParamAdapter)
        ):
            _seg0 = self._segments[0]
            _tfm0 = _seg0.transforms[0]
            _bypass_ok = True
            if _seg0._fast_path == "torchvision":
                try:
                    from fuse_augmentations.adapters.torchvision import is_torchvision_v2_transform

                    _bypass_ok = is_torchvision_v2_transform(_tfm0)
                except ImportError:
                    _bypass_ok = False
            if _bypass_ok:
                self._single_fused_fast_seg = _seg0
        # Single-transform Albumentations direct fast path: for pipelines with exactly
        # one AlbuFusedAffineSegment containing one transform, bypass
        # _forward_albu_native entirely.  Saves ~5-10 us/call: 4 lazy imports,
        # function call overhead, forward_numpy dispatch, and segment isinstance loop.
        # Guards: single segment + single transform + AlbumentationsAdapter only.
        # Not used when aux_targets are passed (kwargs size > 1).
        self._single_albu_direct_seg: AlbuFusedAffineSegment | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], AlbuFusedAffineSegment)
            and len(self._segments[0].transforms) == 1
        ):
            try:
                from fuse_augmentations.adapters.albumentations import (
                    AlbumentationsAdapter as AlbuAdapterCls,
                )

                if isinstance(adapter, AlbuAdapterCls):
                    self._single_albu_direct_seg = self._segments[0]
            except ImportError:
                pass

        # Pre-computed Albumentations dict-input routing flag: avoids the
        # per-call lazy import + isinstance check in __call__ for Albu pipelines.
        _is_albu: bool = False
        if adapter is not None and not isinstance(adapter, _DirectParamAdapter):
            try:
                from fuse_augmentations.adapters.albumentations import (
                    AlbumentationsAdapter as _AlbuAdapterCls,
                )

                _is_albu = isinstance(adapter, _AlbuAdapterCls)
            except ImportError:
                pass
        self._is_albu_native: bool = _is_albu or (adapter is None and not self._segments)

        # Pre-classify segments for the _forward_albu_native hot path: replace
        # per-call isinstance checks with an integer-tagged dispatch list.
        # Tags: 0=AlbuFusedAffineSegment, 1=ExactAffine, 2=FusedColor,
        # 3=CropResize, 4=Passthrough(Albu), 5=Passthrough(non-Albu), -1=unsupported.
        # Built only for Albu pipelines where _forward_albu_native is used.
        self._albu_seg_tags: list[int] | None = None
        if self._is_albu_native and self._segments:
            tags: list[int] = []
            for seg in self._segments:
                if isinstance(seg, AlbuFusedAffineSegment):
                    tags.append(0)
                elif isinstance(seg, ExactAffineSegment):
                    tags.append(1)
                elif isinstance(seg, FusedColorSegment):
                    tags.append(2)
                elif isinstance(seg, CropResizeSegment):
                    tags.append(3)
                elif isinstance(seg, _PassthroughSegment):
                    try:
                        from fuse_augmentations.adapters.albumentations import (
                            AlbumentationsAdapter as _AlbuAdapterCheck,
                        )

                        if isinstance(seg.adapter, _AlbuAdapterCheck):
                            tags.append(4)
                        else:
                            tags.append(5)
                    except ImportError:
                        tags.append(5)
                else:
                    tags.append(-1)
            self._albu_seg_tags = tags

        # Resolve output_backend converter.
        from fuse_augmentations.types import BackendConverter

        self._output_converter: BackendConverter | None
        if output_backend is None:
            self._output_converter = None
        elif output_backend in ("numpy", "numpy_hwc"):
            from fuse_augmentations.converters import TorchToNumpyConverter

            self._output_converter = TorchToNumpyConverter()
        elif output_backend == "torch":
            self._output_converter = None  # identity — already torch
        else:
            msg = f"Unknown output_backend {output_backend!r}; supported: 'numpy', 'numpy_hwc', 'torch', None"
            raise ValueError(msg)
        if self._output_converter is not None and self._output_converter.target_backend not in (
            output_backend,
            "numpy",  # "numpy_hwc" alias maps to TorchToNumpyConverter whose target_backend is "numpy"
        ):
            msg = (
                "Configured output converter target_backend does not match output_backend. "
                f"Requested {output_backend!r}, converter advertises {self._output_converter.target_backend!r}."
            )
            raise RuntimeError(msg)

        if data_keys is not None:
            # Enforce documented contract: first key must map to the image ("input").
            if not data_keys:
                raise ValueError(
                    "data_keys cannot be an empty list. Omit data_keys entirely for single-tensor "
                    "mode, or include 'input' as the first entry for multi-target mode."
                )
            if data_keys[0] != "input":
                raise ValueError(
                    "data_keys[0] must be 'input' (the image tensor), got "
                    f"{data_keys[0]!r}. This prevents silent misrouting of positional arguments "
                    "in multi-target mode."
                )
            for key in data_keys:
                if key not in _KNOWN_DATA_KEYS:
                    warnings.warn(
                        f"Unknown data_key {key!r}; it will be passed through unchanged. "
                        f"Known keys: {sorted(_KNOWN_DATA_KEYS)}",
                        UserWarning,
                        stacklevel=3,
                    )

        # Albumentations fused segments route aux targets through the composed pixel
        # matrix (same coordinate convention as the torch path), so multi-target
        # data_keys are supported for both the cv2 and torch execution strategies.

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore instance state, rebuilding dispatch attributes absent from older pickles.

        The three plan-time dispatch attributes (``_seg_dispatch_tags``,
        ``_multi_target``, ``_aux_keys``) are computed in :meth:`_setup_instance`
        and read directly by :meth:`forward` and :meth:`_build_aux_targets` with no
        per-call fallback. Pickles produced before these attributes existed would
        raise :class:`AttributeError` on the first call after unpickling, so this
        hook reconstructs them from ``data_keys`` and the restored segments. Within
        a single version every attribute is already present, so this is a no-op.

        Args:
            state: The pickled instance ``__dict__`` restored by :mod:`pickle`.

        Examples:
            >>> import pickle
            >>> from fuse_augmentations import Compose  # doctest: +SKIP
            >>> pipe = Compose([...])  # doctest: +SKIP
            >>> reloaded = pickle.loads(pickle.dumps(pipe))  # doctest: +SKIP

        """
        super().__setstate__(state)  # type: ignore[no-untyped-call]
        if not hasattr(self, "_multi_target"):
            self._multi_target = self.data_keys is not None
            self._aux_keys = list(self.data_keys[1:]) if self.data_keys else []
        if getattr(self, "_seg_dispatch_tags", None) is None:
            self._seg_dispatch_tags = self._build_dispatch_tags()
        if not hasattr(self, "execution"):
            # Pickles predating the execution flag default to the cv2 strategy.
            self.execution = "cv2"
        if not hasattr(self, "clip_policy"):
            self.clip_policy = "final"
        if not hasattr(self, "mask_interpolation"):
            # Pickles predating configurable mask sampling retain nearest behavior.
            self.mask_interpolation = "nearest"
        if not hasattr(self, "_fusion_plan_cache"):
            self._fusion_plan_cache = None
        if not hasattr(self, "_fusion_plan_descriptors_cache"):
            self._fusion_plan_descriptors_cache = None

    def __call__(self, *args: object, return_matrix: bool = False, **kwargs: object) -> object:
        """Route to the Albumentations native dict path or the standard tensor path.

        When called with ``image=<numpy.ndarray>`` and the pipeline adapter is an
        :class:`~fuse_augmentations.adapters.albumentations.AlbumentationsAdapter`,
        the call is dispatched to :meth:`_forward_albu_native`, which avoids all
        tensor round-trips and returns a ``{"image": ndarray}`` dict matching the
        :class:`albumentations.Compose` calling convention.

        All other invocations fall through to the standard
        :meth:`~torch.nn.Module.__call__` → :meth:`forward` path.

        Args:
            *args: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments.  When ``"image"`` is present,
                ``isinstance(kwargs["image"], np.ndarray)`` is checked to
                decide routing.
            return_matrix: When ``True``, return the output together with the
                matrix produced by the last fused geometric segment.

        Returns:
            ``dict`` with ``"image"`` key (HWC NumPy) for the Albu native path;
            a ``dict`` keyed by the caller's keyword names for a multi-target
            keyword call (``image=..., mask=...``); or whatever :meth:`forward`
            returns for the positional tensor path.

        Examples:
            >>> import numpy as np
            >>> import albumentations as A
            >>> from fuse_augmentations import Compose
            >>> pipe = Compose([A.HorizontalFlip(p=0.0)])
            >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = pipe(image=img)
            >>> isinstance(out, dict) and out["image"].shape == (8, 8, 3)
            True

        """
        # Dict-out: an Albumentations-style keyword call (image=..., mask=..., etc.)
        # on a multi-target pipeline routes the aux targets through the fused
        # segments and returns a dict keyed by the caller's keyword names. Only
        # the keyword entry adds dict output; the positional data_keys API is
        # unchanged and still returns a tuple.
        if kwargs and "image" in kwargs and self._multi_target and self._aux_keys:
            return self._forward_kwargs_dict(kwargs, return_matrix=return_matrix)

        if (
            self._single_albu_direct_seg is not None
            and len(kwargs) == 1
            and "image" in kwargs
            and isinstance(kwargs["image"], np.ndarray)
        ):
            _clear_current_call_matrix()
            img = self._single_albu_direct_seg.forward_numpy(kwargs["image"])
            self._last_transform_matrix = self._single_albu_direct_seg.last_matrix
            result: object = {"image": img}
            if return_matrix:
                return result, _current_call_matrix()
            return result
        if self._is_albu_native and "image" in kwargs and isinstance(kwargs["image"], np.ndarray):
            if len(kwargs) > 1:
                extra_keys = sorted(key for key in kwargs if key != "image")
                raise NotImplementedError(
                    f"The Albumentations native dict path only transforms 'image'; auxiliary "
                    f"keys {extra_keys} would be silently dropped. Auxiliary targets are not "
                    "supported on this path — pass tensors with data_keys instead."
                )
            _clear_current_call_matrix()
            result = self._forward_albu_native(kwargs["image"])
            if return_matrix:
                return result, _current_call_matrix()
            return result

        # Single-transform tensor fast paths: bypass nn.Module.__call__ overhead
        # (~10-15 us from hook dispatch, _call_impl indirection) for the
        # common single-tensor, single-transform case.  Guards: exactly 1 positional
        # arg, no kwargs, data_keys is None (single-tensor mode).
        if len(args) == 1 and not kwargs and self.data_keys is None:
            _exact_fast = self._single_exact_fast
            if _exact_fast is not None:
                # Exact segments produce no matrix; clear any stale value so the
                # transform_matrix "None after exact/passthrough-only" contract holds.
                self._last_transform_matrix = None
                image = _exact_fast[0].call_nonfused(_exact_fast[1], cast(Tensor, args[0]))
                result = self._convert_primary_output(image)
                return (result, None) if return_matrix else result

            _fused_fast = self._single_fused_fast_seg
            if _fused_fast is not None:
                image = cast(Tensor, _fused_fast.forward(cast(Tensor, args[0]), None))
                self._last_transform_matrix = _current_call_matrix()
                result = self._convert_primary_output(image)
                return (result, _current_call_matrix()) if return_matrix else result

        return super().__call__(*args, return_matrix=return_matrix, **kwargs)

    def _forward_albu_native(self, img_hwc: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """Execute the pipeline in Albumentations native dict-input mode.

        Iterates over all segments in NumPy space — no tensor conversion.
        :class:`~fuse_augmentations.affine.segment.AlbuFusedAffineSegment`
        segments are dispatched via their :meth:`forward_numpy` method;
        :class:`~fuse_augmentations.compose._PassthroughSegment` segments
        are dispatched via
        :meth:`~fuse_augmentations.adapters.albumentations.AlbumentationsAdapter.call_nonfused_numpy`.

        Args:
            img_hwc: ``(H, W, C)`` or ``(H, W)`` NumPy array (uint8 or float32).

        Returns:
            ``{"image": result_hwc}`` matching the :class:`albumentations.Compose`
            output convention.

        Raises:
            RuntimeError: If the pipeline contains a segment type that is not
                supported in the Albumentations native I/O path.

        """
        # Use pre-classified segment tags when available (built at __init__)
        # to avoid per-call isinstance dispatch overhead.
        _tags = self._albu_seg_tags
        if _tags is not None:
            from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

            for idx_segment, seg in enumerate(self._segments):
                tag = _tags[idx_segment]
                if tag == 0:  # AlbuFusedAffineSegment
                    img_hwc = seg.forward_numpy(img_hwc)
                    self._last_transform_matrix = seg.last_matrix
                elif tag == 1:  # ExactAffineSegment
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 2:  # FusedColorSegment
                    for tfm in seg._transforms:
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 3:  # CropResizeSegment
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 4:  # Passthrough(Albu adapter)
                    img_hwc = AlbumentationsAdapter.call_nonfused_numpy(seg.transform, img_hwc)
                elif tag == 5:  # Passthrough(non-Albu adapter)
                    msg = (
                        f"Passthrough segment adapter {type(seg.adapter).__name__!r} does not "
                        "support the Albumentations native I/O path. Use tensor input instead."
                    )
                    raise RuntimeError(msg)
                else:  # unsupported
                    msg = (
                        f"Segment type {type(seg).__name__!r} is not supported in the "
                        "Albumentations native I/O path. Use tensor input instead."
                    )
                    raise RuntimeError(msg)
        else:
            # Fallback: isinstance dispatch (used when _albu_seg_tags was not built).
            from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

            for seg in self._segments:
                if isinstance(seg, AlbuFusedAffineSegment):
                    img_hwc = seg.forward_numpy(img_hwc)
                    self._last_transform_matrix = seg.last_matrix
                    continue
                if isinstance(seg, ExactAffineSegment):
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]  # type: ignore[operator]
                    continue
                if isinstance(seg, FusedColorSegment):
                    for tfm in seg._transforms:
                        img_hwc = tfm(image=img_hwc)["image"]  # type: ignore[operator]
                    continue
                if isinstance(seg, CropResizeSegment):
                    for tfm in seg.transforms:
                        img_hwc = tfm(image=img_hwc)["image"]  # type: ignore[operator]
                    continue
                if isinstance(seg, _PassthroughSegment):
                    if isinstance(seg.adapter, AlbumentationsAdapter):
                        img_hwc = AlbumentationsAdapter.call_nonfused_numpy(seg.transform, img_hwc)
                    else:
                        msg = (
                            f"Passthrough segment adapter {type(seg.adapter).__name__!r} does not "
                            "support the Albumentations native I/O path. Use tensor input instead."
                        )
                        raise RuntimeError(msg)
                    continue
                msg = (
                    f"Segment type {type(seg).__name__!r} is not supported in the "
                    "Albumentations native I/O path. Use tensor input instead."
                )
                raise RuntimeError(msg)

        return {"image": img_hwc}

    def _resolve_kwarg_to_data_key(self, name: str) -> str:
        """Map an Albumentations-style keyword name to a declared data key.

        Args:
            name: The keyword used in the call (e.g. ``"image"``, ``"bboxes"``,
                or an exact data-key name like ``"bbox_xyxy"``).

        Returns:
            The matching entry from the pipeline's ``data_keys``.

        Raises:
            ValueError: If the keyword does not correspond to any declared data key.

        """
        data_keys = cast("list[str]", self.data_keys)
        alias = _KWARG_ALIASES.get(name, name)
        candidates: tuple[str, ...] = alias if isinstance(alias, tuple) else (alias,)
        for candidate in candidates:
            if candidate in data_keys:
                return candidate
        msg = (
            f"Keyword {name!r} does not match any entry in data_keys={data_keys}. "
            "Pass a keyword whose (aliased) name is one of the declared data keys."
        )
        raise ValueError(msg)

    def _forward_kwargs_dict(self, kwargs: dict[str, object], *, return_matrix: bool = False) -> object:
        """Run the pipeline for an Albumentations-style keyword call and return a dict.

        Each keyword is mapped to a declared data key, the targets are ordered to
        match ``data_keys``, routed through :meth:`forward`, and returned in a dict
        keyed by the caller's original keyword names. The positional ``data_keys``
        tuple API is unaffected — this dict form is added only for keyword calls.

        Args:
            kwargs: The keyword arguments from the call; must contain ``"image"``
                and one entry per auxiliary ``data_keys`` target.
            return_matrix: When ``True``, return the output dict and its last
                fused geometric matrix.

        Returns:
            A dict mapping each input keyword name to its transformed output.

        Raises:
            ValueError: If a keyword does not match a declared data key, or if the
                keyword set does not cover exactly the pipeline's ``data_keys``.

        """
        data_keys = cast("list[str]", self.data_keys)
        kwarg_to_key = {name: self._resolve_kwarg_to_data_key(name) for name in kwargs}
        # Reject aliasing collisions (e.g. both ``bboxes`` and ``bbox_xyxy``) before
        # inverting the map — otherwise one input would be silently dropped.
        key_to_kwarg: dict[str, str] = {}
        for name, key in kwarg_to_key.items():
            if key in key_to_kwarg:
                msg = (
                    f"Keywords {key_to_kwarg[key]!r} and {name!r} both resolve to data key {key!r}. "
                    "Pass exactly one keyword per declared data key."
                )
                raise ValueError(msg)
            key_to_kwarg[key] = name
        if sorted(key_to_kwarg) != sorted(data_keys):
            msg = (
                f"Keyword call resolved to data keys {sorted(key_to_kwarg)}, but the pipeline "
                f"declares data_keys={data_keys}. Pass exactly one keyword per declared data key."
            )
            raise ValueError(msg)

        ordered_args = tuple(cast("torch.Tensor", kwargs[key_to_kwarg[key]]) for key in data_keys)
        result = self._forward_multi(ordered_args, return_matrix=return_matrix)
        matrix = None
        if return_matrix:
            result, matrix = cast("tuple[object, Tensor | None]", result)
        outputs = result if isinstance(result, tuple) else (result,)
        output = {key_to_kwarg[key]: outputs[idx] for idx, key in enumerate(data_keys)}
        return (output, matrix) if return_matrix else output

    def _convert_primary_output(self, image: torch.Tensor) -> torch.Tensor | NDArray[Any]:
        """Convert the primary image output to the requested backend."""
        if self._output_converter is None:
            return image

        image_for_conversion = image.unsqueeze(0) if image.ndim == 3 else image
        return self._output_converter.convert(image_for_conversion)  # type: ignore[no-any-return]

    def forward(
        self,
        *args: torch.Tensor,
        return_matrix: bool = False,
    ) -> object:
        """Apply the augmentation pipeline to an image batch and optional auxiliary targets.

        **Single-tensor mode** (``data_keys=None``, default): accepts one image
        tensor and returns one tensor or a NumPy ``ndarray`` when
        ``output_backend="numpy"``. This is the backward-compatible path.

        **Multi-target mode** (``data_keys`` is set): accepts positional
        arguments in the same order as ``data_keys``. The first key must map
        to the image (``"input"``); subsequent keys are auxiliary targets
        (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``).
        Returns a tuple of tensors in ``data_keys`` order, or a bare tensor
        when ``data_keys`` contains exactly one entry.

        Args:
            *args: Positional tensors matching the ``data_keys`` list.
                ``args[0]`` is always the image ``(B, C, H, W)`` float32
                channel-first. Auxiliary args follow in ``data_keys[1:]``
                order: ``"mask"`` as ``(B, C, H, W)`` float/int;
                ``"bbox_xyxy"`` / ``"bbox_xywh"`` as ``(B, N, 4)`` float32;
                ``"keypoints"`` as ``(B, N, 2)`` float32.
            return_matrix: When ``True``, return the output and its last fused
                geometric pixel matrix.

        Returns:
            Single ``Tensor`` or NumPy ``ndarray`` when ``data_keys`` is
            ``None`` or has one entry. NumPy output is returned only when
            ``output_backend="numpy"``. ``tuple[Tensor, ...]`` in
            ``data_keys`` order otherwise.

            When ``return_matrix=True``, returns ``(output, matrix)`` where
            ``matrix`` is the image-dtype pixel matrix from the last fused
            geometric segment, or ``None`` when no such segment ran.

        Raises:
            TypeError: If the number of positional arguments does not match
                the number of ``data_keys`` entries (when ``data_keys`` is set),
                or if more than one argument is passed when ``data_keys`` is
                ``None``.

        Note:
            Passthrough (non-fused) transforms in the pipeline apply to the image
            only. Auxiliary targets skip passthrough segments and retain their
            values from the preceding fused segment. This is by design -
            passthrough backends do not expose a target-routing API.

            ``output_backend`` conversion is applied per target in multi-target
            mode: image and mask outputs are converted to the requested backend,
            while coordinate targets (boxes, keypoints) remain tensors because the
            channel-last image layout does not apply to them.

        """
        # data_keys is constructor state: dispatch to the pre-resolved single-tensor
        # or multi-target path instead of re-parsing it on every call.
        if not self._multi_target:
            if len(args) != 1:
                msg = f"Expected 1 argument (data_keys is None), got {len(args)}"
                raise TypeError(msg)
            return self._forward_single(args[0], return_matrix=return_matrix)
        return self._forward_multi(args, return_matrix=return_matrix)

    def _build_dispatch_tags(self) -> list[int]:
        """Assign one integer dispatch tag to every segment (see the ``_TAG_*`` constants)."""
        tags: list[int] = []
        for seg in self._segments:
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment, ProjectiveSegment, AlbuProjectiveSegment)):
                tags.append(_TAG_MATRIX)
            elif isinstance(seg, (ExactAffineSegment, FusedColorSegment, CropResizeSegment)):
                tags.append(_TAG_PLAIN)
            elif isinstance(seg, _PassthroughSegment):
                tags.append(_TAG_PASSTHROUGH)
            else:
                tags.append(_TAG_LEGACY)
        return tags

    def _dispatch_tags(self) -> list[int]:
        """Return the per-segment dispatch tags, rebuilding them for pre-tag pickles."""
        tags = getattr(self, "_seg_dispatch_tags", None)
        if tags is None:
            tags = self._build_dispatch_tags()
            self._seg_dispatch_tags = tags
        return tags

    def _forward_single(self, image: torch.Tensor, *, return_matrix: bool = False) -> object:
        """Run the pipeline in single-tensor mode (``data_keys is None``)."""
        # Reset per-call state so transform_matrix returns None when only
        # exact/passthrough segments run (no stale matrix from a previous call).
        self._last_transform_matrix = None
        _clear_current_call_matrix()
        call_matrix: Tensor | None = None

        # Whole-pipeline single-segment fast paths: bypass the segment's
        # nn.Module.__call__ and probability machinery entirely.
        _exact_fast = self._single_exact_fast
        if _exact_fast is not None:
            image = _exact_fast[0].call_nonfused(_exact_fast[1], image)
            result = self._convert_primary_output(image)
            return (result, None) if return_matrix else result
        _fast_seg = self._single_fused_fast_seg
        if _fast_seg is not None:
            image = cast(Tensor, _fast_seg.forward(image, None))
            call_matrix = _current_call_matrix()
            self._last_transform_matrix = call_matrix
            result = self._convert_primary_output(image)
            return (result, call_matrix) if return_matrix else result

        # Per-segment: call ``forward`` directly (skips nn.Module.__call__ per
        # segment — this generalises the single-op bypass to every segment in a
        # multi-segment pipeline).
        for seg, tag in zip(self._segments, self._dispatch_tags(), strict=True):
            if tag == _TAG_MATRIX:
                image = cast(Tensor, seg.forward(image, None))
                call_matrix = _current_call_matrix()
                self._last_transform_matrix = call_matrix
            elif tag == _TAG_PLAIN:
                image = cast(Tensor, seg.forward(image, None))
            elif tag == _TAG_PASSTHROUGH:
                image = seg.adapter.call_nonfused(seg.transform, image)
            else:
                image = self._legacy_passthrough(seg, image)
        result = self._convert_primary_output(image)
        return (result, call_matrix) if return_matrix else result

    def _forward_multi(self, args: tuple[torch.Tensor, ...], *, return_matrix: bool = False) -> object:
        """Run the pipeline in multi-target mode (``data_keys`` is set)."""
        self._last_transform_matrix = None
        _clear_current_call_matrix()
        call_matrix: Tensor | None = None
        data_keys = cast("list[str]", self.data_keys)
        if len(args) != len(data_keys):
            msg = f"Expected {len(data_keys)} arguments for data_keys={self.data_keys}, got {len(args)}"
            raise TypeError(msg)
        image = args[0]
        aux_targets = self._build_aux_targets(args)

        for seg, tag in zip(self._segments, self._dispatch_tags(), strict=True):
            if tag == _TAG_MATRIX:
                image, aux_targets = seg.forward(image, aux_targets)
                call_matrix = _current_call_matrix()
                self._last_transform_matrix = call_matrix
            elif tag == _TAG_PLAIN:
                image, aux_targets = seg.forward(image, aux_targets)
            elif tag == _TAG_PASSTHROUGH:
                if aux_targets:
                    self._check_passthrough_aux_policy(seg.transform)
                image = seg.adapter.call_nonfused(seg.transform, image)
            else:
                if aux_targets:
                    self._check_passthrough_aux_policy(seg)
                image = self._legacy_passthrough(seg, image)
        result = self._assemble_multi_output(image, aux_targets, args)
        return (result, call_matrix) if return_matrix else result

    @staticmethod
    def _check_passthrough_aux_policy(transform: object) -> None:
        """Enforce the passthrough / auxiliary-target correctness policy.

        Passthrough (non-fused) transforms apply to the image only; auxiliary targets
        (masks, boxes, keypoints) skip them. Whether that is a bug depends on the op:

        - **Coordinate-changing** passthrough ops (geometric distortion: elastic, grid,
          optical distortion, thin-plate-spline, piecewise-affine) move image content,
          so auxiliary targets that skip them silently desync from the image. This is a
          correctness bug and raises :class:`ValueError`.
        - **Kernel / pointwise** passthrough ops (blur, noise, gamma) leave geometry
          unchanged, so auxiliary targets legitimately pass through untouched — no
          warning is emitted.

        Args:
            transform: The passthrough transform being executed with auxiliary targets
                present.

        Raises:
            ValueError: If ``transform`` is a coordinate-changing passthrough op, since
                the auxiliary targets would silently desync from the image.

        """
        if is_coordinate_changing_passthrough(transform):
            msg = (
                f"Coordinate-changing passthrough transform {type(transform).__name__!r} moves "
                "image content but does NOT route auxiliary targets (masks, boxes, keypoints) — "
                "they would silently desync from the image. Move geometric ops before the "
                "passthrough barrier, transform the auxiliary targets manually, or remove this "
                "op from a multi-target (data_keys) pipeline."
            )
            raise ValueError(msg)

    def _build_aux_targets(self, args: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Build the auxiliary-target dict from positional args, rejecting duplicate keys."""
        aux_keys = self._aux_keys
        # Forbid duplicate auxiliary keys to avoid silent overwrites in aux_targets.
        if len(aux_keys) != len(set(aux_keys)):
            msg = (
                "Duplicate entries detected in auxiliary data_keys "
                f"(data_keys[1:]): {aux_keys}. Auxiliary keys must be unique."
            )
            raise ValueError(msg)
        return dict(zip(aux_keys, args[1:], strict=True))

    def _assemble_multi_output(
        self,
        image: torch.Tensor,
        aux_targets: dict[str, torch.Tensor],
        args: tuple[torch.Tensor, ...],
    ) -> torch.Tensor | NDArray[Any] | tuple[torch.Tensor, ...]:
        """Assemble the multi-target return value in ``data_keys`` order.

        When ``output_backend`` is set, the conversion is applied per target: the
        image and every mask are converted (both are ``(B, C, H, W)`` tensors);
        coordinate targets (boxes, keypoints) are left as tensors because the
        NumPy image converter's channel-last layout does not apply to them.

        """
        data_keys = cast("list[str]", self.data_keys)
        if len(data_keys) == 1:
            return self._convert_primary_output(image)
        result_list: list[torch.Tensor | NDArray[Any]] = []
        for idx, key in enumerate(data_keys):
            value = image if idx == 0 else aux_targets.get(key, args[idx])
            result_list.append(self._convert_target_output(key, value))
        return cast("tuple[torch.Tensor, ...]", tuple(result_list))

    def _convert_target_output(self, key: str, value: torch.Tensor) -> torch.Tensor | NDArray[Any]:
        """Apply the output-backend conversion to a single multi-target output.

        Args:
            key: The target's ``data_keys`` name (``"input"``, ``"mask"``,
                ``"bbox_xyxy"``, ``"bbox_xywh"``, or ``"keypoints"``).
            value: The transformed target tensor.

        Returns:
            The converted value for image/mask targets (a NumPy array when
            ``output_backend="numpy"``), or the unmodified tensor for coordinate
            targets and when no converter is configured.

        """
        if self._output_converter is None or key not in ("input", "mask"):
            return value
        return self._convert_primary_output(value)

    def _legacy_passthrough(self, seg: object, image: torch.Tensor) -> torch.Tensor:
        """Dispatch a legacy (pre-wrapper) passthrough segment by resolving its adapter by index."""
        pt_adapter = None
        for idx, orig_tfm in enumerate(self.original_transforms):
            if orig_tfm is seg:
                pt_adapter = self._transform_adapters.get(idx, self._adapter)
                break
        if pt_adapter is None:
            msg = f"Unknown segment type {type(seg).__name__!r} — update FusedCompose.forward dispatch"
            raise RuntimeError(msg)
        return pt_adapter.call_nonfused(seg, image)

    @property
    def transform_matrix(self) -> torch.Tensor | None:
        """Return the ``(batch_size, 3, 3)`` composed matrix for the last fused segment.

        This is the composed forward transform matrix produced by the last
        fused geometric segment executed in the most recent :meth:`forward`
        call. This includes affine segments and projective segments, so the
        returned matrix may encode either an affine or a full homography-style
        projective warp depending on the last fused segment type. Passthrough
        (non-fused) transforms do not affect this value, and multiple fused
        segments are *not* composed into a single whole-pipeline matrix. In
        mixed-backend pipelines, only the last fused segment across all
        backends contributes to this value.

        Returns:
            The composed matrix for the last fused affine or projective segment, or ``None`` if no such segment has
            been executed yet (including before the first call to :meth:`forward` or if the last forward contained
            only passthrough transforms).

        Note:
            This is per-instance mutable state written on every ``forward``. Reading it from another
            thread while a shared instance is running ``forward`` is racy; use one pipeline instance
            per thread (or read the matrix in the same thread that ran the forward pass).

        """
        return self._last_transform_matrix

    @property
    def n_warps_saved(self) -> int:
        """Return the number of interpolation passes eliminated vs sequential execution.

        For affine fused segments with *n* transforms, *n - 1* warp passes
        are saved. For exact (flip-only) segments with *n* transforms, *n*
        passes are saved because no interpolation is performed at all.
        For color fused segments with *n* transforms, *n - 1* matrix-multiply
        passes are saved (all ops collapse to one ``torch.bmm`` call).
        Single-transform fused segments contribute zero savings.

        Returns:
            Total number of eliminated warp passes across all fused segments.

        """
        total = 0
        for seg in self._segments:
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment, ProjectiveSegment, AlbuProjectiveSegment)):
                # n transforms fused → 1 warp, saving n-1 passes.
                num_transforms = len(seg.transforms)
                if num_transforms > 1:
                    total += num_transforms - 1
                continue

            if isinstance(seg, ExactAffineSegment):
                # Each flip in an ExactAffineSegment avoids grid_sample entirely
                # (uses tensor.flip), so every transform saves exactly 1 warp.
                # This is why ExactAffineSegment contributes n rather than n-1:
                # even a single flip is lossless and free of grid_sample cost.
                total += len(seg.transforms)
                continue

            if isinstance(seg, FusedColorSegment):
                # n color ops fused → 1 matrix multiply, saving n-1 passes.
                num_transforms = len(seg.transforms)
                if num_transforms > 1:
                    total += num_transforms - 1
        return total

    def _plan_device(self) -> torch.device:
        """Return the pipeline's configured device (follows ``.to(device)``)."""
        tracker = getattr(self, "_device_tracker", None)
        if tracker is not None:
            return cast(Tensor, tracker).device
        return torch.device("cpu")

    @property
    def fusion_plan(self) -> str:
        """Return a human-readable summary of what got fused and what didn't.

        When the pipeline is configured on a non-CPU device (via ``.to(device)``),
        each passthrough segment is annotated with a trailing ``" [CPU passthrough]"``
        marker, because a non-fusible op forces a device-to-host round-trip on a
        GPU/MPS pipeline (the "GPU poison pill"). On a CPU pipeline the marker is
        omitted and the string is unchanged.

        Returns:
            Arrow-separated description of segments, e.g.
            ``"fused(RandomRotation, RandomHorizontalFlip) -> passthrough(GaussianBlur)"``.
            On a non-CPU pipeline the passthrough entry reads
            ``"passthrough(GaussianBlur) [CPU passthrough]"``. Returns ``"empty"`` for
            an empty pipeline.

        """
        non_cpu = self._plan_device().type != "cpu"
        cached = self._fusion_plan_cache
        if cached is not None and cached[0] == non_cpu:
            return cached[1]
        marker = " [CPU passthrough]" if non_cpu else ""
        parts: list[str] = []
        for seg in self._segments:
            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"projective({', '.join(names)})")
                continue

            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"fused({', '.join(names)})")
                continue

            if isinstance(seg, ExactAffineSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"exact({', '.join(names)})")
                continue

            if isinstance(seg, FusedColorSegment):
                names = [type(transform).__name__ for transform in seg.transforms]
                parts.append(f"color({', '.join(names)})")
                continue

            if isinstance(seg, CropResizeSegment):
                parts.append(f"crop_resize({type(seg.transform).__name__})")
                continue

            if isinstance(seg, _PassthroughSegment):
                parts.append(f"passthrough({type(seg.transform).__name__}){marker}")
                continue

            parts.append(f"passthrough({type(seg).__name__}){marker}")
        plan = " → ".join(parts) if parts else "empty"
        self._fusion_plan_cache = (non_cpu, plan)
        return plan

    @property
    def fusion_plan_descriptors(self) -> list[SegmentDescriptor]:
        """Return a structured, machine-readable description of the fusion plan.

        Each element corresponds to one segment in the pipeline, in execution
        order. This is the structured counterpart to the human-readable
        :attr:`fusion_plan` string. Available immediately after construction —
        does not require a :meth:`forward` call.

        Returns:
            List of :class:`~fuse_augmentations.types.SegmentDescriptor`
            instances, one per segment. Empty list for an empty pipeline.
            Each descriptor's ``backend`` field is the adapter class name
            (e.g. ``"KorniaAdapter"``) for fused, exact, and projective
            segments, and ``None`` for passthrough segments and backend-free
            pipelines created via :meth:`from_params`.

        Example:
            >>> import torch
            >>> from fuse_augmentations.compose import FusedCompose
            >>> pipe = FusedCompose([])
            >>> pipe.fusion_plan_descriptors
            []

        Note:
            The ``backend`` field on passthrough segments is always ``None``,
            regardless of the pipeline's backend. Only fused, exact, and
            projective segments carry the adapter class name.

        """
        cached = self._fusion_plan_descriptors_cache
        if cached is not None:
            return cached

        def _resolve_backend(seg: object) -> str | None:
            # from_params() uses _DirectParamAdapter; expose backend-free descriptors.
            if isinstance(self._adapter, _DirectParamAdapter):
                return None

            # Mixed-backend mode: resolve by the first transform in this segment.
            if self._transform_adapters and hasattr(seg, "transforms"):
                seg_transforms = getattr(seg, "transforms", None)
                if isinstance(seg_transforms, list) and seg_transforms:
                    first = seg_transforms[0]
                    for idx, orig in enumerate(self.original_transforms):
                        if orig is first:
                            adapter = self._transform_adapters.get(idx)
                            if adapter is not None:
                                return type(adapter).__name__
                            break

            return type(self._adapter).__name__ if self._adapter else None

        descriptors: list[SegmentDescriptor] = []
        for seg in self._segments:
            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="projective",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="fused",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, ExactAffineSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names)  # Each flip saves 1 warp vs grid_sample
                descriptors.append(
                    SegmentDescriptor(
                        kind="exact",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, FusedColorSegment):
                names = tuple(type(transform).__name__ for transform in seg.transforms)
                num_saved = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="color",
                        transforms=names,
                        n_warps_saved=num_saved,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, CropResizeSegment):
                descriptors.append(
                    SegmentDescriptor(
                        kind="crop_resize",
                        transforms=(type(seg.transform).__name__,),
                        n_warps_saved=0,
                        backend=_resolve_backend(seg),
                        # A crop+resize acts as a segment boundary (it changes output
                        # dimensions), so it is a barrier for adjacent geometric runs.
                        barrier="crop_resize",
                        split_reason=self._passthrough_split_reason(seg),
                    )
                )
                continue
            if isinstance(seg, _PassthroughSegment):
                descriptors.append(
                    SegmentDescriptor(
                        kind="passthrough",
                        transforms=(type(seg.transform).__name__,),
                        n_warps_saved=0,
                        barrier=self._passthrough_barrier_reason(seg.transform),
                        split_reason=self._passthrough_split_reason(seg),
                        refused="not_fusible",
                    )
                )
                continue
            # Legacy passthrough
            descriptors.append(
                SegmentDescriptor(
                    kind="passthrough",
                    transforms=(type(seg).__name__,),
                    n_warps_saved=0,
                    barrier=self._passthrough_barrier_reason(seg),
                    split_reason=self._passthrough_split_reason(seg),
                    refused="not_fusible",
                )
            )
        self._fusion_plan_descriptors_cache = descriptors
        return descriptors

    def _passthrough_barrier_reason(self, transform: object) -> str:
        """Classify why a passthrough transform forms a fusion barrier.

        Args:
            transform: The passthrough transform (the wrapped transform for a
                :class:`_PassthroughSegment`, or the raw legacy segment object).

        Returns:
            ``"coordinate_change"`` for a geometric distortion op (elastic/grid/optical
            distortion, thin-plate-spline, piecewise-affine) whose auxiliary targets
            would desync, else ``"spatial_kernel"`` for a kernel/pointwise op.

        """
        if is_coordinate_changing_passthrough(transform):
            return "coordinate_change"
        return "spatial_kernel"

    def _passthrough_split_reason(self, seg: object) -> str | None:
        """Return why a run was split at ``seg``, or ``None`` when not applicable.

        A passthrough that sits between two same-backend fused runs is a natural
        barrier (not a *split* of an otherwise-fusible run). A passthrough at a
        backend boundary in a mixed-backend pipeline additionally marks where the
        backend changed. Only the mixed-backend case is reported here, as the single-
        backend barrier reason is already carried by ``barrier``.

        Args:
            seg: The segment being described.

        Returns:
            ``"backend_boundary"`` when the pipeline is mixed-backend and this segment's
            transform sits at a backend change, else ``None``.

        """
        if not self._transform_adapters:
            return None
        transform = seg.transform if isinstance(seg, _PassthroughSegment) else seg
        seg_idx = next(
            (idx for idx, orig in enumerate(self.original_transforms) if orig is transform),
            None,
        )
        if seg_idx is None:
            return None
        this_adapter = self._transform_adapters.get(seg_idx)
        neighbour_adapters = {
            self._transform_adapters.get(idx)
            for idx in (seg_idx - 1, seg_idx + 1)
            if 0 <= idx < len(self.original_transforms)
        }
        neighbour_adapters.discard(None)
        neighbour_adapters.discard(this_adapter)
        return "backend_boundary" if neighbour_adapters else None

    @classmethod
    def from_config(
        cls,
        specs: list[TransformSpec],
        backend: BackendStr,
        interpolation: InterpolationStr = "bilinear",
        padding_mode: PaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        on_unsupported: Literal["raise", "warn_skip"] = "raise",
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> FusedCompose:
        """Create a FusedCompose pipeline from a list of TransformSpec objects.

        Resolves each spec's operation to the corresponding backend transform
        class, instantiates it with the spec's params and per-sample probability,
        then builds the pipeline via ``cls(transforms, ...)``.

        All specs are validated against the backend's capability matrix **before** any transform is constructed, so an
        unsupported op is reported together with every other offender in a single aggregated error rather than failing
        on the first one.

        Args:
            specs: List of :class:`TransformSpec` objects describing the
                pipeline.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``, or ``"native"``). The native backend
                is fully batched and has no optional dependencies.
            interpolation: Interpolation mode for ``grid_sample`` warp.
            padding_mode: Padding mode for out-of-bounds samples.
            reorder: Reorder policy applied before segmentation.
            data_keys: Key list for auxiliary target routing.
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.
            randomness: Batch randomness policy forwarded to :meth:`__init__`.
            clip_policy: Clamp policy for fused color segments.
            mask_interpolation: Auxiliary mask sampling mode forwarded to
                :meth:`__init__`.
            on_unsupported: Policy for specs whose op the backend cannot build.
                ``"raise"`` (default) aggregates all offenders into one
                ``ValueError``; ``"warn_skip"`` drops each unsupported spec with
                a ``UserWarning`` and builds a pipeline from the rest.

        Returns:
            A configured ``FusedCompose`` instance.

        Raises:
            ValueError: If ``backend`` is not in :data:`~fuse_augmentations.resolver.SUPPORTED_BACKENDS`; or, when
                ``on_unsupported="raise"``, if any spec's operation is unknown or unsupported by the chosen backend
                (all offenders reported together). This validation applies even when ``specs`` is empty.

        Example:
            >>> import torch
            >>> from fuse_augmentations.compose import FusedCompose
            >>> from fuse_augmentations.types import TransformSpec
            >>> spec = TransformSpec(operation="hflip", params={}, prob=0.5)
            >>> pipe = FusedCompose.from_config([spec], backend="kornia")
            >>> pipe(torch.zeros(1, 3, 8, 8)).shape
            torch.Size([1, 3, 8, 8])

        """
        from fuse_augmentations.resolver import SUPPORTED_BACKENDS

        if backend not in SUPPORTED_BACKENDS:
            msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
            raise ValueError(msg)

        kept_specs = cls._validate_specs(specs, backend, on_unsupported)
        if backend == "native":
            return cls._from_param_specs(
                specs=kept_specs,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=_coerce_randomness_policy(randomness),
                route_coords_via_grid=_has_coord_aux(data_keys),
                native=True,
            )
        transforms = [cls._build_transform(spec, backend) for spec in kept_specs]

        return cls(
            transforms,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            reorder=reorder,
            output_backend=output_backend,
            randomness=randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
        )

    @staticmethod
    def _validate_specs(
        specs: list[TransformSpec],
        backend: BackendStr,
        on_unsupported: Literal["raise", "warn_skip"],
    ) -> list[TransformSpec]:
        """Validate every spec via ``resolve_op`` before any construction, aggregating all failures.

        Rejects ``prob`` inside ``params`` (always an error), then probes each op with ``resolve_op`` (the same gate
        construction uses). Under ``"raise"`` every offending spec's message is collected and reported in one
        aggregated ``ValueError``; under ``"warn_skip"`` the offenders are dropped with a ``UserWarning`` and the
        surviving specs are returned.

        Args:
            specs: The specs to validate.
            backend: Target backend name.
            on_unsupported: ``"raise"`` or ``"warn_skip"``.

        Returns:
            The specs to build (all of *specs* under ``"raise"``; only supported ones under ``"warn_skip"``).

        Raises:
            ValueError: On a ``prob``-in-``params`` spec, or (under ``"raise"``) when any op is unsupported.

        """
        from fuse_augmentations.resolver import SUPPORTED_OPS, resolve_op

        offenders: list[str] = []
        kept: list[TransformSpec] = []
        for spec in specs:
            if "prob" in spec.params:
                msg = (
                    f"TransformSpec.params must not include 'prob'; "
                    f"use TransformSpec.prob for operation {spec.operation!r} instead."
                )
                raise ValueError(msg)
            if spec.operation not in SUPPORTED_OPS:
                offenders.append(f"unknown operation {spec.operation!r}; supported: {sorted(SUPPORTED_OPS)}")
                continue
            try:
                resolve_op(cast("OpStr", spec.operation), backend)
            except ValueError as exc:
                offenders.append(str(exc))
            else:
                kept.append(spec)

        if not offenders:
            return kept
        if on_unsupported == "warn_skip":
            warnings.warn(
                "from_config skipping unsupported specs:\n  - " + "\n  - ".join(offenders),
                UserWarning,
                stacklevel=3,
            )
            return kept
        raise ValueError("from_config could not resolve all specs:\n  - " + "\n  - ".join(offenders))

    @staticmethod
    def _build_transform(spec: TransformSpec, backend: BackendStr) -> object:
        """Instantiate one backend transform from a validated spec.

        Applies the per-transform probability via the backend ``p=`` kwarg when accepted, falling back to a ``prob``
        attribute (or a warning) for backends that reject it.

        Args:
            spec: A spec already validated as supported by *backend*.
            backend: Target backend name.

        Returns:
            The constructed backend transform object.

        """
        from fuse_augmentations.resolver import resolve_op, translate_params

        op_name = cast("OpStr", spec.operation)
        tfm_cls = cast(type, resolve_op(op_name, backend))
        kwargs = translate_params(op_name, backend, dict(spec.params))
        # Most backends accept p= for per-transform probability.
        # Some (e.g. TorchVision rotation) don't; pass it and let
        # the backend ignore it via **kwargs or TypeError fallback.
        try:
            return tfm_cls(**kwargs, p=spec.prob)
        except TypeError as exc:
            # Re-raise unless this is specifically a rejected p keyword.
            # Constructor errors about other kwargs/types must not be masked.
            exc_msg = str(exc).lower()
            is_p_keyword_rejection = "'p'" in exc_msg and (
                "unexpected keyword" in exc_msg or "keyword argument" in exc_msg or "does not accept" in exc_msg
            )
            if not is_p_keyword_rejection:
                raise
        # Backend class does not accept p= (e.g. TorchVision RandomRotation)
        tfm = tfm_cls(**kwargs)
        # Attach prob as attribute so the fused engine can read it
        p_set = False
        with contextlib.suppress(AttributeError, TypeError):
            tfm.prob = spec.prob
            p_set = True
        if not p_set and spec.prob != 1.0:
            warnings.warn(
                f"TransformSpec.prob={spec.prob!r} for operation {spec.operation!r} could not be applied to "
                f"{tfm_cls.__name__!r} (backend does not accept p= and attribute is read-only). "
                "The transform will always be applied (prob=1.0 effective).",
                UserWarning,
                stacklevel=3,
            )
        return tfm

    @classmethod
    def supported_ops(cls, backend: BackendStr) -> frozenset[str]:
        """Return the canonical op names *backend* can build.

        Args:
            backend: Backend name (must be in :data:`~fuse_augmentations.resolver.SUPPORTED_BACKENDS`).

        Returns:
            Frozenset of canonical op names the backend supports (empty if the backend's optional dependency is not
            installed).

        Raises:
            KeyError: If *backend* is not in :data:`~fuse_augmentations.resolver.SUPPORTED_BACKENDS`.

        Example:
            >>> from fuse_augmentations.compose import FusedCompose
            >>> "hflip" in FusedCompose.supported_ops("kornia")  # doctest: +SKIP
            True

        """
        from fuse_augmentations.resolver import capability_matrix

        return capability_matrix()[backend]

    @classmethod
    def capability_matrix(cls) -> dict[str, frozenset[str]]:
        """Return the backend -> supported-op-names map across all supported backends.

        Returns:
            Mapping from each supported backend to the frozenset of canonical op names it can build.

        Example:
            >>> from fuse_augmentations.compose import FusedCompose
            >>> sorted(FusedCompose.capability_matrix())
            ['albumentations', 'kornia', 'native', 'torchvision']

        """
        from fuse_augmentations.resolver import capability_matrix

        return capability_matrix()

    @classmethod
    def from_params(
        cls,
        rotation: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        scale_x: tuple[float, float] | None = None,
        scale_y: tuple[float, float] | None = None,
        shear_x: tuple[float, float] | None = None,
        shear_y: tuple[float, float] | None = None,
        translate_x: tuple[float, float] | None = None,
        translate_y: tuple[float, float] | None = None,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        brightness: float | None = None,
        contrast: float | None = None,
        interpolation: InterpolationStr = "bilinear",
        padding_mode: PaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy | Literal["backend", "per_sample"] = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        *,
        specs: list[TransformSpec] | None = None,
        backend: BackendStr | None = None,
        route_coords_via_grid: bool = False,
    ) -> FusedCompose:
        """Create a ``FusedCompose`` pipeline directly from parameter ranges.

        This factory bypasses backend transform objects entirely and samples
        parameters directly using ``_matrix.py`` primitives. Useful for
        backend-agnostic pipelines or when no backend is installed.

        When ``backend`` is provided, the factory delegates to
        :meth:`from_config` semantics instead, using real backend transform
        objects. This allows ``from_params`` to serve as a single entry
        point that works both with and without a backend.

        All geometric parameters are sampled independently per batch item on
        every :meth:`forward` call (i.e. ``same_on_batch=False`` semantics).
        If all geometric params are ``None`` and both flip probabilities are
        0.0, the returned pipeline is an identity passthrough.

        Args:
            rotation: ``(min_deg, max_deg)`` rotation range, or ``None`` to
                disable rotation.
            scale: ``(min_factor, max_factor)`` uniform scale range applied to
                both axes equally, or ``None``. Overridden per-axis by
                ``scale_x``/``scale_y`` when those are also set.
            scale_x: ``(min_factor, max_factor)`` x-axis-only scale range, or
                ``None``.
            scale_y: ``(min_factor, max_factor)`` y-axis-only scale range, or
                ``None``.
            shear_x: ``(min_deg, max_deg)`` x-shear range, or ``None``.
            shear_y: ``(min_deg, max_deg)`` y-shear range, or ``None``.
            translate_x: ``(min_px, max_px)`` x-translation range in pixels,
                or ``None``.
            translate_y: ``(min_px, max_px)`` y-translation range in pixels,
                or ``None``.
            hflip_p: Probability of horizontal flip per sample. Default 0.0.
            vflip_p: Probability of vertical flip per sample. Default 0.0.
            brightness: Maximum multiplicative brightness deviation. A value
                of ``0.1`` samples factors in ``[0.9, 1.1]``.
            contrast: Maximum multiplicative contrast deviation. A value of
                ``0.1`` samples factors in ``[0.9, 1.1]`` around midpoint 0.5.
            interpolation: Interpolation mode for the ``grid_sample`` warp.
                One of ``"bilinear"`` (default), ``"nearest"``, ``"bicubic"``.
            padding_mode: Padding for out-of-bounds samples.
                One of ``"zeros"`` (default), ``"border"``, ``"reflection"``.
            reorder: Reorder policy applied before segmentation.
                Defaults to ``ReorderPolicy.POINTWISE``.
            data_keys: Key list for auxiliary target routing, forwarded to
                :meth:`__init__`. ``None`` preserves single-tensor I/O.
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.
            randomness: Batch randomness policy forwarded to :meth:`__init__`
                or :meth:`from_config`. NOTE: in backend-free mode
                (``backend=None``) this value is stored but has no effect on
                parameter sampling — the direct-param path always draws
                independent per-sample parameters. It only changes behaviour
                when ``backend=`` is set (delegation to :meth:`from_config`).
            clip_policy: Clamp policy forwarded to fused color segments.
            mask_interpolation: Auxiliary mask sampling mode forwarded to
                :meth:`__init__` or :meth:`from_config`.
            route_coords_via_grid: Force coordinate auxiliary targets through
                the grid path for direct-param construction.
            specs: List of :class:`TransformSpec` objects. When provided,
                all other geometric keyword arguments must be at their
                defaults (mutually exclusive). Keyword-only.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``), or ``None`` for backend-free mode.
                When set, delegates to :meth:`from_config` semantics.
                Keyword-only.

        Returns:
            A configured ``FusedCompose`` instance ready for inference or training.

        Raises:
            ValueError: If ``specs`` is provided together with any geometric keyword
                argument (they are mutually exclusive).
            ValueError: If ``specs`` contains an op that is not supported in
                backend-free mode (i.e. not one of ``"rotation"``, ``"scale"``,
                ``"scale_x"``, ``"scale_y"``, ``"shear_x"``, ``"shear_y"``,
                ``"translate_x"``, ``"translate_y"``, ``"hflip"``, ``"vflip"``).
            ValueError: If a backend-native kwarg in ``TransformSpec.params``
                is not accepted by the backend constructor.

        Example:
            >>> import torch
            >>> from fuse_augmentations.compose import FusedCompose
            >>> pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
            >>> x = torch.zeros(2, 3, 64, 64)
            >>> out = pipe(x)
            >>> out.shape
            torch.Size([2, 3, 64, 64])

        """
        # --- specs= overload path ---
        # Note: specs= is a convenience alias for declarative pipeline construction.
        # In a future minor version this dual-path may be split into a from_specs()
        # classmethod; for now mutual exclusivity of specs and geometric kwargs enforces intent.
        if specs is not None:
            # Validate mutual exclusivity
            # Geometric tuple params default to None; use explicit is-not-None
            # checks so invalid falsey values (e.g. empty tuple/list) are still
            # treated as "provided" and rejected in specs mode.
            has_keyword_params = any((
                rotation is not None,
                scale is not None,
                scale_x is not None,
                scale_y is not None,
                shear_x is not None,
                shear_y is not None,
                translate_x is not None,
                translate_y is not None,
                hflip_p != 0.0,
                vflip_p != 0.0,
                brightness is not None,
                contrast is not None,
            ))
            if has_keyword_params:
                msg = "specs and keyword params are mutually exclusive"
                raise ValueError(msg)

            if backend is not None:
                return cls.from_config(
                    specs=specs,
                    backend=backend,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    reorder=reorder,
                    data_keys=data_keys,
                    output_backend=output_backend,
                    randomness=randomness,
                    clip_policy=clip_policy,
                    mask_interpolation=mask_interpolation,
                )

            return cls._from_param_specs(
                specs=specs,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=_coerce_randomness_policy(randomness),
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
                route_coords_via_grid=route_coords_via_grid,
            )

        # When backend is set and geometric kwargs are provided (no specs),
        # convert the kwargs to TransformSpec objects and delegate to from_config.
        if backend is not None:
            if backend != "native" and (brightness is not None or contrast is not None):
                raise NotImplementedError("brightness and contrast from_params require backend='native'")
            config_specs = cls._geometric_kwargs_to_specs(
                rotation=rotation,
                scale=scale,
                scale_x=scale_x,
                scale_y=scale_y,
                shear_x=shear_x,
                shear_y=shear_y,
                translate_x=translate_x,
                translate_y=translate_y,
                hflip_p=hflip_p,
                vflip_p=vflip_p,
            )
            if backend == "native":
                config_specs.extend(
                    TransformSpec(operation=name, params={"factor": value})
                    for name, value in cls._native_color_specs(brightness, contrast)
                )
            return cls.from_config(
                specs=config_specs,
                backend=backend,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
            )

        # Collect geometric param specs
        param_specs: dict[str, tuple[float, float]] = {}
        if rotation is not None:
            param_specs["rotation"] = rotation
        if scale is not None:
            param_specs["scale"] = scale
        if scale_x is not None:
            param_specs["scale_x"] = scale_x
        if scale_y is not None:
            param_specs["scale_y"] = scale_y
        if shear_x is not None:
            param_specs["shear_x"] = shear_x
        if shear_y is not None:
            param_specs["shear_y"] = shear_y
        if translate_x is not None:
            param_specs["translate_x"] = translate_x
        if translate_y is not None:
            param_specs["translate_y"] = translate_y

        color_specs = cls._native_color_specs(brightness, contrast)
        has_affine = bool(param_specs)
        has_flips = hflip_p > 0.0 or vflip_p > 0.0
        has_color = bool(color_specs)

        # NOTE: The identity path (all params None) returns a normal __init__ instance
        # with _adapter=None. The non-identity path uses _DirectParamAdapter.
        # Both handle empty _segments correctly; do not branch on isinstance(_adapter, ...).
        if not has_affine and not has_flips and not has_color:
            return cls(
                transforms=[],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                output_backend=output_backend,
                reorder=reorder,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
            )

        # Build internal transforms and adapter
        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        if has_affine:
            transforms.append(_DirectParamTransform(param_specs, prob=1.0))

        if hflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="hflip", prob=hflip_p))

        if vflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="vflip", prob=vflip_p))

        transforms.extend(_DirectParamTransform(param_specs={key: value}, prob=1.0) for key, value in color_specs)

        # Build instance bypassing detect_backend
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        randomness_policy = _coerce_randomness_policy(randomness)
        segments = build_segments(
            transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness_policy,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            route_coords_via_grid=route_coords_via_grid or _has_coord_aux(data_keys),
        )
        instance._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            output_backend=output_backend,
            randomness=randomness_policy,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
        )

        return instance

    @classmethod
    def _from_param_specs(
        cls,
        specs: list[TransformSpec],
        interpolation: InterpolationStr,
        padding_mode: PaddingModeStr,
        reorder: ReorderPolicy,
        data_keys: list[str] | None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
        mask_interpolation: MaskInterpolationStr = "nearest",
        route_coords_via_grid: bool = False,
        native: bool = False,
    ) -> FusedCompose:
        """Build a from_params pipeline from a list of TransformSpec objects.

        Each spec is converted to the appropriate internal direct-param transform (``_DirectParamTransform`` or
        ``_DirectFlipTransform``).

        """
        op_to_param_key: dict[str, str] = {
            "rotation": "rotation",
            "scale": "scale",
            "scale_x": "scale_x",
            "scale_y": "scale_y",
            "shear_x": "shear_x",
            "shear_y": "shear_y",
            "translate_x": "translate_x",
            "translate_y": "translate_y",
            "brightness": "brightness",
            "contrast": "contrast",
        }

        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        for spec in specs:
            if spec.operation in ("hflip", "vflip"):
                transforms.append(
                    _DirectFlipTransform(flip_type=spec.operation, prob=spec.prob),  # type: ignore[arg-type]
                )
            elif spec.operation in op_to_param_key:
                param_key = op_to_param_key[spec.operation]
                # Resolve and validate one numeric range for this op.
                param_value = cls._extract_param_range_from_spec(spec)
                param_specs = {param_key: param_value}
                transforms.append(
                    _DirectParamTransform(param_specs=param_specs, prob=spec.prob),
                )
            elif native and spec.operation == "shear":
                transforms.append(
                    _DirectParamTransform(
                        param_specs={"shear_x": cls._extract_param_range_from_spec(spec)},
                        prob=spec.prob,
                    )
                )
            elif native and spec.operation == "translate":
                param_value = cls._extract_param_range_from_spec(spec)
                transforms.append(
                    _DirectParamTransform(
                        param_specs={"translate_x": param_value, "translate_y": param_value},
                        prob=spec.prob,
                    )
                )
            else:
                msg = f"Unsupported op for from_params: {spec.operation!r}"
                raise ValueError(msg)

        if not transforms:
            return cls(
                transforms=[],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                output_backend=output_backend,
                reorder=reorder,
                randomness=randomness,
                clip_policy=clip_policy,
                mask_interpolation=mask_interpolation,
            )

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        segments = build_segments(
            transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
            route_coords_via_grid=route_coords_via_grid or _has_coord_aux(data_keys),
        )
        instance._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
            output_backend=output_backend,
            randomness=randomness,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
        )

        return instance

    @staticmethod
    def _native_color_specs(
        brightness: float | None,
        contrast: float | None,
    ) -> list[tuple[str, tuple[float, float]]]:
        """Convert brightness/contrast deviations into native factor ranges."""
        specs: list[tuple[str, tuple[float, float]]] = []
        for name, deviation in (("brightness", brightness), ("contrast", contrast)):
            if deviation is None:
                continue
            if deviation < 0.0:
                raise ValueError(f"{name} must be non-negative, got {deviation!r}")
            specs.append((name, (1.0 - deviation, 1.0 + deviation)))
        return specs

    @staticmethod
    def _extract_param_range_from_spec(spec: TransformSpec) -> tuple[float, float]:
        """Extract one numeric range tuple from a TransformSpec params dict."""
        op_to_allowed_keys: dict[str, tuple[str, ...]] = {
            "rotation": ("degrees", "rotation"),
            "scale": ("factor", "scale"),
            "scale_x": ("factor", "scale_x"),
            "scale_y": ("factor", "scale_y"),
            "shear_x": ("degrees", "shear_x"),
            "shear_y": ("degrees", "shear_y"),
            "shear": ("degrees", "shear"),
            "translate_x": ("pixels", "translate_x"),
            "translate_y": ("pixels", "translate_y"),
            "translate": ("pixels", "translate"),
            "brightness": ("factor", "brightness"),
            "contrast": ("factor", "contrast"),
        }

        allowed_keys = op_to_allowed_keys.get(spec.operation, ())
        for key in allowed_keys:
            if key in spec.params:
                value = spec.params[key]
                if (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and isinstance(value[0], (int, float))
                    and isinstance(value[1], (int, float))
                ):
                    return float(value[0]), float(value[1])
                msg = f"Invalid range for {spec.operation!r}: expected tuple[float, float] in params[{key!r}]"
                raise ValueError(msg)

        msg = f"Missing required range for {spec.operation!r}; expected one of keys: {allowed_keys}"
        raise ValueError(msg)

    @staticmethod
    def _geometric_kwargs_to_specs(
        rotation: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        scale_x: tuple[float, float] | None = None,
        scale_y: tuple[float, float] | None = None,
        shear_x: tuple[float, float] | None = None,
        shear_y: tuple[float, float] | None = None,
        translate_x: tuple[float, float] | None = None,
        translate_y: tuple[float, float] | None = None,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
    ) -> list[TransformSpec]:
        """Convert geometric keyword arguments to a list of TransformSpec objects.

        Used internally by :meth:`from_params` when ``backend`` is set and geometric kwargs (rather than ``specs``) are
        provided.

        """
        if scale_x is not None or scale_y is not None:
            msg = (
                "scale_x and scale_y are not supported when backend= is set. "
                "Use from_config() with an explicit RandomAffine spec for anisotropic scale."
            )
            raise ValueError(msg)

        if any(param is not None for param in (shear_x, shear_y, translate_x, translate_y)):
            msg = (
                "shear_x/shear_y and translate_x/translate_y are not supported when backend= is set. "
                "Use from_config() with an explicit affine spec for per-axis shear/translation."
            )
            raise ValueError(msg)

        specs: list[TransformSpec] = []

        # Map geometric tuple params to their canonical op and param key
        _kwarg_to_op: dict[str, tuple[str, str]] = {
            "rotation": ("rotation", "degrees"),
            "scale": ("scale", "factor"),
        }

        # Geometric tuple params
        for kwarg_name, value in [
            ("rotation", rotation),
            ("scale", scale),
        ]:
            if value is not None:
                op_name, param_key = _kwarg_to_op[kwarg_name]
                specs.append(TransformSpec(operation=op_name, params={param_key: value}, prob=1.0))

        # Flip params
        if hflip_p > 0.0:
            specs.append(TransformSpec(operation="hflip", params={}, prob=hflip_p))
        if vflip_p > 0.0:
            specs.append(TransformSpec(operation="vflip", params={}, prob=vflip_p))

        return specs


# ---------------------------------------------------------------------------
# Mixed-backend helpers
# ---------------------------------------------------------------------------


def _adapter_for_backend(backend: Backend) -> TransformAdapter:
    """Return the adapter instance for a known backend.

    Args:
        backend: A ``Backend`` enum value.

    Returns:
        The corresponding ``TransformAdapter`` instance.

    Raises:
        NotImplementedError: If the backend is not supported.

    """
    if backend == Backend.KORNIA:
        from fuse_augmentations.adapters.kornia import KorniaAdapter

        return KorniaAdapter()
    if backend == Backend.ALBUMENTATIONS:
        from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

        return AlbumentationsAdapter()
    if backend == Backend.TORCHVISION:
        from fuse_augmentations.adapters.torchvision import TorchVisionAdapter

        return TorchVisionAdapter()
    msg = f"Backend '{backend.value}' not yet supported"
    raise NotImplementedError(msg)


def _build_mixed_segments(
    transforms: list[object],
    per_backends: list[Backend | None],
    reorder: ReorderPolicy,
    interpolation: InterpolationStr | None,
    padding_mode: PaddingModeStr | None,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    *,
    route_coords_via_grid: bool = False,
    execution: ExecutionStr = "cv2",
    compile_warp: bool = False,
    antialias: bool = False,
    clip_policy: ClipPolicyStr = "final",
    mask_interpolation: MaskInterpolationStr = "nearest",
) -> tuple[TransformAdapter, list[object], dict[int, TransformAdapter]]:
    """Build segments for a mixed-backend pipeline.

    Groups consecutive transforms that share the same backend, then calls
    ``build_segments`` on each group with the appropriate adapter. Backend
    boundaries act as segment breaks — the same as ``SPATIAL_KERNEL`` barriers.

    Transforms with ``backend=None`` (unrecognised) are emitted as
    passthrough objects directly.

    Args:
        transforms: Full list of transform objects.
        per_backends: Per-transform backend from ``detect_backends_per_transform``.
        reorder: Reorder policy (applied within each same-backend group).
        interpolation: Interpolation mode forwarded to segments.
        padding_mode: Padding mode forwarded to segments.
        randomness: Batch randomness policy forwarded to segments.
        route_coords_via_grid: Forwarded to ``build_segments`` — routes all-exact runs
            through the grid path when box/keypoint aux targets are present.
        execution: Forwarded to ``build_segments`` — Albumentations warp strategy
            (``"cv2"`` default, ``"torch"`` opt-in batched grid_sample).
        compile_warp: Forwarded to ``build_segments`` — opt-in ``torch.compile`` of
            the torch warp core (default off, no-op on CPU / older torch).
        antialias: Forwarded to ``build_segments`` — opt-in downscale antialiasing on
            crop-resize segments (default off, no-op above the scale threshold).
        clip_policy: Forwarded to ``build_segments`` — color-segment clamp policy
            (``"final"`` default, ``"per_op_parity"`` opt-in).
        mask_interpolation: Forwarded to ``build_segments`` for auxiliary masks.

    Returns:
        A 3-tuple ``(primary_adapter, segments, transform_adapters)`` where:

        - ``primary_adapter`` (``TransformAdapter``): the adapter for the first
          recognised backend, used as the fallback adapter on
          ``FusedCompose._adapter``.
        - ``segments`` (``list[object]``): flat, ordered list of fused and
          passthrough segments ready for ``_wrap_passthrough_segments``.
        - ``transform_adapters`` (``dict[int, TransformAdapter]``): maps each
          transform's positional index in the original ``transforms`` list
          to the adapter used for that transform, enabling per-transform
          dispatch in the mixed-backend forward pass.

    """
    # Cache adapter instances per backend to avoid repeated instantiation
    adapter_cache: dict[Backend, TransformAdapter] = {}

    def _get_adapter(backend: Backend) -> TransformAdapter:
        if backend not in adapter_cache:
            adapter_cache[backend] = _adapter_for_backend(backend)
        return adapter_cache[backend]

    # Build per-transform adapter map for passthrough dispatch.
    # Keyed by positional index (stable across pickle) rather than id()
    # or object identity (which break after deserialization).
    transform_adapters: dict[int, TransformAdapter] = {}
    for idx_tfm, (_tfm, backend_name) in enumerate(zip(transforms, per_backends, strict=True)):
        if backend_name is not None:
            transform_adapters[idx_tfm] = _get_adapter(backend_name)

    # Group consecutive transforms by backend
    groups: list[tuple[Backend | None, list[object]]] = []
    for tfm, backend_name in zip(transforms, per_backends, strict=True):
        if groups and groups[-1][0] == backend_name:
            groups[-1][1].append(tfm)
        else:
            groups.append((backend_name, [tfm]))

    # Build segments per group
    all_segments: list[object] = []
    for backend_name, group_transforms in groups:
        if backend_name is None:
            # Unrecognised transforms — emit as passthrough
            all_segments.extend(group_transforms)
            continue

        adapter = _get_adapter(backend_name)
        if reorder == ReorderPolicy.POINTWISE:
            group_transforms = reorder_pointwise(group_transforms, adapter)
        elif reorder == ReorderPolicy.AGGRESSIVE:
            group_transforms = reorder_aggressive(group_transforms, adapter)
        group_segments = build_segments(
            transforms=group_transforms,
            adapter=adapter,
            interpolation=interpolation,
            padding_mode=padding_mode,
            randomness=randomness,
            use_numpy=(backend_name == Backend.ALBUMENTATIONS),
            route_coords_via_grid=route_coords_via_grid,
            execution=execution,
            compile_warp=compile_warp,
            antialias=antialias,
            clip_policy=clip_policy,
            mask_interpolation=mask_interpolation,
        )
        all_segments.extend(group_segments)

    # Primary adapter: first recognised backend, used as fallback for _adapter.
    # This function is only reachable when len(unique_backends) > 1, which
    # guarantees at least one non-None backend exists.
    primary_backend = next((backend_name for backend_name in per_backends if backend_name is not None), None)
    if primary_backend is None:
        raise ValueError(
            "_build_mixed_segments called with no recognised backends. "
            "This is unreachable from FusedCompose.__init__; if you see "
            "this error, a caller is invoking _build_mixed_segments directly "
            "with an all-None per_backends list."
        )
    primary_adapter = _get_adapter(primary_backend)

    return primary_adapter, all_segments, transform_adapters


def _wrap_passthrough_segments(
    segments: list[object],
    default_adapter: TransformAdapter | None,
    transform_adapters: dict[int, TransformAdapter] | None,
    original_transforms: list[object] | None = None,
) -> list[object]:
    """Wrap raw passthrough transforms with their adapter for stable dispatch.

    When ``transform_adapters`` is keyed by positional index (mixed-backend path), ``original_transforms`` is used to
    resolve the index of each raw passthrough transform object.

    """
    wrapped_segments: list[object] = []
    fused_segment_types = (
        FusedAffineSegment,
        AlbuFusedAffineSegment,
        ExactAffineSegment,
        ProjectiveSegment,
        AlbuProjectiveSegment,
        FusedColorSegment,
        CropResizeSegment,
    )

    # Build a reverse lookup: transform object id -> index in original_transforms.
    # Used for index-keyed transform_adapters in the mixed-backend path.
    _id_to_index: dict[int, int] = {}
    if original_transforms is not None:
        for idx, tfm in enumerate(original_transforms):
            if id(tfm) in _id_to_index:
                warnings.warn(
                    f"The same transform object appears at two positions in the pipeline "
                    f"({type(tfm).__name__!r} at indices {_id_to_index[id(tfm)]} and {idx}). "
                    "Reusing the same object is unsupported; only the last occurrence will "
                    "be used for adapter dispatch.",
                    UserWarning,
                    stacklevel=3,
                )
            _id_to_index[id(tfm)] = idx

    for seg in segments:
        if isinstance(seg, (*fused_segment_types, _PassthroughSegment)):
            wrapped_segments.append(seg)
            continue

        seg_adapter = None
        if transform_adapters is not None:
            # Index-keyed lookup: find the transform's position
            seg_idx = _id_to_index.get(id(seg))
            if seg_idx is not None:
                seg_adapter = transform_adapters.get(seg_idx)

        # If we are in the mixed-backend path (transform_adapters is not None)
        # and still have no adapter, do not fall back to default_adapter. This
        # can indicate a backend=None or otherwise unknown/custom transform
        # that would be unsafe to dispatch through a backend-specific adapter
        # (e.g. AlbumentationsAdapter expecting transform(image=...)).
        if seg_adapter is None and transform_adapters is not None:
            msg = (
                "Passthrough transform encountered in mixed-backend mode but no "
                "corresponding adapter was found in transform_adapters. This is "
                "likely caused by a backend=None or custom transform without a "
                "registered adapter and would otherwise be dispatched through an "
                "incompatible default_adapter. Please ensure all passthrough "
                "transforms have an explicit adapter entry."
            )
            raise RuntimeError(msg)

        # Outside of mixed-backend mode, fall back to default_adapter as before.
        if seg_adapter is None:
            seg_adapter = default_adapter
        if seg_adapter is None:
            msg = "Passthrough transform encountered but no adapter found; this is a bug in build_segments"
            raise RuntimeError(msg)
        wrapped_segments.append(_PassthroughSegment(transform=seg, adapter=seg_adapter))

    return wrapped_segments


# ---------------------------------------------------------------------------
# Internal classes for from_params() - NOT exported
# ---------------------------------------------------------------------------


class _DirectParamTransform:
    """Internal transform that holds parameter ranges for from_params().

    Not exported. Implements the minimal interface expected by _DirectParamAdapter.

    """

    def __init__(self, param_specs: dict[str, tuple[float, float]], prob: float = 1.0) -> None:
        self.param_specs = param_specs
        self.prob = prob
        self.same_on_batch = False


class _DirectFlipTransform:
    """Internal transform representing an hflip or vflip for from_params().

    Not exported.

    """

    def __init__(self, flip_type: Literal["hflip", "vflip"], prob: float = 0.5) -> None:
        self.flip_type: Literal["hflip", "vflip"] = flip_type
        self.prob = prob
        self.same_on_batch = False


class _DirectParamAdapter:
    """Internal adapter for from_params() that samples directly from param ranges.

    Not exported. Implements the TransformAdapter protocol for _DirectParamTransform and _DirectFlipTransform objects.

    """

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory for a direct-param transform."""
        if isinstance(transform, _DirectParamTransform):
            if "brightness" in transform.param_specs or "contrast" in transform.param_specs:
                return TransformCategory.POINTWISE_LINEAR
            return TransformCategory.GEOMETRIC_INTERP
        if isinstance(transform, _DirectFlipTransform):
            return TransformCategory.GEOMETRIC_EXACT
        return TransformCategory.SPATIAL_KERNEL

    @staticmethod
    def sample_params(
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample random parameters from the stored ranges."""
        batch_size = input_shape[0]

        if isinstance(transform, _DirectFlipTransform):
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        if isinstance(transform, _DirectParamTransform):
            specs = transform.param_specs
            result: dict[str, torch.Tensor] = {}

            if "rotation" in specs:
                low, high = specs["rotation"]
                low_rad, high_rad = math.radians(low), math.radians(high)
                result["angle_rad"] = torch.empty(batch_size, device=device).uniform_(low_rad, high_rad)

            # Scale: if uniform 'scale' is set, use it for both axes
            # Individual scale_x/scale_y override uniform scale
            scale_x_range = specs.get("scale_x") or specs.get("scale")
            scale_y_range = specs.get("scale_y") or specs.get("scale")
            if scale_x_range is not None and scale_y_range is not None:
                if "scale_x" not in specs and "scale_y" not in specs:
                    # Uniform 'scale' promises isotropic scaling: one draw shared by both axes
                    scale = torch.empty(batch_size, device=device).uniform_(*scale_x_range)
                    result["scale_x"] = scale
                    result["scale_y"] = scale.clone()
                else:
                    result["scale_x"] = torch.empty(batch_size, device=device).uniform_(*scale_x_range)
                    result["scale_y"] = torch.empty(batch_size, device=device).uniform_(*scale_y_range)
            if scale_x_range is not None and scale_y_range is None:
                result["scale_x"] = torch.empty(batch_size, device=device).uniform_(*scale_x_range)
                result["scale_y"] = torch.ones(batch_size, device=device)
            if scale_y_range is not None and scale_x_range is None:
                result["scale_x"] = torch.ones(batch_size, device=device)
                result["scale_y"] = torch.empty(batch_size, device=device).uniform_(*scale_y_range)

            if "shear_x" in specs:
                low, high = specs["shear_x"]
                result["shear_x_rad"] = torch.empty(batch_size, device=device).uniform_(
                    math.radians(low), math.radians(high)
                )

            if "shear_y" in specs:
                low, high = specs["shear_y"]
                result["shear_y_rad"] = torch.empty(batch_size, device=device).uniform_(
                    math.radians(low), math.radians(high)
                )

            if "translate_x" in specs or "translate_y" in specs:
                if "translate_x" in specs:
                    result["translate_x"] = torch.empty(batch_size, device=device).uniform_(*specs["translate_x"])
                else:
                    result["translate_x"] = torch.zeros(batch_size, device=device)
                if "translate_y" in specs:
                    result["translate_y"] = torch.empty(batch_size, device=device).uniform_(*specs["translate_y"])
                else:
                    result["translate_y"] = torch.zeros(batch_size, device=device)

            if "brightness" in specs:
                result["brightness_factor"] = torch.empty(batch_size, device=device).uniform_(*specs["brightness"])
            if "contrast" in specs:
                result["contrast_factor"] = torch.empty(batch_size, device=device).uniform_(*specs["contrast"])

            return result

        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build a (batch_size, 3, 3) forward affine matrix from sampled params."""
        batch_size: int | None = None
        if isinstance(transform, _DirectFlipTransform):
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            if transform.flip_type == "hflip":
                return hflip_matrix(width=width, batch_size=batch_size, device=device, dtype=torch.float32)
            return vflip_matrix(height=height, batch_size=batch_size, device=device, dtype=torch.float32)

        if isinstance(transform, _DirectParamTransform):
            # Determine batch size and device from any param
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

            mtx_acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

            if "angle_rad" in params:
                mtx_acc = matmul3x3(rotation_matrix(params["angle_rad"], height=height, width=width), mtx_acc)

            if "scale_x" in params and "scale_y" in params:
                mtx_acc = matmul3x3(
                    scale_matrix(params["scale_x"], params["scale_y"], height=height, width=width), mtx_acc
                )

            if "shear_x_rad" in params:
                shear_x_tan = torch.tan(params["shear_x_rad"])
                mtx_acc = matmul3x3(shear_x_matrix(shear_x_tan, height=height, width=width), mtx_acc)

            if "shear_y_rad" in params:
                shear_y_tan = torch.tan(params["shear_y_rad"])
                mtx_acc = matmul3x3(shear_y_matrix(shear_y_tan, height=height, width=width), mtx_acc)

            if "translate_x" in params and "translate_y" in params:
                mtx_acc = matmul3x3(translate_matrix(params["translate_x"], params["translate_y"]), mtx_acc)

            return mtx_acc

        msg = (
            f"_DirectParamAdapter.build_matrix: no recognized param keys in {list(params.keys())}. "
            "This is a bug — either sample_params() returned unexpected keys or the transform "
            "was constructed with an unknown operation name."
        )
        raise RuntimeError(msg)

    @staticmethod
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a homogeneous RGB matrix for native brightness or contrast."""
        factor = params.get("brightness_factor", params.get("contrast_factor"))
        if factor is None:
            if isinstance(transform, _DirectParamTransform):
                raise KeyError("native color factor is sampled at forward time")
            msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
            raise NotImplementedError(msg)
        batch_size = factor.shape[0]
        matrix = torch.eye(4, device=factor.device, dtype=factor.dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        matrix[:, 0, 0] = factor
        matrix[:, 1, 1] = factor
        matrix[:, 2, 2] = factor
        if "contrast_factor" in params:
            midpoint = (
                mean.to(device=factor.device, dtype=factor.dtype) if mean is not None else torch.full_like(factor, 0.5)
            )
            bias = (1.0 - factor) * midpoint
            matrix[:, 0, 3] = bias
            matrix[:, 1, 3] = bias
            matrix[:, 2, 3] = bias
        return matrix

    @staticmethod
    def exact_flip_dims(transform: object) -> list[int]:
        """Return the spatial dims to flip for a _DirectFlipTransform."""
        if isinstance(transform, _DirectFlipTransform):
            if transform.flip_type == "hflip":
                return [3]
            return [2]
        msg = f"Cannot determine flip dims for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def exact_apply(transform: object, image: torch.Tensor) -> torch.Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly."""
        if isinstance(transform, _DirectFlipTransform):
            if transform.flip_type == "hflip":
                return image.flip(dims=[3])
            return image.flip(dims=[2])
        msg = f"Cannot apply exact op for {type(transform).__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Passthrough - direct-param transforms are always fusible."""
        return image


# Short alias for convenience; FusedCompose is the canonical name
Compose = FusedCompose
AugmentationSequential = FusedCompose
