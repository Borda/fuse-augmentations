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
    >>> from fuse_augmentations._compose import Compose
    >>> pipe = Compose([])
    >>> x = torch.zeros(1, 3, 8, 8)
    >>> pipe(x).shape
    torch.Size([1, 3, 8, 8])
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from fuse_augmentations._backend import Backend, detect_backends_per_transform
from fuse_augmentations._types import (
    ReorderPolicy,
    SegmentDescriptor,
    TransformAdapter,
    TransformCategory,
    TransformSpec,
)
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
from fuse_augmentations.affine._segment import (
    AlbuFusedAffineSegment,
    AlbuProjectiveSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    ProjectiveSegment,
    build_segments,
    reorder_aggressive,
    reorder_pointwise,
)

if TYPE_CHECKING:
    from torch import Tensor

_KNOWN_DATA_KEYS = {"input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints"}


@dataclass(frozen=True, slots=True)
class _PassthroughSegment:
    """Serializable passthrough segment that carries its adapter explicitly."""

    transform: object
    adapter: TransformAdapter


class FusedCompose(nn.Module):
    """Fused augmentation pipeline that replaces the backend's native Compose.

    Segments the transform list into fused geometric segments and passthrough
    transforms, then executes them sequentially. Consecutive geometric ops are
    grouped and executed as either:

    - A :class:`~fuse_augmentations.affine._segment.FusedAffineSegment` - when the run
      contains at least one ``GEOMETRIC_INTERP`` op. Matrices are composed and
      a single ``grid_sample`` call is used, eliminating redundant interpolation
      passes.
    - An :class:`~fuse_augmentations.affine._segment.ExactAffineSegment` - when the run
      contains *only* ``GEOMETRIC_EXACT`` ops (HFlip, VFlip). Transforms are
      applied via ``tensor.flip`` with zero interpolation error.

    ``SPATIAL_KERNEL`` and ``POINTWISE`` transforms are passed through to the
    backend adapter unchanged.

    ``ReorderPolicy.POINTWISE`` is fully implemented: before segmentation,
    ``POINTWISE`` ops are bubbled past geometric ops within each
    ``SPATIAL_KERNEL``-bounded stretch, maximising the geometric run length
    available for fusion.

    Args:
        transforms: List of augmentation transform objects.
        reorder: Reorder policy applied before segmentation.
            ``NONE`` (default) preserves the original order.
            ``POINTWISE`` reorders pointwise ops after geometric chains.
            ``AGGRESSIVE`` currently aliases ``POINTWISE`` and is kept for API
            compatibility with future stronger reorder semantics.
        interpolation: Interpolation mode override for fused segments
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode override for fused segments
            (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        data_keys: List of key names describing positional arguments to
            :meth:`forward`. The first key should be ``"input"`` (the image).
            Auxiliary keys (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
            ``"keypoints"``) are routed through segments and transformed
            alongside the image. Unknown keys are passed through unchanged
            with a ``UserWarning``. ``None`` preserves backward-compatible
            single-tensor input/output.
        **backend_kwargs: Reserved for backend-specific options (currently unused).

    """

    def __init__(
        self,
        transforms: list[object],
        reorder: ReorderPolicy = ReorderPolicy.NONE,
        interpolation: str | None = None,
        padding_mode: str | None = None,
        data_keys: list[str] | None = None,
        **backend_kwargs: object,
    ) -> None:
        super().__init__()

        if reorder not in (ReorderPolicy.NONE, ReorderPolicy.POINTWISE, ReorderPolicy.AGGRESSIVE):
            msg = f"ReorderPolicy.{reorder.name} not yet supported"
            raise NotImplementedError(msg)

        adapter: TransformAdapter | None
        segments: list[object]
        tfm_adapters: dict[int, TransformAdapter] | None = None

        if not transforms:
            adapter = None
            segments = []
        else:
            per_backends = detect_backends_per_transform(transforms)
            unique_backends = {b for b in per_backends if b is not None}

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
                if reorder == ReorderPolicy.POINTWISE:
                    transforms = reorder_pointwise(transforms, adapter)
                elif reorder == ReorderPolicy.AGGRESSIVE:
                    transforms = reorder_aggressive(transforms, adapter)
                segments = build_segments(
                    transforms,
                    adapter,
                    interpolation,
                    padding_mode,
                    use_numpy=(backend == Backend.ALBUMENTATIONS),
                )
            else:
                # Mixed-backend path: group by backend, build segments per group
                adapter, segments, tfm_adapters = _build_mixed_segments(
                    transforms,
                    per_backends,
                    reorder,
                    interpolation,
                    padding_mode,
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
        )

    def _setup_instance(
        self,
        transforms: list[object],
        reorder: ReorderPolicy,
        interpolation: str | None,
        padding_mode: str | None,
        data_keys: list[str] | None,
        adapter: TransformAdapter | None,
        segments: list[object],
        transform_adapters: dict[int, TransformAdapter] | None = None,
    ) -> None:
        """Assign all instance attributes.

        Called by both ``__init__`` and ``from_params``.

        """
        self.original_transforms: list[object] = list(transforms)
        self.reorder: ReorderPolicy = reorder
        self.interpolation: str | None = interpolation
        self.padding_mode: str | None = padding_mode
        self.data_keys: list[str] | None = data_keys
        self._adapter: TransformAdapter | None = adapter
        self._segments: list[object] = _wrap_passthrough_segments(
            segments,
            adapter,
            transform_adapters,
            original_transforms=transforms,
        )
        self._last_transform_matrix: Tensor | None = None
        self._transform_adapters: dict[int, TransformAdapter] = transform_adapters or {}

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

    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply the augmentation pipeline to an image batch and optional auxiliary targets.

        **Single-tensor mode** (``data_keys=None``, default): accepts one image
        tensor and returns one tensor. This is the backward-compatible path.

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

        Returns:
            Single ``Tensor`` when ``data_keys`` is ``None`` or has one entry;
            ``tuple[Tensor, ...]`` in ``data_keys`` order otherwise.

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

        """
        if self.data_keys is None:
            # Backward-compatible single-tensor path
            if len(args) != 1:
                msg = f"Expected 1 argument (data_keys is None), got {len(args)}"
                raise TypeError(msg)
            image = args[0]
            aux_targets: dict[str, torch.Tensor] | None = None
        else:
            if len(args) != len(self.data_keys):
                msg = f"Expected {len(self.data_keys)} arguments for data_keys={self.data_keys}, got {len(args)}"
                raise TypeError(msg)
            # Build aux_targets dict from positional args
            image = args[0]
            aux_keys = list(self.data_keys[1:])
            # Forbid duplicate auxiliary keys to avoid silent overwrites in aux_targets
            if len(aux_keys) != len(set(aux_keys)):
                msg = (
                    "Duplicate entries detected in auxiliary data_keys "
                    f"(data_keys[1:]): {aux_keys}. Auxiliary keys must be unique."
                )
                raise ValueError(msg)
            aux_targets = {}
            for key, val in zip(aux_keys, args[1:], strict=True):
                aux_targets[key] = val

        for seg in self._segments:
            if isinstance(seg, FusedAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, AlbuFusedAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, ExactAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                continue

            if isinstance(seg, _PassthroughSegment):
                image = seg.adapter.call_nonfused(seg.transform, image)
                continue

            # Legacy passthrough support for pre-wrapper pickles.
            # Try index-based lookup by finding this transform's position.
            pt_adapter = None
            for idx, orig_tfm in enumerate(self.original_transforms):
                if orig_tfm is seg:
                    pt_adapter = self._transform_adapters.get(idx)
                    break
            if pt_adapter is None:
                pt_adapter = self._adapter
            if pt_adapter is None:
                msg = "Passthrough transform encountered but no adapter found; this is a bug in build_segments"
                raise RuntimeError(msg)
            image = pt_adapter.call_nonfused(seg, image)

        if self.data_keys is None:
            return image
        if len(self.data_keys) == 1:
            return image
        # Return tuple in data_keys order (aux_targets is guaranteed non-None here
        # because data_keys is set and has >1 entry)
        _aux: dict[str, torch.Tensor] = aux_targets or {}
        result_list: list[torch.Tensor] = []
        for i, key in enumerate(self.data_keys):
            if i == 0:
                result_list.append(image)
            else:
                result_list.append(_aux.get(key, args[i]))
        return tuple(result_list)

    @property
    def transform_matrix(self) -> torch.Tensor | None:
        """Return the ``(B, 3, 3)`` composed matrix for the last fused segment.

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
            The composed matrix for the last fused affine or projective
            segment, or ``None`` if no such segment has been executed yet
            (including before the first call to :meth:`forward` or if the
            last forward contained only passthrough transforms).

        """
        return self._last_transform_matrix

    @property
    def n_warps_saved(self) -> int:
        """Return the number of interpolation passes eliminated vs sequential execution.

        Each fused segment with *n* transforms saves *n - 1* warp passes.
        Single-transform segments contribute zero savings.

        Returns:
            Total number of eliminated warp passes across all fused segments.

        """
        total = 0
        for seg in self._segments:
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment, ProjectiveSegment, AlbuProjectiveSegment)):
                # n transforms fused → 1 warp, saving n-1 passes.
                n = len(seg.transforms)
                if n > 1:
                    total += n - 1
                continue

            if isinstance(seg, ExactAffineSegment):
                # Each flip in an ExactAffineSegment avoids grid_sample entirely
                # (uses tensor.flip), so every transform saves exactly 1 warp.
                # This is why ExactAffineSegment contributes n rather than n-1:
                # even a single flip is lossless and free of grid_sample cost.
                total += len(seg.transforms)
        return total

    @property
    def fusion_plan(self) -> str:
        """Return a human-readable summary of what got fused and what didn't.

        Returns:
            Arrow-separated description of segments, e.g.
            ``"fused(RandomRotation, RandomHorizontalFlip) -> passthrough(GaussianBlur)"``.
            Returns ``"empty"`` for an empty pipeline.

        """
        parts: list[str] = []
        for seg in self._segments:
            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"projective({', '.join(names)})")
                continue

            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"fused({', '.join(names)})")
                continue

            if isinstance(seg, ExactAffineSegment):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"exact({', '.join(names)})")
                continue

            if isinstance(seg, _PassthroughSegment):
                parts.append(f"passthrough({type(seg.transform).__name__})")
                continue

            parts.append(f"passthrough({type(seg).__name__})")
        return " \u2192 ".join(parts) if parts else "empty"

    @property
    def fusion_plan_descriptors(self) -> list[SegmentDescriptor]:
        """Return a structured, machine-readable description of the fusion plan.

        Each element corresponds to one segment in the pipeline, in execution
        order. This is the structured counterpart to the human-readable
        :attr:`fusion_plan` string. Available immediately after construction —
        does not require a :meth:`forward` call.

        Returns:
            List of :class:`~fuse_augmentations._types.SegmentDescriptor`
            instances, one per segment. Empty list for an empty pipeline.
            Each descriptor's ``backend`` field is the adapter class name
            (e.g. ``"KorniaAdapter"``) for fused, exact, and projective
            segments, and ``None`` for passthrough segments and backend-free
            pipelines created via :meth:`from_params`.

        Example:
            >>> import torch
            >>> from fuse_augmentations._compose import FusedCompose
            >>> pipe = FusedCompose([])
            >>> pipe.fusion_plan_descriptors
            []

        Note:
            The ``backend`` field on passthrough segments is always ``None``,
            regardless of the pipeline's backend. Only fused, exact, and
            projective segments carry the adapter class name.

        """

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
                names = tuple(type(t).__name__ for t in seg.transforms)
                n = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="projective",
                        transforms=names,
                        n_warps_saved=n,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = tuple(type(t).__name__ for t in seg.transforms)
                n = len(names) - 1 if len(names) > 1 else 0
                descriptors.append(
                    SegmentDescriptor(
                        kind="fused",
                        transforms=names,
                        n_warps_saved=n,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, ExactAffineSegment):
                names = tuple(type(t).__name__ for t in seg.transforms)
                n = len(names)  # Each flip saves 1 warp vs grid_sample
                descriptors.append(
                    SegmentDescriptor(
                        kind="exact",
                        transforms=names,
                        n_warps_saved=n,
                        backend=_resolve_backend(seg),
                    )
                )
                continue
            if isinstance(seg, _PassthroughSegment):
                descriptors.append(
                    SegmentDescriptor(
                        kind="passthrough",
                        transforms=(type(seg.transform).__name__,),
                        n_warps_saved=0,
                    )
                )
                continue
            # Legacy passthrough
            descriptors.append(SegmentDescriptor(kind="passthrough", transforms=(type(seg).__name__,), n_warps_saved=0))
        return descriptors

    @classmethod
    def from_config(
        cls,
        specs: list[TransformSpec],
        backend: str,
        interpolation: str = "bilinear",
        padding_mode: str = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
    ) -> FusedCompose:
        """Create a FusedCompose pipeline from a list of TransformSpec objects.

        Resolves each spec's op name to the corresponding backend transform
        class, instantiates it with the spec's params and p, then builds the
        pipeline via ``cls(transforms, ...)``.

        Args:
            specs: List of :class:`TransformSpec` objects describing the
                pipeline.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``).
            interpolation: Interpolation mode for ``grid_sample`` warp.
            padding_mode: Padding mode for out-of-bounds samples.
            reorder: Reorder policy applied before segmentation.
            data_keys: Key list for auxiliary target routing.

        Returns:
            A configured ``FusedCompose`` instance.

        Raises:
            ValueError: If ``backend`` is not in :data:`~fuse_augmentations._resolver.SUPPORTED_BACKENDS`,
                or if a spec's ``op`` is not supported by the chosen backend.

        Example:
            >>> import torch
            >>> from fuse_augmentations._compose import FusedCompose
            >>> from fuse_augmentations._types import TransformSpec
            >>> spec = TransformSpec(op="hflip", params={}, p=0.5)
            >>> pipe = FusedCompose.from_config([spec], backend="kornia")  # doctest: +SKIP
            >>> pipe(torch.zeros(1, 3, 8, 8)).shape  # doctest: +SKIP
            torch.Size([1, 3, 8, 8])

        """
        from fuse_augmentations._resolver import resolve_op

        if not specs:
            return cls(
                [],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                reorder=reorder,
            )

        transforms: list[object] = []
        for spec in specs:
            tfm_cls = resolve_op(spec.op, backend)
            kwargs: dict[str, object] = {**spec.params}
            # Most backends accept p= for per-transform probability.
            # Some (e.g. TorchVision rotation) don't; pass it and let
            # the backend ignore it via **kwargs or TypeError fallback.
            try:
                tfm = tfm_cls(**kwargs, p=spec.p)
            except TypeError:
                # Backend class does not accept p= (e.g. TorchVision RandomRotation)
                tfm = tfm_cls(**kwargs)
                # Attach p as attribute so the fused engine can read it
                with contextlib.suppress(AttributeError, TypeError):
                    tfm.p = spec.p
            transforms.append(tfm)

        return cls(
            transforms,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            reorder=reorder,
        )

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
        interpolation: str = "bilinear",
        padding_mode: str = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        *,
        specs: list[TransformSpec] | None = None,
    ) -> FusedCompose:
        """Create a ``FusedCompose`` pipeline directly from parameter ranges.

        This factory bypasses backend transform objects entirely and samples
        parameters directly using ``_matrix.py`` primitives. Useful for
        backend-agnostic pipelines or when no backend is installed.

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
            brightness: Reserved for v0.4. Raises ``NotImplementedError`` if
                not ``None``.
            contrast: Reserved for v0.4. Raises ``NotImplementedError`` if not
                ``None``.
            interpolation: Interpolation mode for the ``grid_sample`` warp.
                One of ``"bilinear"`` (default), ``"nearest"``, ``"bicubic"``.
            padding_mode: Padding for out-of-bounds samples.
                One of ``"zeros"`` (default), ``"border"``, ``"reflection"``.
            reorder: Reorder policy applied before segmentation.
                Defaults to ``ReorderPolicy.POINTWISE``.
            data_keys: Key list for auxiliary target routing, forwarded to
                :meth:`__init__`. ``None`` preserves single-tensor I/O.
            specs: List of :class:`TransformSpec` objects. When provided,
                all other geometric keyword arguments must be at their
                defaults (mutually exclusive). Keyword-only.

        Returns:
            A configured ``FusedCompose`` instance ready for inference or training.

        Raises:
            NotImplementedError: If ``brightness`` or ``contrast`` is not ``None``.
            ValueError: If ``specs`` is provided together with any geometric keyword
                argument (they are mutually exclusive).
            ValueError: If ``specs`` contains an op that is not supported in
                backend-free mode (i.e. not one of ``"rotation"``, ``"scale"``,
                ``"scale_x"``, ``"scale_y"``, ``"shear_x"``, ``"shear_y"``,
                ``"translate_x"``, ``"translate_y"``, ``"hflip"``, ``"vflip"``).

        Example:
            >>> import torch
            >>> from fuse_augmentations._compose import FusedCompose
            >>> pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
            >>> x = torch.zeros(2, 3, 64, 64)
            >>> out = pipe(x)
            >>> out.shape
            torch.Size([2, 3, 64, 64])

        """
        if brightness is not None:
            msg = "brightness not yet supported, planned v0.4"
            raise NotImplementedError(msg)
        if contrast is not None:
            msg = "contrast not yet supported, planned v0.4"
            raise NotImplementedError(msg)

        # --- specs= overload path (C.4) ---
        if specs is not None:
            # Validate mutual exclusivity
            has_keyword_params = any([
                rotation, scale, scale_x, scale_y,
                shear_x, shear_y, translate_x, translate_y,
                hflip_p > 0.0, vflip_p > 0.0,
            ])
            if has_keyword_params:
                msg = "specs and keyword params are mutually exclusive"
                raise ValueError(msg)
            return cls._from_param_specs(
                specs=specs,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
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

        has_affine = bool(param_specs)
        has_flips = hflip_p > 0.0 or vflip_p > 0.0

        # NOTE: The identity path (all params None) returns a normal __init__ instance
        # with _adapter=None. The non-identity path uses _DirectParamAdapter.
        # Both handle empty _segments correctly; do not branch on isinstance(_adapter, ...).
        if not has_affine and not has_flips:
            return cls(
                [],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                reorder=reorder,
            )

        # Build internal transforms and adapter
        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        if has_affine:
            transforms.append(_DirectParamTransform(param_specs, p=1.0))

        if hflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="hflip", p=hflip_p))

        if vflip_p > 0.0:
            transforms.append(_DirectFlipTransform(flip_type="vflip", p=vflip_p))

        # Build instance bypassing detect_backend
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        segments = build_segments(transforms, adapter, interpolation, padding_mode)
        instance._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
        )

        return instance

    @classmethod
    def _from_param_specs(
        cls,
        specs: list[TransformSpec],
        interpolation: str,
        padding_mode: str,
        reorder: ReorderPolicy,
        data_keys: list[str] | None,
    ) -> FusedCompose:
        """Build a from_params pipeline from a list of TransformSpec objects.

        Each spec is converted to the appropriate internal direct-param
        transform (``_DirectParamTransform`` or ``_DirectFlipTransform``).

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
        }

        adapter = _DirectParamAdapter()
        transforms: list[object] = []

        for spec in specs:
            if spec.op in ("hflip", "vflip"):
                transforms.append(
                    _DirectFlipTransform(flip_type=spec.op, p=spec.p),  # type: ignore[arg-type]
                )
            elif spec.op in op_to_param_key:
                param_key = op_to_param_key[spec.op]
                # Extract the range value from spec.params.
                # The canonical param key varies: "degrees" for rotation,
                # "scale" for scale, etc.
                param_value = next(iter(spec.params.values())) if spec.params else None
                param_specs = {param_key: param_value} if param_value is not None else {}
                transforms.append(
                    _DirectParamTransform(param_specs, p=spec.p),  # type: ignore[arg-type]
                )
            else:
                msg = f"Unsupported op for from_params: {spec.op!r}"
                raise ValueError(msg)

        if not transforms:
            return cls(
                [],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                reorder=reorder,
            )

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        segments = build_segments(transforms, adapter, interpolation, padding_mode)
        instance._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
        )

        return instance


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
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        return KorniaAdapter()
    if backend == Backend.ALBUMENTATIONS:
        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

        return AlbumentationsAdapter()
    if backend == Backend.TORCHVISION:
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        return TorchVisionAdapter()
    msg = f"Backend '{backend.value}' not yet supported"
    raise NotImplementedError(msg)


def _build_mixed_segments(
    transforms: list[object],
    per_backends: list[Backend | None],
    reorder: ReorderPolicy,
    interpolation: str | None,
    padding_mode: str | None,
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
    for idx, (_tfm, bk) in enumerate(zip(transforms, per_backends, strict=True)):
        if bk is not None:
            transform_adapters[idx] = _get_adapter(bk)

    # Group consecutive transforms by backend
    groups: list[tuple[Backend | None, list[object]]] = []
    for tfm, bk in zip(transforms, per_backends, strict=True):
        if groups and groups[-1][0] == bk:
            groups[-1][1].append(tfm)
        else:
            groups.append((bk, [tfm]))

    # Build segments per group
    all_segments: list[object] = []
    for bk, group_transforms in groups:
        if bk is None:
            # Unrecognised transforms — emit as passthrough
            all_segments.extend(group_transforms)
            continue

        adapter = _get_adapter(bk)
        if reorder == ReorderPolicy.POINTWISE:
            group_transforms = reorder_pointwise(group_transforms, adapter)
        elif reorder == ReorderPolicy.AGGRESSIVE:
            group_transforms = reorder_aggressive(group_transforms, adapter)
        group_segments = build_segments(
            group_transforms,
            adapter,
            interpolation,
            padding_mode,
            use_numpy=(bk == Backend.ALBUMENTATIONS),
        )
        all_segments.extend(group_segments)

    # Primary adapter: first recognised backend, used as fallback for _adapter.
    # This function is only reachable when len(unique_backends) > 1, which
    # guarantees at least one non-None backend exists.
    primary_backend = next((b for b in per_backends if b is not None), None)
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

    When ``transform_adapters`` is keyed by positional index (mixed-backend
    path), ``original_transforms`` is used to resolve the index of each raw
    passthrough transform object.

    """
    wrapped_segments: list[object] = []
    fused_segment_types = (
        FusedAffineSegment,
        AlbuFusedAffineSegment,
        ExactAffineSegment,
        ProjectiveSegment,
        AlbuProjectiveSegment,
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

    def __init__(self, param_specs: dict[str, tuple[float, float]], p: float = 1.0) -> None:
        self.param_specs = param_specs
        self.p = p
        self.same_on_batch = False


class _DirectFlipTransform:
    """Internal transform representing an hflip or vflip for from_params().

    Not exported.

    """

    def __init__(self, flip_type: Literal["hflip", "vflip"], p: float = 0.5) -> None:
        self.flip_type: Literal["hflip", "vflip"] = flip_type
        self.p = p
        self.same_on_batch = False


class _DirectParamAdapter:
    """Internal adapter for from_params() that samples directly from param ranges.

    Not exported. Implements the TransformAdapter protocol for _DirectParamTransform and _DirectFlipTransform objects.

    """

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory for a direct-param transform."""
        if isinstance(transform, _DirectParamTransform):
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
        B = input_shape[0]  # noqa: N806

        if isinstance(transform, _DirectFlipTransform):
            return {"_batch_size": torch.tensor([B], device=device, dtype=torch.int64)}

        if isinstance(transform, _DirectParamTransform):
            specs = transform.param_specs
            result: dict[str, torch.Tensor] = {}

            if "rotation" in specs:
                lo, hi = specs["rotation"]
                lo_rad, hi_rad = math.radians(lo), math.radians(hi)
                result["angle_rad"] = torch.empty(B, device=device).uniform_(lo_rad, hi_rad)

            # Scale: if uniform 'scale' is set, use it for both axes
            # Individual scale_x/scale_y override uniform scale
            sx_range = specs.get("scale_x") or specs.get("scale")
            sy_range = specs.get("scale_y") or specs.get("scale")
            if sx_range is not None and sy_range is not None:
                result["scale_x"] = torch.empty(B, device=device).uniform_(*sx_range)
                result["scale_y"] = torch.empty(B, device=device).uniform_(*sy_range)
            if sx_range is not None and sy_range is None:
                result["scale_x"] = torch.empty(B, device=device).uniform_(*sx_range)
                result["scale_y"] = torch.ones(B, device=device)
            if sy_range is not None and sx_range is None:
                result["scale_x"] = torch.ones(B, device=device)
                result["scale_y"] = torch.empty(B, device=device).uniform_(*sy_range)

            if "shear_x" in specs:
                lo, hi = specs["shear_x"]
                result["shear_x_rad"] = torch.empty(B, device=device).uniform_(math.radians(lo), math.radians(hi))

            if "shear_y" in specs:
                lo, hi = specs["shear_y"]
                result["shear_y_rad"] = torch.empty(B, device=device).uniform_(math.radians(lo), math.radians(hi))

            if "translate_x" in specs or "translate_y" in specs:
                if "translate_x" in specs:
                    result["translate_x"] = torch.empty(B, device=device).uniform_(*specs["translate_x"])
                else:
                    result["translate_x"] = torch.zeros(B, device=device)
                if "translate_y" in specs:
                    result["translate_y"] = torch.empty(B, device=device).uniform_(*specs["translate_y"])
                else:
                    result["translate_y"] = torch.zeros(B, device=device)

            return result

        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> torch.Tensor:
        """Build a (B, 3, 3) forward affine matrix from sampled params."""
        if isinstance(transform, _DirectFlipTransform):
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            if transform.flip_type == "hflip":
                return hflip_matrix(W=W, batch_size=B, device=device, dtype=torch.float32)
            return vflip_matrix(H=H, batch_size=B, device=device, dtype=torch.float32)

        if isinstance(transform, _DirectParamTransform):
            # Determine batch size and device from any param
            B = None  # type: ignore[assignment]  # noqa: N806
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

            acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()

            if "angle_rad" in params:
                acc = matmul3x3(rotation_matrix(params["angle_rad"], H=H, W=W), acc)

            if "scale_x" in params and "scale_y" in params:
                acc = matmul3x3(scale_matrix(params["scale_x"], params["scale_y"], H=H, W=W), acc)

            if "shear_x_rad" in params:
                shear_x_tan = torch.tan(params["shear_x_rad"])
                acc = matmul3x3(shear_x_matrix(shear_x_tan, H=H, W=W), acc)

            if "shear_y_rad" in params:
                shear_y_tan = torch.tan(params["shear_y_rad"])
                acc = matmul3x3(shear_y_matrix(shear_y_tan, H=H, W=W), acc)

            if "translate_x" in params and "translate_y" in params:
                acc = matmul3x3(translate_matrix(params["translate_x"], params["translate_y"]), acc)

            return acc

        return torch.eye(3).unsqueeze(0)

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
