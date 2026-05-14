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
from fuse_augmentations._types import (
    InterpolationStr,
    PaddingModeStr,
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
    CropResizeSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    FusedColorSegment,
    ProjectiveSegment,
    build_segments,
    reorder_aggressive,
    reorder_pointwise,
)

__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.from_config"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["FusedCompose.__call__"]

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fuse_augmentations._resolver import BackendStr, OpStr

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

    ``SPATIAL_KERNEL``, ``POINTWISE``, and ``POINTWISE_LINEAR`` transforms are
    passed through to the backend adapter unchanged.

    ``ReorderPolicy.POINTWISE`` is fully implemented: before segmentation,
    ``POINTWISE`` and ``POINTWISE_LINEAR`` ops are bubbled past geometric ops
    within each ``SPATIAL_KERNEL``-bounded stretch, maximising the geometric
    run length available for fusion.

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
            In this release, Albumentations fused affine segments do not
            support ``aux_targets``. If such a segment exists and ``data_keys``
            is set, construction raises ``ValueError``.
        output_backend: Target output format. ``"numpy"`` (or its alias
            ``"numpy_hwc"``) converts the primary image output to a NumPy
            ``ndarray`` with channel-last layout: batched inputs of shape
            ``(batch_size, channels, height, width)`` become ``(batch_size, height, width, channels)``,
            while unbatched inputs of shape ``(channels, height, width)`` become
            ``(height, width, channels)`` (i.e. the batch
            dimension is implicit/squeezed). ``"torch"`` or ``None`` keeps
            the native ``torch.Tensor`` output. Conversion applies only to
            single-tensor output; when ``data_keys`` is active and a tuple is
            returned, conversion is NOT applied.
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
            output_backend=output_backend,
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
    ) -> None:
        """Assign all instance attributes.

        Called by both ``__init__`` and ``from_params``.

        """
        self.original_transforms: list[object] = list(transforms)
        self.reorder: ReorderPolicy = reorder
        self.interpolation: InterpolationStr | None = interpolation
        self.padding_mode: PaddingModeStr | None = padding_mode
        self.data_keys: list[str] | None = data_keys
        self._adapter: TransformAdapter | None = adapter
        self._segments: list[object] = _wrap_passthrough_segments(
            segments,
            adapter,
            transform_adapters,
            original_transforms=transforms,
        )
        self._last_transform_matrix: Tensor | None = None
        self._last_matrix_segment: object | None = None  # deferred resolution
        self._transform_adapters: dict[int, TransformAdapter] = transform_adapters or {}

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
        self._single_fused_fast_seg: tuple[TransformAdapter, object] | None = None
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
                    from fuse_augmentations.adapters._torchvision import _is_torchvision_v2_transform

                    _bypass_ok = _is_torchvision_v2_transform(_tfm0)
                except ImportError:
                    _bypass_ok = False
            if _bypass_ok:
                self._single_fused_fast_seg = (adapter, _tfm0)
        # Pre-cached B=1 CPU float32 identity for _last_transform_matrix in fast paths.
        self._eye_1x3x3_f32: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        # Single-transform Albumentations direct fast path: for pipelines with exactly
        # one AlbuFusedAffineSegment containing one transform, bypass
        # _forward_albu_native entirely.  Saves ~5-10 us/call: 4 lazy imports,
        # function call overhead, forward_numpy dispatch, and segment isinstance loop.
        # Guards: single segment + single transform + AlbumentationsAdapter only.
        # Not used when aux_targets are passed (kwargs size > 1).
        self._single_albu_direct_tfm: object | None = None
        if (
            len(self._segments) == 1
            and isinstance(self._segments[0], AlbuFusedAffineSegment)
            and len(self._segments[0].transforms) == 1
        ):
            try:
                from fuse_augmentations.adapters._albumentations import (
                    AlbumentationsAdapter as AlbuAdapterCls,
                )

                if isinstance(adapter, AlbuAdapterCls):
                    self._single_albu_direct_tfm = self._segments[0].transforms[0]
            except ImportError:
                pass

        # Pre-computed Albumentations dict-input routing flag: avoids the
        # per-call lazy import + isinstance check in __call__ for Albu pipelines.
        _is_albu: bool = False
        if adapter is not None:
            try:
                from fuse_augmentations.adapters._albumentations import (
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
                        from fuse_augmentations.adapters._albumentations import (
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
        from fuse_augmentations._types import BackendConverter

        self._output_converter: BackendConverter | None
        if output_backend is None:
            self._output_converter = None
        elif output_backend in ("numpy", "numpy_hwc"):
            from fuse_augmentations._converters import TorchToNumpyConverter

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

        # Warn early when output_backend and multi-key data_keys are both set: conversion is a no-op in that
        # case (forward() returns a tuple and only the single-tensor path applies conversion).
        if self._output_converter is not None and data_keys is not None and len(data_keys) > 1:
            warnings.warn(
                "output_backend is set but data_keys contains more than one key. "
                "output_backend conversion is NOT applied in this multi-target mode — "
                "the pipeline returns a tuple of raw tensors. "
                "When using multiple data_keys, set output_backend=None or perform conversion manually.",
                UserWarning,
                stacklevel=3,
            )

        # Albumentations fused segments only reject aux_targets (len > 1); single-key
        # data_keys (image only) is fine because no aux dict is passed through.
        if (
            data_keys is not None
            and len(data_keys) > 1
            and any(isinstance(seg, (AlbuFusedAffineSegment, AlbuProjectiveSegment)) for seg in self._segments)
        ):
            raise ValueError(
                "Albumentations fused segments (AlbuFusedAffineSegment, AlbuProjectiveSegment) do not "
                "support aux_targets in this release. Remove extra data_keys or use a non-Albumentations pipeline."
            )

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Route to the Albumentations native dict path or the standard tensor path.

        When called with ``image=<numpy.ndarray>`` and the pipeline adapter is an
        :class:`~fuse_augmentations.adapters._albumentations.AlbumentationsAdapter`,
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

        Returns:
            ``dict`` with ``"image"`` key (HWC NumPy) for the Albu native path,
            or whatever :meth:`forward` returns for the tensor path.

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
        if (
            self._single_albu_direct_tfm is not None
            and len(kwargs) == 1
            and "image" in kwargs
            and isinstance(kwargs["image"], np.ndarray)
        ):
            img = self._single_albu_direct_tfm(image=kwargs["image"])["image"]  # type: ignore[operator]
            self._last_transform_matrix = self._eye_1x3x3_f32
            return {"image": img}
        if self._is_albu_native and "image" in kwargs and isinstance(kwargs["image"], np.ndarray):
            return self._forward_albu_native(kwargs["image"])

        # Single-transform tensor fast paths: bypass nn.Module.__call__ overhead
        # (~10-15 us from hook dispatch, _call_impl indirection) for the
        # common single-tensor, single-transform case.  Guards: exactly 1 positional
        # arg, no kwargs, data_keys is None (single-tensor mode).
        if len(args) == 1 and not kwargs and self.data_keys is None:
            _exact_fast = self._single_exact_fast
            if _exact_fast is not None:
                image = _exact_fast[0].call_nonfused(_exact_fast[1], cast(Tensor, args[0]))
                return self._convert_primary_output(image)

            _fused_fast = self._single_fused_fast_seg
            if _fused_fast is not None:
                image = _fused_fast[0].call_nonfused(_fused_fast[1], cast(Tensor, args[0]))
                _batch_size = image.shape[0]
                _mtx_eye = self._eye_1x3x3_f32
                self._last_transform_matrix = (
                    _mtx_eye if _batch_size == 1 else _mtx_eye.expand(_batch_size, -1, -1).clone()
                )
                return self._convert_primary_output(image)

        return super().__call__(*args, **kwargs)

    def _forward_albu_native(self, img_hwc: NDArray[Any]) -> dict[str, NDArray[Any]]:
        """Execute the pipeline in Albumentations native dict-input mode.

        Iterates over all segments in NumPy space — no tensor conversion.
        :class:`~fuse_augmentations.affine._segment.AlbuFusedAffineSegment`
        segments are dispatched via their :meth:`forward_numpy` method;
        :class:`~fuse_augmentations._compose._PassthroughSegment` segments
        are dispatched via
        :meth:`~fuse_augmentations.adapters._albumentations.AlbumentationsAdapter.call_nonfused_numpy`.

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
            from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

            for idx_segment, seg in enumerate(self._segments):
                tag = _tags[idx_segment]
                if tag == 0:  # AlbuFusedAffineSegment
                    img_hwc = seg.forward_numpy(img_hwc)  # type: ignore[attr-defined]
                    self._last_transform_matrix = seg.last_matrix  # type: ignore[attr-defined]
                elif tag == 1:  # ExactAffineSegment
                    for tfm in seg.transforms:  # type: ignore[attr-defined]
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 2:  # FusedColorSegment
                    for tfm in seg._transforms:  # type: ignore[attr-defined]
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 3:  # CropResizeSegment
                    for tfm in seg.transforms:  # type: ignore[attr-defined]
                        img_hwc = tfm(image=img_hwc)["image"]
                elif tag == 4:  # Passthrough(Albu adapter)
                    img_hwc = AlbumentationsAdapter.call_nonfused_numpy(seg.transform, img_hwc)  # type: ignore[attr-defined]
                elif tag == 5:  # Passthrough(non-Albu adapter)
                    msg = (
                        f"Passthrough segment adapter {type(seg.adapter).__name__!r} does not "  # type: ignore[attr-defined]
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
            from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

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

    def _convert_primary_output(self, image: torch.Tensor) -> torch.Tensor | NDArray[Any]:
        """Convert the primary image output to the requested backend."""
        if self._output_converter is None:
            return image

        image_for_conversion = image.unsqueeze(0) if image.ndim == 3 else image
        return self._output_converter.convert(image_for_conversion)  # type: ignore[no-any-return]

    def forward(self, *args: torch.Tensor) -> torch.Tensor | NDArray[Any] | tuple[torch.Tensor, ...]:
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

        Returns:
            Single ``Tensor`` or NumPy ``ndarray`` when ``data_keys`` is
            ``None`` or has one entry. NumPy output is returned only when
            ``output_backend="numpy"``. ``tuple[Tensor, ...]`` in
            ``data_keys`` order otherwise.

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

            ``output_backend`` conversion applies to the primary image output only.
            When ``data_keys`` has more than one entry a tuple is returned and
            conversion is NOT applied -- aux_targets (masks, bboxes, keypoints)
            require backend-specific handling. Use ``output_backend=None`` when
            multi-target ``data_keys`` mode is active.

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

        # Single ExactAffineSegment fast path: bypass the segment's
        # nn.Module.__call__ and probability machinery by calling the native
        # adapter directly.  Only safe for single-tensor mode (no aux_targets)
        # because ExactAffineSegment has no last_matrix to propagate.
        _exact_fast = self._single_exact_fast
        if _exact_fast is not None and aux_targets is None:
            image = _exact_fast[0].call_nonfused(_exact_fast[1], image)
            return self._convert_primary_output(image)

        # Single-transform FusedAffineSegment fast path: same bypass for
        # single-transform GEOMETRIC_INTERP pipelines (a-group rotate/scale/shear).
        # Sets _last_transform_matrix to identity — the single-transform path always
        # returns identity regardless (no matrix reconstruction for a-group).
        _fast_seg = self._single_fused_fast_seg
        if _fast_seg is not None and aux_targets is None:
            image = _fast_seg[0].call_nonfused(_fast_seg[1], image)
            _batch_size = image.shape[0]
            _mtx_eye = self._eye_1x3x3_f32
            self._last_transform_matrix = _mtx_eye if _batch_size == 1 else _mtx_eye.expand(_batch_size, -1, -1).clone()
            return self._convert_primary_output(image)

        for seg in self._segments:
            if isinstance(seg, FusedAffineSegment):
                # Call forward directly to skip nn.Module.__call__ overhead
                # (~10-15 us: hook dispatch, _call_impl indirection) for the
                # common case of no registered hooks.
                result = seg.forward(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = cast(Tensor, result)
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, AlbuFusedAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = cast(Tensor, result)
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, (ProjectiveSegment, AlbuProjectiveSegment)):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = cast(Tensor, result)
                self._last_transform_matrix = seg.last_matrix
                continue

            if isinstance(seg, ExactAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                continue

            if isinstance(seg, FusedColorSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                continue

            if isinstance(seg, CropResizeSegment):
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
                    pt_adapter = self._transform_adapters.get(idx, self._adapter)
                    break
            if pt_adapter is None:
                msg = f"Unknown segment type {type(seg).__name__!r} — update FusedCompose.forward dispatch"
                raise RuntimeError(msg)
            image = pt_adapter.call_nonfused(seg, image)

        if self.data_keys is None:
            return self._convert_primary_output(image)
        if len(self.data_keys) == 1:
            return self._convert_primary_output(image)
        # Return tuple in data_keys order (aux_targets is guaranteed non-None here
        # because data_keys is set and has >1 entry)
        _aux: dict[str, torch.Tensor] = aux_targets or {}
        result_list: list[torch.Tensor] = []
        for idx, key in enumerate(self.data_keys):
            if idx == 0:
                result_list.append(image)
            else:
                result_list.append(_aux.get(key, args[idx]))
        return tuple(result_list)

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
            The composed matrix for the last fused affine or projective
            segment, or ``None`` if no such segment has been executed yet
            (including before the first call to :meth:`forward` or if the
            last forward contained only passthrough transforms).

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
        backend: BackendStr,
        interpolation: InterpolationStr = "bilinear",
        padding_mode: PaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
    ) -> FusedCompose:
        """Create a FusedCompose pipeline from a list of TransformSpec objects.

        Resolves each spec's operation to the corresponding backend transform
        class, instantiates it with the spec's params and per-sample probability,
        then builds the pipeline via ``cls(transforms, ...)``.

        Args:
            specs: List of :class:`TransformSpec` objects describing the
                pipeline.
            backend: Backend name (``"kornia"``, ``"torchvision"``,
                ``"albumentations"``).
            interpolation: Interpolation mode for ``grid_sample`` warp.
            padding_mode: Padding mode for out-of-bounds samples.
            reorder: Reorder policy applied before segmentation.
            data_keys: Key list for auxiliary target routing.
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.

        Returns:
            A configured ``FusedCompose`` instance.

        Raises:
            ValueError: If ``backend`` is not in :data:`~fuse_augmentations._resolver.SUPPORTED_BACKENDS`,
                or if a spec's operation is not supported by the chosen backend.
                This validation applies even when ``specs`` is empty.

        Example:
            >>> import torch
            >>> from fuse_augmentations._compose import FusedCompose
            >>> from fuse_augmentations._types import TransformSpec
            >>> spec = TransformSpec(operation="hflip", params={}, prob=0.5)
            >>> pipe = FusedCompose.from_config([spec], backend="kornia")
            >>> pipe(torch.zeros(1, 3, 8, 8)).shape
            torch.Size([1, 3, 8, 8])

        """
        from fuse_augmentations._resolver import SUPPORTED_BACKENDS, SUPPORTED_OPS, resolve_op, translate_params

        if backend not in SUPPORTED_BACKENDS:
            msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
            raise ValueError(msg)

        if not specs:
            return cls(
                [],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                reorder=reorder,
                output_backend=output_backend,
            )

        transforms: list[object] = []
        for spec in specs:
            if "prob" in spec.params:
                msg = (
                    f"TransformSpec.params must not include 'prob'; "
                    f"use TransformSpec.prob for operation {spec.operation!r} instead."
                )
                raise ValueError(msg)
            if spec.operation not in SUPPORTED_OPS:
                msg = f"unknown operation {spec.operation!r}; supported: {sorted(SUPPORTED_OPS)}"
                raise ValueError(msg)
            op_name = cast("OpStr", spec.operation)
            tfm_cls = resolve_op(op_name, backend)
            kwargs = translate_params(op_name, backend, dict(spec.params))
            # Most backends accept p= for per-transform probability.
            # Some (e.g. TorchVision rotation) don't; pass it and let
            # the backend ignore it via **kwargs or TypeError fallback.
            try:
                tfm = tfm_cls(**kwargs, p=spec.prob)
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
            transforms.append(tfm)

        return cls(
            transforms,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            reorder=reorder,
            output_backend=output_backend,
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
        interpolation: InterpolationStr = "bilinear",
        padding_mode: PaddingModeStr = "zeros",
        reorder: ReorderPolicy = ReorderPolicy.POINTWISE,
        data_keys: list[str] | None = None,
        output_backend: Literal["numpy", "numpy_hwc", "torch"] | None = None,
        *,
        specs: list[TransformSpec] | None = None,
        backend: BackendStr | None = None,
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
            output_backend: Target output format (``"numpy"``,
                ``"numpy_hwc"``, ``"torch"``, or ``None``). Forwarded to
                :meth:`__init__`.
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
            NotImplementedError: If ``brightness`` or ``contrast`` is not ``None``.
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
            ))
            if has_keyword_params:
                msg = "specs and keyword params are mutually exclusive"
                raise ValueError(msg)

            if backend is not None:
                return cls.from_config(
                    specs,
                    backend=backend,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    reorder=reorder,
                    data_keys=data_keys,
                    output_backend=output_backend,
                )

            return cls._from_param_specs(
                specs=specs,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
            )

        # When backend is set and geometric kwargs are provided (no specs),
        # convert the kwargs to TransformSpec objects and delegate to from_config.
        if backend is not None:
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
            return cls.from_config(
                config_specs,
                backend=backend,
                interpolation=interpolation,
                padding_mode=padding_mode,
                reorder=reorder,
                data_keys=data_keys,
                output_backend=output_backend,
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
                output_backend=output_backend,
                reorder=reorder,
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
            output_backend=output_backend,
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
                    _DirectParamTransform(param_specs, prob=spec.prob),
                )
            else:
                msg = f"Unsupported op for from_params: {spec.operation!r}"
                raise ValueError(msg)

        if not transforms:
            return cls(
                [],
                interpolation=interpolation,
                padding_mode=padding_mode,
                data_keys=data_keys,
                output_backend=output_backend,
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
            output_backend=output_backend,
        )

        return instance

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
            "translate_x": ("pixels", "translate_x"),
            "translate_y": ("pixels", "translate_y"),
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

        Used internally by :meth:`from_params` when ``backend`` is set and
        geometric kwargs (rather than ``specs``) are provided.

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
    interpolation: InterpolationStr | None,
    padding_mode: PaddingModeStr | None,
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
            group_transforms,
            adapter,
            interpolation,
            padding_mode,
            use_numpy=(backend_name == Backend.ALBUMENTATIONS),
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
    ) -> torch.Tensor:
        """_DirectParamAdapter only handles geometric ops — color matrix not supported."""
        msg = f"build_color_matrix not supported for {type(transform).__name__!r}"
        raise NotImplementedError(msg)

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
