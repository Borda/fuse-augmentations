"""Compose -- fused augmentation pipeline replacing the backend's Compose/Sequential.

Wraps a list of augmentation transforms, fuses consecutive geometric ops into a
single ``grid_sample`` pass, and provides the same forward-call interface as the
backend.

v0.3 additions: ``data_keys`` parameter routes auxiliary targets (masks,
bounding boxes, keypoints) alongside the image through every segment.
``from_params()`` classmethod constructs a fused pipeline directly from
numeric parameter ranges, without importing any backend (Kornia-free).

Example:
    >>> import torch
    >>> from fuse_augmentations._compose import Compose
    >>> pipe = Compose([])
    >>> x = torch.zeros(1, 3, 8, 8)
    >>> pipe(x).shape
    torch.Size([1, 3, 8, 8])
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from fuse_augmentations._backend import Backend, detect_backend
from fuse_augmentations._types import ReorderPolicy, TransformAdapter, TransformCategory
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
    ExactSegment,
    FusedAffineSegment,
    build_segments,
    reorder_pointwise,
)

if TYPE_CHECKING:
    from torch import Tensor

_KNOWN_DATA_KEYS = {"input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints"}


class FusedCompose(nn.Module):
    """Fused augmentation pipeline that replaces the backend's native Compose.

    Segments the transform list into fused geometric segments and passthrough
    transforms, then executes them sequentially. Consecutive geometric ops are
    grouped and executed as either:

    - A :class:`~fuse_augmentations._segment.FusedAffineSegment` - when the run
      contains at least one ``GEOMETRIC_INTERP`` op. Matrices are composed and
      a single ``grid_sample`` call is used, eliminating redundant interpolation
      passes.
    - An :class:`~fuse_augmentations._segment.ExactSegment` - when the run
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
            ``AGGRESSIVE`` raises ``NotImplementedError``.
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

    Raises:
        NotImplementedError: If ``reorder`` is ``ReorderPolicy.AGGRESSIVE``.
        NotImplementedError: If the detected backend is not Kornia (only Kornia is currently
            supported).

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

        if reorder not in (ReorderPolicy.NONE, ReorderPolicy.POINTWISE):
            msg = f"ReorderPolicy.{reorder.name} not yet supported"
            raise NotImplementedError(msg)

        adapter: TransformAdapter | None
        segments: list[object]

        if not transforms:
            adapter = None
            segments = []
        else:
            backend = detect_backend(transforms)
            if backend == Backend.KORNIA:
                from fuse_augmentations.adapters._kornia import KorniaAdapter

                adapter = KorniaAdapter()
            elif backend == Backend.ALBUMENTATIONS:
                from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

                adapter = AlbumentationsAdapter()
            else:
                msg = f"Backend '{backend.value}' not yet supported"
                raise NotImplementedError(msg)
            if reorder == ReorderPolicy.POINTWISE:
                transforms = reorder_pointwise(transforms, adapter)
            segments = build_segments(
                transforms,
                adapter,
                interpolation,
                padding_mode,
                use_numpy=(backend == Backend.ALBUMENTATIONS),
            )

        self._setup_instance(
            transforms=transforms,
            reorder=reorder,
            interpolation=interpolation,
            padding_mode=padding_mode,
            data_keys=data_keys,
            adapter=adapter,
            segments=segments,
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
        self._segments: list[object] = segments
        self._last_transform_matrix: Tensor | None = None

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

            if isinstance(seg, ExactSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result
                else:
                    image = result
                continue

            # Passthrough: apply via adapter's call_nonfused (image only)
            if self._adapter is None:
                msg = "Passthrough transform encountered but adapter is None; this is a bug in build_segments"
                raise RuntimeError(msg)
            image = self._adapter.call_nonfused(seg, image)

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
        """Return the ``(B, 3, 3)`` composed matrix for the last fused affine segment.

        This is the composed forward transform matrix produced by the last
        :class:`~fuse_augmentations._segment.FusedAffineSegment` executed in the
        most recent :meth:`forward` call. Passthrough (non-fused) transforms do
        not affect this value, and multiple fused segments are *not* composed into
        a single whole-pipeline matrix.

        Returns:
            The composed matrix for the last fused affine segment, or ``None`` if
            no such segment has been executed yet (including before the first
            call to :meth:`forward` or if the last forward contained only
            passthrough transforms).

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
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                # n transforms fused → 1 warp, saving n-1 passes.
                n = len(seg.transforms)
                if n > 1:
                    total += n - 1
                continue

            if isinstance(seg, ExactSegment):
                # Each flip in an ExactSegment avoids grid_sample entirely
                # (uses tensor.flip), so every transform saves exactly 1 warp.
                # This is why ExactSegment contributes n rather than n-1:
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
            if isinstance(seg, (FusedAffineSegment, AlbuFusedAffineSegment)):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"fused({', '.join(names)})")
                continue

            if isinstance(seg, ExactSegment):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"exact({', '.join(names)})")
                continue

            parts.append(f"passthrough({type(seg).__name__})")
        return " \u2192 ".join(parts) if parts else "empty"

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
    ) -> FusedCompose:
        """Create a ``FusedCompose`` pipeline directly from parameter ranges.

        This factory bypasses backend transform objects entirely and samples
        parameters directly using ``_matrix.py`` primitives. Useful for
        backend-agnostic pipelines or when Kornia is not installed.

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

        Returns:
            A configured ``FusedCompose`` instance ready for inference or training.

        Raises:
            NotImplementedError: If ``brightness`` or ``contrast`` is not ``None``.

        Example:
            >>> import torch
            >>> from fuse_augmentations._compose import FusedCompose
            >>> pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
            >>> x = torch.zeros(2, 3, 64, 64)
            >>> out = pipe(x)
            >>> out.shape
            torch.Size([2, 3, 64, 64])

        """
        if reorder is ReorderPolicy.AGGRESSIVE:
            raise NotImplementedError(
                "ReorderPolicy.AGGRESSIVE is not supported for Compose.from_params(); "
                "use a supported reorder policy, such as ReorderPolicy.POINTWISE."
            )
        if brightness is not None:
            msg = "brightness not yet supported, planned v0.4"
            raise NotImplementedError(msg)
        if contrast is not None:
            msg = "contrast not yet supported, planned v0.4"
            raise NotImplementedError(msg)

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
