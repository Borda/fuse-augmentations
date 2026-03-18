"""Compose -- fused augmentation pipeline replacing the backend's Compose/Sequential.

Wraps a list of augmentation transforms, fuses consecutive geometric ops into a
single ``grid_sample`` pass, and provides the same forward-call interface as the
backend.

Example:
    >>> import torch
    >>> from fuse_augmentations._compose import Compose
    >>> pipe = Compose([])
    >>> x = torch.zeros(1, 3, 8, 8)
    >>> pipe(x).shape
    torch.Size([1, 3, 8, 8])
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from fuse_augmentations._backend import Backend, detect_backend
from fuse_augmentations._segment import ExactSegment, FusedAffineSegment, build_segments, reorder_pointwise
from fuse_augmentations._types import ReorderPolicy, TransformAdapter

if TYPE_CHECKING:
    from torch import Tensor

_KNOWN_DATA_KEYS = {"input", "mask", "bbox_xyxy", "bbox_xywh", "keypoints"}


class FusedCompose(nn.Module):
    """Fused augmentation pipeline that replaces the backend's native Compose.

    Segments the transform list into fused geometric segments and passthrough
    transforms, then executes them sequentially. Consecutive geometric ops are
    grouped and executed as either:

    - A :class:`~fuse_augmentations._segment.FusedAffineSegment` — when the run
      contains at least one ``GEOMETRIC_INTERP`` op. Matrices are composed and
      a single ``grid_sample`` call is used, eliminating redundant interpolation
      passes.
    - An :class:`~fuse_augmentations._segment.ExactSegment` — when the run
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
        NotImplementedError: If the detected backend is not Kornia (only Kornia is supported in
            v0.1/v0.2).

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

        self.original_transforms: list[object] = list(transforms)
        self.reorder: ReorderPolicy = reorder
        self.interpolation: str | None = interpolation
        self.padding_mode: str | None = padding_mode
        self.data_keys: list[str] | None = data_keys

        if data_keys is not None:
            for key in data_keys:
                if key not in _KNOWN_DATA_KEYS:
                    warnings.warn(
                        f"Unknown data_key {key!r}; it will be passed through unchanged. "
                        f"Known keys: {sorted(_KNOWN_DATA_KEYS)}",
                        UserWarning,
                        stacklevel=2,
                    )

        if reorder not in (ReorderPolicy.NONE, ReorderPolicy.POINTWISE):
            msg = f"ReorderPolicy.{reorder.name} not yet supported"
            raise NotImplementedError(msg)

        self._adapter: TransformAdapter | None
        self._segments: list[object]

        if not transforms:
            self._adapter = None
            self._segments = []
        else:
            backend = detect_backend(transforms)
            if backend == Backend.KORNIA:
                from fuse_augmentations.adapters._kornia import KorniaAdapter

                self._adapter = KorniaAdapter()
            else:
                msg = f"Backend '{backend.value}' not yet supported in v0.1; only kornia is implemented"
                raise NotImplementedError(msg)
            if reorder == ReorderPolicy.POINTWISE:
                transforms = reorder_pointwise(transforms, self._adapter)
            self._segments = build_segments(transforms, self._adapter, interpolation, padding_mode)

        self._last_transform_matrix: Tensor | None = None

    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply the augmentation pipeline to an image batch and optional auxiliary targets.

        When ``data_keys`` is ``None`` (default), accepts a single image tensor
        and returns a single tensor (backward-compatible).

        When ``data_keys`` is provided, accepts positional arguments matching
        the key list. The ``"input"`` key corresponds to the image; other keys
        (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``) are
        routed through segments as auxiliary targets. Returns a tuple in
        ``data_keys`` order. If ``data_keys`` has a single entry, the output
        is unwrapped to a single tensor.

        Args:
            *args: Positional tensors. First tensor is always the image
                ``(B, C, H, W)``. Additional tensors correspond to
                ``data_keys[1:]``.

        Returns:
            Single tensor when ``data_keys`` is ``None`` or has one entry;
            tuple of tensors in ``data_keys`` order otherwise.

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
            aux_targets = {}
            for key, val in zip(self.data_keys[1:], args[1:], strict=True):
                aux_targets[key] = val

        for seg in self._segments:
            if isinstance(seg, FusedAffineSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result  # type: ignore[misc]
                else:
                    image = result  # type: ignore[assignment]
                self._last_transform_matrix = seg.last_matrix
            elif isinstance(seg, ExactSegment):
                result = seg(image, aux_targets)
                if aux_targets is not None:
                    image, aux_targets = result  # type: ignore[misc]
                else:
                    image = result  # type: ignore[assignment]
            else:
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
        for key in self.data_keys:
            if key == self.data_keys[0]:
                result_list.append(image)
            else:
                result_list.append(_aux[key])
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
            if isinstance(seg, FusedAffineSegment):
                # n transforms fused → 1 grid_sample, saving n-1 passes.
                # A single-transform FusedAffineSegment saves nothing (n-1 = 0).
                n = len(seg.transforms)
                if n > 1:
                    total += n - 1
            elif isinstance(seg, ExactSegment):
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
            if isinstance(seg, FusedAffineSegment):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"fused({', '.join(names)})")
            elif isinstance(seg, ExactSegment):
                names = [type(t).__name__ for t in seg.transforms]
                parts.append(f"exact({', '.join(names)})")
            else:
                parts.append(f"passthrough({type(seg).__name__})")
        return " \u2192 ".join(parts) if parts else "empty"


# Short alias for convenience; FusedCompose is the canonical name
Compose = FusedCompose
AugmentationSequential = FusedCompose
