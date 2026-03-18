"""Fused affine segment -- vectorised matrix composition and single grid_sample pass.

``FusedAffineSegment`` accumulates per-sample affine matrices for an entire chain
of geometric transforms, inverts the composed matrix once, and executes a single
``grid_sample`` call. No intermediate image warps are performed.

Example:
    >>> import torch
    >>> import kornia.augmentation as K
    >>> from fuse_augmentations._segment import FusedAffineSegment
    >>> from fuse_augmentations.adapters._kornia import KorniaAdapter
    >>> t = K.RandomHorizontalFlip(p=1.0)
    >>> seg = FusedAffineSegment([t], KorniaAdapter())
    >>> out = seg(torch.zeros(1, 3, 8, 8))
    >>> out.shape
    torch.Size([1, 3, 8, 8])

"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from fuse_augmentations._matrix import inv3x3, matmul3x3, normalize_matrix
from fuse_augmentations._types import TransformAdapter, TransformCategory


class ExactSegment(nn.Module):
    """Lossless segment for GEOMETRIC_EXACT-only chains (HFlip, VFlip).

    Used when a run of consecutive geometric transforms consists entirely of
    ``GEOMETRIC_EXACT`` operations (currently: horizontal and vertical flips).
    Applies each transform as a tensor index operation (``tensor.flip``) instead
    of ``grid_sample``, introducing zero interpolation error.

    Per-sample probability masking is implemented with ``torch.where``:
    for each transform, a boolean mask of shape ``(B,)`` is drawn from the
    transform's ``p`` attribute, and only samples whose mask is ``True`` receive
    the flip — the others keep their original values unchanged.

    Args:
        transforms: List of ``GEOMETRIC_EXACT`` transform objects (flips only in v0.2).
        adapter: A ``TransformAdapter`` providing an ``exact_flip_dims`` method
            (duck-typed; required for ``ExactSegment`` use).

    """

    def __init__(self, transforms: list[object], adapter: TransformAdapter) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter

    @property
    def last_matrix(self) -> Tensor | None:
        """Return ``None`` always (ExactSegment does not compute a matrix)."""
        return None

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply flip transforms losslessly via tensor.flip with per-sample masking.

        For each transform, draws a per-sample boolean mask from the transform's
        ``p`` probability, flips the full batch, then selects flipped vs original
        values per sample using ``torch.where``.

        Args:
            image: Input image batch. Shape: ``(B, C, H, W)``, dtype: float32.
                Value range and channel convention follow the calling pipeline.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        bsz = image.shape[0]
        _height, width = image.shape[2], image.shape[3]
        device = image.device

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = getattr(tfm, "same_on_batch", False)
            if same_on_batch:
                # Single Bernoulli draw shared across the entire batch.
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.expand(bsz)
            else:
                # Independent Bernoulli draw per sample.
                active = torch.rand(bsz, device=device) < prob

            flip_dims = self.adapter.exact_flip_dims(tfm)
            flipped = image.flip(dims=flip_dims)
            image = torch.where(active[:, None, None, None], flipped, image)

            # Transform auxiliary targets with the same per-sample active mask
            if aux_targets:
                is_hflip = 3 in flip_dims
                is_vflip = 2 in flip_dims
                for key in list(aux_targets.keys()):
                    val = aux_targets[key]
                    if key == "mask":
                        flipped_val = val.flip(dims=flip_dims)
                        aux_targets[key] = torch.where(active[:, None, None, None], flipped_val, val)
                    elif key == "bbox_xyxy":
                        aux_targets[key] = _flip_bbox_xyxy(val, active, is_hflip, is_vflip, _height, width)
                    elif key == "bbox_xywh":
                        # Convert xywh -> xyxy, flip, convert back
                        xyxy = _xywh_to_xyxy(val)
                        xyxy = _flip_bbox_xyxy(xyxy, active, is_hflip, is_vflip, _height, width)
                        aux_targets[key] = _xyxy_to_xywh(xyxy)
                    elif key == "keypoints":
                        aux_targets[key] = _flip_keypoints(val, active, is_hflip, is_vflip, _height, width)

        if not _has_aux:
            return image
        return image, aux_targets


class FusedAffineSegment(nn.Module):
    """Fused affine segment that composes geometric transforms into one grid_sample call.

    Accumulates per-sample ``(B, 3, 3)`` forward affine matrices for every
    transform in the segment, inverts the composed matrix once, and applies
    a single ``grid_sample`` warp. All operations are vectorised over the
    batch dimension -- no Python loop per sample.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to
            canonical parameters and matrices.
        interpolation: Optional interpolation mode override
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override
            (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: str | None = None,
        padding_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The detached, cloned composed matrix, or ``None`` before the
            first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused affine transform chain via a single grid_sample call.

        Args:
            image: ``(B, C, H, W)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        eye = torch.eye(3, device=device, dtype=dtype)
        acc = eye.unsqueeze(0).expand(bsz, -1, -1).clone()

        input_shape = (bsz, n_ch, height, width)

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = getattr(tfm, "same_on_batch", False)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.expand(bsz)
            else:
                active = torch.rand(bsz, device=device) < prob

            params = self.adapter.sample_params(tfm, input_shape, device)
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)

            # Expand to batch if adapter returned (1, 3, 3)
            if mtx_i.shape[0] == 1 and bsz > 1:
                mtx_i = mtx_i.expand(bsz, -1, -1)

            # Ensure adapter output is on the same device and dtype as the image
            mtx_i = mtx_i.to(device=device, dtype=dtype)

            mtx_i = torch.where(
                active[:, None, None],
                mtx_i,
                eye.unsqueeze(0).expand(bsz, -1, -1),
            )
            acc = matmul3x3(mtx_i, acc)

        self._last_matrix = acc.detach().clone()

        mtx_inv = inv3x3(acc)
        mtx_norm = normalize_matrix(mtx_inv, height, width)

        grid = F.affine_grid(mtx_norm[:, :2, :], [bsz, n_ch, height, width], align_corners=True)
        image = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            from fuse_augmentations._targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_keypoints,
                transform_mask,
            )

            for key in list(aux_targets.keys()):
                val = aux_targets[key]
                if key == "mask":
                    aux_targets[key] = transform_mask(val, grid)
                elif key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(val, acc)
                elif key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(val, acc)
                elif key == "keypoints":
                    aux_targets[key] = transform_keypoints(val, acc)

        if not _has_aux:
            return image
        return image, aux_targets


def reorder_pointwise(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[object]:
    """Reorder transforms so POINTWISE ops are pushed after geometric chains.

    Walks the transform list left to right.  Within each stretch between
    ``SPATIAL_KERNEL`` barriers, geometric ops (``GEOMETRIC_INTERP`` and
    ``GEOMETRIC_EXACT``) are kept in order, while ``POINTWISE`` ops are
    deferred and flushed after the geometric group.  ``POINTWISE`` ops
    never move across a ``SPATIAL_KERNEL`` barrier.

    Args:
        transforms: List of transform objects to reorder.
        adapter: A ``TransformAdapter`` used for category lookup on each transform.

    Returns:
        New list containing the same transforms, possibly reordered so that
        ``POINTWISE`` ops sit after geometric runs within each
        barrier-bounded stretch.

    Example:
        Given a pipeline ``[Rotate, Brightness, HFlip]`` where ``Brightness``
        is ``POINTWISE`` and ``Rotate`` / ``HFlip`` are geometric, the
        ``Brightness`` is pushed after the geometric group:

        Input order:  ``[Rotate, Brightness, HFlip]``
        Output order: ``[Rotate, HFlip, Brightness]``

        Using stub objects (the KorniaAdapter registry does not include any
        POINTWISE transforms in v0.2):

    >>> from fuse_augmentations._segment import reorder_pointwise
    >>> from fuse_augmentations._types import TransformCategory
    >>> class _StubAdapter:
    ...     def category(self, t):
    ...         return t._cat
    ...
    >>> class _T:
    ...     def __init__(self, cat):
    ...         self._cat = cat
    ...
    >>> adapter = _StubAdapter()
    >>> geo = _T(TransformCategory.GEOMETRIC_INTERP)
    >>> pw  = _T(TransformCategory.POINTWISE)
    >>> result = reorder_pointwise([geo, pw, geo], adapter)
    >>> [t._cat.name for t in result]
    ['GEOMETRIC_INTERP', 'GEOMETRIC_INTERP', 'POINTWISE']

    """
    geometric = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT}

    result: list[object] = []
    geo_buf: list[object] = []
    pw_buf: list[object] = []

    def _flush() -> None:
        result.extend(geo_buf)
        result.extend(pw_buf)
        geo_buf.clear()
        pw_buf.clear()

    for tfm in transforms:
        cat = adapter.category(tfm)
        if cat == TransformCategory.POINTWISE:
            pw_buf.append(tfm)
        elif cat in geometric:
            geo_buf.append(tfm)
        else:
            # SPATIAL_KERNEL barrier: flush current stretch, then emit the barrier
            _flush()
            result.append(tfm)

    _flush()
    return result


def build_segments(
    transforms: list[object],
    adapter: TransformAdapter,
    interpolation: str | None = None,
    padding_mode: str | None = None,
) -> list[object]:
    """Split a transform list into fused segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric
    transforms (``GEOMETRIC_INTERP`` or ``GEOMETRIC_EXACT``) into a single
    segment.  Any ``SPATIAL_KERNEL`` or ``POINTWISE`` transform breaks the
    current geometric group and is returned as-is.

    After grouping, each accumulated geometric run is classified:

    - **EXACT-only** — if the run contains *only* ``GEOMETRIC_EXACT`` ops
      (e.g. HFlip, VFlip), it becomes an :class:`ExactSegment` that uses
      ``tensor.flip`` with zero interpolation error.
    - **Mixed / INTERP** — if any op in the run is ``GEOMETRIC_INTERP``, the
      whole run becomes a :class:`FusedAffineSegment` that composes matrices
      and applies one ``grid_sample`` call.

    When ``ReorderPolicy.POINTWISE`` is active in
    :class:`~fuse_augmentations._compose.FusedCompose`, ``reorder_pointwise``
    is called first to bubble pointwise ops out of geometric chains, and
    ``build_segments`` then classifies the reordered list.

    Args:
        transforms: List of transform objects (already reordered if a reorder policy applies).
        adapter: A ``TransformAdapter`` for category lookup and matrix building.
        interpolation: Interpolation mode override forwarded to each :class:`FusedAffineSegment`
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
        padding_mode: Padding mode override forwarded to each :class:`FusedAffineSegment`
            (``"zeros"``, ``"border"``, ``"reflection"``).

    Returns:
        Flat list where each element is a :class:`FusedAffineSegment`
        (mixed/INTERP geometric run), an :class:`ExactSegment`
        (EXACT-only geometric run), or the original transform object
        (passthrough for ``SPATIAL_KERNEL`` and ``POINTWISE`` transforms).

    """
    fusible = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT}

    segments: list[object] = []
    current_geo: list[object] = []

    def _flush_geo() -> None:
        if current_geo:
            has_interp = any(adapter.category(t) == TransformCategory.GEOMETRIC_INTERP for t in current_geo)
            if has_interp:
                segments.append(
                    FusedAffineSegment(
                        list(current_geo),
                        adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                    )
                )
            else:
                segments.append(ExactSegment(list(current_geo), adapter))
            current_geo.clear()

    for tfm in transforms:
        cat = adapter.category(tfm)
        if cat in fusible:
            current_geo.append(tfm)
        else:
            _flush_geo()
            segments.append(tfm)

    _flush_geo()
    return segments


# ---------------------------------------------------------------------------
# Private helpers for ExactSegment auxiliary-target flipping
# ---------------------------------------------------------------------------


def _flip_bbox_xyxy(
    boxes: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip bounding boxes (B, N, 4) xyxy format using direct coordinate arithmetic.

    HFlip: ``x' = W - 1 - x``, swap x1/x2.
    VFlip: ``y' = H - 1 - y``, swap y1/y2.

    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    if is_hflip:
        new_x1 = (width - 1) - x2
        new_x2 = (width - 1) - x1
        x1, x2 = new_x1, new_x2

    if is_vflip:
        new_y1 = (height - 1) - y2
        new_y2 = (height - 1) - y1
        y1, y2 = new_y1, new_y2

    flipped = torch.stack([x1, y1, x2, y2], dim=-1)

    # active shape: (B,) -> (B, 1, 1) for broadcasting with (B, N, 4)
    mask = active[:, None, None]
    return torch.where(mask, flipped, boxes)


def _flip_keypoints(
    kps: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip keypoints (B, N, 2) using direct coordinate arithmetic.

    HFlip: ``x' = W - 1 - x``.
    VFlip: ``y' = H - 1 - y``.

    """
    flipped = kps.clone()
    if is_hflip:
        flipped[..., 0] = (width - 1) - kps[..., 0]
    if is_vflip:
        flipped[..., 1] = (height - 1) - kps[..., 1]

    mask = active[:, None, None]
    return torch.where(mask, flipped, kps)


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (B, N, 4) boxes from xywh to xyxy format."""
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert (B, N, 4) boxes from xyxy to xywh format."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
