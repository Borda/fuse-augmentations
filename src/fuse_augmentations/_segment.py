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

    Applies transforms as tensor index ops (``tensor.flip``) instead of
    ``grid_sample``, introducing zero interpolation error. Uses ``torch.where``
    for per-sample probability masking.

    Args:
        transforms: List of GEOMETRIC_EXACT transform objects (flips).
        adapter: A ``TransformAdapter`` providing ``exact_flip_dims`` (duck-typed).
    """

    def __init__(self, transforms: list[object], adapter: TransformAdapter) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter

    @property
    def last_matrix(self) -> Tensor | None:
        """Return ``None`` always (ExactSegment does not compute a matrix)."""
        return None

    def forward(self, image: Tensor) -> Tensor:
        """Apply flip transforms losslessly via tensor.flip with per-sample masking.

        Args:
            image: ``(B, C, H, W)`` float input tensor.

        Returns:
            ``(B, C, H, W)`` output tensor with flips applied per-sample.
        """
        bsz = image.shape[0]
        device = image.device

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            active = torch.rand(bsz, device=device) < prob

            flip_dims = self.adapter.exact_flip_dims(tfm)  # type: ignore[attr-defined]
            flipped = image.flip(dims=flip_dims)
            image = torch.where(active[:, None, None, None], flipped, image)

        return image


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

    def forward(self, image: Tensor) -> Tensor:
        """Apply the fused affine transform chain via a single grid_sample call.

        Args:
            image: ``(B, C, H, W)`` float input tensor.

        Returns:
            ``(B, C, H, W)`` warped output tensor.
        """
        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        eye = torch.eye(3, device=device, dtype=dtype)
        acc = eye.unsqueeze(0).expand(bsz, -1, -1).clone()

        input_shape = (bsz, n_ch, height, width)

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
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
        return F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )


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
        transforms: List of transform objects.
        adapter: A ``TransformAdapter`` for category lookup.

    Returns:
        New list with the same transforms, possibly reordered so that
        POINTWISE ops sit after geometric runs within each barrier-bounded
        stretch.
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
    """Split a transform list into fused affine segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric
    transforms (``GEOMETRIC_INTERP`` or ``GEOMETRIC_EXACT``) into a single
    :class:`fuse_augmentations._segment.FusedAffineSegment`.  Any ``SPATIAL_KERNEL`` or ``POINTWISE``
    transform breaks the current geometric group and is returned as-is.

    Args:
        transforms: List of transform objects.
        adapter: A ``TransformAdapter`` for category lookup and matrix building.
        interpolation: Optional interpolation mode override for fused segments.
        padding_mode: Optional padding mode override for fused segments.

    Returns:
        Flat list where each element is either a
        :class:`fuse_augmentations._segment.FusedAffineSegment`
        (for fused geometric runs) or the original transform object
        (for passthrough).
    """
    fusible = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT}

    segments: list[object] = []
    current_geo: list[object] = []

    def _flush_geo() -> None:
        if current_geo:
            segments.append(
                FusedAffineSegment(
                    list(current_geo),
                    adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                )
            )
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
