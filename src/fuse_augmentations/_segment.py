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
        B, C, H, W = image.shape  # noqa: N806
        device = image.device
        dtype = image.dtype

        eye = torch.eye(3, device=device, dtype=dtype)
        acc = eye.unsqueeze(0).expand(B, -1, -1).clone()

        input_shape = (B, C, H, W)

        for t in self.transforms:
            p = getattr(t, "p", 1.0)
            active = torch.rand(B, device=device) < p

            params = self.adapter.sample_params(t, input_shape, device)
            M_i = self.adapter.build_matrix(t, params, H, W)  # noqa: N806

            # Expand to batch if adapter returned (1, 3, 3)
            if M_i.shape[0] == 1 and B > 1:
                M_i = M_i.expand(B, -1, -1)  # noqa: N806

            M_i = torch.where(  # noqa: N806
                active[:, None, None],
                M_i.to(dtype=dtype),
                eye.unsqueeze(0).expand(B, -1, -1),
            )
            acc = matmul3x3(M_i, acc)

        self._last_matrix = acc.detach().clone()

        M_inv = inv3x3(acc)  # noqa: N806
        M_norm = normalize_matrix(M_inv, H, W)  # noqa: N806

        grid = F.affine_grid(M_norm[:, :2, :], [B, C, H, W], align_corners=True)
        return F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )


def build_segments(
    transforms: list[object],
    adapter: TransformAdapter,
    interpolation: str | None = None,
    padding_mode: str | None = None,
) -> list[object]:
    """Split a transform list into fused affine segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric
    transforms (``GEOMETRIC_INTERP`` or ``GEOMETRIC_EXACT``) into a single
    :class:`FusedAffineSegment`.  Any ``SPATIAL_KERNEL`` or ``POINTWISE``
    transform breaks the current geometric group and is returned as-is.

    Args:
        transforms: List of transform objects.
        adapter: A ``TransformAdapter`` for category lookup and matrix building.
        interpolation: Optional interpolation mode override for fused segments.
        padding_mode: Optional padding mode override for fused segments.

    Returns:
        Flat list where each element is either a :class:`FusedAffineSegment`
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

    for t in transforms:
        cat = adapter.category(t)
        if cat in fusible:
            current_geo.append(t)
        else:
            _flush_geo()
            segments.append(t)

    _flush_geo()
    return segments
