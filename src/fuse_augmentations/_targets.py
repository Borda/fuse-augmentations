"""Auxiliary target transform helpers for fused augmentation pipelines.

Pure mathematical functions that transform masks, bounding boxes, and keypoints
using precomputed affine matrices or grids from the fused pipeline.

Example:
    >>> import torch
    >>> from fuse_augmentations._targets import transform_keypoints
    >>> kps = torch.tensor([[[10.0, 20.0]]])  # (1, 1, 2)
    >>> M = torch.eye(3).unsqueeze(0)          # identity (1, 3, 3)
    >>> out = transform_keypoints(kps, M)
    >>> torch.allclose(out, kps)
    True

"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def transform_mask(mask: Tensor, grid: Tensor) -> Tensor:
    """Apply a precomputed affine grid to a segmentation mask using nearest-neighbour sampling.

    Parameters
    ----------
    mask : Tensor
        Segmentation mask, shape ``(B, C, H, W)`` (typically ``C=1``).
    grid : Tensor
        Sampling grid from ``F.affine_grid``, shape ``(B, H, W, 2)``.

    Returns:
    -------
    Tensor
        Warped mask with the same shape as ``mask``, sampled with
        ``mode='nearest'`` to preserve integer class labels.

    """
    return F.grid_sample(
        mask,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )


def transform_bbox_xyxy(boxes: Tensor, M_forward: Tensor) -> Tensor:  # noqa: N803
    """Transform ``(B, N, 4)`` xyxy boxes by ``(B, 3, 3)`` forward affine matrix.

    Computes all 4 corners of each box, transforms them through the forward
    matrix, and returns the axis-aligned bounding box (AABB) of the
    transformed corners.

    Parameters
    ----------
    boxes : Tensor
        Bounding boxes in xyxy format, shape ``(B, N, 4)`` where columns are
        ``[x1, y1, x2, y2]``.
    M_forward : Tensor
        Forward affine matrix, shape ``(B, 3, 3)``.

    Returns:
    -------
    Tensor
        Transformed AABB boxes, shape ``(B, N, 4)`` in xyxy format.

    """
    x1 = boxes[..., 0]  # (B, N)
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    # Build all 4 corners: (B, N, 4, 3) homogeneous [x, y, 1]
    ones = torch.ones_like(x1)
    corners_x = torch.stack([x1, x2, x2, x1], dim=-1)  # (B, N, 4)
    corners_y = torch.stack([y1, y1, y2, y2], dim=-1)
    corners_h = torch.stack(
        [corners_x, corners_y, ones.unsqueeze(-1).expand_as(corners_x)],
        dim=-2,
    )  # (B, N, 3, 4)

    # M_forward: (B, 3, 3) -> (B, 1, 3, 3) for broadcasting with (B, N, 3, 4)
    M = M_forward.unsqueeze(1)  # (B, 1, 3, 3)  # noqa: N806
    transformed = M @ corners_h  # (B, N, 3, 4)

    tx = transformed[:, :, 0, :]  # (B, N, 4)
    ty = transformed[:, :, 1, :]

    new_x1 = tx.min(dim=-1).values
    new_y1 = ty.min(dim=-1).values
    new_x2 = tx.max(dim=-1).values
    new_y2 = ty.max(dim=-1).values

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)


def transform_bbox_xywh(boxes: Tensor, M_forward: Tensor) -> Tensor:  # noqa: N803
    """Transform ``(B, N, 4)`` xywh boxes by ``(B, 3, 3)`` forward affine matrix.

    Converts xywh to xyxy, applies :func:`transform_bbox_xyxy`, and converts
    back to xywh.

    Parameters
    ----------
    boxes : Tensor
        Bounding boxes in xywh format, shape ``(B, N, 4)`` where columns are
        ``[x, y, w, h]``.
    M_forward : Tensor
        Forward affine matrix, shape ``(B, 3, 3)``.

    Returns:
    -------
    Tensor
        Transformed boxes in xywh format, shape ``(B, N, 4)``.

    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = torch.stack([x, y, x + w, y + h], dim=-1)
    xyxy_out = transform_bbox_xyxy(xyxy, M_forward)
    x1, y1, x2, y2 = (
        xyxy_out[..., 0],
        xyxy_out[..., 1],
        xyxy_out[..., 2],
        xyxy_out[..., 3],
    )
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def transform_keypoints(kps: Tensor, M_forward: Tensor) -> Tensor:  # noqa: N803
    """Transform ``(B, N, 2)`` keypoints by ``(B, 3, 3)`` forward matrix.

    Applies affine transformation via homogeneous multiply:
    ``p'[b,n] = (M_forward[b] @ [x, y, 1]^T)[:2]``.

    Parameters
    ----------
    kps : Tensor
        Keypoints, shape ``(B, N, 2)`` where columns are ``[x, y]``.
    M_forward : Tensor
        Forward affine matrix, shape ``(B, 3, 3)``.

    Returns:
    -------
    Tensor
        Transformed keypoints, shape ``(B, N, 2)``.

    """
    B, N, _ = kps.shape  # noqa: N806
    ones = torch.ones(B, N, 1, device=kps.device, dtype=kps.dtype)
    kps_h = torch.cat([kps, ones], dim=-1)  # (B, N, 3)

    # M_forward: (B, 3, 3); kps_h: (B, N, 3) -> (B, 3, N) for matmul
    transformed = M_forward @ kps_h.transpose(1, 2)  # (B, 3, N)
    return transformed[:, :2, :].transpose(1, 2)  # (B, N, 2)
