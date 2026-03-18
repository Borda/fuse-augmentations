"""Auxiliary target transform helpers for fused augmentation pipelines.

Pure mathematical functions that transform masks, bounding boxes, and keypoints
using precomputed affine matrices or grids from the fused pipeline.

These helpers are called internally by :class:`~fuse_augmentations._segment.FusedAffineSegment`
and :class:`~fuse_augmentations._segment.ExactSegment` when ``data_keys`` includes auxiliary
targets. They are also exported as public API for callers that want to apply the same
math outside of the pipeline (e.g. to transform a stored transform matrix after the fact).

All functions are stateless and operate on PyTorch tensors with a leading batch
dimension ``B``. No gradient is tracked through ``transform_mask`` (nearest-neighbour
sampling is not differentiable); the other three functions are differentiable.

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

    Uses ``mode='nearest'`` unconditionally so integer class labels are preserved
    without fractional mixing. Out-of-bounds samples are filled with 0 (class 0)
    via ``padding_mode='zeros'``.

    Args:
        mask: Segmentation mask. Shape ``(B, C, H, W)``, typically ``C=1``.
            dtype: Must be ``float32``. Integer class labels should be cast to
            ``float32`` before calling this function. ``F.grid_sample`` with
            ``mode='nearest'`` does not support integer dtypes across all PyTorch
            versions. Value range: integer class indices (e.g. 0, 1, 2, …).
            Channel convention: channel-first (PyTorch).
        grid: Sampling grid from ``torch.nn.functional.affine_grid``.
            Shape ``(B, H, W, 2)``, dtype ``float32``.
            Coordinates in normalised ``[-1, 1]`` space with ``align_corners=True``.

    Returns:
        Warped mask with the same shape and dtype as ``mask``.

    Example:
        Identity grid leaves the mask unchanged:

        >>> import torch
        >>> import torch.nn.functional as F
        >>> mask = torch.zeros(1, 1, 4, 4)
        >>> mask[0, 0, 1, 1] = 1
        >>> eye2 = torch.eye(2, 3).unsqueeze(0)  # identity theta (1, 2, 3)
        >>> grid = F.affine_grid(eye2, [1, 1, 4, 4], align_corners=True)
        >>> out = transform_mask(mask, grid)
        >>> out.shape
        torch.Size([1, 1, 4, 4])
        >>> bool(out[0, 0, 1, 1] == 1)
        True

    """
    return F.grid_sample(
        mask,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )


def transform_bbox_xyxy(boxes: Tensor, M_forward: Tensor) -> Tensor:  # noqa: N803
    """Transform ``(B, N, 4)`` xyxy boxes by a ``(B, 3, 3)`` forward affine matrix.

    Computes all four corners of each box, transforms them through the forward
    matrix using homogeneous multiplication, then returns the axis-aligned
    bounding box (AABB) that tightly wraps the transformed corners.

    The AABB wrapping step means output boxes are always axis-aligned and may be
    larger than the true rotated box. This is the standard trade-off for box
    transforms that must remain in xyxy format.

    Args:
        boxes: Bounding boxes in xyxy format. Shape ``(B, N, 4)``,
            columns ``[x1, y1, x2, y2]`` in pixel coordinates, dtype ``float32``.
        M_forward: Forward (not inverse) affine matrix in pixel coordinates.
            Shape ``(B, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed AABB boxes. Shape ``(B, N, 4)``, xyxy format.

    Example:
        Identity matrix leaves boxes unchanged:

        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 50.0, 80.0]]])  # (1, 1, 4)
        >>> M = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xyxy(boxes, M)
        >>> torch.allclose(out, boxes)
        True

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
    """Transform ``(B, N, 4)`` xywh boxes by a ``(B, 3, 3)`` forward affine matrix.

    Converts boxes from ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``, delegates to
    :func:`transform_bbox_xyxy` (4-corner transform + AABB), then converts back to
    ``[x, y, w, h]``.

    The output ``w`` and ``h`` reflect the AABB after rotation, so they will be
    larger than the input for non-axis-aligned transforms.

    Args:
        boxes: Bounding boxes in xywh format. Shape ``(B, N, 4)``,
            columns ``[x, y, w, h]`` where ``(x, y)`` is the top-left corner,
            dtype ``float32``.
        M_forward: Forward (not inverse) affine matrix in pixel coordinates.
            Shape ``(B, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed boxes in xywh format. Shape ``(B, N, 4)``.

    Example:
        Identity matrix leaves boxes unchanged:

        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 40.0, 60.0]]])  # x, y, w, h
        >>> M = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xywh(boxes, M)
        >>> torch.allclose(out, boxes)
        True

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
    """Transform ``(B, N, 2)`` keypoints by a ``(B, 3, 3)`` forward affine matrix.

    Converts each keypoint to homogeneous coordinates ``[x, y, 1]``, multiplies by
    the forward matrix, and returns the first two components of the result per point::

        p'[b, n] = (M_forward[b] @ [x, y, 1]^T)[:2]

    Unlike bounding boxes, keypoints are transformed exactly (no AABB widening).
    The operation is differentiable with respect to both ``kps`` and ``M_forward``.

    Args:
        kps: Keypoints in pixel coordinates. Shape ``(B, N, 2)``,
            columns ``[x, y]``, dtype ``float32``.
        M_forward: Forward (not inverse) affine matrix in pixel coordinates.
            Shape ``(B, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed keypoints. Shape ``(B, N, 2)``.

    Example:
        Identity matrix leaves keypoints unchanged:

        >>> import torch
        >>> kps = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (1, 2, 2)
        >>> M = torch.eye(3).unsqueeze(0)
        >>> out = transform_keypoints(kps, M)
        >>> torch.allclose(out, kps)
        True

    """
    B, N, _ = kps.shape  # noqa: N806
    ones = torch.ones(B, N, 1, device=kps.device, dtype=kps.dtype)
    kps_h = torch.cat([kps, ones], dim=-1)  # (B, N, 3)

    # M_forward: (B, 3, 3); kps_h: (B, N, 3) -> (B, 3, N) for matmul
    transformed = M_forward @ kps_h.transpose(1, 2)  # (B, 3, N)
    return transformed[:, :2, :].transpose(1, 2)  # (B, N, 2)
