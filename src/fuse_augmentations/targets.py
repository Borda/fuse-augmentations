"""Auxiliary target transform helpers for fused augmentation pipelines.

Pure mathematical functions that transform masks, bounding boxes, and keypoints
using precomputed affine matrices or grids from the fused pipeline.

These helpers are called internally by :class:`~fuse_augmentations.affine.segment.FusedAffineSegment` and
:class:`~fuse_augmentations.affine.segment.ExactAffineSegment` when
``data_keys`` includes auxiliary targets. They are also exported as public API
for callers that want to apply the same math outside of the pipeline
(e.g. to transform a stored transform matrix after the fact).

All functions are stateless and operate on PyTorch tensors with a leading batch
dimension ``batch_size``. Nearest-neighbour mask sampling preserves the historical
non-differentiable behavior; bilinear sampling is available for float soft masks.
The other three functions are differentiable.

Examples:
    ```pycon
    >>> import torch
    >>> from fuse_augmentations.targets import transform_keypoints
    >>> keypoints = torch.tensor([[[10.0, 20.0]]])  # (batch_size=1, num_points=1, 2)
    >>> matrix = torch.eye(3).unsqueeze(0)           # identity (1, 3, 3)
    >>> out = transform_keypoints(keypoints, matrix)
    >>> torch.allclose(out, keypoints)
    True

    ```

"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fuse_augmentations.types import MaskInterpolationStr


def transform_mask(mask: Tensor, grid: Tensor, mode: MaskInterpolationStr = "nearest") -> Tensor:
    """Apply a precomputed affine grid to a segmentation mask.

    The default ``mode='nearest'`` preserves integer class labels without
    fractional mixing and retains the historical no-gradient behavior. Use
    ``mode='bilinear'`` for differentiable float soft masks; labels may mix at
    boundaries. Out-of-bounds samples are filled with 0 via
    ``padding_mode='zeros'``.

    Args:
        mask: Segmentation mask. Shape ``(batch_size, channels, height, width)``, typically ``channels=1``.
            dtype: Any floating or integer dtype. Integer masks are
            automatically cast to a floating dtype for ``grid_sample`` and cast
            back to the original dtype afterward. Value range: integer class
            indices (e.g. 0, 1, 2, …). Channel convention: channel-first
            (PyTorch).
        grid: Sampling grid from ``torch.nn.functional.affine_grid``.
            Shape ``(batch_size, height, width, 2)``. Any floating dtype (``float16``,
            ``float32``, ``float64``) is accepted; integer masks are
            cast to ``float32`` internally regardless of the grid
            dtype to avoid fp16/bf16 rounding while keeping memory
            usage and bandwidth lower than ``float64``. Note: ``float32``
            exactly represents integer class IDs up to ``2**24 - 1``
            (16777215); larger integer IDs may be rounded.
            Coordinates in normalised ``[-1, 1]`` space with ``align_corners=True``.
        mode: Mask sampling mode, either ``"nearest"`` (default) or
            ``"bilinear"``. Bilinear mode requires a floating-point mask.

    Returns:
        Warped mask with the same shape and dtype as ``mask``.

    Examples:
        Identity grid leaves the mask unchanged:

        ```pycon
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

        ```

    """
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"unsupported mask interpolation mode {mode!r}; expected 'nearest' or 'bilinear'")
    if mode == "bilinear" and not mask.is_floating_point():
        raise TypeError("bilinear mask interpolation requires a floating-point mask")

    needs_cast_back = not mask.is_floating_point()
    sample_mask = mask
    sample_grid = grid

    if needs_cast_back:
        # Integer masks must not be sampled through fp16/bf16 in mixed precision,
        # otherwise class IDs can be rounded before being cast back. ``float32``
        # is sufficient to preserve typical class ID ranges while avoiding the
        # memory and bandwidth overhead of ``float64``.
        sample_mask = mask.to(dtype=torch.float32)
        sample_grid = grid.to(dtype=torch.float32)
    elif sample_grid.dtype != sample_mask.dtype:
        # The image warp may use an opt-in low-precision grid while a floating
        # auxiliary mask intentionally stays in its caller-provided dtype.
        sample_grid = grid.to(dtype=sample_mask.dtype)
    if mode == "nearest":
        # Keep the default path exactly as before: nearest sampling is detached.
        with torch.no_grad():
            sampled = F.grid_sample(
                sample_mask,
                sample_grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
    else:
        sampled = F.grid_sample(
            sample_mask,
            sample_grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )
    if needs_cast_back:
        return sampled.to(dtype=mask.dtype)
    return sampled


def transform_bbox_xyxy(boxes: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_boxes, 4)`` xyxy boxes by a ``(batch_size, 3, 3)`` forward homography.

    Computes all four corners of each box, transforms them through the forward
    matrix using homogeneous multiplication, then returns the axis-aligned
    bounding box (AABB) that tightly wraps the transformed corners.

    The AABB wrapping step means output boxes are always axis-aligned and may be
    larger than the true rotated box. This is the standard trade-off for box
    transforms that must remain in xyxy format.

    Args:
        boxes: Bounding boxes in xyxy format. Shape ``(batch_size, num_boxes, 4)``,
            columns ``[x1, y1, x2, y2]`` in pixel coordinates, dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in pixel
            coordinates. Shape ``(batch_size, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed AABB boxes. Shape ``(batch_size, num_boxes, 4)``, xyxy format.

    Examples:
        Identity matrix leaves boxes unchanged:

        ```pycon
        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 50.0, 80.0]]])  # (batch_size=1, num_boxes=1, 4)
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xyxy(boxes, mtx_identity)
        >>> torch.allclose(out, boxes)
        True

        ```

    """
    box_x1 = boxes[..., 0]  # (batch_size, num_boxes)
    box_y1 = boxes[..., 1]
    box_x2 = boxes[..., 2]
    box_y2 = boxes[..., 3]

    # Build all 4 corners: (B, N, 3, 4) homogeneous [x, y, 1]
    ones = torch.ones_like(box_x1)
    corners_x = torch.stack([box_x1, box_x2, box_x2, box_x1], dim=-1)  # (B, N, 4)
    corners_y = torch.stack([box_y1, box_y1, box_y2, box_y2], dim=-1)
    corners_h = torch.stack(
        [corners_x, corners_y, ones.unsqueeze(-1).expand_as(corners_x)],
        dim=-2,
    )  # (B, N, 3, 4)

    # mtx_forward: (B, 3, 3) -> (B, 1, 3, 3) for broadcasting with (B, N, 3, 4)
    mtx_unsqueezed = mtx_forward.unsqueeze(1)  # (B, 1, 3, 3)
    transformed = mtx_unsqueezed @ corners_h  # (B, N, 3, 4)

    # Perspective division (for affine, homogeneous_w_raw=1 so this is a no-op)
    homogeneous_w_raw = transformed[:, :, 2, :]  # (B, N, 4) — homogeneous w
    # Guard against zero or extremely small |w| to avoid inf/NaN from division.
    # Use finfo.eps (not finfo.tiny): dividing by tiny overflows float32 to inf,
    # so a tiny-based clamp does not actually prevent non-finite outputs.
    eps = torch.finfo(homogeneous_w_raw.dtype).eps
    small_mask = homogeneous_w_raw.abs() < eps
    sign_val = torch.sign(homogeneous_w_raw)
    sign_val = torch.where(sign_val == 0, torch.ones_like(sign_val), sign_val)
    safe_homogeneous_w = torch.where(small_mask, eps * sign_val, homogeneous_w_raw)
    transformed_x = transformed[:, :, 0, :] / safe_homogeneous_w
    transformed_y = transformed[:, :, 1, :] / safe_homogeneous_w

    new_x1 = transformed_x.min(dim=-1).values
    new_y1 = transformed_y.min(dim=-1).values
    new_x2 = transformed_x.max(dim=-1).values
    new_y2 = transformed_y.max(dim=-1).values

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)


def transform_bbox_xywh(boxes: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_boxes, 4)`` xywh boxes by a ``(batch_size, 3, 3)`` forward homography.

    Converts boxes from ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``, delegates to
    :func:`transform_bbox_xyxy` (4-corner transform + AABB), then converts back to
    ``[x, y, w, h]``.

    The output ``width`` and ``height`` reflect the AABB after rotation, so they will be
    larger than the input for non-axis-aligned transforms.

    Args:
        boxes: Bounding boxes in xywh format. Shape ``(batch_size, num_boxes, 4)``,
            columns ``[x, y, w, h]`` where ``(x, y)`` is the top-left corner,
            dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in pixel
            coordinates. Shape ``(batch_size, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed boxes in xywh format. Shape ``(batch_size, num_boxes, 4)``.

    Examples:
        Identity matrix leaves boxes unchanged:

        ```pycon
        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 20.0, 40.0, 60.0]]])  # x, y, w, h
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_bbox_xywh(boxes, mtx_identity)
        >>> torch.allclose(out, boxes)
        True

        ```

    """
    box_left, box_top, box_width, box_height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = torch.stack([box_left, box_top, box_left + box_width, box_top + box_height], dim=-1)
    xyxy_out = transform_bbox_xyxy(xyxy, mtx_forward)
    box_x1, box_y1, box_x2, box_y2 = (
        xyxy_out[..., 0],
        xyxy_out[..., 1],
        xyxy_out[..., 2],
        xyxy_out[..., 3],
    )
    return torch.stack([box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1], dim=-1)


def transform_keypoints(keypoints: Tensor, mtx_forward: Tensor) -> Tensor:
    """Transform ``(batch_size, num_points, 2)`` keypoints by a ``(batch_size, 3, 3)`` forward homography.

    Converts each keypoint to homogeneous coordinates ``[x, y, 1]``, multiplies by
    the forward matrix, and returns the first two components of the result per point::

        keypoints'[batch_size, num_points] = (mtx_forward[batch_size] @ [x, y, 1]^T)[:2]

    Unlike bounding boxes, keypoints are transformed exactly (no AABB widening).
    The operation is differentiable with respect to both ``keypoints`` and ``mtx_forward``.

    Args:
        keypoints: Keypoints in pixel coordinates. Shape ``(batch_size, num_points, 2)``,
            columns ``[coord_x, coord_y]``, dtype ``float32``.
        mtx_forward: Forward (not inverse) affine or projective matrix in pixel
            coordinates. Shape ``(batch_size, 3, 3)``, dtype ``float32``.

    Returns:
        Transformed keypoints. Shape ``(batch_size, num_points, 2)``.

    Examples:
        Identity matrix leaves keypoints unchanged:

        ```pycon
        >>> import torch
        >>> keypoints = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (batch_size=1, num_points=2, 2)
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> out = transform_keypoints(keypoints, mtx_identity)
        >>> torch.allclose(out, keypoints)
        True

        ```

    """
    batch_size, num_kps, _ = keypoints.shape
    ones = torch.ones(batch_size, num_kps, 1, device=keypoints.device, dtype=keypoints.dtype)
    keypoints_h = torch.cat([keypoints, ones], dim=-1)  # (batch_size, num_points, 3)

    # mtx_forward: (B, 3, 3); keypoints_h: (B, num_points, 3) -> (B, 3, num_points) for matmul
    transformed = mtx_forward @ keypoints_h.transpose(1, 2)  # (batch_size, 3, num_points)
    # Perspective division (for affine, homogeneous_w=1 so this is a no-op).
    # Clamp homogeneous_w away from 0 to avoid Inf/NaN for degenerate homographies.
    homogeneous_w = transformed[:, 2:3, :]  # (batch_size, 1, num_points)
    eps = torch.finfo(keypoints.dtype).eps
    abs_homogeneous_w = homogeneous_w.abs()
    sign_homogeneous_w = torch.sign(homogeneous_w)
    # Ensure we have a non-zero sign so clamped values keep a consistent direction.
    sign_homogeneous_w = torch.where(sign_homogeneous_w == 0, torch.ones_like(sign_homogeneous_w), sign_homogeneous_w)
    safe_homogeneous_w = torch.where(abs_homogeneous_w < eps, sign_homogeneous_w * eps, homogeneous_w)
    return (transformed[:, :2, :] / safe_homogeneous_w).transpose(1, 2)  # (B, N, 2)
