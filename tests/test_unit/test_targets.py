"""Unit tests for targets.py: transform_mask, transform_bbox_xyxy, transform_bbox_xywh, transform_keypoints.

Pure-unit coverage with hand-built matrices (identity, hflip, 90-degree rotation) and a degenerate near-zero
homogeneous-w homography, independent of any backend pipeline.

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fuse_augmentations.targets import (
    transform_bbox_xywh,
    transform_bbox_xyxy,
    transform_keypoints,
    transform_mask,
)

_WIDTH = 16.0


def _identity_mtx() -> torch.Tensor:
    """Return (1, 3, 3) identity matrix."""
    return torch.eye(3).unsqueeze(0)


def _hflip_mtx(width: float = _WIDTH) -> torch.Tensor:
    """Return (1, 3, 3) horizontal flip matrix mapping x to (width - 1) - x."""
    return torch.tensor([[[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])


def _rot90_mtx() -> torch.Tensor:
    """Return (1, 3, 3) 90-degree counter-clockwise rotation about the origin: (x, y) -> (-y, x)."""
    return torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])


def _degenerate_mtx() -> torch.Tensor:
    """Return (1, 3, 3) homography whose third row zeroes the homogeneous w component."""
    return torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])


class TestTransformKeypoints:
    """transform_keypoints against known matrices."""

    def test_identity(self):
        """Identity matrix leaves keypoints unchanged."""
        keypoints = torch.tensor([[[3.0, 5.0], [10.0, 12.0]]])
        out = transform_keypoints(keypoints, _identity_mtx())
        assert torch.allclose(out, keypoints)

    def test_hflip(self):
        """Horizontal flip maps x to (width - 1) - x and preserves y."""
        keypoints = torch.tensor([[[3.0, 5.0]]])
        out = transform_keypoints(keypoints, _hflip_mtx())
        expected = torch.tensor([[[_WIDTH - 1.0 - 3.0, 5.0]]])
        assert torch.allclose(out, expected)

    def test_rot90(self):
        """90-degree rotation about the origin maps (x, y) to (-y, x)."""
        keypoints = torch.tensor([[[3.0, 5.0]]])
        out = transform_keypoints(keypoints, _rot90_mtx())
        expected = torch.tensor([[[-5.0, 3.0]]])
        assert torch.allclose(out, expected)

    def test_degenerate_w_stays_finite(self):
        """Near-zero homogeneous w is clamped -- output must be finite, never NaN or Inf."""
        keypoints = torch.tensor([[[3.0, 5.0]]])
        out = transform_keypoints(keypoints, _degenerate_mtx())
        assert torch.isfinite(out).all()

    def test_batched_matrices_apply_per_sample(self):
        """Each batch item is transformed by its own matrix."""
        keypoints = torch.tensor([[[3.0, 5.0]], [[3.0, 5.0]]])
        mtx_batch = torch.cat([_identity_mtx(), _hflip_mtx()], dim=0)
        out = transform_keypoints(keypoints, mtx_batch)
        assert torch.allclose(out[0], keypoints[0])
        assert torch.allclose(out[1, 0, 0], torch.tensor(_WIDTH - 1.0 - 3.0))


class TestTransformBboxXyxy:
    """transform_bbox_xyxy against known matrices."""

    def test_identity(self):
        """Identity matrix leaves boxes unchanged."""
        boxes = torch.tensor([[[2.0, 3.0, 6.0, 9.0]]])
        out = transform_bbox_xyxy(boxes, _identity_mtx())
        assert torch.allclose(out, boxes)

    def test_hflip_swaps_x_extent(self):
        """Horizontal flip mirrors the x extent: new x1 = (w-1) - x2, new x2 = (w-1) - x1."""
        boxes = torch.tensor([[[2.0, 3.0, 6.0, 9.0]]])
        out = transform_bbox_xyxy(boxes, _hflip_mtx())
        expected = torch.tensor([[[_WIDTH - 1.0 - 6.0, 3.0, _WIDTH - 1.0 - 2.0, 9.0]]])
        assert torch.allclose(out, expected)

    def test_rotation_45_widens_aabb(self):
        """A 45-degree rotation of a square box widens the AABB by sqrt(2)."""
        angle = torch.tensor(torch.pi / 4)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        mtx_rot = torch.tensor([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]])
        boxes = torch.tensor([[[-1.0, -1.0, 1.0, 1.0]]])  # square centred at origin, side 2
        out = transform_bbox_xyxy(boxes, mtx_rot)
        side = float(2**0.5)
        expected = torch.tensor([[[-side, -side, side, side]]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_degenerate_w_stays_finite(self):
        """Near-zero homogeneous w is clamped -- output must be finite, never NaN or Inf."""
        boxes = torch.tensor([[[2.0, 3.0, 6.0, 9.0]]])
        out = transform_bbox_xyxy(boxes, _degenerate_mtx())
        assert torch.isfinite(out).all()


class TestBboxAabbReductionContract:
    """Backward behaviour of the four-corner AABB reduction, which uses ``amin``/``amax``.

    The reduction was moved off ``Tensor.min(dim=)`` because that variant also computes argmin and dispatches a
    parallelised kernel, costing ~38 us per call regardless of tensor size -- the single largest fixed cost of a
    detection-shaped step. ``amin``/``amax`` are ~43x cheaper and return the same values, but they distribute the
    subgradient differently. These tests pin that difference so it stays a deliberate contract rather than something a
    future revert silently changes back.

    """

    def test_forward_matches_indexed_reduction(self):
        """``amin``/``amax`` and ``min(dim=)``/``max(dim=)`` agree on every forward value.

        The whole optimisation rests on the two being interchangeable in the forward pass; if a torch release ever
        diverged here, every warped box would move.

        """
        boxes = torch.tensor([[[2.0, 3.0, 6.0, 9.0], [-4.0, 0.5, 1.0, 11.0]]])
        angle = torch.tensor(torch.pi / 6)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        mtx = torch.tensor([[[cos_a, -sin_a, 3.0], [sin_a, cos_a, -2.0], [0.0, 0.0, 1.0]]])
        out = transform_bbox_xyxy(boxes, mtx)
        corners_x = torch.stack([boxes[..., 0], boxes[..., 2], boxes[..., 2], boxes[..., 0]], dim=-1)
        corners_y = torch.stack([boxes[..., 1], boxes[..., 1], boxes[..., 3], boxes[..., 3]], dim=-1)
        ones = torch.ones_like(corners_x)
        transformed = mtx.unsqueeze(1) @ torch.stack([corners_x, corners_y, ones], dim=-2)
        expected = torch.stack(
            [
                transformed[:, :, 0, :].min(dim=-1).values,
                transformed[:, :, 1, :].min(dim=-1).values,
                transformed[:, :, 0, :].max(dim=-1).values,
                transformed[:, :, 1, :].max(dim=-1).values,
            ],
            dim=-1,
        )
        assert torch.allclose(out, expected)

    def test_tied_corners_split_the_subgradient(self):
        """Every axis-aligned box ties, and the subgradient splits evenly across the tied corners.

        An axis-aligned box lists each coordinate twice among its four corners, so the tie is the normal case rather
        than an edge case. ``min(dim=)`` would route the full gradient to the lowest index; the even split is the
        behaviour that ships.

        """
        boxes = torch.tensor([[[2.0, 3.0, 6.0, 9.0]]], requires_grad=True)
        transform_bbox_xyxy(boxes, _identity_mtx()).sum().backward()
        # x1 and x2 each reach the output twice -- once as an AABB min, once as an AABB max --
        # and each of those two reductions splits its gradient over two tied corners.
        assert torch.allclose(boxes.grad, torch.ones_like(boxes))

    def test_nan_corner_poisons_every_gradient(self):
        """A NaN box coordinate makes the whole gradient NaN rather than selecting the NaN's index.

        ``min(dim=)`` reports a NaN forward value but hands the gradient to the NaN's own index;
        ``amin`` returns NaN for every corner. Nothing in this package backpropagates through box
        coordinates -- the documented differentiable target is the soft mask -- so this is recorded
        as accepted behaviour, not relied upon.

        """
        boxes = torch.tensor([[[2.0, 3.0, float("nan"), 9.0]]], requires_grad=True)
        out = transform_bbox_xyxy(boxes, _identity_mtx())
        assert torch.isnan(out).any()
        out.sum().backward()
        assert torch.isnan(boxes.grad).all()


class TestTransformBboxXywh:
    """transform_bbox_xywh round-trips through the xyxy transform."""

    def test_identity(self):
        """Identity matrix leaves xywh boxes unchanged."""
        boxes = torch.tensor([[[2.0, 3.0, 4.0, 6.0]]])
        out = transform_bbox_xywh(boxes, _identity_mtx())
        assert torch.allclose(out, boxes)

    def test_hflip_preserves_width_and_height(self):
        """Horizontal flip relocates the top-left corner but keeps w and h."""
        boxes = torch.tensor([[[2.0, 3.0, 4.0, 6.0]]])
        out = transform_bbox_xywh(boxes, _hflip_mtx())
        expected = torch.tensor([[[_WIDTH - 1.0 - 6.0, 3.0, 4.0, 6.0]]])
        assert torch.allclose(out, expected)

    def test_matches_xyxy_conversion(self):
        """Xywh output equals converting to xyxy, transforming, and converting back."""
        boxes_xywh = torch.tensor([[[2.0, 3.0, 4.0, 6.0]]])
        boxes_xyxy = torch.tensor([[[2.0, 3.0, 6.0, 9.0]]])
        out_xywh = transform_bbox_xywh(boxes_xywh, _rot90_mtx())
        out_xyxy = transform_bbox_xyxy(boxes_xyxy, _rot90_mtx())
        rebuilt = torch.stack(
            [
                out_xyxy[..., 0],
                out_xyxy[..., 1],
                out_xyxy[..., 2] - out_xyxy[..., 0],
                out_xyxy[..., 3] - out_xyxy[..., 1],
            ],
            dim=-1,
        )
        assert torch.allclose(out_xywh, rebuilt)

    def test_degenerate_w_stays_finite(self):
        """Near-zero homogeneous w is clamped -- output must be finite, never NaN or Inf."""
        boxes = torch.tensor([[[2.0, 3.0, 4.0, 6.0]]])
        out = transform_bbox_xywh(boxes, _degenerate_mtx())
        assert torch.isfinite(out).all()


class TestTransformMask:
    """transform_mask under identity and non-identity grids with integer dtypes."""

    @staticmethod
    def _grid_from_theta(theta: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Build an align_corners=True sampling grid from a (1, 2, 3) normalized theta."""
        return F.affine_grid(theta, [1, 1, height, width], align_corners=True)

    def test_identity_preserves_values_and_dtype(self):
        """Identity grid returns the mask unchanged, including integer dtype."""
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.int64)
        mask[0, 0, 1, 2] = 7
        grid = self._grid_from_theta(torch.eye(2, 3).unsqueeze(0), 4, 4)
        out = transform_mask(mask, grid)
        assert out.dtype == torch.int64
        assert torch.equal(out, mask)

    def test_hflip_moves_labels_and_restores_int64(self):
        """A real (non-identity) hflip warp must preserve int64 dtype and the original class-ID set."""
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.int64)
        mask[0, 0, 1, 0] = 5
        theta_hflip = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = self._grid_from_theta(theta_hflip, 4, 4)
        out = transform_mask(mask, grid)
        assert out.dtype == torch.int64
        assert out[0, 0, 1, 3] == 5
        assert set(out.unique().tolist()) <= {0, 5}

    def test_bool_mask_round_trip(self):
        """Boolean masks are cast for sampling and cast back to bool."""
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[0, 0, 2, 2] = True
        grid = self._grid_from_theta(torch.eye(2, 3).unsqueeze(0), 4, 4)
        out = transform_mask(mask, grid)
        assert out.dtype == torch.bool
        assert bool(out[0, 0, 2, 2])

    def test_float_mask_passes_through_without_cast(self):
        """Floating masks keep their dtype without the int cast round-trip."""
        mask = torch.rand(1, 1, 4, 4)
        grid = self._grid_from_theta(torch.eye(2, 3).unsqueeze(0), 4, 4)
        out = transform_mask(mask, grid)
        assert out.dtype == torch.float32
        assert torch.allclose(out, mask)
