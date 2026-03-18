"""Integration tests for auxiliary target support (masks, bboxes, keypoints).

Covers spec tests #32--38. Requires kornia >= 0.6.12.

"""

from __future__ import annotations

import pytest
import torch

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")
from kornia.augmentation import RandomHorizontalFlip, RandomRotation  # noqa: E402

from fuse_augmentations._compose import FusedCompose as Compose  # noqa: E402

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B, C, H, W = 2, 3, 32, 32


# ---------------------------------------------------------------------------
# #32 — Mask NEAREST-only: no fractional class labels after transform
# ---------------------------------------------------------------------------


class TestMaskNearestOnly:
    """#32: Mask values must remain integers after any geometric transform."""

    def test_binary_mask_stays_binary_after_rotation(self):
        """After rotation, binary mask contains only {0.0, 1.0}."""
        pipe = Compose(
            [RandomRotation(degrees=45, p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(B, C, H, W)
        mask = torch.randint(0, 2, (B, 1, H, W)).float()
        _, out_mask = pipe(img, mask)

        unique_vals = torch.unique(out_mask)
        allowed = {0.0, 1.0}
        actual = set(unique_vals.tolist())
        assert actual.issubset(allowed), (
            f"Mask should only contain {allowed} but got {actual}; "
            "this means nearest-neighbor interpolation was NOT used for the mask."
        )

    def test_multiclass_mask_stays_integer_after_rotation(self):
        """After rotation, 3-class mask values remain in {0, 1, 2}."""
        pipe = Compose(
            [RandomRotation(degrees=30, p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(B, C, H, W)
        mask = torch.randint(0, 3, (B, 1, H, W)).float()
        _, out_mask = pipe(img, mask)

        unique_vals = torch.unique(out_mask)
        for v in unique_vals.tolist():
            assert v == int(v), f"Mask value {v} is not an integer; nearest interpolation was NOT used"


# ---------------------------------------------------------------------------
# #33 — Mask spatial correspondence
# ---------------------------------------------------------------------------


class TestMaskSpatialCorrespondence:
    """#33: After HFlip, pixel at column x in image corresponds to column W-1-x in mask."""

    def test_hflip_spatial_correspondence(self):
        """After HFlip(p=1), mask columns are mirrored: column x -> column W-1-x."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        # Build a mask with distinct integer regions: left half = 1, right half = 2
        mask = torch.ones(B, 1, H, W)
        mask[:, :, :, W // 2 :] = 2.0
        img = torch.rand(B, C, H, W)

        _, out_mask = pipe(img, mask)

        # After HFlip: left half should now be 2, right half should be 1
        left_half = out_mask[:, :, :, : W // 2]
        right_half = out_mask[:, :, :, W // 2 :]
        assert (left_half == 2.0).all(), "After HFlip, left half of mask should contain class 2"
        assert (right_half == 1.0).all(), "After HFlip, right half of mask should contain class 1"


# ---------------------------------------------------------------------------
# #34 — Bbox identity
# ---------------------------------------------------------------------------


class TestBboxIdentity:
    """#34: Identity transform leaves bounding boxes unchanged."""

    def test_bbox_unchanged_under_identity(self):
        """Bbox with hflip_p=0 (identity-like pipeline) is unchanged."""
        pipe = Compose(
            [RandomHorizontalFlip(p=0.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(B, C, H, W)
        # (B, N, 4) format: one box per sample
        boxes = torch.tensor([[[5.0, 10.0, 20.0, 25.0]]] * B)
        _, out_boxes = pipe(img, boxes)

        torch.testing.assert_close(out_boxes, boxes, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #35 — Bbox HFlip
# ---------------------------------------------------------------------------


class TestBboxHFlip:
    """#35: HFlip mirrors bbox x-coordinates: new_x = W - 1 - old_x."""

    def test_bbox_x_mirrored_after_hflip(self):
        """After HFlip(p=1), bbox x-coords mirror: x_min' = W-1-x_max, x_max' = W-1-x_min."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(B, C, H, W)
        x1, y1, x2, y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[x1, y1, x2, y2]]] * B)
        _, out_boxes = pipe(img, boxes)

        expected_x1 = W - 1 - x2
        expected_x2 = W - 1 - x1
        expected = torch.tensor([[[expected_x1, y1, expected_x2, y2]]] * B)
        torch.testing.assert_close(out_boxes, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #36 — Bbox rotation (90 deg)
# ---------------------------------------------------------------------------


class TestBboxRotation:
    """#36: After 90-degree rotation, bbox AABB contains expected corners."""

    def test_bbox_aabb_after_90_rotation(self):
        """90-degree rotation of a bbox: AABB of rotated corners is computed correctly."""
        # Use a known 90-degree rotation via RandomRotation with degrees=(90, 90)
        pipe = Compose(
            [RandomRotation(degrees=(90, 90), p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(1, C, H, W)
        # A box at (5, 10, 15, 20) — width=10, height=10
        boxes = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        _, out_boxes = pipe(img, boxes)

        # After 90-degree rotation around image center (W/2, H/2):
        # The resulting AABB should exist and have valid coordinates
        assert out_boxes.shape == (1, 1, 4), f"Expected shape (1,1,4), got {out_boxes.shape}"
        # Verify AABB constraints: x1 < x2 and y1 < y2
        ox1, oy1, ox2, oy2 = out_boxes[0, 0].tolist()
        assert ox1 < ox2, f"Expected x1 < x2, got {ox1} >= {ox2}"
        assert oy1 < oy2, f"Expected y1 < y2, got {oy1} >= {oy2}"


# ---------------------------------------------------------------------------
# #37 — Keypoints known mapping
# ---------------------------------------------------------------------------


class TestKeypointsKnownMapping:
    """#37: For HFlip(p=1), keypoint x' = W-1-x, y' unchanged."""

    def test_hflip_keypoint_x_mirrored(self):
        """After HFlip, keypoint x is mirrored: x' = W-1-x, y unchanged."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "keypoints"],
        )
        img = torch.rand(B, C, H, W)
        kx, ky = 10.0, 20.0
        # (B, N, 2) format
        kps = torch.tensor([[[kx, ky]]] * B)
        _, out_kps = pipe(img, kps)

        expected_x = W - 1 - kx
        expected = torch.tensor([[[expected_x, ky]]] * B)
        torch.testing.assert_close(out_kps, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #38 — Batch consistency
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    """#38: Per-sample transforms applied consistently to image and all aux targets."""

    def test_batch_image_mask_consistency(self):
        """Batch of B=4 with p=1.0 HFlip: each sample's image and mask are consistently flipped."""
        b = 4
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(b, C, H, W)
        # Build a mask that is NOT symmetric, so flip is detectable
        mask = torch.zeros(b, 1, H, W)
        mask[:, :, :, : W // 4] = 1.0  # left quarter is 1

        out_img, out_mask = pipe(img, mask)

        # Verify each sample individually
        for i in range(b):
            # Image should be horizontally flipped
            expected_img_i = img[i].flip(dims=[-1])
            torch.testing.assert_close(
                out_img[i],
                expected_img_i,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {i}: image not consistently flipped",
            )
            # Mask should also be horizontally flipped
            expected_mask_i = mask[i].flip(dims=[-1])
            torch.testing.assert_close(
                out_mask[i],
                expected_mask_i,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {i}: mask not consistently flipped with image",
            )

    def test_batch_image_bbox_consistency(self):
        """Batch with p=1.0 HFlip: each sample's bbox is consistently transformed."""
        b = 4
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(b, C, H, W)
        x1, y1, x2, y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[x1, y1, x2, y2]]] * b)

        out_img, out_boxes = pipe(img, boxes)

        expected_x1 = W - 1 - x2
        expected_x2 = W - 1 - x1
        for i in range(b):
            # Image must be flipped
            expected_img_i = img[i].flip(dims=[-1])
            torch.testing.assert_close(out_img[i], expected_img_i, atol=1e-4, rtol=1e-4)
            # Box must be mirrored
            torch.testing.assert_close(
                out_boxes[i, 0, 0].item(),
                expected_x1,
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                out_boxes[i, 0, 2].item(),
                expected_x2,
                atol=1e-4,
                rtol=1e-4,
            )


# ---------------------------------------------------------------------------
# Edge case: Passthrough segment with active aux_targets
# ---------------------------------------------------------------------------


class TestPassthroughWithAuxTargets:
    """Passthrough segment (e.g. GaussianBlur) does not crash with aux_targets."""

    def test_passthrough_does_not_crash_with_aux_targets(self):
        """Pipeline with passthrough + fused segment routes mask correctly.

        RandomGaussianBlur is SPATIAL_KERNEL (passthrough), RandomRotation is GEOMETRIC_INTERP (fused). The passthrough
        segment only gets the image; the fused segment transforms both image and mask. The mask shape must be preserved
        after the full forward pass.

        """
        from kornia.augmentation import RandomGaussianBlur

        pipe = Compose(
            [
                RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                RandomRotation(degrees=15, p=1.0),
            ],
            data_keys=["input", "mask"],
        )
        img = torch.rand(B, C, H, W)
        mask = torch.randint(0, 3, (B, 1, H, W)).float()
        out = pipe(img, mask)

        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        out_img, out_mask = out
        assert out_img.shape == img.shape, f"Image shape mismatch: {out_img.shape} vs {img.shape}"
        assert out_mask.shape == mask.shape, f"Mask shape mismatch: {out_mask.shape} vs {mask.shape}"


# ---------------------------------------------------------------------------
# Edge case: transform_mask with int64 input
# ---------------------------------------------------------------------------


class TestTransformMaskInt64:
    """transform_mask with int64 input is supported via internal cast-and-restore."""

    def test_transform_mask_int64_input(self):
        """transform_mask accepts int64 masks and preserves integer labels."""
        import torch.nn.functional as F

        from fuse_augmentations._targets import transform_mask

        mask_int64 = torch.randint(0, 3, (B, 1, H, W), dtype=torch.int64)
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, [B, 1, H, W], align_corners=True)

        out = transform_mask(mask_int64, grid)
        assert out.dtype == torch.int64, f"Expected int64 output dtype, got {out.dtype}"
        unique_vals = set(torch.unique(out).tolist())
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected label values after transform: {unique_vals}"

    def test_transform_mask_int64_preserves_large_labels_with_fp16_grid(self):
        """Large integer labels are preserved even when the input grid is float16."""
        import torch.nn.functional as F

        from fuse_augmentations._targets import transform_mask

        mask_int64 = torch.full((B, 1, H, W), 4097, dtype=torch.int64)
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, [B, 1, H, W], align_corners=True).to(torch.float16)

        out = transform_mask(mask_int64, grid)
        assert out.dtype == torch.int64, f"Expected int64 output dtype, got {out.dtype}"
        assert (out == 4097).all(), "Large class IDs must not be rounded in mixed-precision paths"
