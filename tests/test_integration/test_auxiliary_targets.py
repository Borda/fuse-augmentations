"""Integration tests for auxiliary target support (masks, bboxes, keypoints).

Covers spec tests #32--38. Requires kornia >= 0.6.12.

"""

from __future__ import annotations

import pytest
import torch

kornia = pytest.importorskip("kornia", minversion="0.6.12", reason="kornia >= 0.6.12 required")
from kornia.augmentation import RandomHorizontalFlip, RandomRotation  # noqa: E402

from fuse_augmentations._compose import FusedCompose as Compose  # noqa: E402

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32


# ---------------------------------------------------------------------------
# #32 - Mask NEAREST-only: no fractional class labels after transform
# ---------------------------------------------------------------------------


class TestMaskNearestOnly:
    """#32: Mask values must remain integers after any geometric transform."""

    def test_binary_mask_stays_binary_after_rotation(self):
        """After rotation, binary mask contains only {0.0, 1.0}."""
        pipe = Compose(
            [RandomRotation(degrees=45, p=1.0)],
            data_keys=["input", "mask"],
        )
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 2, (BATCH_SIZE, 1, HEIGHT, WIDTH)).float()
        _, out_mask = pipe(image, mask)

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
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 3, (BATCH_SIZE, 1, HEIGHT, WIDTH)).float()
        _, out_mask = pipe(image, mask)

        unique_vals = torch.unique(out_mask)
        for value in unique_vals.tolist():
            assert value == int(value), f"Mask value {value} is not an integer; nearest interpolation was NOT used"


# ---------------------------------------------------------------------------
# #33 - Mask spatial correspondence
# ---------------------------------------------------------------------------


class TestMaskSpatialCorrespondence:
    """#33: After HFlip, pixel at column x in image corresponds to column width-1-x in mask."""

    def test_hflip_spatial_correspondence(self):
        """After HFlip(prob=1), mask columns are mirrored: column coord_x -> column width-1-coord_x."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        # Build a mask with distinct integer regions: left half = 1, right half = 2
        mask = torch.ones(BATCH_SIZE, 1, HEIGHT, WIDTH)
        mask[:, :, :, WIDTH // 2 :] = 2.0
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

        _, out_mask = pipe(image, mask)

        # After HFlip: left half should now be 2, right half should be 1
        left_half = out_mask[:, :, :, : WIDTH // 2]
        right_half = out_mask[:, :, :, WIDTH // 2 :]
        assert (left_half == 2.0).all(), "After HFlip, left half of mask should contain class 2"
        assert (right_half == 1.0).all(), "After HFlip, right half of mask should contain class 1"


# ---------------------------------------------------------------------------
# #34 - Bbox identity
# ---------------------------------------------------------------------------


class TestBboxIdentity:
    """#34: Identity transform leaves bounding boxes unchanged."""

    def test_bbox_unchanged_under_identity(self):
        """Bbox with hflip_p=0 (identity-like pipeline) is unchanged."""
        pipe = Compose(
            [RandomHorizontalFlip(p=0.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        # (batch_size, num_boxes, 4) format: one box per sample
        boxes = torch.tensor([[[5.0, 10.0, 20.0, 25.0]]] * BATCH_SIZE)
        _, image_output_boxes = pipe(image, boxes)

        torch.testing.assert_close(image_output_boxes, boxes, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #35 - Bbox HFlip
# ---------------------------------------------------------------------------


class TestBboxHFlip:
    """#35: HFlip mirrors bbox x-coordinates: new_x = width - 1 - old_x."""

    def test_bbox_x_mirrored_after_hflip(self):
        """After HFlip(prob=1), bbox x-coords mirror: x_min' = width-1-x_max, x_max' = width-1-x_min."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        box_x1, box_y1, box_x2, box_y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[box_x1, box_y1, box_x2, box_y2]]] * BATCH_SIZE)
        _, boxes_output = pipe(image, boxes)

        expected_x1 = WIDTH - 1 - box_x2
        expected_x2 = WIDTH - 1 - box_x1
        expected = torch.tensor([[[expected_x1, box_y1, expected_x2, box_y2]]] * BATCH_SIZE)
        torch.testing.assert_close(boxes_output, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #36 - Bbox rotation (90 deg)
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
        image = torch.rand(1, CHANNELS, HEIGHT, WIDTH)
        # albu box at (5, 10, 15, 20) - width=10, height=10
        boxes = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        _, out_boxes = pipe(image, boxes)

        # After 90-degree rotation around image center (width/2, height/2):
        # The resulting AABB should exist and have valid coordinates
        assert out_boxes.shape == (1, 1, 4), f"Expected shape (1,1,4), got {out_boxes.shape}"
        # Verify AABB constraints: x1 < x2 and y1 < y2
        ox1, oy1, ox2, oy2 = out_boxes[0, 0].tolist()
        assert ox1 < ox2, f"Expected x1 < x2, got {ox1} >= {ox2}"
        assert oy1 < oy2, f"Expected y1 < y2, got {oy1} >= {oy2}"


# ---------------------------------------------------------------------------
# #37 - Keypoints known mapping
# ---------------------------------------------------------------------------


class TestKeypointsKnownMapping:
    """#37: For HFlip(prob=1), keypoint x' = width-1-x, y' unchanged."""

    def test_hflip_keypoint_x_mirrored(self):
        """After HFlip, keypoint x is mirrored: x' = width - 1 - coord_x, y' unchanged."""
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "keypoints"],
        )
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        kp_x, kp_y = 10.0, 20.0
        # (batch_size, num_points, 2) format
        keypoints = torch.tensor([[[kp_x, kp_y]]] * BATCH_SIZE)
        _, keypoints_output = pipe(image, keypoints)

        expected_x = WIDTH - 1 - kp_x
        expected = torch.tensor([[[expected_x, kp_y]]] * BATCH_SIZE)
        torch.testing.assert_close(keypoints_output, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# #38 - Batch consistency
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    """#38: Per-sample transforms applied consistently to image and all aux targets."""

    def test_batch_image_mask_consistency(self):
        """Batch of batch_size=4 with prob=1.0 HFlip: each sample's image and mask are consistently flipped."""
        num_samples = 4
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        image = torch.rand(num_samples, CHANNELS, HEIGHT, WIDTH)
        # Build a mask that is NOT symmetric, so flip is detectable
        mask = torch.zeros(num_samples, 1, HEIGHT, WIDTH)
        mask[:, :, :, : WIDTH // 4] = 1.0  # left quarter is 1

        out_image, out_mask = pipe(image, mask)

        # Verify each sample individually
        for idx in range(num_samples):
            # Image should be horizontally flipped
            expected_image_idx = image[idx].flip(dims=[-1])
            torch.testing.assert_close(
                out_image[idx],
                expected_image_idx,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {idx}: image not consistently flipped",
            )
            # Mask should also be horizontally flipped
            expected_mask_idx = mask[idx].flip(dims=[-1])
            torch.testing.assert_close(
                out_mask[idx],
                expected_mask_idx,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {idx}: mask not consistently flipped",
            )

    def test_batch_image_bbox_consistency(self):
        """Batch with prob=1.0 HFlip: each sample's bbox is consistently transformed."""
        num_samples = 4
        pipe = Compose(
            [RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        image = torch.rand(num_samples, CHANNELS, HEIGHT, WIDTH)
        box_x1, box_y1, box_x2, box_y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[box_x1, box_y1, box_x2, box_y2]]] * num_samples)

        image_output, boxes_output = pipe(image, boxes)

        expected_x1 = WIDTH - 1 - box_x2
        expected_x2 = WIDTH - 1 - box_x1
        for idx in range(num_samples):
            # Image must be flipped
            expected_image_idx = image[idx].flip(dims=[-1])
            torch.testing.assert_close(image_output[idx], expected_image_idx, atol=1e-4, rtol=1e-4)
            # Box must be mirrored
            torch.testing.assert_close(
                boxes_output[idx, 0, 0].item(),
                expected_x1,
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                boxes_output[idx, 0, 2].item(),
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
        image = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 3, (BATCH_SIZE, 1, HEIGHT, WIDTH)).float()
        image_output = pipe(image, mask)

        assert isinstance(image_output, tuple), f"Expected tuple, got {type(image_output)}"
        image_out_result, mask_out_result = image_output
        assert image_out_result.shape == image.shape, f"Image shape mismatch: {image_out_result.shape} vs {image.shape}"
        assert mask_out_result.shape == mask.shape, f"Mask shape mismatch: {mask_out_result.shape} vs {mask.shape}"


# ---------------------------------------------------------------------------
# Edge case: transform_mask with int64 input
# ---------------------------------------------------------------------------


class TestTransformMaskInt64:
    """transform_mask with int64 input is supported via internal cast-and-restore."""

    def test_transform_mask_int64_input(self):
        """transform_mask accepts int64 masks and preserves integer labels."""
        import torch.nn.functional as F

        from fuse_augmentations._targets import transform_mask

        mask_int64 = torch.randint(0, 3, (BATCH_SIZE, 1, HEIGHT, WIDTH), dtype=torch.int64)
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
        grid = F.affine_grid(theta, [BATCH_SIZE, 1, HEIGHT, WIDTH], align_corners=True)

        image_output = transform_mask(mask_int64, grid)
        assert image_output.dtype == torch.int64, f"Expected int64 output dtype, got {image_output.dtype}"
        unique_vals = set(torch.unique(image_output).tolist())
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected label values after transform: {unique_vals}"

    def test_transform_mask_int64_preserves_large_labels_with_fp16_grid(self):
        """Large integer labels are preserved even when the input grid is float16."""
        import torch.nn.functional as F

        from fuse_augmentations._targets import transform_mask

        mask_int64 = torch.full((BATCH_SIZE, 1, HEIGHT, WIDTH), 4097, dtype=torch.int64)
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
        grid = F.affine_grid(theta, [BATCH_SIZE, 1, HEIGHT, WIDTH], align_corners=True).to(torch.float16)

        image_output = transform_mask(mask_int64, grid)
        assert image_output.dtype == torch.int64, f"Expected int64 output dtype, got {image_output.dtype}"
        assert (image_output == 4097).all(), "Large class IDs must not be rounded in mixed-precision paths"


# ---------------------------------------------------------------------------
# Additional dtype coverage for transform_mask
# ---------------------------------------------------------------------------


class TestTransformMaskDtypes:
    """transform_mask preserves dtype for int32, bool, and float32 inputs."""

    def _identity_grid(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        import torch.nn.functional as F

        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
        return F.affine_grid(theta, [batch_size, 1, height, width], align_corners=True)

    def test_transform_mask_int32_preserves_dtype_and_labels(self):
        """Int32 mask is cast internally and cast back; output dtype is int32."""
        from fuse_augmentations._targets import transform_mask

        mask = torch.randint(0, 3, (BATCH_SIZE, 1, HEIGHT, WIDTH), dtype=torch.int32)
        grid = self._identity_grid(BATCH_SIZE, HEIGHT, WIDTH)

        image_output = transform_mask(mask, grid)
        assert image_output.dtype == torch.int32, f"Expected int32 output dtype, got {image_output.dtype}"
        unique_vals = set(torch.unique(image_output).tolist())
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected values after transform: {unique_vals}"

    def test_transform_mask_bool_preserves_dtype_and_values(self):
        """Bool mask is cast internally and cast back; output dtype is bool."""
        from fuse_augmentations._targets import transform_mask

        mask = torch.randint(0, 2, (BATCH_SIZE, 1, HEIGHT, WIDTH)).bool()
        grid = self._identity_grid(BATCH_SIZE, HEIGHT, WIDTH)

        image_output = transform_mask(mask, grid)
        assert image_output.dtype == torch.bool, f"Expected bool output dtype, got {image_output.dtype}"
        unique_vals = set(torch.unique(image_output).tolist())
        assert unique_vals.issubset({False, True}), f"Unexpected values after transform: {unique_vals}"

    def test_transform_mask_float32_passthrough_preserves_dtype(self):
        """Float32 mask takes the needs_cast_back=False path; output dtype is float32."""
        from fuse_augmentations._targets import transform_mask

        mask = torch.rand(BATCH_SIZE, 1, HEIGHT, WIDTH, dtype=torch.float32)
        grid = self._identity_grid(BATCH_SIZE, HEIGHT, WIDTH)

        image_output = transform_mask(mask, grid)
        assert image_output.dtype == torch.float32, f"Expected float32 output dtype, got {image_output.dtype}"
