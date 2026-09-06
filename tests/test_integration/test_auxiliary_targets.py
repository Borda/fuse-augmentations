"""Integration tests for auxiliary target support (masks, bboxes, keypoints).

Covers spec tests #32--38. Requires kornia >= 0.6.12.

"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.compose import FusedCompose as Compose
from fuse_augmentations.targets import transform_mask

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = pytest.mark.integration


def _half_grid_sample_works_on_cpu() -> bool:
    """Return whether this torch build can run ``grid_sample`` on a half-precision CPU tensor.

    Probed rather than compared against a version number: support arrived in a minor release, and the
    supported floor (torch 2.2) raises ``RuntimeError: grid_sampler_2d_cpu not implemented for Half``.

    """
    try:
        F.grid_sample(
            torch.zeros(1, 1, 2, 2, dtype=torch.float16),
            torch.zeros(1, 2, 2, 2, dtype=torch.float16),
            align_corners=False,
        )
    except RuntimeError:
        return False
    return True


#: Whether the half-precision image path can be exercised at all on this build.
_HALF_GRID_SAMPLE_ON_CPU = _half_grid_sample_works_on_cpu()

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestMaskNearestOnly:
    """#32: Mask values must remain integers after any geometric transform."""

    def test_binary_mask_stays_binary_after_rotation(self):
        """After rotation, binary mask contains only {0.0, 1.0}."""
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=45, p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 2, (BATCH, 1, HEIGHT, WIDTH)).float()
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
            [kornia_aug.RandomRotation(degrees=30, p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH)).float()
        _, out_mask = pipe(img, mask)

        unique_vals = torch.unique(out_mask)
        for val in unique_vals.tolist():
            assert val == int(val), f"Mask value {val} is not an integer; nearest interpolation was NOT used"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestMaskSpatialCorrespondence:
    """#33: After HFlip, pixel at column x in image corresponds to column W-1-x in mask."""

    def test_hflip_spatial_correspondence(self):
        """After HFlip(p=1), mask columns are mirrored: column x -> column W-1-x."""
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        # Build a mask with distinct integer regions: left half = 1, right half = 2
        mask = torch.ones(BATCH, 1, HEIGHT, WIDTH)
        mask[:, :, :, WIDTH // 2 :] = 2.0
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)

        _, out_mask = pipe(img, mask)

        # After HFlip: left half should now be 2, right half should be 1
        left_half = out_mask[:, :, :, : WIDTH // 2]
        right_half = out_mask[:, :, :, WIDTH // 2 :]
        assert (left_half == 2.0).all(), "After HFlip, left half of mask should contain class 2"
        assert (right_half == 1.0).all(), "After HFlip, right half of mask should contain class 1"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestBboxIdentity:
    """#34: Identity transform leaves bounding boxes unchanged."""

    def test_bbox_unchanged_under_identity(self):
        """Bbox with hflip_p=0 (identity-like pipeline) is unchanged."""
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=0.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        # (B, N, 4) format: one box per sample
        boxes = torch.tensor([[[5.0, 10.0, 20.0, 25.0]]] * BATCH)
        _, out_boxes = pipe(img, boxes)

        torch.testing.assert_close(out_boxes, boxes, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestBboxHFlip:
    """#35: HFlip mirrors bbox pixel-edge x-coordinates: new_x = W - old_x."""

    def test_bbox_x_mirrored_after_hflip(self):
        """After HFlip(p=1), bbox edges mirror: x_min' = W-x_max, x_max' = W-x_min."""
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        x1, y1, x2, y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[x1, y1, x2, y2]]] * BATCH)
        _, out_boxes = pipe(img, boxes)

        expected_x1 = WIDTH - x2
        expected_x2 = WIDTH - x1
        expected = torch.tensor([[[expected_x1, y1, expected_x2, y2]]] * BATCH)
        torch.testing.assert_close(out_boxes, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestBboxRotation:
    """#36: After 90-degree rotation, bbox AABB contains expected corners."""

    def test_bbox_aabb_after_90_rotation(self):
        """90-degree rotation of a bbox produces a valid axis-aligned bounding box.

        Rotating a bbox by 90 degrees swaps its width and height; the implementation must compute the AABB enclosing all
        four rotated corners rather than naively transforming the original (x1, y1, x2, y2) pair, which would yield an
        inverted box.

        """
        # Use a known 90-degree rotation via kornia_aug.RandomRotation with degrees=(90, 90)
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=(90, 90), p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(1, CHANNELS, HEIGHT, WIDTH)
        # A box at (5, 10, 15, 20) - width=10, height=10
        boxes = torch.tensor([[[5.0, 10.0, 15.0, 20.0]]])
        _, out_boxes = pipe(img, boxes)

        # After 90-degree rotation around image center (W/2, H/2):
        # The resulting AABB should exist and have valid coordinates
        assert out_boxes.shape == (1, 1, 4), f"Expected shape (1,1,4), got {out_boxes.shape}"
        # Verify AABB constraints: x1 < x2 and y1 < y2
        ox1, oy1, ox2, oy2 = out_boxes[0, 0].tolist()
        assert ox1 < ox2, f"Expected x1 < x2, got {ox1} >= {ox2}"
        assert oy1 < oy2, f"Expected y1 < y2, got {oy1} >= {oy2}"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKeypointsKnownMapping:
    """#37: For HFlip(p=1), keypoint x' = W-1-x, y' unchanged."""

    def test_hflip_keypoint_x_mirrored(self):
        """After HFlip, keypoint x is mirrored: x' = W-1-x, y unchanged."""
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "keypoints"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        kx, ky = 10.0, 20.0
        # (B, N, 2) format
        kps = torch.tensor([[[kx, ky]]] * BATCH)
        _, out_kps = pipe(img, kps)

        expected_x = WIDTH - 1 - kx
        expected = torch.tensor([[[expected_x, ky]]] * BATCH)
        torch.testing.assert_close(out_kps, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestBatchConsistency:
    """#38: Per-sample transforms applied consistently to image and all aux targets."""

    def test_batch_image_mask_consistency(self):
        """Batch of B=4 with p=1.0 HFlip: each sample's image and mask are consistently flipped."""
        batch_size = 4
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        img = torch.rand(batch_size, CHANNELS, HEIGHT, WIDTH)
        # Build a mask that is NOT symmetric, so flip is detectable
        mask = torch.zeros(batch_size, 1, HEIGHT, WIDTH)
        mask[:, :, :, : WIDTH // 4] = 1.0  # left quarter is 1

        out_img, out_mask = pipe(img, mask)

        # Verify each sample individually
        for batch_idx in range(batch_size):
            # Image should be horizontally flipped
            expected_img_i = img[batch_idx].flip(dims=[-1])
            torch.testing.assert_close(
                out_img[batch_idx],
                expected_img_i,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {batch_idx}: image not consistently flipped",
            )
            # Mask should also be horizontally flipped
            expected_mask_i = mask[batch_idx].flip(dims=[-1])
            torch.testing.assert_close(
                out_mask[batch_idx],
                expected_mask_i,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Sample {batch_idx}: mask not consistently flipped with image",
            )

    def test_batch_image_bbox_consistency(self):
        """Batch with p=1.0 HFlip: each sample's bbox is consistently transformed."""
        batch_size = 4
        pipe = Compose(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy"],
        )
        img = torch.rand(batch_size, CHANNELS, HEIGHT, WIDTH)
        x1, y1, x2, y2 = 5.0, 10.0, 20.0, 25.0
        boxes = torch.tensor([[[x1, y1, x2, y2]]] * batch_size)

        out_img, out_boxes = pipe(img, boxes)

        expected_x1 = WIDTH - x2
        expected_x2 = WIDTH - x1
        for batch_idx in range(batch_size):
            # Image must be flipped
            expected_img_i = img[batch_idx].flip(dims=[-1])
            torch.testing.assert_close(out_img[batch_idx], expected_img_i, atol=1e-4, rtol=1e-4)
            # Box must be mirrored
            torch.testing.assert_close(
                out_boxes[batch_idx, 0, 0].item(),
                expected_x1,
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                out_boxes[batch_idx, 0, 2].item(),
                expected_x2,
                atol=1e-4,
                rtol=1e-4,
            )


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestPassthroughWithAuxTargets:
    """Passthrough segment (e.g. GaussianBlur) does not crash with aux_targets."""

    def test_passthrough_does_not_crash_with_aux_targets(self):
        """Pipeline with passthrough + fused segment routes mask correctly.

        kornia_aug.RandomGaussianBlur is SPATIAL_KERNEL (passthrough), kornia_aug.RandomRotation is GEOMETRIC_INTERP
        (fused). The passthrough segment only gets the image; the fused segment transforms both image and mask. The mask
        shape must be preserved after the full forward pass.

        """
        pipe = Compose(
            [
                kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                kornia_aug.RandomRotation(degrees=15, p=1.0),
            ],
            data_keys=["input", "mask"],
        )
        img = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)
        mask = torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH)).float()
        out = pipe(img, mask)

        assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
        out_img, out_mask = out
        assert out_img.shape == img.shape, f"Image shape mismatch: {out_img.shape} vs {img.shape}"
        assert out_mask.shape == mask.shape, f"Mask shape mismatch: {out_mask.shape} vs {mask.shape}"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestTransformMask:
    """transform_mask dtype cast-and-restore for int64, int32, bool, and float32."""

    def _identity_grid(self, batch: int, height: int, width: int) -> torch.Tensor:
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1)
        return F.affine_grid(theta, [batch, 1, height, width], align_corners=True)

    def test_int64_preserves_labels(self):
        """transform_mask accepts int64 masks and preserves integer labels."""
        mask_int64 = torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH), dtype=torch.int64)
        grid = self._identity_grid(BATCH, HEIGHT, WIDTH)

        out = transform_mask(mask_int64, grid)
        assert out.dtype == torch.int64, f"Expected int64 output dtype, got {out.dtype}"
        unique_vals = set(torch.unique(out).tolist())
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected label values after transform: {unique_vals}"

    def test_int64_preserves_large_labels_with_fp16_grid(self):
        """Large integer labels are preserved even when the input grid is float16.

        Mixed-precision paths can quietly round class IDs above the float16 mantissa limit (2048). Casting to float32
        internally before the warp ensures large labels survive the round-trip without truncation.

        """
        mask_int64 = torch.full((BATCH, 1, HEIGHT, WIDTH), 4097, dtype=torch.int64)
        theta = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).expand(BATCH, -1, -1)
        grid = F.affine_grid(theta, [BATCH, 1, HEIGHT, WIDTH], align_corners=True).to(torch.float16)

        out = transform_mask(mask_int64, grid)
        assert out.dtype == torch.int64, f"Expected int64 output dtype, got {out.dtype}"
        assert (out == 4097).all(), "Large class IDs must not be rounded in mixed-precision paths"

    def test_int32_preserves_dtype_and_labels(self):
        """Int32 mask is cast internally and cast back; output dtype is int32."""
        mask = torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH), dtype=torch.int32)
        grid = self._identity_grid(BATCH, HEIGHT, WIDTH)

        out = transform_mask(mask, grid)
        assert out.dtype == torch.int32, f"Expected int32 output dtype, got {out.dtype}"
        unique_vals = set(torch.unique(out).tolist())
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected values after transform: {unique_vals}"

    def test_bool_preserves_dtype_and_values(self):
        """Bool mask is cast internally and cast back; output dtype is bool."""
        mask = torch.randint(0, 2, (BATCH, 1, HEIGHT, WIDTH)).bool()
        grid = self._identity_grid(BATCH, HEIGHT, WIDTH)

        out = transform_mask(mask, grid)
        assert out.dtype == torch.bool, f"Expected bool output dtype, got {out.dtype}"
        unique_vals = set(torch.unique(out).tolist())
        assert unique_vals.issubset({False, True}), f"Unexpected values after transform: {unique_vals}"

    def test_float32_passthrough_preserves_dtype(self):
        """Float32 mask takes the needs_cast_back=False path; output dtype is float32."""
        mask = torch.rand(BATCH, 1, HEIGHT, WIDTH, dtype=torch.float32)
        grid = self._identity_grid(BATCH, HEIGHT, WIDTH)

        out = transform_mask(mask, grid)
        assert out.dtype == torch.float32, f"Expected float32 output dtype, got {out.dtype}"


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAuxGeometryDtype:
    """Auxiliary coordinate routing keeps full geometric precision regardless of image dtype.

    The composed matrix used to place boxes, keypoints and rotated boxes used to inherit the
    image's own dtype. That is fine while images are float32, and quietly wrong the moment they are
    not: a half-precision image rounds the affine coefficients that position its own targets, and an
    integer image would truncate them to whole pixels. Nothing raises -- every shape still lines up
    -- so these tests pin the decoupling rather than the symptom.

    """

    @staticmethod
    def _rotation_pipeline() -> Compose:
        """Build a multi-target pipeline whose matrix has deliberately non-representable entries.

        A flip or a 90-degree rotation would not catch anything: their matrices are integer-valued
        and survive any dtype exactly. The fractional angle and scale here produce coefficients that
        float16 cannot hold. ``execution="torch"`` because ``cv2.warpAffine`` rejects a float16
        image outright, and the point of the test is the coordinate routing, not the image warp.

        """
        import albumentations as albu

        return Compose(
            [albu.Affine(rotate=(17.0, 17.0), scale=(1.13, 1.13), p=1.0)],
            data_keys=["input", "bbox_xyxy"],
            execution="torch",
        )

    @pytest.mark.skipif(not _HALF_GRID_SAMPLE_ON_CPU, reason="torch build has no half-precision CPU grid_sample")
    def test_half_precision_image_does_not_round_box_geometry(self):
        """A float16 image routes its boxes through the same geometry a float32 image would.

        A half-precision training pipeline is the realistic way to hit this: the image dtype is
        chosen for memory, and there is no reason the box coordinates that ride along with it should
        inherit that choice.

        """
        boxes = torch.tensor([[[5.0, 10.0, 20.0, 25.0]]])
        pipe = self._rotation_pipeline()

        _, boxes_f32 = pipe(torch.rand(1, CHANNELS, HEIGHT, WIDTH), boxes)
        _, boxes_f16 = pipe(torch.rand(1, CHANNELS, HEIGHT, WIDTH, dtype=torch.float16), boxes)

        assert boxes_f16.dtype == torch.float32
        torch.testing.assert_close(boxes_f16, boxes_f32)

    def test_geometry_dtype_maps_low_precision_to_float32(self):
        """The geometry dtype keeps float64 and sends every narrower dtype to float32."""
        from fuse_augmentations.affine.segment import _matrix_geometry_dtype

        assert _matrix_geometry_dtype(torch.float64) is torch.float64
        assert _matrix_geometry_dtype(torch.float32) is torch.float32
        assert _matrix_geometry_dtype(torch.float16) is torch.float32
        assert _matrix_geometry_dtype(torch.bfloat16) is torch.float32
        assert _matrix_geometry_dtype(torch.uint8) is torch.float32
        assert _matrix_geometry_dtype(torch.int64) is torch.float32
