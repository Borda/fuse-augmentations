"""Limitations-as-tests harness (M1): README limitations encoded as executable specs.

Each README limitation from the ``## ⚠️ Limitations`` section is mapped to a
blueprint work package (WP) in ``.plans/active/blueprint_arch_redesign_superior_speed.md``
section 0.4 (rows L1--L8). For every row whose closure is a concrete future
behavior, this module holds ONE test asserting the DESIRED (post-closure)
behavior, marked ``@pytest.mark.xfail(strict=True, ...)``.

Contract:

- These xfail tests MUST fail today -- the features they assert do not exist yet,
  so each one XFAILs.
- When a future WP closes a row, its test starts to pass. Because ``strict=True``,
  an unexpected XPASS FAILS CI, forcing the closing PR to remove the marker and
  update the matching README limitation row in the same change. The test IS the
  machine-checkable definition of "closed".

Two rows (L4, L7) are BEHAVIOR/INHERENT -- documented semantics that are kept
deliberately, not defects. Those are regular (non-xfail) tests locking today's
contract; they must always pass.

Requires kornia (and albumentations for the L6 row).

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

try:
    import albumentations as albu

    _ALBU_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when albu is absent
    _ALBU_AVAILABLE = False

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required"),
]

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 16, 16


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 16, 16)`` float32 image batch."""
    return torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH)


def _mask() -> torch.Tensor:
    """Return a deterministic ``(2, 1, 16, 16)`` integer-valued mask batch."""
    return torch.randint(0, 2, (BATCH, 1, HEIGHT, WIDTH)).float()


class TestFusibleClosures:
    """L1--L3, L5, L6, L8: rows whose closure is a concrete future behavior (strict-xfail)."""

    @pytest.mark.xfail(strict=True, reason="L1: pixel-wise non-linear ops not fusible -- closed by WP-19")
    def test_l1_pointwise_nonlinear_chain_fuses(self):
        """DESIRED: a chain of pixel-wise non-linear ops collapses (no passthrough).

        Today each ``RandomSaturation`` is a passthrough segment. WP-19's
        LUT-composition fusion collapses per-channel scalar maps into one lookup,
        so the chain should fuse and save at least one pass.
        """
        pipe = Compose([
            kornia_aug.RandomSaturation((0.8, 1.2), p=1.0),
            kornia_aug.RandomSaturation((0.8, 1.2), p=1.0),
        ])
        assert "passthrough" not in pipe.fusion_plan
        assert pipe.n_warps_saved >= 1

    @pytest.mark.xfail(strict=True, reason="L2: spatial-kernel ops are fusion barriers -- closed by WP-17")
    def test_l2_blur_chain_stays_one_segment(self):
        """DESIRED: [Rotate, GaussianBlur, Scale] executes as ONE fused segment.

        Today the blur is a barrier splitting the chain into three segments.
        WP-17 lets blur commute through affine warps (Sigma' = A Sigma A^T), so
        the whole chain keeps K->1 fusion as a single segment.
        """
        pipe = Compose([
            kornia_aug.RandomRotation(degrees=25, p=1.0),
            kornia_aug.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            kornia_aug.RandomAffine(degrees=0, scale=(0.8, 1.2), p=1.0),
        ])
        assert len(pipe.fusion_plan_descriptors) == 1
        assert "passthrough" not in pipe.fusion_plan

    @pytest.mark.xfail(
        strict=True, reason="L3: padding mode is segment-level, no per-transform override -- closed by WP-18"
    )
    def test_l3_mixed_border_modes_split_per_transform(self):
        """DESIRED: transforms with different border modes are honored per-transform.

        Today a Compose-level padding mode overrides everything, so two affines
        with different ``padding_mode`` values still fuse into one segment. WP-18
        (Tier 1) splits the segment on a border-mode change
        (``split_reason="border_mode_change"``) to reproduce native semantics.
        """
        pipe = Compose([
            kornia_aug.RandomAffine(degrees=25, padding_mode="reflection", p=1.0),
            kornia_aug.RandomAffine(degrees=20, padding_mode="zeros", p=1.0),
        ])
        assert len(pipe.fusion_plan_descriptors) >= 2

    @pytest.mark.xfail(
        strict=True, reason="L5: non-flip exact ops image-only through ExactAffineSegment -- closed by WP-5"
    )
    def test_l5_rotate90_routes_aux_mask_without_raising(self):
        """DESIRED: a non-flip exact op routes an aux mask without raising.

        Today ``RandomRotation90`` with a mask raises RuntimeError in
        ``ExactAffineSegment`` (image-only). WP-5's D4-group dispatch routes aux
        exactly through rot90/flip, so the forward pass returns image and mask.
        """
        pipe = Compose(
            [kornia_aug.RandomRotation90(times=(1, 1), p=1.0)],
            data_keys=["input", "mask"],
        )
        out_image, out_mask = pipe(_image(), _mask())
        assert out_image.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
        assert out_mask.shape == (BATCH, 1, HEIGHT, WIDTH)

    @pytest.mark.xfail(
        strict=True, reason="L6: Albu fused segments + multi-target data_keys raise ValueError -- closed by WP-4a"
    )
    @pytest.mark.skipif(not _ALBU_AVAILABLE, reason="albumentations required")
    def test_l6_albu_pipeline_with_mask_constructs(self):
        """DESIRED: an Albumentations pipeline with a mask data_key constructs.

        Today ``Compose`` raises ValueError at construction when an Albumentations
        pipeline is combined with ``data_keys`` beyond the image key. WP-4a routes
        aux through the composed pixel matrix, so construction succeeds.
        """
        pipe = Compose(
            [albu.Affine(rotate=15, p=1.0), albu.HorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        assert pipe is not None

    @pytest.mark.xfail(
        strict=True, reason="L8: output_backend skipped for multi-target tuple outputs -- closed by WP-4c"
    )
    def test_l8_output_backend_applies_to_multi_target(self):
        """DESIRED: output_backend conversion is applied to every multi-target output.

        Today, with more than one ``data_key``, the pipeline returns raw tensors
        and skips ``output_backend`` conversion. WP-4c converts each target, so a
        numpy backend yields numpy arrays for both image and mask.
        """
        import numpy as np

        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=30, p=1.0)],
            data_keys=["input", "mask"],
            output_backend="numpy",
        )
        out_image, out_mask = pipe(_image(), _mask())
        assert isinstance(out_image, np.ndarray)
        assert isinstance(out_mask, np.ndarray)


class TestDocumentedSemantics:
    """L4, L7: BEHAVIOR/INHERENT rows -- lock today's documented contract (must always pass)."""

    def test_l4_crop_resize_changes_output_dims(self):
        """L4 (BEHAVIOR): crop+resize produces the requested output size, not the input size.

        Changing spatial dimensions is what crop+resize MEANS; this is documented behavior, not a defect. This test
        locks the size contract so a future refactor cannot silently drop it.

        """
        pipe = Compose([kornia_aug.RandomResizedCrop((8, 8), p=1.0)])
        out = pipe(_image())
        assert out.shape == (BATCH, CHANNELS, 8, 8)

    def test_l7_mask_sampling_is_not_differentiable(self):
        """L7 (INHERENT): nearest-neighbour mask sampling carries no gradient.

        Hard integer labels sampled with ``mode='nearest'`` are non-differentiable
        everywhere -- mathematically unavoidable and shared by native backends.
        The image path stays differentiable; the mask path does not.

        """
        image = _image().requires_grad_(True)
        mask = _mask().requires_grad_(True)
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=30, p=1.0)],
            data_keys=["input", "mask"],
        )
        out_image, out_mask = pipe(image, mask)
        assert out_image.grad_fn is not None
        assert out_mask.grad_fn is None
