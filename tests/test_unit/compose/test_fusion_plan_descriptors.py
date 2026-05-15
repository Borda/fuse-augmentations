"""Unit tests for FusedCompose.fusion_plan_descriptors property."""

from __future__ import annotations

import pytest

from fuse_augmentations import Compose, FusedCompose, SegmentDescriptor
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug


class TestFusionPlanDescriptorsBasic:
    """Basic existence and shape checks for fusion_plan_descriptors property."""

    def test_property_exists_on_fused_compose(self):
        """Compose exposes a fusion_plan_descriptors attribute."""
        pipe = Compose([])
        assert hasattr(pipe, "fusion_plan_descriptors")

    def test_empty_pipeline_returns_empty_list(self):
        """Empty pipeline has zero descriptors (no segments to describe)."""
        pipe = Compose([])
        assert pipe.fusion_plan_descriptors == []

    def test_returns_list_of_segment_descriptor(self):
        """fusion_plan_descriptors returns a list whose items are SegmentDescriptor instances."""
        pipe = FusedCompose.from_params(rotation=(-30, 30))
        descriptors = pipe.fusion_plan_descriptors
        assert isinstance(descriptors, list)
        assert all(isinstance(descriptor, SegmentDescriptor) for descriptor in descriptors)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFusionPlanDescriptorsKornia:
    """Tests requiring kornia -- skipped if not installed."""

    def test_single_fused_segment(self):
        """Single GEOMETRIC_INTERP transform produces one 'fused' descriptor naming the original transform."""
        pipe = Compose([kornia_aug.RandomRotation(degrees=30, p=1.0)])
        descriptors = pipe.fusion_plan_descriptors
        assert len(descriptors) == 1
        assert descriptors[0].kind == "fused"
        assert "RandomRotation" in descriptors[0].transforms

    def test_exact_segment_kind(self):
        """Single GEOMETRIC_EXACT transform (HFlip) produces one 'exact' descriptor."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        descriptors = pipe.fusion_plan_descriptors
        assert len(descriptors) == 1
        assert descriptors[0].kind == "exact"

    def test_n_warps_saved_for_fused_segment(self):
        """Three transforms fused into one segment report n_warps_saved == 2 (n - 1).

        Validates that fused segment warp accounting follows the n-1 rule: a chain
        of n geometric ops collapsed into a single grid_sample saves n-1 warp passes
        compared to executing them sequentially.
        """
        pipe = Compose([
            kornia_aug.RandomRotation(degrees=30, p=1.0),
            kornia_aug.RandomAffine(degrees=0, p=1.0),
            kornia_aug.RandomRotation(degrees=15, p=1.0),
        ])
        descriptors = pipe.fusion_plan_descriptors
        fused = [descriptor for descriptor in descriptors if descriptor.kind == "fused"]
        assert len(fused) == 1
        assert fused[0].n_warps_saved == 2


class TestFusionPlanDescriptorsConsistency:
    """fusion_plan_descriptors must be derivable from fusion_plan string."""

    def test_from_params_descriptors_derivable(self):
        """Reconstructing the plan string from descriptor list matches the fusion_plan property exactly.

        Guards against drift between the human-readable fusion_plan string and the structured descriptor list — both
        must describe the same segments in the same order.

        """
        pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
        descriptors = pipe.fusion_plan_descriptors
        # Reconstruct the string from descriptors and compare to fusion_plan
        parts = [f"{descriptor.kind}({', '.join(descriptor.transforms)})" for descriptor in descriptors]
        reconstructed = " → ".join(parts)
        assert reconstructed == pipe.fusion_plan

    def test_descriptors_total_warps_matches_n_warps_saved_property(self):
        """Sum of per-descriptor n_warps_saved equals the pipe-level n_warps_saved aggregate."""
        pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
        total_from_descriptors = sum(d.n_warps_saved for d in pipe.fusion_plan_descriptors)
        assert total_from_descriptors == pipe.n_warps_saved
