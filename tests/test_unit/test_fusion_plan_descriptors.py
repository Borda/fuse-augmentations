"""Unit tests for FusedCompose.fusion_plan_descriptors property (Phase B.2)."""
from __future__ import annotations

import pytest

from fuse_augmentations import SegmentDescriptor


class TestFusionPlanDescriptorsBasic:
    def test_property_exists_on_fused_compose(self):
        from fuse_augmentations import Compose

        pipe = Compose([])
        assert hasattr(pipe, "fusion_plan_descriptors")

    def test_empty_pipeline_returns_empty_list(self):
        from fuse_augmentations import Compose

        pipe = Compose([])
        assert pipe.fusion_plan_descriptors == []

    def test_returns_list_of_segment_descriptor(self):
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-30, 30))
        descriptors = pipe.fusion_plan_descriptors
        assert isinstance(descriptors, list)
        assert all(isinstance(d, SegmentDescriptor) for d in descriptors)


class TestFusionPlanDescriptorsKornia:
    """Tests requiring kornia -- skipped if not installed."""

    kornia = pytest.importorskip("kornia", reason="kornia required")

    def test_single_fused_segment(self):
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        pipe = Compose([K.RandomRotation(degrees=30, p=1.0)])
        descriptors = pipe.fusion_plan_descriptors
        assert len(descriptors) == 1
        assert descriptors[0].kind == "fused"
        assert "RandomRotation" in descriptors[0].transforms

    def test_exact_segment_kind(self):
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        descriptors = pipe.fusion_plan_descriptors
        assert len(descriptors) == 1
        assert descriptors[0].kind == "exact"

    def test_n_warps_saved_for_fused_segment(self):
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        # 3 transforms in one fused segment -> 2 warps saved
        pipe = Compose([
            K.RandomRotation(degrees=30, p=1.0),
            K.RandomAffine(degrees=0, p=1.0),
            K.RandomRotation(degrees=15, p=1.0),
        ])
        descriptors = pipe.fusion_plan_descriptors
        fused = [d for d in descriptors if d.kind == "fused"]
        assert len(fused) == 1
        assert fused[0].n_warps_saved == 2


class TestFusionPlanDescriptorsConsistency:
    """fusion_plan_descriptors must be derivable from fusion_plan string."""

    def test_from_params_descriptors_derivable(self):
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
        descriptors = pipe.fusion_plan_descriptors
        # Reconstruct the string from descriptors and compare to fusion_plan
        parts = [f"{d.kind}({', '.join(d.transforms)})" for d in descriptors]
        reconstructed = " \u2192 ".join(parts)
        assert reconstructed == pipe.fusion_plan

    def test_descriptors_total_warps_matches_n_warps_saved_property(self):
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-30, 30), hflip_p=0.5)
        total_from_descriptors = sum(
            d.n_warps_saved for d in pipe.fusion_plan_descriptors
        )
        assert total_from_descriptors == pipe.n_warps_saved
