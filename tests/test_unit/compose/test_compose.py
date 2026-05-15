"""Unit tests for _compose.py: empty pipeline, fusion_plan, n_warps_saved.

Pure-unit tests use stub transforms and do NOT require Kornia. Integration tests (marked @pytest.mark.integration)
require kornia >= 0.6.12.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations._compose import AugmentationSequential, Compose, FusedCompose
from fuse_augmentations._types import ReorderPolicy

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug


class TestEmptyPipeline:
    """Verify empty Compose pipeline passthrough behaviour."""

    def test_shape_preserved(self):
        """Empty Compose returns input unchanged with correct shape."""
        pipe = Compose([])
        image = torch.zeros(1, 3, 8, 8)
        assert pipe(image).shape == torch.Size([1, 3, 8, 8])

    def test_values_unchanged(self):
        """Empty Compose returns exactly the same tensor."""
        pipe = Compose([])
        image = torch.rand(2, 3, 16, 16)
        assert torch.equal(pipe(image), image)

    def test_fusion_plan(self):
        """Empty pipeline reports 'empty' fusion plan."""
        pipe = Compose([])
        assert pipe.fusion_plan == "empty"

    def test_n_warps_saved(self):
        """Empty pipeline saves zero warps."""
        pipe = Compose([])
        assert pipe.n_warps_saved == 0

    def test_transform_matrix_none(self):
        """Empty pipeline has no transform matrix before or after forward."""
        pipe = Compose([])
        assert pipe.transform_matrix is None
        pipe(torch.zeros(1, 3, 4, 4))
        assert pipe.transform_matrix is None


class TestReorderPolicyAggressive:
    """All ReorderPolicy values are accepted by Compose."""

    @pytest.mark.parametrize("policy", list(ReorderPolicy))
    def test_all_policies_accepted(self, policy):
        """Every ReorderPolicy member is accepted without error and stored on the pipe."""
        pipe = Compose([], reorder=policy)
        assert pipe.reorder is policy


class TestAliases:
    """Verify public API aliases point to Compose."""

    def test_fused_affine_compose_is_compose(self):
        """FusedCompose is an alias for Compose."""
        assert FusedCompose is Compose

    def test_augmentation_sequential_is_compose(self):
        """AugmentationSequential is an alias for Compose."""
        assert AugmentationSequential is Compose


class TestOriginalTransforms:
    """Verify original_transforms is a defensive copy."""

    def test_is_copy(self):
        """Mutating the original list after construction does not affect the pipe."""
        transforms: list[object] = []
        pipe = Compose(transforms)
        transforms.append("should_not_appear")
        assert len(pipe.original_transforms) == 0


class TestNNModuleIntegration:
    """Verify Compose integrates with torch.nn.Module."""

    def test_compose_is_nn_module(self):
        """Compose is a subclass of torch.nn.Module."""
        pipe = Compose([])
        assert isinstance(pipe, torch.nn.Module)

    def test_compose_eval_mode(self):
        """Compose works correctly in eval mode."""
        pipe = Compose([])
        pipe.eval()
        image = torch.zeros(1, 3, 4, 4)
        assert torch.equal(pipe(image), image)


@pytest.mark.integration
@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestNWarpsSavedWithPolicy:
    """n_warps_saved with ReorderPolicy.NONE and barrier prevention."""

    def test_fused_three_saves_two(self):
        """Three consecutive ops fused into one segment save two warp passes (n - 1)."""
        pipe = Compose(
            [
                kornia_aug.RandomRotation(30, p=1.0),
                kornia_aug.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
                kornia_aug.RandomHorizontalFlip(p=1.0),
            ],
            reorder=ReorderPolicy.NONE,
        )
        assert pipe.n_warps_saved == 2, f"3 ops fused into 1 segment should save 2 warps, got {pipe.n_warps_saved}"

    def test_barrier_prevents_fusion(self):
        """A SPATIAL_KERNEL barrier between two geometric ops prevents fusion; n_warps_saved is 0."""
        pipe = Compose(
            [
                kornia_aug.RandomRotation(30, p=1.0),
                kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                kornia_aug.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            ],
            reorder=ReorderPolicy.NONE,
        )
        assert pipe.n_warps_saved == 0, (
            f"Barrier should prevent fusion, expected 0 warps saved, got {pipe.n_warps_saved}"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFusionPlanFormat:
    """fusion_plan string format with fused, exact, passthrough segments."""

    def test_fused_segment_format(self):
        """Fused segment shows fused(TransformA, TransformB) in plan."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0), kornia_aug.RandomAffine(0, scale=(0.8, 1.2), p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("fused(")
        assert "RandomRotation" in plan
        assert "RandomAffine" in plan

    def test_exact_segment_format(self):
        """Exact-only segment shows exact(TransformName) in plan."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("exact(")
        assert "RandomHorizontalFlip" in plan

    def test_passthrough_segment_format(self):
        """Passthrough segment shows passthrough(TransformName) in plan."""
        pipe = Compose([
            kornia_aug.RandomRotation(30, p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
        ])
        plan = pipe.fusion_plan
        assert "passthrough(" in plan
        assert "RandomGaussianBlur" in plan

    def test_mixed_plan_format(self):
        """Mixed pipeline: fused -> passthrough -> exact segments all described."""
        pipe = Compose([
            kornia_aug.RandomRotation(30, p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            kornia_aug.RandomHorizontalFlip(p=1.0),
        ])
        plan = pipe.fusion_plan
        # Should contain all three segment types
        assert "fused(" in plan
        assert "passthrough(" in plan
        assert "exact(" in plan
        # Arrow separator
        assert "→" in plan
