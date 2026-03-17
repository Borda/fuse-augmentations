"""Unit tests for _compose.py -- spec tests #23, #58-60 (empty pipeline, fusion_plan, n_warps_saved).

Pure-unit tests use stub transforms and do NOT require Kornia.
Integration tests (marked @pytest.mark.integration) require kornia >= 0.6.12.
"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compose import AugmentationSequential, Compose, FusedAffineCompose
from fuse_augmentations._types import ReorderPolicy


class TestEmptyPipeline:
    """Verify empty Compose pipeline passthrough behaviour."""

    def test_shape_preserved(self):
        """Empty Compose returns input unchanged with correct shape."""
        pipe = Compose([])
        x = torch.zeros(1, 3, 8, 8)
        assert pipe(x).shape == torch.Size([1, 3, 8, 8])

    def test_values_unchanged(self):
        """Empty Compose returns exactly the same tensor."""
        pipe = Compose([])
        x = torch.rand(2, 3, 16, 16)
        assert torch.equal(pipe(x), x)

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


class TestDataKeysNotImplemented:
    """Verify data_keys raises NotImplementedError."""

    def test_raises(self):
        """Passing data_keys raises NotImplementedError with descriptive message."""
        import pytest

        with pytest.raises(NotImplementedError, match="data_keys"):
            Compose([], data_keys=["input"])


class TestReorderPolicyAggressive:
    """Verify reorder policy acceptance and rejection."""

    def test_aggressive_reorder_raises(self):
        """AGGRESSIVE reorder policy raises NotImplementedError in v0.1."""
        import pytest

        with pytest.raises(NotImplementedError, match="AGGRESSIVE"):
            Compose([], reorder=ReorderPolicy.AGGRESSIVE)

    def test_none_reorder_accepted(self):
        """NONE reorder policy is accepted without error."""
        pipe = Compose([], reorder=ReorderPolicy.NONE)
        assert pipe.reorder is ReorderPolicy.NONE

    def test_pointwise_reorder_accepted(self):
        """POINTWISE reorder policy is accepted without error."""
        pipe = Compose([], reorder=ReorderPolicy.POINTWISE)
        assert pipe.reorder is ReorderPolicy.POINTWISE


class TestAliases:
    """Verify public API aliases point to Compose."""

    def test_fused_affine_compose_is_compose(self):
        """FusedAffineCompose is an alias for Compose."""
        assert FusedAffineCompose is Compose

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
        x = torch.zeros(1, 3, 4, 4)
        assert torch.equal(pipe(x), x)


# ---------------------------------------------------------------------------
# Spec tests #58-60: n_warps_saved and fusion_plan with real Kornia transforms
# ---------------------------------------------------------------------------

_kornia_available = pytest.importorskip.__module__ is not None  # always True; guard below


@pytest.mark.integration
class TestNWarpsSavedWithPolicy:
    """#58-59: n_warps_saved with ReorderPolicy.NONE and barrier prevention."""

    @pytest.fixture(autouse=True)
    def _import_kornia(self):
        pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")

    def test_fused_three_saves_two(self):
        """#58: [Rotate, Scale, Flip] with ReorderPolicy.NONE -> n_warps_saved == 2."""
        from kornia.augmentation import RandomAffine, RandomHorizontalFlip, RandomRotation

        pipe = Compose(
            [
                RandomRotation(30, p=1.0),
                RandomAffine(0, scale=(0.8, 1.2), p=1.0),
                RandomHorizontalFlip(p=1.0),
            ],
            reorder=ReorderPolicy.NONE,
        )
        assert pipe.n_warps_saved == 2, (
            f"3 ops fused into 1 segment should save 2 warps, got {pipe.n_warps_saved}"
        )

    def test_barrier_prevents_fusion(self):
        """#59: [Rotate, GaussianBlur, Scale] -> n_warps_saved == 0 (barrier prevents fusion)."""
        from kornia.augmentation import RandomAffine, RandomGaussianBlur, RandomRotation

        pipe = Compose(
            [
                RandomRotation(30, p=1.0),
                RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            ],
            reorder=ReorderPolicy.NONE,
        )
        assert pipe.n_warps_saved == 0, (
            f"Barrier should prevent fusion, expected 0 warps saved, got {pipe.n_warps_saved}"
        )


@pytest.mark.integration
class TestFusionPlanFormat:
    """#60: fusion_plan string format with fused, exact, passthrough segments."""

    @pytest.fixture(autouse=True)
    def _import_kornia(self):
        pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")

    def test_fused_segment_format(self):
        """Fused segment shows fused(TransformA, TransformB) in plan."""
        from kornia.augmentation import RandomAffine, RandomRotation

        pipe = Compose([RandomRotation(30, p=1.0), RandomAffine(0, scale=(0.8, 1.2), p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("fused(")
        assert "RandomRotation" in plan
        assert "RandomAffine" in plan

    def test_exact_segment_format(self):
        """Exact-only segment shows exact(TransformName) in plan."""
        from kornia.augmentation import RandomHorizontalFlip

        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("exact(")
        assert "RandomHorizontalFlip" in plan

    def test_passthrough_segment_format(self):
        """Passthrough segment shows passthrough(TransformName) in plan."""
        from kornia.augmentation import RandomGaussianBlur, RandomRotation

        pipe = Compose([
            RandomRotation(30, p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
        ])
        plan = pipe.fusion_plan
        assert "passthrough(" in plan
        assert "RandomGaussianBlur" in plan

    def test_mixed_plan_format(self):
        """Mixed pipeline: fused -> passthrough -> exact segments all described."""
        from kornia.augmentation import RandomGaussianBlur, RandomHorizontalFlip, RandomRotation

        pipe = Compose([
            RandomRotation(30, p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            RandomHorizontalFlip(p=1.0),
        ])
        plan = pipe.fusion_plan
        # Should contain all three segment types
        assert "fused(" in plan
        assert "passthrough(" in plan
        assert "exact(" in plan
        # Arrow separator
        assert "\u2192" in plan
