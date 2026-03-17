"""Integration tests for _compose.py: shape, dtype, device, passthrough, and warp-count correctness.

Requires kornia >= 0.6.12.

"""

from __future__ import annotations

import io
import pickle

import pytest
import torch

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")
from kornia.augmentation import RandomAffine, RandomGaussianBlur, RandomHorizontalFlip, RandomRotation  # noqa: E402

from fuse_augmentations._compose import FusedCompose as Compose  # noqa: E402

pytestmark = pytest.mark.integration


class TestSingleTransformNoFusion:
    """Validate behavior for one-transform pipelines."""

    def test_n_warps_saved_zero(self):
        """A single GEOMETRIC_INTERP transform saves zero warps (no fusion)."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        assert pipe.n_warps_saved == 0

    def test_n_warps_saved_exact(self):
        """A single GEOMETRIC_EXACT transform saves 1 warp (no grid_sample at all)."""
        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        assert pipe.n_warps_saved == 1

    def test_forward_runs(self):
        """Single transform Compose produces valid output."""
        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        x = torch.rand(1, 3, 8, 8)
        out = pipe(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestMixedBackend:
    """Validate mixed-backend rejection in Compose construction."""

    @pytest.mark.parametrize(
        "second_module",
        [
            "torchvision.transforms",
            "albumentations.augmentations",
            "torchvision.transforms.v2",
        ],
    )
    def test_raises(self, second_module):
        """Mixing kornia with any other backend raises ValueError or NotImplementedError."""
        cls = type("FakeTransform", (), {"__module__": second_module, "__qualname__": "FakeTransform"})

        with pytest.raises((ValueError, NotImplementedError)):
            Compose([RandomRotation(degrees=30), cls()])


class TestSerialization:
    """Validate pickle and torch.save serialization round-trips."""

    def test_pickle_roundtrip(self):
        """Compose survives pickle dump/load and produces identical output."""
        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        loaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301

        x = torch.rand(1, 3, 8, 8)
        torch.manual_seed(42)
        out1 = pipe(x)
        torch.manual_seed(42)
        out2 = loaded(x)
        assert torch.allclose(out1, out2)

    def test_torch_save_load(self):
        """Compose survives torch.save/torch.load via BytesIO."""
        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        buf = io.BytesIO()
        torch.save(pipe, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert isinstance(loaded, Compose)


class TestNWarpsSaved:
    """Validate warp-savings accounting for representative pipelines."""

    def test_three_fused_saves_two(self):
        """Three consecutive geometric transforms fused -> 2 warps saved."""
        pipe = Compose([
            RandomRotation(30, p=1.0),
            RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            RandomHorizontalFlip(p=1.0),
        ])
        assert pipe.n_warps_saved == 2

    def test_single_transform_saves_zero(self):
        """Single geometric transform -> 0 warps saved."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        assert pipe.n_warps_saved == 0


class TestFusionPlan:
    """Validate human-readable fusion-plan formatting."""

    def test_contains_transform_names(self):
        """fusion_plan names each transform in fused segments."""
        pipe = Compose([RandomRotation(30, p=1.0), RandomHorizontalFlip(p=1.0)])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "RandomRotation" in plan
        assert "RandomHorizontalFlip" in plan

    def test_single_segment(self):
        """Single fused segment shows one fused(...) group."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("fused(")
        assert "RandomRotation" in plan


class TestTransformMatrix:
    """Validate `transform_matrix` lifecycle and shape semantics."""

    def test_none_before_forward(self):
        """transform_matrix is None before any forward call."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        assert pipe.transform_matrix is None

    def test_populated_after_forward(self):
        """transform_matrix is populated with correct shape after forward."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        pipe(torch.rand(1, 3, 8, 8))
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == torch.Size([1, 3, 3])

    def test_batch_shape(self):
        """transform_matrix batch dimension matches input batch size."""
        pipe = Compose([RandomRotation(30, p=1.0)])
        pipe(torch.rand(4, 3, 8, 8))
        assert pipe.transform_matrix.shape == torch.Size([4, 3, 3])

    def test_none_for_exact_only(self):
        """transform_matrix is None when pipeline has only ExactSegments."""
        pipe = Compose([RandomHorizontalFlip(p=1.0)])
        pipe(torch.rand(1, 3, 8, 8))
        assert pipe.transform_matrix is None


class TestPassthroughPath:
    """Validate passthrough behavior for non-fusible transforms."""

    def test_spatial_kernel_passthrough_shape(self):
        """Pipeline with a SPATIAL_KERNEL transform (GaussianBlur) executes the passthrough branch."""
        pipe = Compose([
            RandomHorizontalFlip(p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            RandomRotation(degrees=15, p=1.0),
        ])
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_spatial_kernel_passthrough_segments(self):
        """GaussianBlur between two geometric transforms breaks into 3 segments."""
        pipe = Compose([
            RandomHorizontalFlip(p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            RandomRotation(degrees=15, p=1.0),
        ])
        # Should have 3 segments: fused(hflip), passthrough(GaussianBlur), fused(rotation)
        assert "passthrough" in pipe.fusion_plan

    def test_spatial_kernel_warps_saved(self):
        """Single-transform segments on each side of a passthrough save appropriately."""
        pipe = Compose([
            RandomRotation(30, p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            RandomRotation(30, p=1.0),
        ])
        # Each fused segment has only one INTERP transform, so 0 warps saved per segment
        assert pipe.n_warps_saved == 0

    def test_spatial_kernel_exact_warps_saved(self):
        """ExactSegment single-transform on each side of a passthrough: 1 warp saved each."""
        pipe = Compose([
            RandomHorizontalFlip(p=1.0),
            RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            RandomHorizontalFlip(p=1.0),
        ])
        # Each ExactSegment with 1 transform saves 1 warp (no grid_sample at all)
        assert pipe.n_warps_saved == 2


class TestUnknownBackend:
    """Validate behavior for unsupported augmentation backends."""

    def test_raises(self):
        """Transforms from an unknown backend raise NotImplementedError."""

        class FakeTransform:
            pass

        FakeTransform.__module__ = "unknown_lib.transforms"

        with pytest.raises(NotImplementedError, match="not yet supported"):
            Compose([FakeTransform()])


class TestForwardMultiTransform:
    """Validate end-to-end forward pass for multi-transform pipelines."""

    def test_three_transform_forward(self):
        """Three-transform pipeline produces valid (B,C,H,W) output."""
        pipe = Compose([
            RandomRotation(30, p=1.0),
            RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            RandomHorizontalFlip(p=1.0),
        ])
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
