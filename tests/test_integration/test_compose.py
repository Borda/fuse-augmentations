"""Integration tests for _compose.py: shape, dtype, device, passthrough, and warp-count correctness.

Requires kornia >= 0.6.12.

"""

from __future__ import annotations

import io
import pickle

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia >= 0.6.12 required")]


@pytest.fixture
def image8x8_batch1() -> torch.Tensor:
    return torch.rand(1, 3, 8, 8)


@pytest.fixture
def image32x32_batch2() -> torch.Tensor:
    return torch.rand(2, 3, 32, 32)


class TestSingleTransformNoFusion:
    """Validate behavior for one-transform pipelines."""

    def test_n_warps_saved_zero(self):
        """A single GEOMETRIC_INTERP transform saves zero warps (no fusion)."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0)])
        assert pipe.n_warps_saved == 0

    def test_n_warps_saved_exact(self):
        """A single GEOMETRIC_EXACT transform saves 1 warp (no grid_sample at all)."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        assert pipe.n_warps_saved == 1

    def test_forward_runs(self, image8x8_batch1):
        """Single transform Compose produces valid output."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        out = pipe(image8x8_batch1)
        assert out.shape == image8x8_batch1.shape
        assert not torch.isnan(out).any()


class TestMixedBackend:
    """Validate mixed-backend acceptance in Compose construction (v0.5+)."""

    @pytest.mark.parametrize(
        "second_module",
        [
            "torchvision.transforms",
            "albumentations.augmentations",
            "torchvision.transforms.v2",
        ],
    )
    def test_mixed_backend_constructs_successfully(self, second_module):
        """Mixing kornia with another backend constructs without raising (v0.5+)."""
        cls = type("FakeTransform", (), {"__module__": second_module, "__qualname__": "FakeTransform"})

        pipe = Compose([kornia_aug.RandomRotation(degrees=30), cls()])
        assert pipe is not None


class TestSerialization:
    """Validate pickle and torch.save serialization round-trips."""

    def test_pickle_roundtrip(self, image8x8_batch1):
        """Compose survives pickle dump/load and produces identical output."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        loaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301

        torch.manual_seed(42)
        out1 = pipe(image8x8_batch1)
        torch.manual_seed(42)
        out2 = loaded(image8x8_batch1)
        assert torch.allclose(out1, out2)

    def test_torch_save_load(self):
        """Compose survives torch.save/torch.load via BytesIO."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
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
            kornia_aug.RandomRotation(30, p=1.0),
            kornia_aug.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            kornia_aug.RandomHorizontalFlip(p=1.0),
        ])
        assert pipe.n_warps_saved == 2


class TestFusionPlan:
    """Validate human-readable fusion-plan formatting."""

    def test_contains_transform_names(self):
        """fusion_plan names each transform in fused segments."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0), kornia_aug.RandomHorizontalFlip(p=1.0)])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "RandomRotation" in plan
        assert "RandomHorizontalFlip" in plan

    def test_single_segment(self):
        """Single fused segment shows one fused(...) group."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("fused(")
        assert "RandomRotation" in plan


class TestTransformMatrix:
    """Validate `transform_matrix` lifecycle and shape semantics."""

    def test_none_before_forward(self):
        """transform_matrix is None before any forward call."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0)])
        assert pipe.transform_matrix is None

    def test_populated_after_forward(self):
        """transform_matrix is populated with correct shape after forward."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0)])
        pipe(torch.rand(1, 3, 8, 8))
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == torch.Size([1, 3, 3])

    def test_single_transform_matrix_preserves_sampled_rotation(self):
        """A single-transform fast path records the sampled matrix, not identity."""
        pipe = Compose([kornia_aug.RandomRotation(degrees=(30.0, 30.0), p=1.0)])
        pipe(torch.rand(1, 3, 16, 16))

        matrix = pipe.transform_matrix
        assert matrix is not None
        identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device).unsqueeze(0)
        assert not torch.allclose(matrix, identity, rtol=1e-4, atol=1e-6)

    def test_batch_shape(self):
        """transform_matrix batch dimension matches input batch size."""
        pipe = Compose([kornia_aug.RandomRotation(30, p=1.0)])
        pipe(torch.rand(4, 3, 8, 8))
        assert pipe.transform_matrix.shape == torch.Size([4, 3, 3])

    def test_none_for_exact_only(self):
        """transform_matrix is None when pipeline has only ExactAffineSegments."""
        pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
        pipe(torch.rand(1, 3, 8, 8))
        assert pipe.transform_matrix is None


class TestPassthroughPath:
    """Validate passthrough behavior for non-fusible transforms."""

    def test_spatial_kernel_passthrough_shape(self, image32x32_batch2):
        """Pipeline with a SPATIAL_KERNEL transform (GaussianBlur) executes the passthrough branch."""
        pipe = Compose([
            kornia_aug.RandomHorizontalFlip(p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            kornia_aug.RandomRotation(degrees=15, p=1.0),
        ])
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape
        assert not torch.isnan(out).any()

    def test_spatial_kernel_passthrough_segments(self):
        """GaussianBlur between two geometric transforms breaks into 3 segments."""
        pipe = Compose([
            kornia_aug.RandomHorizontalFlip(p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            kornia_aug.RandomRotation(degrees=15, p=1.0),
        ])
        # Should have 3 segments: fused(hflip), passthrough(GaussianBlur), fused(rotation)
        assert "passthrough" in pipe.fusion_plan

    def test_spatial_kernel_warps_saved(self):
        """Single-transform segments on each side of a passthrough save appropriately."""
        pipe = Compose([
            kornia_aug.RandomRotation(30, p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            kornia_aug.RandomRotation(30, p=1.0),
        ])
        # Each fused segment has only one INTERP transform, so 0 warps saved per segment
        assert pipe.n_warps_saved == 0

    def test_spatial_kernel_exact_warps_saved(self):
        """ExactAffineSegment single-transform on each side of a passthrough: 1 warp saved each."""
        pipe = Compose([
            kornia_aug.RandomHorizontalFlip(p=1.0),
            kornia_aug.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            kornia_aug.RandomHorizontalFlip(p=1.0),
        ])
        # Each ExactAffineSegment with 1 transform saves 1 warp (no grid_sample at all)
        assert pipe.n_warps_saved == 2


class TestUnknownBackend:
    """Validate behavior for unsupported augmentation backends."""

    def test_raises(self):
        """Transforms from an unknown backend raise ValueError."""

        class FakeTransform:
            pass

        FakeTransform.__module__ = "unknown_lib.transforms"

        with pytest.raises(ValueError, match="No recognised backend"):
            Compose([FakeTransform()])


class TestForwardMultiTransform:
    """Validate end-to-end forward pass for multi-transform pipelines."""

    def test_three_transform_forward(self, image32x32_batch2):
        """Three-transform pipeline produces valid (B,C,H,W) output."""
        pipe = Compose([
            kornia_aug.RandomRotation(30, p=1.0),
            kornia_aug.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            kornia_aug.RandomHorizontalFlip(p=1.0),
        ])
        out = pipe(image32x32_batch2)
        assert out.shape == image32x32_batch2.shape
        assert not torch.isnan(out).any()
