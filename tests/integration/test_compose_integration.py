"""Integration tests for _compose.py -- spec tests #24-25, #48-52, #58-60.

Requires kornia >= 0.6.12.
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")
import kornia.augmentation as K  # noqa: E402

from fuse_augmentations._compose import Compose  # noqa: E402

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Test #24: Single transform, no fusion overhead
# ---------------------------------------------------------------------------


class TestSingleTransformNoFusion:
    def test_single_transform_n_warps_saved_zero(self):
        """A single geometric transform saves zero warps (no fusion)."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        assert pipe.n_warps_saved == 0

    def test_single_transform_forward_runs(self):
        """Single transform Compose produces valid output."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        x = torch.rand(1, 3, 8, 8)
        out = pipe(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Test #25: Mixed backend raises ValueError
# ---------------------------------------------------------------------------


class TestMixedBackend:
    def test_mixed_backend_raises(self):
        """Mixing backends (e.g., Kornia + TorchVision stub) raises."""

        class FakeTV:
            pass

        FakeTV.__module__ = "torchvision.transforms"

        with pytest.raises((ValueError, NotImplementedError)):
            Compose([K.RandomRotation(degrees=30), FakeTV()])


# ---------------------------------------------------------------------------
# Test #48: pickle round-trip
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    def test_pickle_roundtrip(self):
        """Compose survives pickle dump/load and produces identical output."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        loaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301

        x = torch.rand(1, 3, 8, 8)
        torch.manual_seed(42)
        out1 = pipe(x)
        torch.manual_seed(42)
        out2 = loaded(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Test #49: torch.save / torch.load
# ---------------------------------------------------------------------------


class TestTorchSaveLoad:
    def test_torch_save_load(self):
        """Compose survives torch.save/torch.load via BytesIO."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        buf = io.BytesIO()
        torch.save(pipe, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert isinstance(loaded, Compose)


# ---------------------------------------------------------------------------
# Test #50 / #58: n_warps_saved for [Rotate, Scale, Flip] (3 fused = 2 saved)
# ---------------------------------------------------------------------------


class TestNWarpsSavedThreeTransforms:
    def test_three_fused_saves_two_warps(self):
        """Three consecutive geometric transforms fused -> 2 warps saved."""
        pipe = Compose([
            K.RandomRotation(30, p=1.0),
            K.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            K.RandomHorizontalFlip(p=1.0),
        ])
        assert pipe.n_warps_saved == 2


# ---------------------------------------------------------------------------
# Test #51 / #59: n_warps_saved with single transform (no savings)
# ---------------------------------------------------------------------------


class TestNWarpsSavedSingleTransform:
    def test_single_transform_zero_warps_saved(self):
        """Single geometric transform -> 0 warps saved."""
        pipe = Compose([K.RandomRotation(30, p=1.0)])
        assert pipe.n_warps_saved == 0


# ---------------------------------------------------------------------------
# Test #52 / #60: fusion_plan string
# ---------------------------------------------------------------------------


class TestFusionPlan:
    def test_fusion_plan_contains_transform_names(self):
        """fusion_plan names each transform in fused segments."""
        pipe = Compose([K.RandomRotation(30, p=1.0), K.RandomHorizontalFlip(p=1.0)])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "RandomRotation" in plan
        assert "RandomHorizontalFlip" in plan

    def test_fusion_plan_single_segment(self):
        """Single fused segment shows one fused(...) group."""
        pipe = Compose([K.RandomRotation(30, p=1.0)])
        plan = pipe.fusion_plan
        assert plan.startswith("fused(")
        assert "RandomRotation" in plan


# ---------------------------------------------------------------------------
# transform_matrix property
# ---------------------------------------------------------------------------


class TestTransformMatrix:
    def test_transform_matrix_none_before_forward(self):
        """transform_matrix is None before any forward call."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        assert pipe.transform_matrix is None

    def test_transform_matrix_populated_after_forward(self):
        """transform_matrix is populated with correct shape after forward."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        pipe(torch.rand(1, 3, 8, 8))
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == torch.Size([1, 3, 3])

    def test_transform_matrix_batch_shape(self):
        """transform_matrix batch dimension matches input batch size."""
        pipe = Compose([K.RandomHorizontalFlip(p=1.0)])
        pipe(torch.rand(4, 3, 8, 8))
        assert pipe.transform_matrix.shape == torch.Size([4, 3, 3])


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------


class TestUnknownBackend:
    def test_unknown_backend_raises(self):
        """Transforms from an unknown backend raise NotImplementedError."""

        class FakeTransform:
            pass

        FakeTransform.__module__ = "unknown_lib.transforms"

        with pytest.raises(NotImplementedError, match="not yet supported"):
            Compose([FakeTransform()])


# ---------------------------------------------------------------------------
# Forward produces valid output for multi-transform pipeline
# ---------------------------------------------------------------------------


class TestForwardMultiTransform:
    def test_three_transform_forward(self):
        """Three-transform pipeline produces valid (B,C,H,W) output."""
        pipe = Compose([
            K.RandomRotation(30, p=1.0),
            K.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            K.RandomHorizontalFlip(p=1.0),
        ])
        x = torch.rand(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
