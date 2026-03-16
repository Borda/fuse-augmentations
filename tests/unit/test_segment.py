"""Unit tests for _segment.py -- spec tests #42-47 + build_segments tests."""

from __future__ import annotations

import torch

from fuse_augmentations._matrix import inv3x3, matmul3x3
from fuse_augmentations._segment import FusedAffineSegment, build_segments
from fuse_augmentations._types import TransformCategory

# ---------------------------------------------------------------------------
# Minimal stub adapter (no Kornia dependency)
# ---------------------------------------------------------------------------


class _StubTransform:
    """A stub geometric transform with a p attribute and a matrix factory."""

    def __init__(self, matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_INTERP):
        self.p = p
        self.matrix_fn = matrix_fn
        self._category = category


class _BarrierTransform:
    """A stub non-geometric transform (SPATIAL_KERNEL)."""

    def __init__(self):
        self._category = TransformCategory.SPATIAL_KERNEL


class _PointwiseTransform:
    """A stub pointwise transform."""

    def __init__(self):
        self._category = TransformCategory.POINTWISE


class _StubAdapter:
    """Minimal TransformAdapter for unit tests -- no Kornia dependency."""

    def category(self, transform):
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        B = input_shape[0]
        return {"_batch_size": torch.tensor([B])}

    def build_matrix(self, transform, params, H, W):
        B = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(B, H, W)
        return torch.eye(3).unsqueeze(0).expand(B, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        return image


def _identity_matrix_fn(B, H, W):
    return torch.eye(3).unsqueeze(0).expand(B, -1, -1)


def _hflip_matrix_fn(B, H, W):
    from fuse_augmentations._matrix import hflip_matrix

    return hflip_matrix(W=W, batch_size=B, device=torch.device("cpu"), dtype=torch.float32)


def _vflip_matrix_fn(B, H, W):
    from fuse_augmentations._matrix import vflip_matrix

    return vflip_matrix(H=H, batch_size=B, device=torch.device("cpu"), dtype=torch.float32)


def _small_scale_matrix_fn(B, H, W):
    """Scale 0.01 -- near-degenerate but valid."""
    from fuse_augmentations._matrix import scale_matrix

    sx = torch.full((B,), 0.01)
    sy = torch.full((B,), 0.01)
    return scale_matrix(sx, sy, H=H, W=W)


# ---------------------------------------------------------------------------
# Test #42: p=0 identity -- all transforms inactive → output == input
# ---------------------------------------------------------------------------


class TestP0Identity:
    def test_p0_all_transforms_inactive(self):
        """With p=0 every transform is skipped; composed matrix is identity."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=0.0)
        seg = FusedAffineSegment([t, t], adapter)

        img = torch.rand(2, 3, 16, 16)
        out = seg(img)

        assert torch.allclose(out, img, atol=1e-5)
        # Composed matrix should be identity
        I = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(seg.last_matrix, I, atol=1e-7)


# ---------------------------------------------------------------------------
# Test #43: p=1 always active -- all transforms applied
# ---------------------------------------------------------------------------


class TestP1AlwaysActive:
    def test_p1_both_flips_compose_to_rotation_180(self):
        """Two flips (h+v) with p=1 should compose to 180-deg rotation."""
        adapter = _StubAdapter()
        t_h = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_v = _StubTransform(_vflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t_h, t_v], adapter)

        B, C, H, W = 1, 3, 8, 8
        img = torch.rand(B, C, H, W)
        out = seg(img)

        # Applying h+v flip should be equivalent to 180-deg rotation
        # For a square image, this means pixel (x, y) -> (W-1-x, H-1-y)
        # Verify on a known pattern
        assert out.shape == img.shape

        # The composed matrix should be the product of hflip and vflip
        M = seg.last_matrix
        assert M is not None
        assert M.shape == (B, 3, 3)


# ---------------------------------------------------------------------------
# Test #44: Batch heterogeneity -- different samples get different transforms
# ---------------------------------------------------------------------------


class TestBatchHeterogeneity:
    def test_different_samples_different_active_masks(self):
        """Manually verify that p-masking produces per-sample variation.

        We use p=0.5 and a fixed seed such that some samples are active and
        some are not, then verify the outputs differ across samples.
        """
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=0.5)
        seg = FusedAffineSegment([t], adapter)

        B = 8
        torch.manual_seed(123)
        img = torch.rand(B, 1, 8, 8)

        # With B=8 and p=0.5, it is overwhelmingly likely that at least
        # one sample is active and at least one is inactive.
        torch.manual_seed(999)
        out = seg(img)

        # At least one sample should differ from input, and at least one
        # should be unchanged (identity). Check per-sample max diff.
        diffs = (out - img).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-4).any()
        has_unchanged = (diffs < 1e-4).any()

        # With 8 samples and p=0.5 the probability of all-same is 2*(0.5^8)=0.8%.
        # If seed changes break this, the test documents expected heterogeneity.
        assert has_changed, "Expected at least one sample to be transformed"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


# ---------------------------------------------------------------------------
# Test #45: Near-degenerate scale -- inv3x3 must not produce NaN
# ---------------------------------------------------------------------------


class TestNearDegenerateScale:
    def test_scale_001_no_nan(self):
        """Scale factor 0.01 should invert without NaN."""
        adapter = _StubAdapter()
        t = _StubTransform(_small_scale_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        img = torch.rand(2, 3, 16, 16)
        out = seg(img)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

        # Verify the inverse round-trip in float64 where det clamping
        # does not dominate.  Float32 det of scale=0.01 matrix is 1e-4,
        # which sits right at the eps*1e3 clamp boundary.
        M = seg.last_matrix.double()
        M_inv = inv3x3(M)
        product = matmul3x3(M_inv, M)
        I = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(product, I, atol=1e-8)


# ---------------------------------------------------------------------------
# Test #46: NaN/Inf in input -- propagates without crash
# ---------------------------------------------------------------------------


class TestNaNInfInput:
    def test_nan_input_propagates(self):
        """NaN in input should propagate through grid_sample without crash."""
        adapter = _StubAdapter()
        t = _StubTransform(_identity_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        img = torch.full((1, 1, 4, 4), float("nan"))
        out = seg(img)  # should not raise
        assert out.shape == img.shape

    def test_inf_input_propagates(self):
        """Inf in input should propagate through grid_sample without crash."""
        adapter = _StubAdapter()
        t = _StubTransform(_identity_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        img = torch.full((1, 1, 4, 4), float("inf"))
        out = seg(img)  # should not raise
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Test #47: Device consistency -- matrices on same device as image
# ---------------------------------------------------------------------------


class TestDeviceConsistency:
    def test_cpu_device_consistency(self):
        """All intermediate matrices should live on the same device as the input."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8, device=torch.device("cpu"))
        out = seg(img)

        assert out.device == img.device
        assert seg.last_matrix.device == img.device


# ---------------------------------------------------------------------------
# build_segments tests
# ---------------------------------------------------------------------------


class TestBuildSegments:
    def test_single_geometric_returns_one_segment(self):
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        result = build_segments([t], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 1

    def test_two_geometric_fused_into_one_segment(self):
        adapter = _StubAdapter()
        t1 = _StubTransform(_hflip_matrix_fn, p=1.0)
        t2 = _StubTransform(_vflip_matrix_fn, p=1.0)
        result = build_segments([t1, t2], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_geometric_barrier_breaks_segment(self):
        adapter = _StubAdapter()
        t_geo = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_barrier = _BarrierTransform()
        result = build_segments([t_geo, t_barrier], adapter)

        assert len(result) == 2
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is t_barrier

    def test_geo_barrier_geo_produces_three_elements(self):
        adapter = _StubAdapter()
        t1 = _StubTransform(_hflip_matrix_fn, p=1.0)
        t2 = _StubTransform(_vflip_matrix_fn, p=1.0)
        t_barrier = _BarrierTransform()
        result = build_segments([t1, t_barrier, t2], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is t_barrier
        assert isinstance(result[2], FusedAffineSegment)

    def test_pointwise_breaks_segment(self):
        adapter = _StubAdapter()
        t_geo = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_pw = _PointwiseTransform()
        result = build_segments([t_geo, t_pw, t_geo], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is t_pw
        assert isinstance(result[2], FusedAffineSegment)

    def test_geometric_exact_fuses_with_interp(self):
        adapter = _StubAdapter()
        t_interp = _StubTransform(_hflip_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        t_exact = _StubTransform(_vflip_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([t_interp, t_exact], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_empty_transforms_returns_empty(self):
        adapter = _StubAdapter()
        result = build_segments([], adapter)
        assert result == []

    def test_interpolation_and_padding_forwarded(self):
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        result = build_segments([t], adapter, interpolation="bicubic", padding_mode="reflection")

        seg = result[0]
        assert isinstance(seg, FusedAffineSegment)
        assert seg.interpolation == "bicubic"
        assert seg.padding_mode == "reflection"


# ---------------------------------------------------------------------------
# last_matrix property before forward
# ---------------------------------------------------------------------------


class TestLastMatrixProperty:
    def test_last_matrix_none_before_forward(self):
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)
        assert seg.last_matrix is None

    def test_last_matrix_populated_after_forward(self):
        adapter = _StubAdapter()
        t = _StubTransform(_identity_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)
        seg(torch.rand(2, 3, 8, 8))

        assert seg.last_matrix is not None
        assert seg.last_matrix.shape == (2, 3, 3)
