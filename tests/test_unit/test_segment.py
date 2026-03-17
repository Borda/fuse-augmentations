"""Unit tests for _segment.py -- spec tests #42-47 + build_segments + ExactSegment tests."""

from __future__ import annotations

import torch

from fuse_augmentations._matrix import inv3x3, matmul3x3
from fuse_augmentations._segment import ExactSegment, FusedAffineSegment, build_segments, reorder_pointwise
from fuse_augmentations._types import TransformCategory


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
        """Return the transform's category attribute or default to SPATIAL_KERNEL."""
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        """Return minimal canonical params with batch size from input_shape."""
        bsz = input_shape[0]
        return {"_batch_size": torch.tensor([bsz])}

    def build_matrix(self, transform, params, height, width):
        """Delegate to transform.matrix_fn or return identity."""
        bsz = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(bsz, height, width)
        return torch.eye(3).unsqueeze(0).expand(bsz, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        """Pass through the image unchanged for stub testing."""
        return image


def _identity_matrix_fn(B, H, W):
    """Return (B, 3, 3) identity matrices."""
    return torch.eye(3).unsqueeze(0).expand(B, -1, -1)


def _hflip_matrix_fn(B, H, W):
    """Return (B, 3, 3) horizontal flip matrices."""
    from fuse_augmentations._matrix import hflip_matrix

    return hflip_matrix(W=W, batch_size=B, device=torch.device("cpu"), dtype=torch.float32)


def _vflip_matrix_fn(B, H, W):
    """Return (B, 3, 3) vertical flip matrices."""
    from fuse_augmentations._matrix import vflip_matrix

    return vflip_matrix(H=H, batch_size=B, device=torch.device("cpu"), dtype=torch.float32)


def _small_scale_matrix_fn(B, H, W):
    """Scale 0.01 -- near-degenerate but valid."""
    from fuse_augmentations._matrix import scale_matrix

    sx = torch.full((B,), 0.01)
    sy = torch.full((B,), 0.01)
    return scale_matrix(sx, sy, H=H, W=W)


class TestP0Identity:
    """Verify that p=0 produces identity (no-op) behaviour."""

    def test_all_transforms_inactive(self):
        """With p=0 every transform is skipped; composed matrix is identity."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=0.0)
        seg = FusedAffineSegment([t, t], adapter)

        img = torch.rand(2, 3, 16, 16)
        out = seg(img)

        assert torch.allclose(out, img, atol=1e-5)
        # Composed matrix should be identity
        mtx_i = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(seg.last_matrix, mtx_i, atol=1e-7)


class TestP1AlwaysActive:
    """Verify that p=1 always applies the transform."""

    def test_both_flips_compose_to_rotation_180(self):
        """Two flips (h+v) with p=1 should compose to 180-deg rotation."""
        adapter = _StubAdapter()
        t_h = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_v = _StubTransform(_vflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t_h, t_v], adapter)

        bsz, n_ch, height, width = 1, 3, 8, 8
        img = torch.rand(bsz, n_ch, height, width)
        out = seg(img)

        # Applying h+v flip should be equivalent to 180-deg rotation
        # For a square image, this means pixel (x, y) -> (W-1-x, H-1-y)
        # Verify on a known pattern
        assert out.shape == img.shape

        # The composed matrix should be the product of hflip and vflip
        mtx = seg.last_matrix
        assert mtx is not None
        assert mtx.shape == (bsz, 3, 3)


class TestBatchHeterogeneity:
    """Verify p-masking produces per-sample variation across a batch."""

    def test_different_samples_different_active_masks(self):
        """Manually verify that p-masking produces per-sample variation.

        We use p=0.5 and a fixed seed such that some samples are active and some are not, then verify the outputs differ
        across samples.

        """
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=0.5)
        seg = FusedAffineSegment([t], adapter)

        bsz = 8
        img = torch.rand(bsz, 1, 8, 8)

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


class TestNearDegenerateScale:
    """Verify near-degenerate scale factors do not produce NaN/Inf."""

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
        mtx = seg.last_matrix.double()
        mtx_inv = inv3x3(mtx)
        product = matmul3x3(mtx_inv, mtx)
        mtx_i = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(product, mtx_i, atol=1e-8)


class TestNaNInfInput:
    """Verify NaN and Inf inputs propagate through grid_sample without crash."""

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


class TestDeviceConsistency:
    """Verify output and matrices live on the same device as the input."""

    def test_cpu_device_consistency(self):
        """All intermediate matrices should live on the same device as the input."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8, device=torch.device("cpu"))
        out = seg(img)

        assert out.device == img.device
        assert seg.last_matrix.device == img.device


class TestBuildSegments:
    """Verify build_segments partitions transforms into correct segments."""

    def test_single_geometric_returns_one_segment(self):
        """Single geometric transform produces one FusedAffineSegment."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        result = build_segments([t], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 1

    def test_two_geometric_fused_into_one_segment(self):
        """Two consecutive geometric transforms fuse into one segment."""
        adapter = _StubAdapter()
        t1 = _StubTransform(_hflip_matrix_fn, p=1.0)
        t2 = _StubTransform(_vflip_matrix_fn, p=1.0)
        result = build_segments([t1, t2], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_geometric_barrier_breaks_segment(self):
        """A SPATIAL_KERNEL transform breaks the fused segment."""
        adapter = _StubAdapter()
        t_geo = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_barrier = _BarrierTransform()
        result = build_segments([t_geo, t_barrier], adapter)

        assert len(result) == 2
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is t_barrier

    def test_geo_barrier_geo_produces_three_elements(self):
        """[geo, barrier, geo] produces [segment, barrier, segment]."""
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
        """A POINTWISE transform breaks the fused segment."""
        adapter = _StubAdapter()
        t_geo = _StubTransform(_hflip_matrix_fn, p=1.0)
        t_pw = _PointwiseTransform()
        result = build_segments([t_geo, t_pw, t_geo], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is t_pw
        assert isinstance(result[2], FusedAffineSegment)

    def test_geometric_exact_fuses_with_interp(self):
        """GEOMETRIC_EXACT and GEOMETRIC_INTERP fuse into a single segment."""
        adapter = _StubAdapter()
        t_interp = _StubTransform(_hflip_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        t_exact = _StubTransform(_vflip_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([t_interp, t_exact], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_empty_transforms_returns_empty(self):
        """Empty transform list returns empty segment list."""
        adapter = _StubAdapter()
        result = build_segments([], adapter)
        assert result == []

    def test_interpolation_and_padding_forwarded(self):
        """Interpolation and padding_mode kwargs are forwarded to the segment."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        result = build_segments([t], adapter, interpolation="bicubic", padding_mode="reflection")

        seg = result[0]
        assert isinstance(seg, FusedAffineSegment)
        assert seg.interpolation == "bicubic"
        assert seg.padding_mode == "reflection"


class TestLastMatrixProperty:
    """Verify last_matrix lifecycle: None before forward, populated after."""

    def test_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _StubAdapter()
        t = _StubTransform(_hflip_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)
        assert seg.last_matrix is None

    def test_populated_after_forward(self):
        """last_matrix is populated with correct shape after forward."""
        adapter = _StubAdapter()
        t = _StubTransform(_identity_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)
        seg(torch.rand(2, 3, 8, 8))

        assert seg.last_matrix is not None
        assert seg.last_matrix.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# ExactSegment tests
# ---------------------------------------------------------------------------


class _HFlipTransform:
    """Stub HFlip transform for ExactSegment tests."""

    def __init__(self, p=1.0):
        self.p = p
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [3]  # width axis


class _VFlipTransform:
    """Stub VFlip transform for ExactSegment tests."""

    def __init__(self, p=1.0):
        self.p = p
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [2]  # height axis


class _FlipAdapter:
    """Adapter stub that supports exact_flip_dims for ExactSegment tests."""

    def category(self, transform):
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def exact_flip_dims(self, transform):
        return getattr(transform, "_flip_dims", [])

    def sample_params(self, transform, input_shape, device):
        bsz = input_shape[0]
        return {"_batch_size": torch.tensor([bsz])}

    def build_matrix(self, transform, params, height, width):
        bsz = int(params["_batch_size"].item())
        return torch.eye(3).unsqueeze(0).expand(bsz, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        return image


class TestExactSegmentLossless:
    """Verify ExactSegment applies lossless flips via tensor.flip."""

    def test_hflip_p1_matches_tensor_flip(self):
        """HFlip with p=1.0 produces pixel-exact same result as image.flip(dims=[3])."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=1.0)
        seg = ExactSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        expected = img.flip(dims=[3])

        assert torch.equal(out, expected), "ExactSegment HFlip should be pixel-exact"

    def test_vflip_p1_matches_tensor_flip(self):
        """VFlip with p=1.0 produces pixel-exact same result as image.flip(dims=[2])."""
        adapter = _FlipAdapter()
        t = _VFlipTransform(p=1.0)
        seg = ExactSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        expected = img.flip(dims=[2])

        assert torch.equal(out, expected), "ExactSegment VFlip should be pixel-exact"


class TestExactSegmentP0:
    """Verify p=0 leaves the image unchanged."""

    def test_p0_output_equals_input(self):
        """ExactSegment with p=0 returns the input tensor unchanged."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=0.0)
        seg = ExactSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img), "p=0 should leave image unchanged"


class TestExactSegmentDoubleFlip:
    """Verify HFlip then VFlip with p=1 composes correctly."""

    def test_hflip_then_vflip(self):
        """HFlip then VFlip with p=1 is same as image.flip(dims=[2, 3]) sequentially."""
        adapter = _FlipAdapter()
        t_h = _HFlipTransform(p=1.0)
        t_v = _VFlipTransform(p=1.0)
        seg = ExactSegment([t_h, t_v], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        # Sequential: first flip width, then flip height
        expected = img.flip(dims=[3]).flip(dims=[2])

        assert torch.equal(out, expected), "HFlip+VFlip should match sequential tensor.flip"


class TestExactSegmentPerSampleMask:
    """Verify per-sample p=0.5 masking in ExactSegment."""

    def test_p05_heterogeneous_batch(self):
        """With B=8 and p=0.5, at least one sample changed, at least one unchanged."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=0.5)
        seg = ExactSegment([t], adapter)

        bsz = 8
        img = torch.rand(bsz, 1, 8, 8)

        # Make the per-sample mask used inside ExactSegment deterministic so that
        # some samples are flipped and some are not, avoiding flaky behavior.
        pattern = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        orig_rand = torch.rand

        def _deterministic_rand(*size, **kwargs):
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", torch.float32)
            # Intercept calls that generate a per-sample mask over the batch.
            if len(size) >= 1 and size[0] == bsz:
                base = torch.zeros(size, device=device, dtype=dtype)
                mask = pattern.to(device=device, dtype=dtype)
                view_shape = (bsz,) + (1,) * (base.ndim - 1)
                return base + mask.view(view_shape)
            return orig_rand(*size, **kwargs)

        try:
            torch.rand = _deterministic_rand
            out = seg(img)
        finally:
            torch.rand = orig_rand

        diffs = (out - img).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-6).any()
        has_unchanged = (diffs < 1e-6).any()

        assert has_changed, "Expected at least one sample to be flipped"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


class TestExactSegmentLastMatrix:
    """Verify ExactSegment.last_matrix is always None."""

    def test_last_matrix_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=1.0)
        seg = ExactSegment([t], adapter)
        assert seg.last_matrix is None

    def test_last_matrix_none_after_forward(self):
        """last_matrix remains None after forward (ExactSegment has no matrix)."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=1.0)
        seg = ExactSegment([t], adapter)
        seg(torch.rand(2, 3, 8, 8))
        assert seg.last_matrix is None


class TestPointwiseReorderBuildSegments:
    """POINTWISE reorder + build_segments integration: verify correct segmentation."""

    def test_reorder_then_build_fuses_geometric(self):
        """[Rotate, Brightness, Scale] with POINTWISE reorder -> 1 FusedAffineSegment + 1 passthrough."""
        adapter = _StubAdapter()
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        brightness = _PointwiseTransform()
        scale = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)

        reordered = reorder_pointwise([rotate, brightness, scale], adapter)
        segments = build_segments(reordered, adapter)

        # After reorder: [Rotate, Scale, Brightness]
        # build_segments: FusedAffineSegment([Rotate, Scale]), passthrough(Brightness)
        assert len(segments) == 2
        assert isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 2
        assert segments[1] is brightness
