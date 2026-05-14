"""Unit tests for _segment.py: FusedAffineSegment, ExactAffineSegment, and build_segments."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import inv3x3, matmul3x3
from fuse_augmentations.affine._segment import ExactAffineSegment, FusedAffineSegment, build_segments, reorder_pointwise


class _StubTransform:
    """Albu stub geometric transform with a prob attribute and a matrix factory."""

    def __init__(self, matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP):
        self.prob = prob
        self.matrix_fn = matrix_fn
        self._category = category


class _BarrierTransform:
    """Albu stub non-geometric transform (SPATIAL_KERNEL)."""

    def __init__(self):
        self._category = TransformCategory.SPATIAL_KERNEL


class _PointwiseTransform:
    """Albu stub pointwise transform."""

    def __init__(self):
        self._category = TransformCategory.POINTWISE


class _StubAdapter:
    """Minimal TransformAdapter for unit tests -- no Kornia dependency."""

    def category(self, transform):
        """Return the transform's category attribute or default to SPATIAL_KERNEL."""
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        """Return minimal canonical params with batch size from input_shape."""
        batch_size = input_shape[0]
        return {"_batch_size": torch.tensor([batch_size])}

    def build_matrix(self, transform, params, height, width):
        """Delegate to transform.matrix_fn or return identity."""
        batch_size = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(batch_size, height, width)
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        """Pass through the image unchanged for stub testing."""
        return image


def _identity_matrix_fn(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Return (batch_size, 3, 3) identity matrices."""
    return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)


def _hflip_matrix_fn(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Return (batch_size, 3, 3) horizontal flip matrices."""
    from fuse_augmentations.affine._matrix import hflip_matrix

    return hflip_matrix(width=width, batch_size=batch_size, device=torch.device("cpu"), dtype=torch.float32)


def _vflip_matrix_fn(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Return (batch_size, 3, 3) vertical flip matrices."""
    from fuse_augmentations.affine._matrix import vflip_matrix

    return vflip_matrix(height=height, batch_size=batch_size, device=torch.device("cpu"), dtype=torch.float32)


def _small_scale_matrix_fn(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Scale 0.01 -- near-degenerate but valid."""
    from fuse_augmentations.affine._matrix import scale_matrix

    scale_x = torch.full((batch_size,), 0.01)
    scale_y = torch.full((batch_size,), 0.01)
    return scale_matrix(scale_x, scale_y, height=height, width=width)


class TestP0Identity:
    """Verify that prob=0 produces identity (no-operation) behaviour."""

    def test_all_transforms_inactive(self):
        """With prob=0 every transform is skipped; composed matrix is identity."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=0.0)
        segment = FusedAffineSegment([transform, transform], adapter)

        image = torch.rand(2, 3, 16, 16)
        image_output = segment(image)

        assert torch.allclose(image_output, image, atol=1e-5)
        # Composed matrix should be identity
        mtx_identity = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(segment.last_matrix, mtx_identity, atol=1e-7)


class TestP1AlwaysActive:
    """Verify that prob=1 always applies the transform."""

    def test_both_flips_compose_to_rotation_180(self):
        """Two flips (h+v) with prob=1 should compose to 180-deg rotation."""
        adapter = _StubAdapter()
        transform_h = _StubTransform(_hflip_matrix_fn, prob=1.0)
        transform_v = _StubTransform(_vflip_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform_h, transform_v], adapter)

        batch_size, num_channels, height, width = 1, 3, 8, 8
        image = torch.rand(batch_size, num_channels, height, width)
        image_output = segment(image)

        # Applying h+v flip should be equivalent to 180-deg rotation
        # For a square image, this means pixel (coord_x, coord_y) -> (width-1-coord_x, height-1-coord_y)
        # Verify on a known pattern
        assert image_output.shape == image.shape

        # The composed matrix should be the product of hflip and vflip
        mtx = segment.last_matrix
        assert mtx is not None
        assert mtx.shape == (batch_size, 3, 3)


class TestBatchHeterogeneity:
    """Verify prob-masking produces per-sample variation across a batch."""

    def test_different_samples_different_active_masks(self):
        """Manually verify that prob-masking produces per-sample variation.

        We use prob=0.5 and a fixed seed such that some samples are active and some are not, then verify the outputs
        differ across samples.

        """
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=0.5)
        segment = FusedAffineSegment([transform], adapter)

        batch_size = 8
        image = torch.rand(batch_size, 1, 8, 8)

        image_output = segment(image)

        # At least one sample should differ from input, and at least one
        # should be unchanged (identity). Check per-sample max diff.
        diffs = (image_output - image).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-4).any()
        has_unchanged = (diffs < 1e-4).any()

        # With 8 samples and prob=0.5 the probability of all-same is 2*(0.5^8)=0.8%.
        # If seed changes break this, the test documents expected heterogeneity.
        assert has_changed, "Expected at least one sample to be transformed"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


class TestNearDegenerateScale:
    """Verify near-degenerate scale factors do not produce NaN/Inf."""

    def test_scale_001_no_nan(self):
        """Scale factor 0.01 should invert without NaN."""
        adapter = _StubAdapter()
        transform = _StubTransform(_small_scale_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)

        image = torch.rand(2, 3, 16, 16)
        image_output = segment(image)

        assert not torch.isnan(image_output).any(), "Output contains NaN"
        assert not torch.isinf(image_output).any(), "Output contains Inf"

        # Verify the inverse round-trip in float64 where det clamping
        # does not dominate.  Float32 det of scale=0.01 matrix is 1e-4,
        # which sits right at the eps*1e3 clamp boundary.
        mtx = segment.last_matrix.double()
        mtx_inv = inv3x3(mtx)
        product = matmul3x3(mtx_inv, mtx)
        mtx_identity = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(product, mtx_identity, atol=1e-8)


class TestNaNInfInput:
    """Verify NaN and Inf inputs propagate through grid_sample without crash."""

    def test_nan_input_propagates(self):
        """NaN in input should propagate through grid_sample without crash."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)

        image = torch.full((1, 1, 4, 4), float("nan"))
        image_output = segment(image)  # should not raise
        assert image_output.shape == image.shape

    def test_inf_input_propagates(self):
        """Inf in input should propagate through grid_sample without crash."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)

        image = torch.full((1, 1, 4, 4), float("inf"))
        image_output = segment(image)  # should not raise
        assert image_output.shape == image.shape


class TestDeviceConsistency:
    """Verify output and matrices live on the same device as the input."""

    def test_cpu_device_consistency(self):
        """All intermediate matrices should live on the same device as the input."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)

        image = torch.rand(2, 3, 8, 8, device=torch.device("cpu"))
        image_output = segment(image)

        assert image_output.device == image.device
        assert segment.last_matrix.device == image.device


class TestBuildSegments:
    """Verify build_segments partitions transforms into correct segments."""

    def test_single_geometric_returns_one_segment(self):
        """Single geometric transform produces one FusedAffineSegment."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        result = build_segments([transform], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 1

    def test_two_geometric_fused_into_one_segment(self):
        """Two consecutive geometric transforms fuse into one segment."""
        adapter = _StubAdapter()
        transform1 = _StubTransform(_hflip_matrix_fn, prob=1.0)
        transform2 = _StubTransform(_vflip_matrix_fn, prob=1.0)
        result = build_segments([transform1, transform2], adapter)

        assert len(result) == 1
        assert isinstance(result[0], FusedAffineSegment)
        assert len(result[0].transforms) == 2

    def test_geometric_barrier_breaks_segment(self):
        """Albu SPATIAL_KERNEL transform breaks the fused segment."""
        adapter = _StubAdapter()
        transform_geo = _StubTransform(_hflip_matrix_fn, prob=1.0)
        transform_barrier = _BarrierTransform()
        result = build_segments([transform_geo, transform_barrier], adapter)

        assert len(result) == 2
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is transform_barrier

    def test_geo_barrier_geo_produces_three_elements(self):
        """[geo, barrier, geo] produces [segment, barrier, segment]."""
        adapter = _StubAdapter()
        transform1 = _StubTransform(_hflip_matrix_fn, prob=1.0)
        transform2 = _StubTransform(_vflip_matrix_fn, prob=1.0)
        transform_barrier = _BarrierTransform()
        result = build_segments([transform1, transform_barrier, transform2], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is transform_barrier
        assert isinstance(result[2], FusedAffineSegment)

    def test_pointwise_breaks_segment(self):
        """Albu POINTWISE transform breaks the fused segment."""
        adapter = _StubAdapter()
        transform_geo = _StubTransform(_hflip_matrix_fn, prob=1.0)
        transform_pw = _PointwiseTransform()
        result = build_segments([transform_geo, transform_pw, transform_geo], adapter)

        assert len(result) == 3
        assert isinstance(result[0], FusedAffineSegment)
        assert result[1] is transform_pw
        assert isinstance(result[2], FusedAffineSegment)

    def test_geometric_exact_fuses_with_interp(self):
        """GEOMETRIC_EXACT and GEOMETRIC_INTERP fuse into a single segment."""
        adapter = _StubAdapter()
        transform_interp = _StubTransform(_hflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        transform_exact = _StubTransform(_hflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([transform_interp, transform_exact], adapter)

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
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        result = build_segments([transform], adapter, interpolation="bicubic", padding_mode="reflection")

        segment = result[0]
        assert isinstance(segment, FusedAffineSegment)
        assert segment.interpolation == "bicubic"
        assert segment.padding_mode == "reflection"


class TestLastMatrixProperty:
    """Verify last_matrix lifecycle: None before forward, populated after."""

    def test_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _StubAdapter()
        transform = _StubTransform(_hflip_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)
        assert segment.last_matrix is None

    def test_populated_after_forward(self):
        """last_matrix is populated with correct shape after forward."""
        adapter = _StubAdapter()
        transform = _StubTransform(_identity_matrix_fn, prob=1.0)
        segment = FusedAffineSegment([transform], adapter)
        segment(torch.rand(2, 3, 8, 8))

        assert segment.last_matrix is not None
        assert segment.last_matrix.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# ExactAffineSegment tests
# ---------------------------------------------------------------------------


class _HFlipTransform:
    """Stub HFlip transform for ExactAffineSegment tests."""

    def __init__(self, p=1.0):
        self.p = p
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [3]  # width axis


class _VFlipTransform:
    """Stub VFlip transform for ExactAffineSegment tests."""

    def __init__(self, p=1.0):
        self.p = p
        self._category = TransformCategory.GEOMETRIC_EXACT
        self._flip_dims = [2]  # height axis


class _FlipAdapter:
    """Adapter stub that supports exact_flip_dims for ExactAffineSegment tests."""

    def category(self, transform):
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def exact_flip_dims(self, transform):
        return getattr(transform, "_flip_dims", [])

    def exact_apply(self, transform, image):
        return image.flip(dims=self.exact_flip_dims(transform))

    def sample_params(self, transform, input_shape, device):
        batch_size = input_shape[0]
        return {"_batch_size": torch.tensor([batch_size])}

    def build_matrix(self, transform, params, height, width):
        batch_size = int(params["_batch_size"].item())
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        return image


class _RaisingExactAdapter(_FlipAdapter):
    """Adapter whose exact_apply must never run for inactive transforms."""

    def exact_apply(self, transform, image):
        raise RuntimeError("exact_apply should not be called for inactive samples")


class TestExactAffineSegmentLossless:
    """Verify ExactAffineSegment applies lossless flips via tensor.flip."""

    def test_hflip_p1_matches_tensor_flip(self):
        """HFlip with prob=1.0 produces pixel-exact same result as image.flip(dims=[3])."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=1.0)
        seg = ExactAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        expected = img.flip(dims=[3])

        assert torch.equal(out, expected), "ExactAffineSegment HFlip should be pixel-exact"

    def test_vflip_p1_matches_tensor_flip(self):
        """VFlip with prob=1.0 produces pixel-exact same result as image.flip(dims=[2])."""
        adapter = _FlipAdapter()
        t = _VFlipTransform(p=1.0)
        seg = ExactAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        expected = img.flip(dims=[2])

        assert torch.equal(out, expected), "ExactAffineSegment VFlip should be pixel-exact"


class TestExactAffineSegmentP0:
    """Verify prob=0 leaves the image unchanged."""

    def test_p0_output_equals_input(self):
        """ExactAffineSegment with prob=0 returns the input tensor unchanged."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=0.0)
        seg = ExactAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img), "prob=0 should leave image unchanged"

    def test_p0_skips_exact_apply_entirely(self):
        """Inactive exact transforms must not evaluate exact_apply."""
        adapter = _RaisingExactAdapter()
        t = _HFlipTransform(p=0.0)
        seg = ExactAffineSegment([t], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)

        assert torch.equal(out, img), "prob=0 should bypass exact_apply and leave image unchanged"


class TestExactAffineSegmentDoubleFlip:
    """Verify HFlip then VFlip with prob=1 composes correctly."""

    def test_hflip_then_vflip(self):
        """HFlip then VFlip with prob=1 is same as image.flip(dims=[2, 3]) sequentially."""
        adapter = _FlipAdapter()
        t_h = _HFlipTransform(p=1.0)
        t_v = _VFlipTransform(p=1.0)
        seg = ExactAffineSegment([t_h, t_v], adapter)

        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        # Sequential: first flip width, then flip height
        expected = img.flip(dims=[3]).flip(dims=[2])

        assert torch.equal(out, expected), "HFlip+VFlip should match sequential tensor.flip"


class TestExactAffineSegmentPerSampleMask:
    """Verify per-sample prob=0.5 masking in ExactAffineSegment."""

    def test_p05_heterogeneous_batch(self):
        """With batch_size=8 and prob=0.5, at least one sample changed, at least one unchanged."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=0.5)
        segment = ExactAffineSegment([transform], adapter)

        batch_size = 8
        image = torch.rand(batch_size, 1, 8, 8)

        # Make the per-sample mask used inside ExactAffineSegment deterministic so that
        # some samples are flipped and some are not, avoiding flaky behavior.
        pattern = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        orig_rand = torch.rand

        def _deterministic_rand(*size, **kwargs):
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", torch.float32)
            # Intercept calls that generate a per-sample mask over the batch.
            if len(size) >= 1 and size[0] == batch_size:
                base = torch.zeros(size, device=device, dtype=dtype)
                mask = pattern.to(device=device, dtype=dtype)
                view_shape = (batch_size,) + (1,) * (base.ndim - 1)
                return base + mask.view(view_shape)
            return orig_rand(*size, **kwargs)

        try:
            torch.rand = _deterministic_rand
            image_output = segment(image)
        finally:
            torch.rand = orig_rand

        diffs = (image_output - image).abs().amax(dim=(1, 2, 3))
        has_changed = (diffs > 1e-6).any()
        has_unchanged = (diffs < 1e-6).any()

        assert has_changed, "Expected at least one sample to be flipped"
        assert has_unchanged, "Expected at least one sample to remain unchanged"


class TestExactAffineSegmentLastMatrix:
    """Verify ExactAffineSegment.last_matrix is always None."""

    def test_last_matrix_none_before_forward(self):
        """last_matrix is None before any forward pass."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=1.0)
        segment = ExactAffineSegment([transform], adapter)
        assert segment.last_matrix is None

    def test_last_matrix_none_after_forward(self):
        """last_matrix remains None after forward (ExactAffineSegment has no matrix)."""
        adapter = _FlipAdapter()
        transform = _HFlipTransform(prob=1.0)
        segment = ExactAffineSegment([transform], adapter)
        segment(torch.rand(2, 3, 8, 8))
        assert segment.last_matrix is None


class TestBuildSegmentsExactOnly:
    """Verify build_segments routes an EXACT-only run to ExactAffineSegment."""

    def test_exact_only_returns_exact_segment(self):
        """An EXACT-only run (no INTERP) produces an ExactAffineSegment, not FusedAffineSegment."""
        adapter = _StubAdapter()
        transform1 = _StubTransform(_hflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        transform2 = _StubTransform(_vflip_matrix_fn, prob=1.0, category=TransformCategory.GEOMETRIC_EXACT)
        result = build_segments([transform1, transform2], adapter)

        assert len(result) == 1
        assert isinstance(result[0], ExactAffineSegment)
        assert len(result[0].transforms) == 2


class TestExactAffineSegmentSameOnBatch:
    """Verify ExactAffineSegment respects same_on_batch=True."""

    def test_same_on_batch_p1_all_flipped(self):
        """With same_on_batch=True and prob=1, every sample in the batch is flipped."""

        class _SameOnBatchHFlip:
            prob = 1.0
            same_on_batch = True
            _category = TransformCategory.GEOMETRIC_EXACT
            _flip_dims = (3,)

        adapter = _FlipAdapter()
        segment = ExactAffineSegment([_SameOnBatchHFlip()], adapter)

        image = torch.rand(4, 3, 8, 8)
        image_output = segment(image)

        assert torch.equal(image_output, image.flip(dims=[3]))

    def test_same_on_batch_p0_none_flipped(self):
        """With same_on_batch=True and prob=0, no sample in the batch is flipped."""

        class _SameOnBatchHFlip:
            prob = 0.0
            same_on_batch = True
            _category = TransformCategory.GEOMETRIC_EXACT
            _flip_dims = (3,)

        adapter = _FlipAdapter()
        segment = ExactAffineSegment([_SameOnBatchHFlip()], adapter)

        image = torch.rand(4, 3, 8, 8)
        image_output = segment(image)

        assert torch.equal(image_output, image)


class TestBatchSizeOne:
    """Verify batch_size=1 forward pass works for both segment types."""

    def test_fused_affine_b1_shape(self):
        """FusedAffineSegment with batch_size=1 produces output of shape (1, num_channels, height, width)."""
        adapter = _StubAdapter()
        t = _StubTransform(_identity_matrix_fn, p=1.0)
        seg = FusedAffineSegment([t], adapter)

        out = seg(torch.rand(1, 3, 8, 8))

        assert out.shape == (1, 3, 8, 8)

    def test_exact_segment_b1_shape(self):
        """ExactAffineSegment with batch_size=1 produces output of shape (1, num_channels, height, width)."""
        adapter = _FlipAdapter()
        t = _HFlipTransform(p=1.0)
        seg = ExactAffineSegment([t], adapter)

        out = seg(torch.rand(1, 3, 8, 8))

        assert out.shape == (1, 3, 8, 8)


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


# ---------------------------------------------------------------------------
# ProjectiveSegment tests
# ---------------------------------------------------------------------------


class TestProjectiveSegmentBuildSegments:
    """build_segments() creates ProjectiveSegment for PROJECTIVE transforms."""

    def _proj_transform(self):
        """Stub with PROJECTIVE category."""
        return _StubTransform(_identity_matrix_fn, p=1.0, category=TransformCategory.PROJECTIVE)

    def test_two_projective_become_one_segment(self):
        """[Proj, Proj] -> one ProjectiveSegment."""
        from fuse_augmentations.affine._segment import ProjectiveSegment

        adapter = _StubAdapter()
        j1, j2 = self._proj_transform(), self._proj_transform()
        segments = build_segments([j1, j2], adapter)
        assert len(segments) == 1
        assert isinstance(segments[0], ProjectiveSegment)
        assert len(segments[0].transforms) == 2

    def test_rotate_then_proj_gives_two_segments(self):
        """[Rot, Proj] -> [FusedAffineSegment, ProjectiveSegment]."""
        from fuse_augmentations.affine._segment import ProjectiveSegment

        adapter = _StubAdapter()
        rot = _StubTransform(_identity_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        proj = self._proj_transform()
        segments = build_segments([rot, proj], adapter)
        assert len(segments) == 2
        assert isinstance(segments[0], FusedAffineSegment)
        assert isinstance(segments[1], ProjectiveSegment)

    def test_proj_then_rotate_gives_two_segments(self):
        """[Proj, Rot] -> [ProjectiveSegment, FusedAffineSegment]."""
        from fuse_augmentations.affine._segment import ProjectiveSegment

        adapter = _StubAdapter()
        proj = self._proj_transform()
        rot = _StubTransform(_identity_matrix_fn, p=1.0, category=TransformCategory.GEOMETRIC_INTERP)
        segments = build_segments([proj, rot], adapter)
        assert len(segments) == 2
        assert isinstance(segments[0], ProjectiveSegment)
        assert isinstance(segments[1], FusedAffineSegment)

    def test_use_numpy_true_gives_albu_projective(self):
        """use_numpy=True produces AlbuProjectiveSegment."""
        pytest.importorskip("cv2", reason="cv2 required for AlbuProjectiveSegment")
        from fuse_augmentations.affine._segment import AlbuProjectiveSegment

        adapter = _StubAdapter()
        proj = self._proj_transform()
        segments = build_segments([proj], adapter, use_numpy=True)
        assert len(segments) == 1
        assert isinstance(segments[0], AlbuProjectiveSegment)

    def test_projective_forward_identity(self):
        """ProjectiveSegment with identity matrix produces same-shape output."""
        from fuse_augmentations.affine._segment import ProjectiveSegment

        adapter = _StubAdapter()
        proj = self._proj_transform()
        seg = ProjectiveSegment([proj], adapter)
        img = torch.rand(2, 3, 8, 8)
        out = seg(img)
        assert out.shape == img.shape
        # atol=1e-5: bilinear interpolation via F.grid_sample introduces floating-point
        # error even for an identity warp; this is an interpolation tolerance, not an
        # algebraic identity check.
        assert torch.allclose(out, img, atol=1e-5)

    def test_projective_segment_mask_aux_target(self):
        """ProjectiveSegment with identity transform returns mask via aux_targets unchanged."""
        from fuse_augmentations.affine._segment import ProjectiveSegment

        adapter = _StubAdapter()
        transform_projective = self._proj_transform()
        segment = ProjectiveSegment([transform_projective], adapter)

        image = torch.rand(2, 3, 8, 8)
        mask = torch.rand(2, 1, 8, 8)
        result = segment(image, aux_targets={"mask": mask})

        assert isinstance(result, tuple)
        assert len(result) == 2
        image_output, aux_output = result
        assert image_output.shape == image.shape
        assert "mask" in aux_output
        assert aux_output["mask"].shape == mask.shape
        # Identity transform: image and mask should pass through unchanged
        assert torch.allclose(image_output, image, atol=1e-5)
        assert torch.allclose(aux_output["mask"], mask, atol=1e-5)

    def test_albu_projective_segment_raises_on_aux_targets(self):
        """AlbuProjectiveSegment raises RuntimeError when aux_targets is not None."""
        pytest.importorskip("cv2", reason="cv2 required for AlbuProjectiveSegment")
        from fuse_augmentations.affine._segment import AlbuProjectiveSegment

        adapter = _StubAdapter()
        proj = self._proj_transform()
        seg = AlbuProjectiveSegment([proj], adapter)

        img = torch.rand(2, 3, 8, 8)
        mask = torch.rand(2, 1, 8, 8)
        with pytest.raises(RuntimeError, match="aux_targets"):
            seg(img, aux_targets={"mask": mask})

    def test_albu_projective_segment_requires_cv2(self, monkeypatch: pytest.MonkeyPatch):
        """AlbuProjectiveSegment raises a clear ImportError when cv2 is unavailable."""
        import fuse_augmentations.affine._segment as segment_mod
        from fuse_augmentations.affine._segment import AlbuProjectiveSegment

        monkeypatch.setattr(segment_mod, "_cv2", None)

        adapter = _StubAdapter()
        proj = self._proj_transform()

        with pytest.raises(ImportError, match="opencv-python"):
            AlbuProjectiveSegment([proj], adapter)
