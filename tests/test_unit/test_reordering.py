"""Unit tests for reorder_pointwise and build_segments reordering."""

from __future__ import annotations

import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._segment import (
    ExactSegment,
    FusedAffineSegment,
    build_segments,
    reorder_pointwise,
)


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
        bsz = input_shape[0]
        return {"_batch_size": torch.tensor([bsz])}

    def build_matrix(self, transform, params, height, width):
        bsz = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(bsz, height, width)
        return torch.eye(3).unsqueeze(0).expand(bsz, -1, -1)

    def call_nonfused(self, transform, image, **kwargs):
        return image

    def exact_flip_dims(self, transform):
        return [3]


def _identity_matrix_fn(B, H, W):
    return torch.eye(3).unsqueeze(0).expand(B, -1, -1)


class TestPointwiseReorder:
    """POINTWISE reorder moves pointwise ops after geometric chains."""

    def test_pointwise_moved_after_geometric(self):
        """[Rotate, Brightness, Scale] reorders to [Rotate, Scale], [Brightness]."""
        adapter = _StubAdapter()
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        brightness = _PointwiseTransform()
        scale = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)

        result = reorder_pointwise([rotate, brightness, scale], adapter)

        assert len(result) == 3
        # Geometric ops first, then pointwise
        assert adapter.category(result[0]) == TransformCategory.GEOMETRIC_INTERP
        assert adapter.category(result[1]) == TransformCategory.GEOMETRIC_INTERP
        assert adapter.category(result[2]) == TransformCategory.POINTWISE
        # Preserve identity of original objects
        assert result[0] is rotate
        assert result[1] is scale
        assert result[2] is brightness

    def test_no_pointwise_unchanged(self):
        """All-geometric list is returned unchanged."""
        adapter = _StubAdapter()
        t1 = _StubTransform(_identity_matrix_fn)
        t2 = _StubTransform(_identity_matrix_fn)
        result = reorder_pointwise([t1, t2], adapter)
        assert result == [t1, t2]

    def test_all_pointwise_unchanged(self):
        """All-pointwise list stays in original order."""
        adapter = _StubAdapter()
        p1 = _PointwiseTransform()
        p2 = _PointwiseTransform()
        result = reorder_pointwise([p1, p2], adapter)
        assert result == [p1, p2]

    def test_empty_list(self):
        """Empty list returns empty."""
        adapter = _StubAdapter()
        assert reorder_pointwise([], adapter) == []


class TestBarrierSplits:
    """Barriers prevent pointwise from crossing; each barrier-bounded stretch reorders independently."""

    def test_barrier_prevents_crossing(self):
        """[Rotate, GaussianBlur, Brightness, Scale] stays barrier-split.

        Brightness cannot cross GaussianBlur barrier, so after reorder: [Rotate] -> [GaussianBlur] -> [Scale,
        Brightness] (Brightness deferred within the second stretch only)

        """
        adapter = _StubAdapter()
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        blur = _BarrierTransform()
        brightness = _PointwiseTransform()
        scale = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)

        result = reorder_pointwise([rotate, blur, brightness, scale], adapter)

        # rotate stays before barrier
        assert result[0] is rotate
        # barrier stays in place
        assert result[1] is blur
        # In the second stretch: [Brightness, Scale] -> [Scale, Brightness]
        assert result[2] is scale
        assert result[3] is brightness

    def test_multiple_barriers(self):
        """Multiple barriers each create independent stretches."""
        adapter = _StubAdapter()
        geo1 = _StubTransform(_identity_matrix_fn)
        pw1 = _PointwiseTransform()
        barrier1 = _BarrierTransform()
        geo2 = _StubTransform(_identity_matrix_fn)
        pw2 = _PointwiseTransform()
        barrier2 = _BarrierTransform()
        geo3 = _StubTransform(_identity_matrix_fn)

        result = reorder_pointwise([geo1, pw1, barrier1, geo2, pw2, barrier2, geo3], adapter)

        # First stretch: [geo1, pw1] -> [geo1, pw1] (geo first, then pw)
        assert result[0] is geo1
        assert result[1] is pw1
        # Barrier 1
        assert result[2] is barrier1
        # Second stretch: [geo2, pw2] -> [geo2, pw2]
        assert result[3] is geo2
        assert result[4] is pw2
        # Barrier 2
        assert result[5] is barrier2
        # Third stretch: [geo3]
        assert result[6] is geo3


class TestExactOnlyDetection:
    """EXACT-only transforms produce ExactSegment (not FusedAffineSegment)."""

    def test_exact_only_creates_exact_segment(self):
        """[HFlip, VFlip] -> ExactSegment (no FusedAffineSegment)."""
        adapter = _StubAdapter()
        hflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)
        vflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)

        segments = build_segments([hflip, vflip], adapter)

        assert len(segments) == 1
        assert isinstance(segments[0], ExactSegment)
        assert not isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 2

    def test_single_exact_creates_exact_segment(self):
        """Single GEOMETRIC_EXACT transform produces ExactSegment."""
        adapter = _StubAdapter()
        hflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)

        segments = build_segments([hflip], adapter)

        assert len(segments) == 1
        assert isinstance(segments[0], ExactSegment)


class TestExactWithInterp:
    """EXACT with INTERP present fuses everything into a single FusedAffineSegment."""

    def test_exact_with_interp_creates_fused_segment(self):
        """[HFlip, Rotate, VFlip] -> single FusedAffineSegment (INTERP present)."""
        adapter = _StubAdapter()
        hflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        vflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)

        segments = build_segments([hflip, rotate, vflip], adapter)

        assert len(segments) == 1
        assert isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 3

    def test_interp_then_exact_fuses(self):
        """[Rotate, HFlip] -> single FusedAffineSegment."""
        adapter = _StubAdapter()
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)
        hflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)

        segments = build_segments([rotate, hflip], adapter)

        assert len(segments) == 1
        assert isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 2

    def test_exact_then_interp_fuses(self):
        """[HFlip, Rotate] -> single FusedAffineSegment."""
        adapter = _StubAdapter()
        hflip = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_EXACT)
        rotate = _StubTransform(_identity_matrix_fn, category=TransformCategory.GEOMETRIC_INTERP)

        segments = build_segments([hflip, rotate], adapter)

        assert len(segments) == 1
        assert isinstance(segments[0], FusedAffineSegment)
        assert len(segments[0].transforms) == 2
