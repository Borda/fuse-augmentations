"""Tests for POINTWISE_LINEAR TransformCategory -- enum, reordering, and color-space fusion algebra.

Covers:
- TransformCategory.POINTWISE_LINEAR enum member exists with correct string value
- reorder_pointwise treats POINTWISE_LINEAR identically to POINTWISE (reorderable past geometric ops)
- build_segments treats POINTWISE_LINEAR as a segment barrier (pass-through)
- Property-based tests for the 4x4 homogeneous color-space affine fusion law:
    (M2, b2) o (M1, b1) = (M2*M1, M2*b1 + b2)
  This validates the mathematical foundation proved in docs/math/fusible-categories-proofs.md
  and will be the correctness invariant for FusedColorSegment when implemented.

"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from fuse_augmentations.affine.segment import (
    ExactAffineSegment,
    FusedAffineSegment,
    build_segments,
    reorder_pointwise,
)
from fuse_augmentations.types import TransformCategory


class _GeoTransform:
    """Stub GEOMETRIC_INTERP transform."""

    def __init__(self):
        self._category = TransformCategory.GEOMETRIC_INTERP
        self.p = 1.0


class _ExactTransform:
    """Stub GEOMETRIC_EXACT transform."""

    def __init__(self):
        self._category = TransformCategory.GEOMETRIC_EXACT
        self.p = 1.0


class _PointwiseTransform:
    """Stub POINTWISE transform."""

    def __init__(self):
        self._category = TransformCategory.POINTWISE


class _PointwiseLinearTransform:
    """Stub POINTWISE_LINEAR transform."""

    def __init__(self):
        self._category = TransformCategory.POINTWISE_LINEAR


class _BarrierTransform:
    """Stub SPATIAL_KERNEL transform."""

    def __init__(self):
        self._category = TransformCategory.SPATIAL_KERNEL


class _StubAdapter:
    """Minimal TransformAdapter for unit tests -- no backend dependency."""

    def category(self, transform):
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        batch_size = input_shape[0]
        return {"_batch_size": torch.tensor([batch_size])}

    def build_matrix(self, transform, params, height, width):
        batch_size = int(params["_batch_size"].item())
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).clone()

    def call_nonfused(self, transform, image, **kwargs):
        return image

    def exact_flip_dims(self, transform):
        return [3]

    def exact_apply(self, transform, image):
        return image.flip(dims=[3])


class TestPointwiseLinearEnum:
    """POINTWISE_LINEAR enum member -- existence and value."""

    def test_member_exists(self):
        """TransformCategory.POINTWISE_LINEAR is accessible."""
        assert hasattr(TransformCategory, "POINTWISE_LINEAR")

    def test_string_value(self):
        """TransformCategory.POINTWISE_LINEAR has value 'pointwise_linear'."""
        assert TransformCategory.POINTWISE_LINEAR.value == "pointwise_linear"

    def test_is_enum_member(self):
        """POINTWISE_LINEAR is a proper TransformCategory member."""
        assert TransformCategory.POINTWISE_LINEAR in set(TransformCategory)

    def test_distinct_from_pointwise(self):
        """POINTWISE_LINEAR and POINTWISE are distinct enum members."""
        assert TransformCategory.POINTWISE_LINEAR is not TransformCategory.POINTWISE
        assert TransformCategory.POINTWISE_LINEAR != TransformCategory.POINTWISE


class TestPointwiseLinearReordering:
    """reorder_pointwise treats POINTWISE_LINEAR as reorderable (like POINTWISE)."""

    def test_pl_moved_after_geometric(self):
        """[Rotate, LinearColor, Scale] reorders to [Rotate, Scale, LinearColor]"""
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        linear = _PointwiseLinearTransform()
        scale = _GeoTransform()

        result = reorder_pointwise([rotate, linear, scale], adapter)

        cats = [adapter.category(tfm) for tfm in result]
        assert cats == [
            TransformCategory.GEOMETRIC_INTERP,
            TransformCategory.GEOMETRIC_INTERP,
            TransformCategory.POINTWISE_LINEAR,
        ]

    def test_pl_and_pw_both_moved_after_geometric(self):
        """[Rotate, POINTWISE, POINTWISE_LINEAR, Scale] -> [Rotate, Scale, pw, pl] or same deferred order.

        Both POINTWISE and POINTWISE_LINEAR are reorderable past geometric ops; this verifies that mixing them does not
        break the invariant that all geometric ops precede all reorderable ops in the result.

        """
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        pointwise = _PointwiseTransform()
        pointwise_linear = _PointwiseLinearTransform()
        scale = _GeoTransform()

        result = reorder_pointwise([rotate, pointwise, pointwise_linear, scale], adapter)

        # All geometric ops must come before all reorderable ops
        geo_indices = [
            idx for idx, tfm in enumerate(result) if adapter.category(tfm) == TransformCategory.GEOMETRIC_INTERP
        ]
        pw_indices = [idx for idx, tfm in enumerate(result) if adapter.category(tfm) == TransformCategory.POINTWISE]
        pl_indices = [
            idx for idx, tfm in enumerate(result) if adapter.category(tfm) == TransformCategory.POINTWISE_LINEAR
        ]
        assert len(result) == 4
        assert max(geo_indices) < min(pw_indices + pl_indices), "All geometric ops should precede all reorderable ops"

    def test_pl_does_not_cross_spatial_kernel_barrier(self):
        """POINTWISE_LINEAR is not reordered across a SPATIAL_KERNEL barrier.

        SPATIAL_KERNEL ops (blur, sharpen) read pixel neighborhoods, so a color-space op crossing them would change the
        result. The reorderer must respect this barrier and only reorder within the post-barrier sub-sequence.

        """
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        barrier = _BarrierTransform()
        linear = _PointwiseLinearTransform()
        scale = _GeoTransform()

        result = reorder_pointwise([rotate, barrier, linear, scale], adapter)

        cats = [adapter.category(tfm) for tfm in result]
        # rotate before barrier, linear before scale after the barrier (linear IS moved past scale)
        assert cats[0] == TransformCategory.GEOMETRIC_INTERP
        assert cats[1] == TransformCategory.SPATIAL_KERNEL
        # After barrier: [linear, scale] -> reordered to [scale, linear]
        after_barrier = cats[2:]
        geo_after = [cat for cat in after_barrier if cat == TransformCategory.GEOMETRIC_INTERP]
        pl_after = [cat for cat in after_barrier if cat == TransformCategory.POINTWISE_LINEAR]
        assert geo_after
        assert pl_after
        # geometric comes before POINTWISE_LINEAR in the post-barrier stretch
        assert after_barrier.index(TransformCategory.GEOMETRIC_INTERP) < after_barrier.index(
            TransformCategory.POINTWISE_LINEAR
        )

    def test_pl_already_after_geometric_is_stable(self):
        """[Rotate, Scale, POINTWISE_LINEAR] is already stable under reordering."""
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        scale = _GeoTransform()
        linear = _PointwiseLinearTransform()

        original = [rotate, scale, linear]
        result = reorder_pointwise(original, adapter)

        assert result == original

    def test_pl_only_pipeline_unchanged(self):
        """Albu pipeline with only POINTWISE_LINEAR ops is unchanged."""
        adapter = _StubAdapter()
        pl1 = _PointwiseLinearTransform()
        pl2 = _PointwiseLinearTransform()

        result = reorder_pointwise([pl1, pl2], adapter)

        assert result == [pl1, pl2]


class TestPointwiseLinearSegmentation:
    """build_segments treats POINTWISE_LINEAR as a pass-through barrier."""

    def test_pl_breaks_geometric_segment(self):
        """[Rotate, POINTWISE_LINEAR, Scale] -> [FusedAffine, pl, FusedAffine] (two separate segments)

        Before color-segment fusion was implemented, POINTWISE_LINEAR acted as a barrier that prevented adjacent
        geometric ops from collapsing into one warp. This test guards the pre-fusion behaviour.

        """
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        linear = _PointwiseLinearTransform()
        scale = _GeoTransform()

        segments = build_segments([rotate, linear, scale], adapter)

        assert len(segments) == 3
        assert isinstance(segments[0], FusedAffineSegment)
        assert segments[1] is linear
        assert isinstance(segments[2], FusedAffineSegment)

    def test_pl_passes_through_as_is(self):
        """POINTWISE_LINEAR transforms are returned verbatim in build_segments output."""
        adapter = _StubAdapter()
        linear = _PointwiseLinearTransform()

        segments = build_segments([linear], adapter)

        assert len(segments) == 1
        assert segments[0] is linear

    def test_pl_does_not_merge_with_geometric(self):
        """Albu POINTWISE_LINEAR between two geometric ops creates two separate fused segments.

        Compares the split case [geo, pl, geo] (3 segments) against the merged case [geo, geo] (1 segment) to confirm
        that POINTWISE_LINEAR genuinely partitions the geometric run rather than just hiding inside it.

        """
        adapter = _StubAdapter()
        geo1 = _GeoTransform()
        geo2 = _GeoTransform()
        linear = _PointwiseLinearTransform()

        segments_split = build_segments([geo1, linear, geo2], adapter)
        segments_merged = build_segments([geo1, geo2], adapter)

        # Split version: 3 elements (fused, pl, fused)
        assert len(segments_split) == 3
        # Merged version: 1 fused segment
        assert len(segments_merged) == 1

    def test_consecutive_pl_both_pass_through(self):
        """Two consecutive POINTWISE_LINEAR ops both appear in output unchanged."""
        adapter = _StubAdapter()
        pl1 = _PointwiseLinearTransform()
        pl2 = _PointwiseLinearTransform()

        segments = build_segments([pl1, pl2], adapter)

        assert len(segments) == 2
        assert segments[0] is pl1
        assert segments[1] is pl2

    def test_mixed_pointwise_and_pointwise_linear_both_act_as_barriers(self):
        """POINTWISE and POINTWISE_LINEAR both act as pass-through barriers in segmentation.

        [Rotate, POINTWISE, POINTWISE_LINEAR, Scale] -> [FusedAffine(Rotate), pw, pl, FusedAffine(Scale)]

        """
        adapter = _StubAdapter()
        rotate = _GeoTransform()
        pointwise = _PointwiseTransform()
        pointwise_linear = _PointwiseLinearTransform()
        scale = _GeoTransform()

        segments = build_segments([rotate, pointwise, pointwise_linear, scale], adapter)

        # 4 segments: fused(rotate), pointwise, pointwise_linear, fused(scale)
        assert len(segments) == 4
        assert isinstance(segments[0], FusedAffineSegment)
        assert segments[1] is pointwise
        assert segments[2] is pointwise_linear
        assert isinstance(segments[3], FusedAffineSegment)

    def test_geometric_exact_pl_geometric_interp(self):
        """[ExactOp, POINTWISE_LINEAR, InterpOp] -> [ExactSegment, pl, FusedSegment]

        Exact and interp geometric ops live in separate segment classes; a POINTWISE_LINEAR between them keeps the
        segment kinds distinct and emits the color op as a pass-through middle element.

        """
        adapter = _StubAdapter()
        exact = _ExactTransform()
        linear = _PointwiseLinearTransform()
        interp = _GeoTransform()

        segments = build_segments([exact, linear, interp], adapter)

        assert len(segments) == 3
        assert isinstance(segments[0], ExactAffineSegment)
        assert segments[1] is linear
        assert isinstance(segments[2], FusedAffineSegment)


#
# These tests validate the mathematical foundation of POINTWISE_LINEAR fusion
# proved in docs/math/fusible-categories-proofs.md section 1.2:
#
#   (M2, b2) o (M1, b1) = (M2*M1, M2*b1 + b2)
#
# in homogeneous 4x4 form:
#   A_fused = A2 * A1
#   where albu = [[M, b], [0^T, 1]]
#
# These tests confirm the algebra is correct in float64. When FusedColorSegment
# is implemented in a later phase, these properties should hold for its output.

DEFAULT_DTYPE = torch.float64


def _make_color_matrix(mat: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Build a (1, 4, 4) homogeneous color-space affine matrix from (3,3) matrix and (3,) bias."""
    hom_mat = torch.eye(4, dtype=mat.dtype)
    hom_mat[:3, :3] = mat
    hom_mat[:3, 3] = bias
    return hom_mat.unsqueeze(0)


def _apply_color_matrix(mat: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Apply a (1, 4, 4) color matrix to a (3,) RGB vector; returns (3,) result."""
    c_hom = torch.cat([color, torch.ones(1, dtype=color.dtype)])  # (4,)
    return (mat[0] @ c_hom)[:3]


class TestColorMatrixFusionAlgebra:
    """Property-based tests for the 4x4 homogeneous color-space affine fusion law.

    Validates (M2, b2) ∘ (M1, b1) = (M2*M1, M2*b1 + b2) as proved in docs/math/fusible-categories-proofs.md section 1.2.

    """

    @given(
        mult1=floats(min_value=0.1, max_value=3.0),
        mult2=floats(min_value=0.1, max_value=3.0),
        bias1=floats(min_value=-1.0, max_value=1.0),
        bias2=floats(min_value=-1.0, max_value=1.0),
        red=floats(min_value=0.0, max_value=1.0),
        green=floats(min_value=0.0, max_value=1.0),
        b_val=floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_brightness_fusion_equals_sequential(
        self, mult1: float, mult2: float, bias1: float, bias2: float, red: float, green: float, b_val: float
    ) -> None:
        """Two diagonal brightness ops fuse: A2*A1 gives same result as sequential application

        Op1: c' = mult1*c + bias1*1 (brightness scale + offset, applied identically per channel)
        Op2: c'' = mult2*c' + bias2*1

        Sequential: c'' = mult2*(mult1*c + bias1) + bias2 = (mult2*mult1)*c + (mult2*bias1 + bias2)
        Fused:      A_fused = A2 * A1 applied once.
        """
        mat1 = mult1 * torch.eye(3, dtype=DEFAULT_DTYPE)
        bvec1 = bias1 * torch.ones(3, dtype=DEFAULT_DTYPE)
        mat2 = mult2 * torch.eye(3, dtype=DEFAULT_DTYPE)
        bvec2 = bias2 * torch.ones(3, dtype=DEFAULT_DTYPE)

        color_mat1 = _make_color_matrix(mat1, bvec1)
        color_mat2 = _make_color_matrix(mat2, bvec2)

        color_vec = torch.tensor([red, green, b_val], dtype=DEFAULT_DTYPE)

        color_seq = _apply_color_matrix(color_mat2, _apply_color_matrix(color_mat1, color_vec))
        fused_mat = torch.bmm(color_mat2, color_mat1)
        color_fused = _apply_color_matrix(fused_mat, color_vec)

        assert torch.allclose(color_seq, color_fused, atol=1e-9), (
            f"Fused and sequential results differ: seq={color_seq.tolist()}, fused={color_fused.tolist()}"
        )

    @given(
        seed=integers(min_value=0, max_value=9999),
        n_ops=integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100)
    def test_n_color_ops_fuse_to_single_matrix(self, seed: int, n_ops: int) -> None:
        """N consecutive linear color ops: A_N*...*A_1 applied once equals sequential application

        Validates the inductive proof from D.5: any chain of POINTWISE_LINEAR ops can be collapsed to a single 4x4
        matrix.
        """
        torch.manual_seed(seed)

        matrices = []
        biases = []
        for _ in range(n_ops):
            col_mat = torch.randn(3, 3, dtype=DEFAULT_DTYPE) * 0.3 + torch.eye(3, dtype=DEFAULT_DTYPE)
            bias_vec = torch.randn(3, dtype=DEFAULT_DTYPE) * 0.1
            matrices.append(col_mat)
            biases.append(bias_vec)

        color_vec = torch.rand(3, dtype=DEFAULT_DTYPE)

        color_seq = color_vec.clone()
        for col_mat, bias_vec in zip(matrices, biases, strict=True):
            color_seq = col_mat @ color_seq + bias_vec

        fused_mat = _make_color_matrix(matrices[0], biases[0])
        for col_mat, bias_vec in zip(matrices[1:], biases[1:], strict=True):
            color_mat_i = _make_color_matrix(col_mat, bias_vec)
            fused_mat = torch.bmm(color_mat_i, fused_mat)

        color_fused = _apply_color_matrix(fused_mat, color_vec)

        assert torch.allclose(color_seq, color_fused, atol=1e-8), (
            f"N={n_ops} ops: fused and sequential differ; seed={seed}"
        )

    @given(
        alpha=floats(min_value=0.1, max_value=3.0),
        beta=floats(min_value=0.1, max_value=3.0),
        mean=floats(min_value=0.0, max_value=1.0),
        red=floats(min_value=0.0, max_value=1.0),
        green=floats(min_value=0.0, max_value=1.0),
        b_val=floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=150)
    def test_brightness_then_contrast_non_commutative(
        self, alpha: float, beta: float, mean: float, red: float, green: float, b_val: float
    ) -> None:
        """Brightness(a) then Contrast(b,mean) != Contrast then Brightness in general.

        Validates the non-commutativity result from D.5 section 1.3. The counter-example uses diagonal ops with bias,
        confirming that POINTWISE_LINEAR order must be preserved.

        """
        mat_brightness = alpha * torch.eye(3, dtype=DEFAULT_DTYPE)
        bias_brightness = torch.zeros(3, dtype=DEFAULT_DTYPE)

        mat_contrast = beta * torch.eye(3, dtype=DEFAULT_DTYPE)
        bias_contrast = (1.0 - beta) * mean * torch.ones(3, dtype=DEFAULT_DTYPE)

        brightness_mat = _make_color_matrix(mat_brightness, bias_brightness)
        contrast_mat = _make_color_matrix(mat_contrast, bias_contrast)

        color_vec = torch.tensor([red, green, b_val], dtype=DEFAULT_DTYPE)

        bc_mat = torch.bmm(contrast_mat, brightness_mat)
        color_bc = _apply_color_matrix(bc_mat, color_vec)

        cb_mat = torch.bmm(brightness_mat, contrast_mat)
        color_cb = _apply_color_matrix(cb_mat, color_vec)

        color_seq_bc = _apply_color_matrix(contrast_mat, _apply_color_matrix(brightness_mat, color_vec))
        assert torch.allclose(color_bc, color_seq_bc, atol=1e-9), "Fused brightness->contrast should match sequential"

        expected_diff = abs((alpha - 1.0) * (1.0 - beta) * mean)
        if expected_diff > 1e-3:
            assert not torch.allclose(color_bc, color_cb, atol=expected_diff * 0.5), (
                f"brightness->contrast should differ from contrast->brightness when "
                f"alpha={alpha:.3f}, beta={beta:.3f}, mean={mean:.3f}: expected per-channel diff~{expected_diff:.3e}, "
                f"got color_bc={color_bc.tolist()}, color_cb={color_cb.tolist()}"
            )

    @given(
        seed=integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=100)
    def test_color_matrix_composition_law(self, seed: int) -> None:
        """A_fused = A2 * A1 satisfies M_fused = M2*M1 and b_fused = M2*b1 + b2.

        Direct verification of the composition formula from D.5 section 1.2.

        """
        torch.manual_seed(seed)
        M1 = torch.randn(3, 3, dtype=DEFAULT_DTYPE)
        b1 = torch.randn(3, dtype=DEFAULT_DTYPE)
        M2 = torch.randn(3, 3, dtype=DEFAULT_DTYPE)
        b2 = torch.randn(3, dtype=DEFAULT_DTYPE)

        A1 = _make_color_matrix(M1, b1)
        A2 = _make_color_matrix(M2, b2)
        A_fused = torch.bmm(A2, A1)[0]

        M_fused_expected = M2 @ M1
        b_fused_expected = M2 @ b1 + b2

        assert torch.allclose(A_fused[:3, :3], M_fused_expected, atol=1e-10), "M_fused = M2*M1 violated"
        assert torch.allclose(A_fused[:3, 3], b_fused_expected, atol=1e-10), "b_fused = M2*b1 + b2 violated"
