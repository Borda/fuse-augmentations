"""Rotated boxes as a first-class target modality (FA-7).

A rotated box is ``(cx, cy, w, h, theta)``. A general affine does not map a rectangle to a rectangle — only a similarity
does — so the transport expands to corners, maps those, and re-fits. That fit is exact under a similarity and
approximate under shear, and these tests pin both halves of that statement rather than only the convenient one.

The package deliberately imposes no canonical form: which representative of a rectangle is "the" one (long-edge, bounded
angle, some other reading) belongs to the caller that also owns the assigner, the loss and the evaluation kernel.

"""

from __future__ import annotations

import math

import pytest
import torch

from fuse_augmentations import (
    FusedCompose,
    corners_to_rboxes,
    mirror_rboxes,
    rbox_envelopes,
    rboxes_to_corners,
    shift_rboxes,
    transform_rboxes,
)


def _matrix(values: list[list[float]]) -> torch.Tensor:
    """Return a ``(1, 3, 3)`` float64 matrix from a nested list."""
    return torch.tensor(values, dtype=torch.float64).unsqueeze(0)


def _rotation(angle: float) -> torch.Tensor:
    """Return a ``(1, 3, 3)`` rotation about the origin by ``angle`` radians."""
    cos, sin = math.cos(angle), math.sin(angle)
    return _matrix([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def box() -> torch.Tensor:
    """Return one ``(1, 1, 5)`` axis-aligned 8x4 rotated box centred at ``(10, 6)``."""
    return torch.tensor([[[10.0, 6.0, 8.0, 4.0, 0.0]]], dtype=torch.float64)


class TestCornerRoundTrip:
    """Expanding to corners and fitting back is lossless for an actual rectangle."""

    def test_corners_then_fit_returns_the_same_box(self, box: torch.Tensor) -> None:
        """A box expanded to corners and re-fitted comes back unchanged.

        Everything else in this module is built on that round trip, so a fit that quietly rotated the parameterisation
        by a quarter turn — swapping ``w`` and ``h`` and adding ``pi/2`` — would corrupt every result while still
        describing the same rectangle.

        """
        assert torch.allclose(corners_to_rboxes(rboxes_to_corners(box)), box, atol=1e-12)

    def test_an_oblique_box_round_trips_too(self) -> None:
        """A box at an angle that is neither axis-aligned nor 45 degrees round-trips.

        The axis-aligned case above hides sign errors in the ``sin`` terms: they contribute zero there and dominate
        here.

        """
        oblique = torch.tensor([[[3.0, -2.0, 7.0, 2.5, 0.7]]], dtype=torch.float64)

        assert torch.allclose(corners_to_rboxes(rboxes_to_corners(oblique)), oblique, atol=1e-12)

    def test_the_fit_uses_every_vertex(self) -> None:
        """A quad perturbed on one vertex moves the fit by less than the perturbation.

        The fit averages opposite sides rather than reading two corners, so a hand-drawn or numerically noisy quad
        degrades gracefully instead of inheriting one bad vertex.

        """
        exact = rboxes_to_corners(torch.tensor([[[0.0, 0.0, 4.0, 2.0, 0.0]]], dtype=torch.float64))
        noisy = exact.clone()
        noisy[0, 0, 0, 0] += 1.0

        drift = (corners_to_rboxes(noisy) - corners_to_rboxes(exact)).abs().max()

        assert 0.0 < float(drift) < 1.0


class TestTransportUnderMatrices:
    """Exact under a similarity, bounded under shear."""

    @pytest.mark.parametrize(
        "angle",
        [pytest.param(0.3, id="oblique"), pytest.param(math.pi / 2, id="quarter-turn")],
    )
    def test_a_pure_rotation_is_exact(self, box: torch.Tensor, angle: float) -> None:
        """Rotating a box gives exactly the rotated rectangle: same extents, angle plus ``angle``.

        A rotation is a similarity, so the corner fit has nothing to approximate. Extents that drift here would mean the
        fit is measuring something other than the side lengths.

        """
        warped = transform_rboxes(box, _rotation(angle))

        assert warped[0, 0, 2].item() == pytest.approx(8.0, abs=1e-9)
        assert warped[0, 0, 3].item() == pytest.approx(4.0, abs=1e-9)
        assert warped[0, 0, 4].item() == pytest.approx(angle, abs=1e-9)

    def test_a_uniform_scale_is_exact(self, box: torch.Tensor) -> None:
        """A uniform scale multiplies both extents and the centre, leaving the angle alone.

        The other similarity worth checking separately: a fit that normalised its direction
        vector incorrectly would still pass the rotation test.

        """
        warped = transform_rboxes(box, _matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))

        assert warped[0, 0, :4].tolist() == pytest.approx([20.0, 12.0, 16.0, 8.0], abs=1e-9)
        assert warped[0, 0, 4].item() == pytest.approx(0.0, abs=1e-9)

    def test_a_non_uniform_scale_is_not_a_rectangle_preserving_map_but_stays_axis_aligned(
        self, box: torch.Tensor
    ) -> None:
        """An axis-aligned box under an anisotropic scale still fits exactly.

        Anisotropic scaling maps *axis-aligned* rectangles to rectangles even though it is not a similarity — worth
        pinning so the "only a similarity is exact" statement is read as being about the general case, not this one.

        """
        warped = transform_rboxes(box, _matrix([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]))

        assert warped[0, 0, :4].tolist() == pytest.approx([20.0, 3.0, 16.0, 2.0], abs=1e-9)

    def test_the_shear_residual_matches_the_documented_expression(self, box: torch.Tensor) -> None:
        """Under a known shear the fitted corners sit exactly where the docstring says.

        Shear sends a rectangle to a parallelogram, which no ``(cx, cy, w, h, theta)`` describes, so the fit *must* have
        a residual. Pinning it to the closed form — rather than to a comfortable inequality — is what makes the
        approximation documented instead of merely tolerated, and it is how the first-order shorthand ``h * sin(s / 2)``
        was found to understate the true value from the third order on.

        """
        shear_angle = 0.2
        matrix = _matrix([[1.0, math.tan(shear_angle), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        exact_corners = torch.einsum("ij,bnkj->bnki", matrix[0, :2, :2], rboxes_to_corners(box))
        fitted_corners = rboxes_to_corners(transform_rboxes(box, matrix))

        residual = (fitted_corners - exact_corners).norm(dim=-1).max().item()
        height = float(box[0, 0, 3])
        closed_form = (height / 2) * math.sqrt(math.tan(shear_angle) ** 2 + (1 - 1 / math.cos(shear_angle)) ** 2)
        assert residual == pytest.approx(closed_form, rel=1e-9)
        assert residual > height * math.sin(shear_angle / 2)

    def test_a_translation_needs_no_corner_round_trip(self, box: torch.Tensor) -> None:
        """``shift_rboxes`` moves only the centre and matches the matrix path exactly.

        A translation introduces no fitting residual at all, so the dedicated path must agree with the general one bit
        for bit rather than merely closely.

        """
        shifted = shift_rboxes(box, 4.0, -3.0)
        warped = transform_rboxes(box, _matrix([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]))

        assert torch.allclose(shifted, warped, atol=1e-12)


class TestMirror:
    """A mirror is an isometry, and twice a mirror is the identity."""

    def test_mirroring_twice_is_the_identity(self) -> None:
        """Two mirrors about the same axis return the original box exactly.

        The cheapest check that the reflection is a true involution rather than an approximate one that drifts under
        repetition.

        """
        oblique = torch.tensor([[[3.0, 5.0, 8.0, 4.0, 0.25]]], dtype=torch.float64)

        assert torch.allclose(mirror_rboxes(mirror_rboxes(oblique, 32), 32), oblique, atol=1e-12)

    def test_the_mirror_axis_matches_the_image_flip(self) -> None:
        """A mirrored box lands where the pipeline's own horizontal flip puts the image.

        The mirror line is ``(width - 1) / 2``, this package's ``align_corners=True`` flip
        axis — not ``width / 2``, which an extent-convention implementation uses. Half a
        pixel of disagreement between image and boxes is exactly the kind of drift that
        survives every shape check.

        """
        boxes = torch.tensor([[[4.0, 6.0, 8.0, 4.0, 0.3]]])
        pipe = FusedCompose.from_params(hflip_p=1.0, data_keys=["input", "rboxes"])

        _, routed = pipe(torch.rand(1, 3, 16, 32), boxes)

        assert torch.allclose(routed, mirror_rboxes(boxes, 32), atol=1e-5)


class TestPipelineRouting:
    """``rboxes`` is an ordinary data key, routed by every path that routes boxes."""

    def test_rboxes_route_through_a_fused_affine_warp(self) -> None:
        """A rotation routes rboxes through the composed matrix, agreeing with the helper.

        The pipeline must not grow a second, subtly different rotated-box implementation: routing and the public helper
        have to be the same arithmetic on the same matrix.

        """
        boxes = torch.tensor([[[16.0, 8.0, 8.0, 4.0, 0.0]]])
        pipe = FusedCompose.from_params(rotation=(30.0, 30.0), data_keys=["input", "rboxes"])

        _, routed = pipe(torch.rand(1, 3, 16, 32), boxes, return_matrix=False)
        matrix = pipe.transform_matrix

        assert torch.allclose(routed, transform_rboxes(boxes, matrix), atol=1e-5)

    def test_rboxes_survive_a_letterbox(self) -> None:
        """A letterbox scales the extents by its ratio and leaves the angle alone.

        The letterbox is a similarity, so a rotated box passes through it exactly — which is what makes an oriented-box
        eval loop able to un-letterbox its predictions.

        """
        boxes = torch.tensor([[[20.0, 10.0, 8.0, 4.0, 0.0]]])
        pipe = FusedCompose.from_params(letterbox=(32, 32), data_keys=["input", "rboxes"])

        _, routed = pipe(torch.rand(1, 3, 20, 40), boxes)

        assert routed[0, 0, 2].item() == pytest.approx(8.0 * 0.8, abs=1e-4)
        assert routed[0, 0, 4].item() == pytest.approx(0.0, abs=1e-6)

    def test_rboxes_and_boxes_route_consistently(self) -> None:
        """A rotated box's envelope after a warp matches the warped axis-aligned box.

        The two modalities describe the same instance, so an oriented pipeline that also carries plain boxes must not
        see them drift apart under the same matrix.

        """
        rboxes = torch.tensor([[[16.0, 8.0, 8.0, 4.0, 0.0]]])
        boxes = torch.tensor([[[12.0, 6.0, 20.0, 10.0]]])
        pipe = FusedCompose.from_params(scale=(1.5, 1.5), data_keys=["input", "bbox_xyxy", "rboxes"])

        _, routed_boxes, routed_rboxes = pipe(torch.rand(1, 3, 16, 32), boxes, rboxes)

        assert torch.allclose(rbox_envelopes(routed_rboxes), routed_boxes, atol=1e-4)

    def test_the_pipeline_returns_un_canonicalized_boxes(self) -> None:
        """A rotation past a quarter turn is reported as-is, not folded into a long-edge form.

        The package promises no canonical form; a caller applying its own convention must be able to see what actually
        came out. Folding silently here would hide a rotation the caller's assigner needs to know about.

        """
        boxes = torch.tensor([[[16.0, 8.0, 8.0, 4.0, 0.0]]])
        pipe = FusedCompose.from_params(rotation=(80.0, 80.0), data_keys=["input", "rboxes"])

        _, routed = pipe(torch.rand(1, 3, 16, 32), boxes)

        assert routed[0, 0, 2].item() == pytest.approx(8.0, abs=1e-3)
        assert abs(routed[0, 0, 4].item()) > math.pi / 4

    def test_a_canonicalizer_callback_is_applied_when_given(self) -> None:
        """``transform_rboxes`` applies a caller-supplied convention and nothing else.

        The hook exists so a caller can pass its own long-edge rule through one place rather than post-processing every
        call site — and so this package never has to hold an opinion it would then have to keep in sync with a
        downstream loss.

        """
        calls: list[torch.Tensor] = []

        def canonicalize(fitted: torch.Tensor) -> torch.Tensor:
            calls.append(fitted)
            return fitted * 0.0

        result = transform_rboxes(
            torch.tensor([[[1.0, 2.0, 4.0, 2.0, 0.0]]], dtype=torch.float64),
            _rotation(0.0),
            canonicalize=canonicalize,
        )

        assert len(calls) == 1
        assert torch.equal(result, torch.zeros_like(result))


class TestShapeContracts:
    """Malformed tables raise instead of being interpreted as something else."""

    def test_a_four_column_table_is_rejected(self) -> None:
        """An ``(N, 4)`` axis-aligned table passed as rboxes raises.

        Passing plain boxes where rotated ones are expected is the likely mistake, and the arithmetic would otherwise
        run on whatever the fifth column happened to alias to.

        """
        with pytest.raises(ValueError, match="trailing dimension of 5"):
            rboxes_to_corners(torch.zeros(1, 2, 4))

    def test_a_malformed_corner_table_is_rejected(self) -> None:
        """A corner table that is not ``(..., 4, 2)`` raises."""
        with pytest.raises(ValueError, match=r"\(4, 2\)"):
            corners_to_rboxes(torch.zeros(1, 2, 3, 2))

    def test_the_envelope_is_the_tight_axis_aligned_box(self) -> None:
        """A 45-degree box's envelope is the square its corners span.

        The envelope is the bridge to the axis-aligned keep-mask machinery, so it has to be the corner extent rather
        than the box's own ``w``/``h``, which would understate a rotated instance.

        """
        rotated = torch.tensor([[[0.0, 0.0, math.sqrt(2.0), math.sqrt(2.0), math.pi / 4]]], dtype=torch.float64)

        envelope = rbox_envelopes(rotated)[0, 0]

        assert envelope.tolist() == pytest.approx([-1.0, -1.0, 1.0, 1.0], abs=1e-9)
