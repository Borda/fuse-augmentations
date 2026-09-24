"""Aspect-preserving letterbox with an exact analytic inverse (FA-4).

A letterbox scales by one ratio and pads the slack, so its map is a pure scale plus translation and inverts exactly —
which is what lets it serve as an inference preprocessor rather than only a training-time resize. It is classified
``CROP_RESIZE_FIXED``, so a geometric run in front of it composes into one matrix and one resample instead of two, and
`return_matrix` hands back the whole chain.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import (
    FusedCompose,
    letterbox_geometry,
    letterbox_matrix,
    transform_bbox_xyxy,
    transform_keypoints,
)
from fuse_augmentations.affine.matrix import inv3x3

GREY = 114.0 / 255.0


def _image(height: int, width: int, value: float = 0.8) -> torch.Tensor:
    """Return a ``(1, 3, height, width)`` constant batch, so pad pixels are unmistakable."""
    return torch.full((1, 3, height, width), value)


class TestLetterboxGeometry:
    """The fit follows the resize-then-pad convention it has to agree with."""

    def test_a_wide_image_pads_top_and_bottom(self) -> None:
        """A 20x40 source into a 32x32 canvas scales by 0.8 and pads the short axis.

        Padding the wrong axis is the classic letterbox bug and stays invisible on the square inputs most tests use, so
        both orientations are checked here and below.

        """
        geometry = letterbox_geometry(height_in=20, width_in=40, height_out=32, width_out=32)

        assert geometry.r == pytest.approx(0.8)
        assert (geometry.new_h, geometry.new_w) == (16, 32)
        assert (geometry.pad_top, geometry.pad_left) == (8, 0)

    def test_a_tall_image_pads_left_and_right(self) -> None:
        """A 40x20 source into a 32x32 canvas pads horizontally instead.

        The mirror of the case above: a fit that transposed height and width anywhere would
        pass one of these two and fail the other.

        """
        geometry = letterbox_geometry(height_in=40, width_in=20, height_out=32, width_out=32)

        assert (geometry.new_h, geometry.new_w) == (32, 16)
        assert (geometry.pad_top, geometry.pad_left) == (0, 8)

    def test_odd_slack_puts_the_extra_pixel_after_the_content(self) -> None:
        """A slack of 5 px splits 2 before and 3 after, not 2.5 each.

        Pads are integers — the floor of half the slack — because the fit has to agree with a resize-then-pad
        implementation pixel for pixel. "Symmetric" here means symmetric up to that floor, and the residual pixel lands
        on the right or the bottom.

        """
        geometry = letterbox_geometry(height_in=10, width_in=10, height_out=15, width_out=15, allow_upscale=False)

        assert geometry.new_h == 10
        assert geometry.pad_top == 2
        assert geometry.out_h - geometry.new_h - geometry.pad_top == 3

    def test_allow_upscale_false_never_magnifies(self) -> None:
        """A source smaller than the canvas is padded, not enlarged, when upscaling is off.

        Eval preprocessors often refuse to invent resolution; the flag has to cap the ratio at exactly ``1.0`` rather
        than merely reduce it.

        """
        capped = letterbox_geometry(height_in=8, width_in=8, height_out=32, width_out=32, allow_upscale=False)
        uncapped = letterbox_geometry(height_in=8, width_in=8, height_out=32, width_out=32)

        assert capped.r == 1.0
        assert capped.pad_top == 12
        assert uncapped.r == 4.0

    def test_a_non_positive_dimension_is_refused(self) -> None:
        """A zero or negative canvas raises rather than producing an infinite ratio."""
        with pytest.raises(ValueError, match="height_in"):
            letterbox_geometry(height_in=0, width_in=8, height_out=8, width_out=8)


class TestAnalyticInverse:
    """The map is a pure scale plus translation, so it inverts exactly."""

    def test_the_matrix_is_scale_plus_translation_only(self) -> None:
        """The forward matrix is exactly ``[[r, 0, px], [0, r, py], [0, 0, 1]]``.

        Any rotation or shear term would break the closed-form inverse downstream's eval loop relies on, and would not
        be visible in a round-trip test that only checks points the same matrix produced.

        """
        matrix = letterbox_matrix(height_in=20, width_in=40, height_out=32, width_out=32, dtype=torch.float64)[0]

        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        assert matrix[0, 0] == matrix[1, 1] == pytest.approx(0.8)
        assert matrix[0, 2].item() == 0.0
        assert matrix[1, 2].item() == 8.0
        assert matrix[2].tolist() == [0.0, 0.0, 1.0]

    @pytest.mark.parametrize(
        ("height_in", "width_in", "allow_upscale"),
        [
            pytest.param(20, 40, True, id="wide-downscale"),
            pytest.param(40, 20, True, id="tall-downscale"),
            pytest.param(8, 8, False, id="pad-only"),
            pytest.param(15, 37, True, id="odd-sizes"),
        ],
    )
    def test_forward_then_inverse_round_trips_arbitrary_points(
        self, height_in: int, width_in: int, allow_upscale: bool
    ) -> None:
        """Points anywhere — content or pad region — return to their source coordinates.

        The pad region matters: a prediction in letterboxed space can legitimately land
        outside the content, and mapping it back must stay well defined rather than clamp
        or diverge. Float64 makes this an exactness check rather than a tolerance one.

        """
        forward = letterbox_matrix(
            height_in=height_in,
            width_in=width_in,
            height_out=32,
            width_out=32,
            allow_upscale=allow_upscale,
            dtype=torch.float64,
        )
        points = torch.tensor(
            [[[0.0, 0.0], [7.5, 3.25], [float(width_in - 1), float(height_in - 1)], [-4.0, -6.0]]],
            dtype=torch.float64,
        )

        letterboxed = transform_keypoints(points, forward)
        recovered = transform_keypoints(letterboxed, inv3x3(forward))

        assert torch.allclose(recovered, points, atol=1e-12)


class TestPipelineIntegration:
    """The letterbox is a segment in the ordinary pipeline, not a bolt-on."""

    def test_the_output_canvas_has_the_requested_size(self) -> None:
        """A letterbox-only pipeline resamples to the target canvas.

        The plainest contract, and the one that proves the transform reaches the shape-changing machinery at all rather
        than being treated as a passthrough.

        """
        out = FusedCompose.from_params(letterbox=(32, 32))(_image(20, 40))

        assert out.shape == (1, 3, 32, 32)

    def test_the_pad_region_takes_the_fill(self) -> None:
        """Pads carry the FA-2 fill, content keeps its value.

        The letterbox is the reason a constant fill exists at all — a padded canvas with a black border rather than the
        recipe's grey is a silent train/eval mismatch.

        """
        out = FusedCompose.from_params(letterbox=(32, 32), fill=GREY)(_image(20, 40))

        assert out[0, :, 0, 16].tolist() == pytest.approx([GREY] * 3)
        assert out[0, :, 16, 16].tolist() == pytest.approx([0.8] * 3)

    def test_an_affine_run_and_a_letterbox_fuse_into_one_segment(self) -> None:
        """Rotation plus letterbox is one segment, one resample, one composed matrix.

        This is the acceptance the downstream work rests on: its hand-rolled ``FusedAffineLetterbox`` exists only
        because two transforms meant two resamples, and its rings are warped by whatever ``return_matrix`` hands back.
        If the two landed as separate segments, the returned matrix would describe the letterbox alone and the rings
        would drift from the image with nothing raising.

        """
        pipe = FusedCompose.from_params(rotation=(20.0, 20.0), letterbox=(32, 32), fill=GREY)

        out, matrix = pipe(_image(20, 40), return_matrix=True)

        assert out.shape == (1, 3, 32, 32)
        assert len(pipe.fusion_plan_descriptors) == 1
        assert matrix.shape == (1, 3, 3)

    def test_the_composed_matrix_is_the_letterbox_after_the_geometry(self) -> None:
        """For a deterministic shift the returned matrix is exactly ``M_letterbox @ M_shift``.

        Composition order is the failure this pins: ``M_letterbox @ M_geo`` and ``M_geo @ M_letterbox`` are both
        well-formed 3x3s, and only one of them describes what the single resample actually did.

        """
        pipe = FusedCompose.from_params(translate_x=(4.0, 4.0), letterbox=(32, 32))

        _, matrix = pipe(_image(20, 40), return_matrix=True)

        shift = torch.eye(3, dtype=torch.float64)
        shift[0, 2] = 4.0
        expected = letterbox_matrix(20, 40, 32, 32, dtype=torch.float64)[0] @ shift
        assert torch.allclose(matrix[0].to(torch.float64), expected, atol=1e-6)

    def test_an_exact_flip_run_also_fuses_with_the_letterbox(self) -> None:
        """A flip-only run in front of a letterbox is one segment too, with a mirrored matrix.

        Flips normally take the exact, interpolation-free path, which builds no matrix at all; ahead of a letterbox they
        have to join the composed warp instead, or a mosaic-shaped pipeline (flip then letterbox) would resample twice
        and hand back a matrix describing only half the chain.

        """
        pipe = FusedCompose.from_params(hflip_p=1.0, letterbox=(32, 32))

        out, matrix = pipe(_image(20, 40), return_matrix=True)

        assert out.shape == (1, 3, 32, 32)
        assert len(pipe.fusion_plan_descriptors) == 1
        assert matrix[0, 0, 0].item() == pytest.approx(-0.8, abs=1e-6)

    def test_the_composed_matrix_maps_source_pixels_to_the_letterboxed_canvas(self) -> None:
        """The returned matrix takes a source-canvas point to where the image content went.

        Checking the matrix against the *image* rather than against another matrix is what
        catches a chain composed in the wrong order: ``M_letterbox @ M_geo`` and
        ``M_geo @ M_letterbox`` are both plausible-looking 3x3s.

        """
        image = torch.zeros(1, 1, 20, 40)
        image[0, 0, 10, 30] = 1.0
        pipe = FusedCompose.from_params(translate_x=(4.0, 4.0), letterbox=(32, 32))

        out, matrix = pipe(image, return_matrix=True)

        point = torch.tensor([[[30.0, 10.0]]], dtype=matrix.dtype)
        mapped = transform_keypoints(point, matrix)[0, 0]
        peak = out[0, 0].flatten().argmax().item()
        assert (peak // 32, peak % 32) == (round(mapped[1].item()), round(mapped[0].item()))

    @pytest.mark.parametrize(
        "shape,output,upscale", [((5, 7), (9, 11), False), ((5, 7), (10, 12), False), ((20, 40), (32, 32), True)]
    )
    def test_letterbox_reports_actual_coordinate_matrix(self, shape, output, upscale) -> None:
        """A standalone letterbox exposes the same geometry its pixels and targets use."""
        height, width = shape
        out_height, out_width = output
        image = torch.zeros(1, 1, height, width)
        image[0, 0, 2, 3] = 1.0
        pipe = FusedCompose.from_params(letterbox=output, allow_upscale=upscale)
        result, matrix = pipe(image, return_matrix=True)
        expected = letterbox_matrix(height, width, out_height, out_width, allow_upscale=upscale)
        assert matrix is not None
        torch.testing.assert_close(matrix, expected, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(matrix, pipe.transform_matrix, rtol=1e-4, atol=1e-6)
        point = torch.tensor([[[3.0, 2.0]]])
        mapped = transform_keypoints(point, matrix)
        recovered = transform_keypoints(mapped, inv3x3(matrix))
        torch.testing.assert_close(recovered, point, rtol=1e-4, atol=1e-6)
        peak = result[0, 0].flatten().argmax().item()
        assert (peak // out_width, peak % out_width) == (round(mapped[0, 0, 1].item()), round(mapped[0, 0, 0].item()))
        boxes = torch.tensor([[[0.0, 0.0, float(width), float(height)]]])
        recovered_boxes = transform_bbox_xyxy(transform_bbox_xyxy(boxes, matrix), inv3x3(matrix))
        torch.testing.assert_close(recovered_boxes, boxes, rtol=1e-4, atol=1e-6)
        with pytest.raises(ValueError, match="crop-resize"):
            pipe.inverse(result)

    def test_boxes_route_through_the_same_letterbox_map(self) -> None:
        """A routed box equals the box mapped by the public ``letterbox_matrix``.

        This is the downstream-agreement check in miniature: the eval loop maps predictions
        back with a matrix it recomputes from sizes alone, so the pipeline's own routing has
        to agree with that matrix rather than with an internal variant of it.

        """
        boxes = torch.tensor([[[2.0, 3.0, 30.0, 15.0]]])
        pipe = FusedCompose.from_params(letterbox=(32, 32), data_keys=["input", "bbox_xyxy"])

        _, routed = pipe(_image(20, 40), boxes)

        expected = transform_bbox_xyxy(boxes, letterbox_matrix(20, 40, 32, 32))
        assert torch.allclose(routed, expected, atol=1e-5)

    def test_a_seeded_pipeline_with_a_letterbox_still_reproduces(self) -> None:
        """The deterministic letterbox consumes no randomness and breaks no seeding.

        A shape-changing op that quietly drew from the stream would desynchronise every later draw; sampling identical
        output twice from one seed rules that out.

        """

        def run() -> torch.Tensor:
            generator = torch.Generator().manual_seed(5)
            pipe = FusedCompose.from_params(rotation=(-30.0, 30.0), letterbox=(32, 32), generator=generator)
            return pipe(_image(20, 40))

        assert torch.equal(run(), run())

    def test_a_backend_pipeline_refuses_a_letterbox(self) -> None:
        """``letterbox=`` with ``backend=`` raises instead of being dropped.

        The backend path routes through ``from_config``, whose op vocabulary has no letterbox; accepting the argument
        there would resize nothing and report success.

        """
        with pytest.raises(NotImplementedError, match="letterbox"):
            FusedCompose.from_params(letterbox=(32, 32), backend="native")
