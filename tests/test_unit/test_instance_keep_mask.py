"""Post-warp instance survival: the caller gets a mask, never pre-filtered boxes (FA-5).

A warp pushes instances off the canvas and clips others to slivers. Deciding which ones a training recipe should still
see is the caller's policy, so this package supplies the geometry — clip to the canvas, measure what is left — and
returns a boolean mask over the input instance axis. Returning filtered boxes instead would silently desynchronise every
other modality sharing that axis.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import clip_bbox_xyxy, instance_keep_mask

CANVAS = 8.0


class TestSurvivalRules:
    """Size and visibility thresholds decide survival, and both come from the caller."""

    def test_an_instance_warped_fully_off_canvas_is_dropped(self) -> None:
        """A box entirely outside the canvas clips to nothing and does not survive.

        This is the case the whole helper exists for: after a translate or a downscale the
        instance has no pixels left in the image, yet its unclipped box is still a perfectly
        well-formed rectangle that every shape check accepts.

        """
        warped = torch.tensor([[[-20.0, -20.0, -12.0, -12.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.1)

        assert keep.tolist() == [[False]]

    def test_an_untouched_instance_is_kept(self) -> None:
        """A box fully inside the canvas keeps its whole area and survives any threshold.

        The complement of the case above — without it, a helper that dropped everything would pass the drop test.

        """
        warped = torch.tensor([[[2.0, 2.0, 6.0, 6.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.9)

        assert keep.tolist() == [[True]]

    @pytest.mark.parametrize(
        ("min_size", "expected"),
        [pytest.param(1.9, True, id="just-above"), pytest.param(2.1, False, id="just-below")],
    )
    def test_the_size_threshold_is_a_boundary_not_a_gradient(self, min_size: float, expected: bool) -> None:
        """A box clipped to exactly 2 px survives ``min_size=1.9`` and fails ``2.1``.

        Straddling the threshold from both sides is what proves the comparison is on the clipped extent, in pixels,
        rather than on the unclipped box or on a normalised fraction that happens to correlate.

        """
        warped = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=min_size, min_visibility=0.0)

        assert keep.tolist() == [[expected]]

    @pytest.mark.parametrize(
        ("min_visibility", "expected"),
        [pytest.param(0.24, True, id="just-above"), pytest.param(0.26, False, id="just-below")],
    )
    def test_the_visibility_threshold_is_a_boundary_not_a_gradient(self, min_visibility: float, expected: bool) -> None:
        """A box with exactly a quarter of its area left survives ``0.24`` and fails ``0.26``.

        The 4x4 warped box overlaps the canvas corner by 2x2, so visibility is exactly ``0.25``: a ratio taken against
        the clipped area instead of the unclipped one, or against width rather than area, lands somewhere else and fails
        one of these two.

        """
        warped = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=0.0, min_visibility=min_visibility)

        assert keep.tolist() == [[expected]]

    def test_zero_thresholds_drop_nothing_at_all(self) -> None:
        """With both thresholds at zero every instance survives, clipped away or not.

        Zero means "no opinion", not "a sensible minimum": a fully clipped-away box has
        width, height and visibility of exactly zero, and zero clears a zero threshold. The
        defaults therefore encode no recipe, and a caller who wants off-canvas instances
        dropped states the threshold that does it — as the drop test above does.

        """
        warped = torch.tensor([[[7.9, 7.9, 12.0, 12.0], [-5.0, -5.0, -1.0, -1.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        assert instance_keep_mask(warped, clipped).tolist() == [[True, True]]
        assert instance_keep_mask(warped, clipped, min_visibility=1e-6).tolist() == [[True, False]]

    def test_a_degenerate_pre_clip_box_has_zero_visibility(self) -> None:
        """A zero-area warped box does not divide by zero and does not survive a threshold.

        A fully collapsed instance (a scale of zero, or a box that was already degenerate) has no area to keep a
        fraction of; treating it as fully visible would let it through every recipe.

        """
        warped = torch.tensor([[[4.0, 4.0, 4.0, 4.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=0.0, min_visibility=0.1)

        assert keep.tolist() == [[False]]


class TestMaskContract:
    """The mask is a per-instance flag in input order, whatever the batch shape."""

    def test_the_mask_length_always_equals_the_instance_count(self) -> None:
        """Every input instance gets exactly one flag, kept or not.

        The caller filters labels, keypoints and rotated boxes with this mask, so a mask that were itself already
        filtered would corrupt the modalities it is meant to keep in step.

        """
        warped = torch.tensor([[[2.0, 2.0, 6.0, 6.0], [-20.0, -20.0, -12.0, -12.0], [1.0, 1.0, 3.0, 3.0]]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.5)

        assert keep.shape == (1, 3)
        assert keep.tolist() == [[True, False, True]]

    def test_a_batch_keeps_its_leading_dimensions(self) -> None:
        """A ``(B, N, 4)`` input yields a ``(B, N)`` mask, per sample and per instance.

        Every other target helper in this package is batch-leading; a helper that collapsed the batch would force the
        caller to re-derive which sample a flag belonged to.

        """
        warped = torch.tensor([
            [[2.0, 2.0, 6.0, 6.0], [-20.0, -20.0, -12.0, -12.0]],
            [[-20.0, -20.0, -12.0, -12.0], [1.0, 1.0, 7.0, 7.0]],
        ])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.5)

        assert keep.tolist() == [[True, False], [False, True]]

    def test_an_unbatched_box_table_is_accepted(self) -> None:
        """An ``(N, 4)`` table yields an ``(N,)`` mask.

        Downstream holds per-sample tables rather than batched ones at the point it filters, so requiring a batch axis
        would make it reshape around this helper for no reason.

        """
        warped = torch.tensor([[2.0, 2.0, 6.0, 6.0], [-20.0, -20.0, -12.0, -12.0]])
        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.5)

        assert keep.tolist() == [True, False]

    def test_mismatched_box_tables_are_rejected(self) -> None:
        """Clipped and unclipped tables of different shapes raise instead of broadcasting.

        Broadcasting a ``(1, 4)`` against an ``(N, 4)`` would silently compare every instance against one box's area and
        return a plausible mask.

        """
        warped = torch.tensor([[2.0, 2.0, 6.0, 6.0], [1.0, 1.0, 3.0, 3.0]])

        with pytest.raises(ValueError, match="same shape"):
            instance_keep_mask(warped, warped[:1])

    def test_a_non_xyxy_table_is_rejected(self) -> None:
        """A trailing dimension other than 4 raises rather than indexing past the end.

        Passing ``(cx, cy, w, h)`` boxes here is the likely mistake; five-column tables with a score appended are the
        other. Both must fail loudly.

        """
        with pytest.raises(ValueError, match="trailing dimension"):
            instance_keep_mask(torch.zeros(2, 5), torch.zeros(2, 5))


class TestClipping:
    """Clipping is geometry, and the canvas is the pixel extent."""

    def test_clipping_clamps_to_the_canvas_extent(self) -> None:
        """A box overhanging every edge comes back spanning exactly the canvas.

        The extent convention (``[0, width]``, not ``[0, width - 1]``) is what makes the
        visibility ratio an area ratio; the off-by-one version silently biases every
        visibility measurement on boxes that touch an edge.

        """
        warped = torch.tensor([[[-3.0, -3.0, 11.0, 11.0]]])

        clipped = clip_bbox_xyxy(warped, CANVAS, CANVAS)

        assert clipped.tolist() == [[[0.0, 0.0, 8.0, 8.0]]]

    def test_clipping_does_not_mutate_its_input(self) -> None:
        """The caller's warped boxes survive the call unchanged.

        The unclipped table is the denominator of the visibility ratio, so clipping in place would make every instance
        look fully visible.

        """
        warped = torch.tensor([[[-3.0, -3.0, 11.0, 11.0]]])

        clip_bbox_xyxy(warped, CANVAS, CANVAS)

        assert warped.tolist() == [[[-3.0, -3.0, 11.0, 11.0]]]
