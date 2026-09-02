"""Keypoint pair swapping on an orientation-reversing transform (FA-8).

A mirrored image has its left and right anatomy swapped: the coordinates are correct after the warp, but the *identity*
of each keypoint slot has to follow. Which slots pair with which is dataset schema and stays with the caller; this
package decides only when to apply the permutation, and decides it from the sign of the composed matrix's determinant.

That distinction is the whole point. After fusion a mirror is not a discrete operation any more — it is part of a larger
matrix — and two mirrors compose back to a rotation that must not swap at all.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import FusedCompose, orientation_reversed, permute_keypoint_pairs

#: Three slots: a centre that maps to itself, and one left/right pair.
PAIRS = (0, 2, 1)
#: Distinct points so a swap is visible in the values, not only in the order.
POINTS = torch.tensor([[[4.0, 4.0], [8.0, 6.0], [12.0, 10.0]]])


def _pipe(**kwargs: object) -> FusedCompose:
    """Build a keypoint-carrying pipeline with the pair table installed."""
    return FusedCompose.from_params(data_keys=["input", "keypoints"], keypoint_flip_index=PAIRS, **kwargs)  # type: ignore[arg-type]


def _image() -> torch.Tensor:
    """Return a ``(1, 3, 16, 16)`` batch; content is irrelevant to keypoint routing."""
    return torch.zeros(1, 3, 16, 16)


def _swapped(points: torch.Tensor) -> torch.Tensor:
    """Return ``points`` with the pair table applied, coordinates untouched."""
    return points.index_select(-2, torch.tensor(PAIRS))


class TestDeterminantDrivesTheSwap:
    """Orientation reversal is read off the matrix, never off the transform list."""

    def test_a_mirror_is_detected(self) -> None:
        """A matrix with a negative determinant is reported as orientation-reversing."""
        mirror = torch.tensor([[[-1.0, 0.0, 15.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

        assert orientation_reversed(mirror).tolist() == [True]

    def test_a_rotation_is_not(self) -> None:
        """A rotation keeps the determinant positive however large the angle.

        A rotation past 90 degrees looks like a flip in the image and is not one; keying off appearance rather than the
        determinant would swap here and corrupt the labels.

        """
        angle = torch.tensor(2.5)
        rotation = torch.tensor([
            [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        ])

        assert orientation_reversed(rotation).tolist() == [False]

    def test_the_permutation_only_touches_selected_samples(self) -> None:
        """Samples whose mask is ``False`` keep their keypoint order.

        The gate is per sample because the probability draw is per sample: in one batch some
        images mirror and others do not.

        """
        batch = POINTS.expand(2, -1, -1).contiguous()

        out = permute_keypoint_pairs(batch, torch.tensor(PAIRS), torch.tensor([True, False]))

        assert torch.equal(out[0], _swapped(POINTS)[0])
        assert torch.equal(out[1], POINTS[0])

    def test_a_mismatched_table_is_rejected(self) -> None:
        """A pair table shorter than the keypoint axis raises rather than indexing past it."""
        with pytest.raises(ValueError, match="slots"):
            permute_keypoint_pairs(POINTS, torch.tensor([0, 1]), torch.tensor([True]))

    def test_a_non_permutation_is_rejected_at_construction(self) -> None:
        """A table that repeats or drops a slot raises when the pipeline is built.

        ``(0, 1, 1)`` would duplicate one landmark and silently delete another — a corruption with no shape signature at
        all, so it has to be caught at construction.

        """
        with pytest.raises(ValueError, match="permutation"):
            FusedCompose.from_params(hflip_p=1.0, keypoint_flip_index=(0, 1, 1))


class TestPipelineBehaviour:
    """The swap fires exactly once per orientation reversal, on every routing path."""

    def test_a_mirror_composed_with_a_rotation_swaps_once(self) -> None:
        """A flip fused into a rotation still swaps the pair exactly once.

        This is the case a "is there a flip transform in the list" implementation gets wrong
        in the other direction: here the flip *is* in the list but is no longer visible as a
        discrete op by the time the keypoints are routed, because it has been composed into
        the rotation's matrix.

        """
        pipe = _pipe(rotation=(25.0, 25.0), hflip_p=1.0)

        _, routed = pipe(_image(), POINTS)

        unswapped = FusedCompose.from_params(rotation=(25.0, 25.0), hflip_p=1.0, data_keys=["input", "keypoints"])(
            _image(), POINTS
        )[1]
        assert torch.allclose(routed, _swapped(unswapped), atol=1e-5)

    def test_two_mirrors_compose_to_no_swap(self) -> None:
        """A horizontal and a vertical flip together leave the keypoint order alone.

        Two mirrors are a half turn — determinant ``+1`` — so the pair table must not fire, even though the pipeline
        contains two flip transforms. Counting flips instead of reading the determinant swaps twice or once here, both
        wrong.

        """
        pipe = _pipe(hflip_p=1.0, vflip_p=1.0)
        plain = FusedCompose.from_params(hflip_p=1.0, vflip_p=1.0, data_keys=["input", "keypoints"])

        _, routed = pipe(_image(), POINTS)
        _, expected = plain(_image(), POINTS)

        assert torch.allclose(routed, expected, atol=1e-5)

    def test_a_rotation_alone_does_not_swap(self) -> None:
        """A pipeline with no mirror leaves the keypoint order untouched."""
        pipe = _pipe(rotation=(40.0, 40.0))
        plain = FusedCompose.from_params(rotation=(40.0, 40.0), data_keys=["input", "keypoints"])

        _, routed = pipe(_image(), POINTS)
        _, expected = plain(_image(), POINTS)

        assert torch.allclose(routed, expected, atol=1e-5)

    def test_the_default_leaves_the_axis_untouched(self) -> None:
        """Without a pair table a mirrored pipeline returns keypoints in input order.

        FA-8 must be inert by default: a caller with no left/right pairs — or with a
        symmetric schema — sees exactly the previous behaviour.

        """
        plain = FusedCompose.from_params(hflip_p=1.0, data_keys=["input", "keypoints"])

        _, routed = plain(_image(), POINTS)

        assert routed[0, 0, 0].item() == pytest.approx(15.0 - 4.0)
        assert routed[0, 1, 0].item() == pytest.approx(15.0 - 8.0)

    def test_the_d4_fast_path_permutes_like_the_interpolating_path(self) -> None:
        """A pure flip (D4 fast path, no grid_sample) swaps like a flip inside a real warp.

        A pure flip is classified as a D4 element and applied by tensor reversal, skipping the sampling grid entirely; a
        flip composed with an oblique rotation goes through the interpolating warp instead. The permutation has to fire
        on both, or a mirrored image trains with unswapped left/right labels on whichever path the pipeline happened to
        take — with no shape error anywhere.

        """
        for params in ({"hflip_p": 1.0}, {"hflip_p": 1.0, "rotation": (17.0, 17.0)}):
            with_pairs = _pipe(**params)
            without_pairs = FusedCompose.from_params(data_keys=["input", "keypoints"], **params)

            _, swapped_points = with_pairs(_image(), POINTS)
            _, plain_points = without_pairs(_image(), POINTS)

            assert torch.allclose(swapped_points, _swapped(plain_points), atol=1e-5)
            assert not torch.allclose(swapped_points, plain_points, atol=1e-3)

    def test_a_letterbox_after_a_mirror_still_swaps_once(self) -> None:
        """A mirror fused with a letterbox reverses orientation once and swaps once.

        The letterbox composes into the same matrix, so its positive determinant must not cancel or double the mirror's
        — the combined sign is what decides.

        """
        pipe = _pipe(hflip_p=1.0, letterbox=(32, 32))
        plain = FusedCompose.from_params(hflip_p=1.0, letterbox=(32, 32), data_keys=["input", "keypoints"])

        _, routed = pipe(_image(), POINTS)
        _, expected = plain(_image(), POINTS)

        assert torch.allclose(routed, _swapped(expected), atol=1e-5)
