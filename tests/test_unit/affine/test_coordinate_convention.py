"""The half-pixel convention is pinned by measurement, not by assertion (FA-3).

This package samples with ``align_corners=True`` and normalizes with the sandwich derived
for that same flag (``normalize_matrix``: ``2 / (W - 1)`` scale, ``(W - 1) / 2`` offset).
Because the sandwich and the sampling flag agree, the *pixel-space* map they produce is
the one either convention would produce — a pixel-space matrix carries no convention of
its own, only its normalization into ``[-1, 1]`` does.

The tests below measure that against an independently constructed ``align_corners=False``
reference (``p = ((g + 1) * L - 1) / 2``, the mapping TorchVision, Albumentations and
downstream ``lucid_yolo`` all use) rather than asserting it, and they pin the two places
where the equivalence genuinely stops holding: reflection padding, whose reflection axis
*is* defined by the flag, and canvases thinner than two pixels, where the ``True``
normalization is singular and this package refuses the warp by name.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations import FusedCompose
from fuse_augmentations.affine.matrix import inv3x3, normalize_matrix_io

#: Float32 sampling of two independently composed grids differs in the last bits only.
ROUNDING = 1e-5


def _normalize_false(height: int, width: int) -> torch.Tensor:
    """Return the pixel-to-normalized map for ``align_corners=False``: ``g = (2p + 1) / L - 1``."""
    return torch.tensor(
        [
            [2.0 / width, 0.0, 1.0 / width - 1.0],
            [0.0, 2.0 / height, 1.0 / height - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _warp_false(
    image: torch.Tensor,
    matrix_forward: torch.Tensor,
    padding_mode: str = "zeros",
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Warp ``image`` by a forward pixel matrix under the ``align_corners=False`` convention."""
    _, channels, height_in, width_in = image.shape
    height_out, width_out = out_size or (height_in, width_in)
    inverse = torch.inverse(matrix_forward.to(torch.float64))
    normalized = (
        _normalize_false(height_in, width_in) @ inverse @ torch.inverse(_normalize_false(height_out, width_out))
    )
    grid = F.affine_grid(normalized[None, :2].float(), [1, channels, height_out, width_out], align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)


@pytest.fixture
def noise() -> torch.Tensor:
    """Return a deterministic ``(1, 1, 12, 16)`` non-square batch with structure everywhere."""
    return torch.rand(1, 1, 12, 16, generator=torch.Generator().manual_seed(0))


class TestConventionEquivalence:
    """A pixel-space matrix produces the same image under either half-pixel convention."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"translate_x": (1.0, 1.0)}, id="integer-shift"),
            pytest.param({"translate_x": (2.5, 2.5)}, id="half-pixel-shift"),
            pytest.param({"translate_x": (-3.25, -3.25)}, id="quarter-pixel-shift"),
            pytest.param({"scale": (0.5, 0.5)}, id="downscale"),
            pytest.param({"scale": (2.0, 2.0)}, id="upscale"),
            pytest.param({"rotation": (23.0, 23.0)}, id="rotation"),
        ],
    )
    def test_warp_matches_the_false_convention_reference(self, noise: torch.Tensor, params: dict) -> None:
        """Each warp matches an ``align_corners=False`` reference to float32 rounding.

        This is the measurement FA-3 turns on: if the two conventions produced different
        pixel maps, a caller porting matrices from a ``False`` codebase would see a
        half-pixel drift, and this package would owe them a convention flag. They do not,
        so it does not — the sandwich and the sampling flag cancel.

        """
        pipe = FusedCompose.from_params(**params)
        got = pipe(noise)

        reference = _warp_false(noise, pipe.transform_matrix[0])

        assert torch.allclose(got, reference, atol=ROUNDING)

    @pytest.mark.parametrize("padding_mode", ["zeros", "border"])
    def test_constant_and_replicated_borders_match(self, noise: torch.Tensor, padding_mode: str) -> None:
        """Zero and replicate padding agree between conventions, including at the canvas border.

        Out-of-canvas contributions are decided by pixel-space neighbour positions, which both conventions place
        identically, so the border region is not a residual for these two modes — unlike reflection below.

        """
        pipe = FusedCompose.from_params(translate_x=(2.5, 2.5), padding_mode=padding_mode)
        got = pipe(noise)

        reference = _warp_false(noise, pipe.transform_matrix[0], padding_mode=padding_mode)

        assert torch.allclose(got, reference, atol=ROUNDING)

    def test_differing_input_and_output_sizes_match(self, noise: torch.Tensor) -> None:
        """``normalize_matrix_io`` is convention-free too, for a letterbox-shaped map.

        This is the normalization a shape-changing segment uses, and the one FA-4's
        letterbox will be built on: if the convention leaked anywhere, differing input and
        output sizes is where it would, since the two canvases scale independently.

        """
        forward = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)[None]
        out_h, out_w = 8, 8

        normalized = normalize_matrix_io(inv3x3(forward), 12, 16, out_h, out_w)
        grid = F.affine_grid(normalized[:, :2].float(), [1, 1, out_h, out_w], align_corners=True)
        got = F.grid_sample(noise, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        reference = _warp_false(noise, forward[0], out_size=(out_h, out_w))

        assert torch.allclose(got, reference, atol=ROUNDING)


class TestConventionResiduals:
    """The two places the equivalence stops, both pinned rather than left to be discovered."""

    def test_reflection_padding_differs_between_conventions(self, noise: torch.Tensor) -> None:
        """Reflection is the one padding mode whose result the flag actually changes.

        ``align_corners=True`` reflects about the outer pixel *centres* (OpenCV's ``BORDER_REFLECT_101``) while
        ``False`` reflects about the outer pixel *edges* (``BORDER_REFLECT``); the two disagree by a whole pixel of
        phase in the mirrored band. This is a documented property, not a bug — but a caller matching another
        implementation's reflected border must know it, so the divergence is asserted to exist rather than tolerated
        silently.

        """
        pipe = FusedCompose.from_params(translate_x=(2.5, 2.5), padding_mode="reflection")
        got = pipe(noise)

        reference = _warp_false(noise, pipe.transform_matrix[0], padding_mode="reflection")

        assert not torch.allclose(got, reference, atol=1e-2)

    @pytest.mark.parametrize(
        ("shape", "axis"),
        [
            pytest.param((1, 1, 4, 1), "width", id="single-column"),
            pytest.param((1, 1, 1, 4), "height", id="single-row"),
        ],
    )
    def test_a_canvas_thinner_than_two_pixels_is_refused_by_name(self, shape: tuple[int, ...], axis: str) -> None:
        """A one-pixel axis raises naming the axis instead of warping with an infinite scale.

        The ``True`` normalization divides by ``L - 1``, which is singular at ``L == 1``;
        the ``False`` one divides by ``L`` and would carry on. Refusing is the honest
        outcome — an infinite normalization scale silently yields an all-zero warp — and
        the error names the offending axis so the caller can see which one it is.

        """
        pipe = FusedCompose.from_params(rotation=(10.0, 10.0))

        with pytest.raises(ValueError, match=axis):
            pipe(torch.rand(*shape))


class TestCoordinateColocation:
    """Image sampling and coordinate transport are pinned against each other, not against a constant."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"translate_x": (2.0, 2.0)}, id="translation"),
            pytest.param({"rotation": (90.0, 90.0)}, id="quarter-turn"),
            pytest.param({"rotation": (15.0, 15.0)}, id="oblique-rotation"),
            pytest.param({"scale": (2.0, 2.0)}, id="upscale"),
        ],
    )
    def test_a_keypoint_on_a_bright_pixel_stays_on_it(self, params: dict) -> None:
        """The warped keypoint lands on the warped image's brightest pixel.

        This is the invariant worth having whichever way the convention question lands: it
        constrains ``targets.py``'s coordinate transport (which multiplies the pixel matrix
        directly, with no normalization at all) against the image sampling (which goes
        through the normalized grid), rather than checking either against a constant. A
        half-pixel error in one and not the other would move the peak off the keypoint.

        """
        image = torch.zeros(1, 1, 32, 32)
        image[0, 0, 12, 20] = 1.0
        keypoint = torch.tensor([[[20.0, 12.0]]])
        pipe = FusedCompose.from_params(data_keys=["input", "keypoints"], **params)

        warped_image, warped_keypoint = pipe(image, keypoint)

        x, y = warped_keypoint[0, 0].tolist()
        at_keypoint = float(warped_image[0, 0, round(y), round(x)])
        assert at_keypoint == pytest.approx(float(warped_image[0, 0].max()), abs=1e-6)

    def test_a_mask_follows_the_image_through_the_same_grid(self) -> None:
        """A mask marking the bright pixel still marks it after the warp.

        Masks are resampled on the image's own grid rather than through the pixel matrix, so this pins the third
        transport path — grid-sampled aux — to the same convention as the image it accompanies.

        """
        image = torch.zeros(1, 1, 32, 32)
        image[0, 0, 12, 20] = 1.0
        mask = torch.zeros(1, 1, 32, 32)
        mask[0, 0, 12, 20] = 1.0
        pipe = FusedCompose.from_params(rotation=(90.0, 90.0), data_keys=["input", "mask"])

        warped_image, warped_mask = pipe(image, mask)

        assert torch.equal(warped_image.argmax(), warped_mask.argmax())
