"""Constant-value border padding: ``fill=`` paints the out-of-canvas region (FA-2).

``grid_sample`` offers no constant padding mode, so the torch path subtracts the fill, samples against the zero border,
and adds it back; the cv2 paths pass it as ``borderValue``. Both must land on the same value, because which one runs is
a function of batch size, device and execution strategy rather than of anything the caller wrote.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import FusedCompose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.affine.segment import _cv2_border_value

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

#: Deterministic shift: the first six columns of the output map outside the source.
SHIFT_PX = 6.0
#: Downstream's letterbox grey, as float and as its uint8 original.
GREY = 114.0 / 255.0
CONTENT = 0.8


@pytest.fixture
def image() -> torch.Tensor:
    """Return a ``(1, 3, 16, 16)`` constant-valued batch, so any non-content pixel is obvious."""
    return torch.full((1, 3, 16, 16), CONTENT)


def _shifted(fill: float | tuple[float, ...] | None, **kwargs: object) -> FusedCompose:
    """Build a pipeline that shifts content right by ``SHIFT_PX`` and fills what it uncovers."""
    return FusedCompose.from_params(translate_x=(SHIFT_PX, SHIFT_PX), fill=fill, **kwargs)  # type: ignore[arg-type]


class TestConstantFillOnTheTorchPath:
    """A scalar or per-channel fill is written exactly into the uncovered region."""

    def test_scalar_fill_paints_every_channel(self, image: torch.Tensor) -> None:
        """A scalar fill lands in the uncovered columns of every channel, content untouched.

        The subtract/add construction is only correct if the sampled zero comes back as exactly the fill; a value that
        is merely close would show here as drift on a region that has no source pixel contributing to it at all.

        """
        out = _shifted(GREY)(image)

        assert torch.allclose(out[0, :, :, :2], torch.full((3, 16, 2), GREY))
        assert torch.allclose(out[0, :, :, 10:], torch.full((3, 16, 6), CONTENT))

    def test_per_channel_fill_paints_each_channel_separately(self, image: torch.Tensor) -> None:
        """A three-value fill writes a different constant per channel.

        Per-channel is the shape a real letterbox colour takes; a broadcast bug that wrote the first value everywhere
        would pass the scalar test above.

        """
        out = _shifted((0.1, 0.2, 0.3))(image)

        border = out[0, :, 8, 0]
        assert torch.allclose(border, torch.tensor([0.1, 0.2, 0.3]))

    def test_default_keeps_the_zero_border(self, image: torch.Tensor) -> None:
        """``fill=None`` leaves the historical zero padding in place.

        FA-2 must be inert by default: an existing pipeline's border stays black with no
        code change and no extra tensor arithmetic on the warp path.

        """
        out = _shifted(None)(image)

        assert torch.equal(out[0, :, :, :2], torch.zeros(3, 16, 2))

    def test_mask_keeps_zero_padding_while_the_image_is_filled(self, image: torch.Tensor) -> None:
        """The routed mask pads with zero even when the image pads with the fill.

        A mask's out-of-canvas region means "no instance here", not "the letterbox colour". Filling it would invent
        instances of class ``114`` at the border, which no shape check would ever catch.

        """
        mask = torch.ones(1, 1, 16, 16)
        pipe = _shifted(GREY, data_keys=["input", "mask"])

        out_image, out_mask = pipe(image, mask)

        assert torch.allclose(out_image[0, :, 8, 0], torch.full((3,), GREY))
        assert float(out_mask[0, 0, 8, 0]) == 0.0

    def test_fill_length_must_match_the_channel_count(self, image: torch.Tensor) -> None:
        """A fill whose length is neither 1 nor the channel count raises at forward.

        The channel count is not known at construction, so this is the one fill error that cannot be caught early — it
        must still raise rather than broadcast into something plausible.

        """
        with pytest.raises(ValueError, match="channel"):
            _shifted((0.1, 0.2))(image)


class TestExecutorParity:
    """Which warp executor runs is invisible to the caller, so the fill must not depend on it."""

    def test_cv2_fast_path_matches_the_torch_path(self, image: torch.Tensor) -> None:
        """A B=1 CPU multi-transform segment (cv2) fills identically to the B=2 batch (torch).

        The cv2 fast path activates on batch size, device and transform count — never on anything in the pipeline config
        — so a fill threaded through only one of the two warp backends would produce a border that changes with the
        batch size.

        """
        pipe = FusedCompose.from_params(translate_x=(SHIFT_PX, SHIFT_PX), hflip_p=1.0, fill=GREY)

        single = pipe(image)
        batched = pipe(image.expand(2, -1, -1, -1).contiguous())

        assert torch.allclose(single[0, :, 8, -1], torch.full((3,), GREY))
        assert torch.allclose(single[0], batched[0], atol=1e-6)

    @pytest.mark.integration
    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
    def test_albumentations_cv2_and_torch_execution_agree(self, image: torch.Tensor) -> None:
        """The Albumentations segment fills the same under ``execution="cv2"`` and ``"torch"``.

        These are two genuinely different implementations of the same warp — one ``cv2.warpAffine`` per sample against
        one batched ``grid_sample`` — and the border is where they are most likely to diverge.

        """

        def run(execution: str) -> torch.Tensor:
            transform = albu.Affine(translate_px={"x": int(SHIFT_PX), "y": 0}, p=1.0)
            return FusedCompose([transform], execution=execution, fill=GREY)(image)  # type: ignore[arg-type]

        cv2_out, torch_out = run("cv2"), run("torch")

        assert torch.allclose(cv2_out[0, :, 8, 0], torch.full((3,), GREY), atol=1e-6)
        assert torch.allclose(cv2_out, torch_out, atol=1e-5)

    @pytest.mark.integration
    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
    def test_uint8_numpy_path_fills_in_the_images_own_range(self) -> None:
        """The native NumPy uint8 path takes the fill in uint8 units, not in ``[0, 1]``.

        This path warps the uint8 array directly with no rescaling, so the fill is whatever the caller wrote — ``114``
        here, against ``114 / 255`` on the float paths. Getting that wrong yields a black border from a value rounded to
        zero.

        """
        array = np.full((16, 16, 3), 200, dtype=np.uint8)
        transform = albu.Affine(translate_px={"x": int(SHIFT_PX), "y": 0}, p=1.0)
        pipe = FusedCompose([transform], fill=114.0)

        out = np.asarray(pipe(image=array)["image"])

        assert out.dtype == np.uint8
        assert out[8, 0].tolist() == [114, 114, 114]
        assert out[8, 15].tolist() == [200, 200, 200]


class TestFillValidation:
    """A fill that cannot be honoured is refused at construction, not approximated."""

    @pytest.mark.parametrize("padding_mode", ["border", "reflection", "per_transform"])
    def test_non_constant_padding_modes_are_rejected(self, padding_mode: str) -> None:
        """A fill with a padding mode that has no constant raises.

        ``border`` replicates the edge pixel and ``reflection`` mirrors the image, so neither has a constant to replace;
        ``per_transform`` picks a mode per transform, so one fill cannot describe the outcome. Silently ignoring the
        argument on those modes would produce a border the caller never asked for.

        """
        with pytest.raises(ValueError, match="constant border"):
            FusedCompose.from_params(rotation=(-10.0, 10.0), padding_mode=padding_mode, fill=GREY)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "fill",
        [
            pytest.param((), id="empty"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param("grey", id="string"),
        ],
    )
    def test_malformed_fill_values_are_rejected(self, fill: object) -> None:
        """An empty, non-finite, or non-numeric fill raises at construction.

        A NaN fill propagates through the subtract/add construction into every pixel of the warped image, not only the
        border — the failure surfaces far from its cause unless it is caught here.

        """
        with pytest.raises(ValueError, match="fill"):
            FusedCompose.from_params(rotation=(-10.0, 10.0), fill=fill)  # type: ignore[arg-type]

    def test_cv2_cannot_express_a_fill_beyond_four_channels(self) -> None:
        """A per-channel fill on a >4-channel image raises for the cv2 warp path.

        cv2's ``borderValue`` is a four-component ``Scalar``; a five-channel fill has nowhere to go, and cv2 would
        quietly drop the tail. The torch path has no such limit, which is what the message points the caller at.

        """
        with pytest.raises(ValueError, match="cv2"):
            _cv2_border_value((0.1, 0.2, 0.3, 0.4, 0.5), 5)

    def test_a_scalar_fill_is_broadcast_to_the_cv2_scalar(self) -> None:
        """A scalar fill becomes a full-length cv2 ``borderValue`` tuple, not a bare number.

        cv2 reads a bare number as ``Scalar(v, 0, 0, 0)``, so a scalar grey would fill only the first channel and leave
        the other two black — a fill bug that looks like a channel-order bug.

        """
        assert _cv2_border_value((114.0,), 3) == (114.0, 114.0, 114.0)


class TestFillUnderLowPrecision:
    """The subtract/add construction still lands on the fill in half precision."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_half_precision_border_is_the_fill_within_tolerance(self, image: torch.Tensor) -> None:
        """``pipeline_dtype="float16"`` keeps the border at the fill to half-precision tolerance.

        The shift happens in the warp's working dtype, so a half-precision run cannot be bit-exact; it must still be the
        fill rather than drifting toward zero, which is what an un-shifted or double-shifted implementation would show.

        """
        pipe = _shifted(GREY, pipeline_dtype="float16")

        out = pipe(image.to("mps"))

        assert torch.allclose(out[0, :, 8, 0].float().cpu(), torch.full((3,), GREY), atol=1e-3)
