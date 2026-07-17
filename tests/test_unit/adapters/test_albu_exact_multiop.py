"""Tests for multi-op Albumentations exact-apply fusion (``ExactAffineSegment``).

A ``Compose`` of two or more ``GEOMETRIC_EXACT`` Albumentations ops (flips,
``RandomRotate90``, ``D4``, ``Transpose``) fuses into a single
``ExactAffineSegment`` that applies each op in sequence via
``AlbumentationsAdapter.exact_apply``. Before this file, no test chained 2+
such ops together -- every existing multi-op Compose paired an exact op with
an interpolating one (``Rotate``, ``Affine``), which routes through a
different (matrix) fusion path entirely. That left ``_apply_discrete_exact``
(``adapters/albumentations.py:650-711``) and ``_apply_d4_element``
(``adapters/albumentations.py:811-844``) with zero coverage from any test
that actually exercises the multi-op fused segment.

Ground truth for every case here is built directly from primitive ``torch``
ops (``.flip``, ``torch.rot90``, ``.permute``) chained in the same order as
the Compose transform list -- independent of the adapter's own
implementation -- since these ops are lossless index permutations, fused
output must be bit-exact against that chain.

Requires albumentations.

"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as A

    from fuse_augmentations import Compose

pytestmark = pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")

HEIGHT = WIDTH = 32


def _smooth_image(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Smooth asymmetric single-sample image so op order/direction is observable."""
    yy, xx = torch.meshgrid(torch.linspace(0, 3.14, height), torch.linspace(0, 3.14, width), indexing="ij")
    plane = 0.5 + 0.3 * torch.sin(3 * xx) * torch.cos(2 * yy) + 0.15 * xx / 3.14
    return plane.reshape(1, 1, height, width).expand(1, 3, height, width).contiguous()


def _smooth_batch(batch_size: int, height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Batch of smooth asymmetric images, phase-shifted per sample so batch-dim mixups are observable."""
    samples = [_smooth_image(height, width) * (1.0 + 0.1 * idx) for idx in range(batch_size)]
    return torch.cat(samples, dim=0)


def _smooth_mask(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    """Single-channel asymmetric mask, distinct pattern from the image so its own routing is verifiable."""
    yy, xx = torch.meshgrid(torch.linspace(0, 3.14, height), torch.linspace(0, 3.14, width), indexing="ij")
    plane = 0.5 + 0.4 * torch.cos(2 * xx) * torch.sin(3 * yy)
    return plane.reshape(1, 1, height, width)


# Hand-written D4 group-element ground truth, independent of the adapter's own
# ``_apply_d4_element`` implementation -- mirrors the algebra, not the code.
_D4_NATIVE_OPS = {
    "e": lambda img: img,
    "r90": lambda img: torch.rot90(img, k=1, dims=(-2, -1)),
    "r180": lambda img: torch.rot90(img, k=2, dims=(-2, -1)),
    "r270": lambda img: torch.rot90(img, k=3, dims=(-2, -1)),
    "h": lambda img: img.flip(dims=[3]),
    "v": lambda img: img.flip(dims=[2]),
    "t": lambda img: img.permute(0, 1, 3, 2).contiguous(),
    "hvt": lambda img: img.flip(dims=[2, 3]).permute(0, 1, 3, 2).contiguous(),
}


class TestTwoOpExactChainFusion:
    """Two-op GEOMETRIC_EXACT chains fuse into one ExactAffineSegment applied in list order."""

    def test_hflip_then_rotate90_matches_native_sequential(self):
        """HorizontalFlip followed by RandomRotate90 equals flip-then-rot90 chained natively."""
        image = _smooth_image()
        with patch.object(A.RandomRotate90, "get_params", return_value={"factor": 1}):
            pipe = Compose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            out = pipe(image)

        expected = torch.rot90(image.flip(dims=[3]), k=1, dims=(-2, -1))
        assert torch.equal(out, expected)

    def test_rotate90_then_hflip_matches_native_sequential(self):
        """RandomRotate90 followed by HorizontalFlip equals rot90-then-flip -- confirms order is not commuted."""
        image = _smooth_image()
        with patch.object(A.RandomRotate90, "get_params", return_value={"factor": 1}):
            pipe = Compose([A.RandomRotate90(p=1.0), A.HorizontalFlip(p=1.0)])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            out = pipe(image)

        expected = torch.rot90(image, k=1, dims=(-2, -1)).flip(dims=[3])
        assert torch.equal(out, expected)

    def test_vflip_then_transpose_matches_native_sequential(self):
        """VerticalFlip followed by Transpose equals flip-then-permute chained natively."""
        image = _smooth_image()
        pipe = Compose([A.VerticalFlip(p=1.0), A.Transpose(p=1.0)])
        assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"

        out = pipe(image)

        expected = image.flip(dims=[2]).permute(0, 1, 3, 2).contiguous()
        assert torch.equal(out, expected)


class TestThreeOpExactChainFusion:
    """A three-op GEOMETRIC_EXACT chain (D4 + Transpose + HorizontalFlip) fuses into one segment."""

    def test_d4_transpose_hflip_matches_native_sequential(self):
        """D4(r90) -> Transpose -> HorizontalFlip equals the same chain of native torch ops."""
        image = _smooth_image()
        with patch.object(A.D4, "get_params", return_value={"group_element": "r90"}):
            pipe = Compose([A.D4(p=1.0), A.Transpose(p=1.0), A.HorizontalFlip(p=1.0)])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            assert len(pipe._segments[0].transforms) == 3
            out = pipe(image)

        expected = _D4_NATIVE_OPS["r90"](image).permute(0, 1, 3, 2).contiguous().flip(dims=[3])
        assert torch.equal(out, expected)


class TestD4DispatchCoverage:
    """Every D4 group element, paired with RandomRotate90, exercises the D4 branch of _apply_discrete_exact."""

    @pytest.mark.parametrize(
        "elem",
        [
            pytest.param("e", id="identity"),
            pytest.param("r90", id="r90"),
            pytest.param("r180", id="r180"),
            pytest.param("r270", id="r270"),
            pytest.param("h", id="hflip-elem"),
            pytest.param("v", id="vflip-elem"),
            pytest.param("t", id="transpose-elem"),
            pytest.param("hvt", id="anti-transpose-elem"),
        ],
    )
    def test_d4_element_then_rotate90_matches_native_sequential(self, elem):
        """D4(elem) followed by RandomRotate90(factor=1) equals D4-native-op-then-rot90 chained natively."""
        image = _smooth_image()
        with (
            patch.object(A.D4, "get_params", return_value={"group_element": elem}),
            patch.object(A.RandomRotate90, "get_params", return_value={"factor": 1}),
        ):
            pipe = Compose([A.D4(p=1.0), A.RandomRotate90(p=1.0)])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            out = pipe(image)

        expected = torch.rot90(_D4_NATIVE_OPS[elem](image), k=1, dims=(-2, -1))
        assert torch.equal(out, expected)


class TestAuxMaskExactChain:
    """A mask aux target transforms identically to the image through a multi-op exact chain."""

    def test_mask_matches_image_through_two_op_chain(self):
        """RandomRotate90(factor=3) -> D4(hvt) applies identically to image and mask."""
        image = _smooth_image()
        mask = _smooth_mask()
        with (
            patch.object(A.RandomRotate90, "get_params", return_value={"factor": 3}),
            patch.object(A.D4, "get_params", return_value={"group_element": "hvt"}),
        ):
            pipe = Compose([A.RandomRotate90(p=1.0), A.D4(p=1.0)], data_keys=["input", "mask"])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            out_image, out_mask = pipe(image, mask)

        expected_image = _D4_NATIVE_OPS["hvt"](torch.rot90(image, k=3, dims=(-2, -1)))
        expected_mask = _D4_NATIVE_OPS["hvt"](torch.rot90(mask, k=3, dims=(-2, -1)))
        assert torch.equal(out_image, expected_image)
        assert torch.equal(out_mask, expected_mask)


class TestBatchUniformParams:
    """Batch>1: real Albumentations transforms expose no ``same_on_batch`` attribute, so ``_apply_discrete_exact``
    always takes the per-sample ``get_params()`` loop (adapters/albumentations.py:666-668).

    Pinning ``get_params`` to a fixed
    ``return_value`` makes every sampled draw identical across the batch by
    construction of the mock -- this is the "per-batch uniform" case; per-sample
    *differing* params are never exercised for genuine Albumentations transforms.

    """

    def test_batch_size_two_uniform_params_applied_per_sample(self):
        """HorizontalFlip -> RandomRotate90(factor=1), uniform across a 2-sample batch with distinct content."""
        images = _smooth_batch(batch_size=2)
        with patch.object(A.RandomRotate90, "get_params", return_value={"factor": 1}):
            pipe = Compose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)])
            assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
            out = pipe(images)

        expected = torch.rot90(images.flip(dims=[3]), k=1, dims=(-2, -1))
        assert torch.equal(out, expected)
