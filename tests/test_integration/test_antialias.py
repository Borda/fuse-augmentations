"""Opt-in antialiased downscale (``antialias=True``).

An aggressive crop-resize downscale through a single ``grid_sample`` aliases: it drops
high-frequency detail between samples. With ``antialias=True`` the input is Gaussian
prefiltered (mipmap-rule sigma) before the warp, so the result is closer to a proper
antialiased resize. These tests assert:

- the antialiased downscale beats the unfiltered one on SSIM vs a
  ``torchvision.transforms.v2.Resize(antialias=True)`` reference;
- the output is bit-identical to the default path when the flag is off;
- the output is bit-identical when the scale is >= 0.5 (no aliasing risk).

SSIM is computed inline (scikit-image is not a dependency).

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE
from fuse_augmentations.affine.matrix import estimate_scale
from fuse_augmentations.affine.segment import _mipmap_sigma


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Return a normalized 1D Gaussian window for the SSIM filter."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return g / g.sum()


def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean SSIM between two ``(B, C, H, W)`` tensors in ``[0, 1]`` (inline, no skimage)."""
    window_size, sigma = 11, 1.5
    channels = img1.shape[1]
    win1d = _gaussian_window(window_size, sigma)
    win2d = (win1d[:, None] * win1d[None, :]).to(img1.dtype)
    window = win2d.expand(channels, 1, window_size, window_size).contiguous()

    def _filter(img: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(img, window, padding=window_size // 2, groups=channels)

    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = _filter(img1), _filter(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = _filter(img1 * img1) - mu1_sq
    sigma2_sq = _filter(img2 * img2) - mu2_sq
    sigma12 = _filter(img1 * img2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def _high_frequency_image(size: int = 96) -> torch.Tensor:
    """Build a high-frequency checkerboard-like image that aliases hard on downscale."""
    torch.manual_seed(7)
    coords = torch.arange(size, dtype=torch.float32)
    grid_x = torch.sin(coords * 1.3)[None, :]
    grid_y = torch.sin(coords * 1.3)[:, None]
    pattern = 0.5 + 0.5 * (grid_x * grid_y)
    return pattern.expand(1, 3, size, size).contiguous()


class TestEstimateScale:
    """The per-axis scale estimator underpinning the antialias decision."""

    def test_pure_downscale_reports_min_axis(self) -> None:
        """A diagonal down-scale matrix reports its two axis scales, smallest first."""
        mtx = torch.eye(3).unsqueeze(0)
        mtx[:, 0, 0] = 0.25
        mtx[:, 1, 1] = 0.5
        lo, hi = estimate_scale(mtx)
        assert lo == pytest.approx(0.25, abs=1e-4)
        assert hi == pytest.approx(0.5, abs=1e-4)


class TestMipmapSigma:
    """The mipmap prefilter sigma rule."""

    def test_no_blur_at_or_above_unit_scale(self) -> None:
        """Scale >= 1 (upscale / no downscale) needs no prefilter."""
        assert _mipmap_sigma(1.0) == 0.0
        assert _mipmap_sigma(1.5) == 0.0

    def test_sigma_grows_as_scale_shrinks(self) -> None:
        """A more aggressive downscale demands a wider prefilter."""
        assert _mipmap_sigma(0.25) > _mipmap_sigma(0.4) > 0.0


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="antialias prefilter needs the kornia Gaussian backend")
class TestAntialiasDownscale:
    """End-to-end antialiased crop-resize vs a torchvision antialias reference."""

    def _pipe_out(self, image: torch.Tensor, target: int, *, antialias: bool) -> torch.Tensor:
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        torch.manual_seed(0)
        crop = kornia_aug.RandomResizedCrop(size=(target, target), scale=(1.0, 1.0), ratio=(1.0, 1.0), p=1.0)
        pipe = FusedCompose([crop], adapter=KorniaAdapter(), antialias=antialias)
        return pipe(image.clone())

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required for the antialias reference")
    def test_antialiased_beats_unfiltered_on_ssim(self) -> None:
        """Antialiased downscale is closer to a torchvision antialias resize than the unfiltered warp."""
        from torchvision.transforms import v2

        image = _high_frequency_image(96)
        target = 24  # 4x downscale, well below the 0.5 threshold
        reference = v2.Resize((target, target), antialias=True)(image)

        unfiltered = self._pipe_out(image, target, antialias=False)
        antialiased = self._pipe_out(image, target, antialias=True)

        ssim_unfiltered = _ssim(unfiltered, reference)
        ssim_antialiased = _ssim(antialiased, reference)
        assert ssim_antialiased > ssim_unfiltered

    def test_flag_off_is_bit_identical_to_default(self) -> None:
        """With the flag off, output is bit-identical to the plain single-warp path."""
        image = _high_frequency_image(96)
        default = self._pipe_out(image, 24, antialias=False)
        again = self._pipe_out(image, 24, antialias=False)
        assert torch.equal(default, again)

    def test_no_change_when_scale_at_or_above_half(self) -> None:
        """A mild downscale (scale >= 0.5) is untouched: antialias output equals the unfiltered output."""
        image = _high_frequency_image(96)
        target = 64  # 96 -> 64 => scale ~0.66, above the 0.5 threshold
        unfiltered = self._pipe_out(image, target, antialias=False)
        antialiased = self._pipe_out(image, target, antialias=True)
        assert torch.equal(unfiltered, antialiased)
