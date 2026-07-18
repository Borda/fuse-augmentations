"""Parity tests for mean-relative fused contrast against native ColorJitter.

TorchVision and Kornia ``ColorJitter`` contrast is relative to the per-image mean luminance
(``c' = cf*c + (1-cf)*luma_mean``), not a fixed 0.5 midpoint. The fused color matrix now bakes the
same per-image luminance into its contrast bias, so a single 4x4 matmul reproduces the native output.

Two contracts are checked:

1. Single ColorJitter contrast, applied via ``build_color_matrix`` with the per-image luminance, matches
   the native transform fed the SAME sampled params (reorder=NONE, one op — no reordering involved).
2. A mid-chain ``[brightness, contrast]`` fused color run matches a native sequential chain: the contrast
   op must take its midpoint from the brightened image's mean, which the segment propagates through the
   prefix matrices (``mean(a*x+b) = a*mean(x)+b``).

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters.kornia import KorniaAdapter

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms.v2 as tv_v2
    import torchvision.transforms.v2.functional as tv_f

    from fuse_augmentations.adapters.torchvision import TorchVisionAdapter


def _weighted_luma(image: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    """Return per-image weighted-luminance mean, shape ``(batch_size,)``."""
    w = torch.tensor(weights, dtype=image.dtype)
    mean_ch = image.reshape(image.shape[0], image.shape[1], -1).mean(dim=2)  # (B, C)
    return (mean_ch * w).sum(dim=1)


def _apply_matrix(matrix: torch.Tensor, image: torch.Tensor, *, clip: bool = True) -> torch.Tensor:
    """Apply a ``(B, 4, 4)`` color matrix to a ``(B, 3, H, W)`` image."""
    batch_size, channels, height, width = image.shape
    pixels = image.reshape(batch_size, channels, height * width)
    ones = torch.ones(batch_size, 1, height * width, dtype=image.dtype)
    result = torch.bmm(matrix, torch.cat([pixels, ones], dim=1))[:, :channels, :]
    out = result.reshape(batch_size, channels, height, width)
    return out.clamp(0.0, 1.0) if clip else out


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKorniaContrastMeanParity:
    """Fused Kornia ColorJitter contrast matches native mean-relative contrast."""

    adapter = KorniaAdapter() if _KORNIA_AVAILABLE else None
    weights = (0.299, 0.587, 0.114)

    @pytest.mark.parametrize("seed", [0, 7, 21])
    def test_contrast_only_matches_native(self, seed: int) -> None:
        """ColorJitter(contrast) fused matrix matches native for the same params (reorder=NONE)."""
        torch.manual_seed(seed)
        transform = kornia_aug.ColorJitter(brightness=0.0, contrast=(0.5, 1.5), p=1.0)
        image = torch.rand(2, 3, 16, 16) * 0.6 + 0.2  # inside gamut, no clamp

        params = self.adapter.sample_params(transform, image.shape, torch.device("cpu"))
        luma = _weighted_luma(image, self.weights)
        matrix = self.adapter.build_color_matrix(transform, params, mean=luma)
        fused = _apply_matrix(matrix, image)

        native = transform(image, params=params)

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)

    def test_mean_midpoint_beats_fixed_half(self) -> None:
        """The per-image mean midpoint is strictly closer to native than the old fixed 0.5."""
        torch.manual_seed(0)
        transform = kornia_aug.ColorJitter(brightness=0.0, contrast=(0.5, 1.5), p=1.0)
        image = torch.rand(2, 3, 16, 16) * 0.4 + 0.05  # mean far from 0.5

        params = self.adapter.sample_params(transform, image.shape, torch.device("cpu"))
        native = transform(image, params=params)

        luma = _weighted_luma(image, self.weights)
        fused_mean = _apply_matrix(self.adapter.build_color_matrix(transform, params, mean=luma), image)
        fused_half = _apply_matrix(self.adapter.build_color_matrix(transform, params, mean=None), image)

        err_mean = (fused_mean - native).abs().max()
        err_half = (fused_half - native).abs().max()
        assert err_mean < err_half


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestTorchVisionContrastMeanParity:
    """Fused TorchVision ColorJitter contrast matches native mean-relative contrast."""

    adapter = TorchVisionAdapter() if _TORCHVISION_AVAILABLE else None
    weights = (0.2989, 0.587, 0.114)

    @pytest.mark.parametrize("seed", [0, 11, 29])
    def test_contrast_only_matches_native(self, seed: int) -> None:
        """ColorJitter(contrast) fused matrix matches native functional contrast (reorder=NONE)."""
        torch.manual_seed(seed)
        transform = tv_v2.ColorJitter(contrast=0.5)
        image = torch.rand(2, 3, 16, 16) * 0.6 + 0.2

        params = self.adapter.sample_params(transform, image.shape, torch.device("cpu"))
        luma = _weighted_luma(image, self.weights)
        matrix = self.adapter.build_color_matrix(transform, params, mean=luma)
        fused = _apply_matrix(matrix, image)

        # Native: apply the sampled contrast factor per image with the functional kernel.
        cf = params["contrast_factor"]
        native = torch.cat(
            [tv_f.adjust_contrast(image[b : b + 1], float(cf[0 if cf.shape[0] == 1 else b])) for b in range(2)],
            dim=0,
        )

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestMidChainMeanPropagation:
    """A brightness op before a contrast op shifts the mean the contrast midpoint uses."""

    adapter = TorchVisionAdapter() if _TORCHVISION_AVAILABLE else None
    weights = (0.2989, 0.587, 0.114)

    def test_brightness_then_contrast_matches_sequential(self) -> None:
        """[Brightness, Contrast] fused order matches native sequential (mid-chain mean propagation)."""
        torch.manual_seed(13)
        image = torch.rand(2, 3, 16, 16) * 0.5 + 0.25

        # Fixed factors isolate the propagation math from RNG-stream ordering.
        bright = torch.tensor([1.3, 0.7])
        contr = torch.tensor([1.4, 0.6])
        p_b = {"brightness_factor": bright, "contrast_factor": torch.ones(2), "order": torch.tensor([0])}
        p_c = {"brightness_factor": torch.ones(2), "contrast_factor": contr, "order": torch.tensor([1])}
        tf = tv_v2.ColorJitter(brightness=0.4, contrast=0.5)

        # Fused: brightness matrix, advance the per-channel mean through it, then contrast off that mean.
        mat_b = self.adapter.build_color_matrix(tf, p_b, mean=_weighted_luma(image, self.weights))
        mean_ch = image.reshape(2, 3, -1).mean(dim=2)
        mean_ch2 = torch.bmm(mat_b, torch.cat([mean_ch, torch.ones(2, 1)], dim=1).unsqueeze(2)).squeeze(2)[:, :3]
        luma_c = (mean_ch2 * torch.tensor(self.weights)).sum(dim=1)
        mat_c = self.adapter.build_color_matrix(tf, p_c, mean=luma_c)
        acc = torch.bmm(mat_c, mat_b)
        fused = _apply_matrix(acc, image)

        # Native sequential: brightness then contrast, each with its own per-op mean.
        bright_img = torch.cat([tv_f.adjust_brightness(image[b : b + 1], float(bright[b])) for b in range(2)], dim=0)
        native = torch.cat(
            [tv_f.adjust_contrast(bright_img[b : b + 1], float(contr[b])) for b in range(2)], dim=0
        ).clamp(0.0, 1.0)

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
def test_per_operation_clip_policy_matches_native_colorjitter_chain() -> None:
    """Per-operation clipping reproduces TorchVision's gamut-clamped brightness chain.

    A brightening operation takes 0.8 outside the valid range before the subsequent darkening operation. Native
    TorchVision clamps between the operations, whereas the default one-matmul mode intentionally does not. Fixed factor
    intervals keep this test independent of the backend RNG stream.

    """
    image = torch.full((1, 3, 4, 4), 0.8)
    transforms = [
        tv_v2.ColorJitter(brightness=(1.5, 1.5)),
        tv_v2.ColorJitter(brightness=(0.5, 0.5)),
    ]

    native = tv_v2.Compose(transforms)(image)
    per_op = Compose(transforms, clip_policy="per_op_parity")(image)
    final = Compose(transforms, clip_policy="final")(image)

    torch.testing.assert_close(per_op, native, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(final, native, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
def test_per_operation_clip_recomputes_contrast_mean_after_clamp() -> None:
    """A contrast after a parity clamp uses the clamped image's luminance mean."""
    image = torch.tensor(
        [[[[0.30, 0.55], [0.75, 0.90]], [[0.25, 0.50], [0.70, 0.95]], [[0.20, 0.45], [0.65, 0.85]]]],
        dtype=torch.float32,
    )
    transforms = [
        tv_v2.ColorJitter(brightness=(1.5, 1.5)),
        tv_v2.ColorJitter(contrast=(0.5, 0.5)),
    ]

    native = tv_v2.Compose(transforms)(image)
    per_op = Compose(transforms, clip_policy="per_op_parity")(image)
    final = Compose(transforms, clip_policy="final")(image)

    torch.testing.assert_close(per_op, native, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(final, native, atol=1e-6, rtol=1e-6)
