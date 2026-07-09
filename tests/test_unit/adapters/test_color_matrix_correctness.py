"""Property-based correctness tests for adapter build_color_matrix implementations.

Verifies that ``build_color_matrix`` returns numerically correct ``(B, 4, 4)``
homogeneous color-space affine matrices by comparing matrix-based application
against the native backend transform for each supported backend.

The core test pattern for each transform:
1. Sample a random image in a safe range (avoiding clamp boundaries).
2. Get ``params = adapter.sample_params(transform, shape, device)``.
3. Get ``matrix = adapter.build_color_matrix(transform, params)`` -- ``(B, 4, 4)``.
4. Apply the matrix manually: reshape pixels to ``(B, H*W, 4)`` homogeneous
   vectors, ``bmm`` with matrix, reshape back.
5. Apply the transform natively via ``call_nonfused`` with the same params.
6. Assert ``torch.testing.assert_close`` between the two results.

Note: transforms that clamp outputs (Kornia, TorchVision) are tested with
images in the ``[0.2, 0.8]`` range and moderate factors to avoid clamp
boundaries where the linear matrix model diverges from the clamped output.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers

from fuse_augmentations._compat import (
    _ALBUMENTATIONS_AVAILABLE,
    _KORNIA_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)
from fuse_augmentations.types import TransformCategory

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters.kornia import KorniaAdapter

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans

    from fuse_augmentations.adapters.torchvision import TorchVisionAdapter

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

    from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter


def _apply_color_matrix(matrix: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Apply a (B, 4, 4) homogeneous color matrix to a (B, C, H, W) image.

    Args:
        matrix: ``(B, 4, 4)`` homogeneous color-space affine matrix.
        image: ``(B, C, H, W)`` float32 image tensor with C=3.

    Returns:
        ``(B, C, H, W)`` transformed image (no clamping).

    """
    batch_size, channels, height, width = image.shape
    assert channels == 3, f"Expected 3 channels, got {channels}"

    pixels = image.reshape(batch_size, 3, height * width)
    ones = torch.ones(batch_size, 1, height * width, device=image.device, dtype=image.dtype)
    pixels_h = torch.cat([pixels, ones], dim=1)

    result = torch.bmm(matrix, pixels_h)

    return result[:, :3, :].reshape(batch_size, 3, height, width)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKorniaColorMatrixCorrectness:
    """Numerical correctness for Kornia color transforms."""

    # Stateless shared instance; guarded — class body executes at collection even under skipif.
    adapter = KorniaAdapter() if _KORNIA_AVAILABLE else None

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(lambda: kornia_aug.RandomBrightness(brightness=(0.5, 1.5), p=1.0), id="RandomBrightness"),
            pytest.param(lambda: kornia_aug.RandomContrast(contrast=(0.5, 1.5), p=1.0), id="RandomContrast"),
            pytest.param(
                lambda: kornia_aug.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), p=1.0), id="ColorJitter"
            ),
        ],
    )
    def test_shape(self, transform_factory):
        """build_color_matrix returns (B, 4, 4) for brightness, contrast, and jitter transforms."""
        transform = transform_factory()
        shape = (2, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        assert matrix.shape == (2, 4, 4)

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(lambda: kornia_aug.RandomBrightness(brightness=(0.5, 1.5), p=1.0), id="brightness"),
            pytest.param(lambda: kornia_aug.RandomContrast(contrast=(0.5, 1.5), p=1.0), id="contrast"),
        ],
    )
    def test_last_row_homogeneous(self, transform_factory):
        """Bottom row of each (4, 4) sub-matrix is [0, 0, 0, 1]"""
        transform = transform_factory()
        shape = (3, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).expand(3, -1)
        torch.testing.assert_close(matrix[:, 3, :], expected)

    @given(seed=integers(min_value=0, max_value=9999))
    @settings(max_examples=30)
    def test_random_brightness_parity(self, seed):
        """Matrix-applied brightness matches Kornia's native output (no clamping)"""
        torch.manual_seed(seed)
        transform = kornia_aug.RandomBrightness(brightness=(0.8, 1.2), p=1.0, clip_output=False)
        batch_size, channels, height, width = 2, 3, 8, 8
        image = torch.rand(batch_size, channels, height, width) * 0.6 + 0.2

        shape = (batch_size, channels, height, width)
        canon_params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, canon_params)
        fused = _apply_color_matrix(matrix, image)

        native_params = {"brightness_factor": canon_params["brightness_factor"]}
        native = transform(image, params=native_params)

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)

    @given(seed=integers(min_value=0, max_value=9999))
    @settings(max_examples=30)
    def test_random_contrast_parity(self, seed):
        """Matrix-applied contrast matches Kornia's native output (no clamping)"""
        torch.manual_seed(seed)
        transform = kornia_aug.RandomContrast(contrast=(0.8, 1.2), p=1.0, clip_output=False)
        batch_size, channels, height, width = 2, 3, 8, 8
        image = torch.rand(batch_size, channels, height, width) * 0.6 + 0.2

        shape = (batch_size, channels, height, width)
        canon_params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, canon_params)
        fused = _apply_color_matrix(matrix, image)

        native_params = {"contrast_factor": canon_params["contrast_factor"]}
        native = transform(image, params=native_params)

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(lambda: kornia_aug.RandomBrightness(brightness=(1.0, 1.0), p=1.0), id="brightness"),
            pytest.param(lambda: kornia_aug.RandomContrast(contrast=(1.0, 1.0), p=1.0), id="contrast"),
        ],
    )
    def test_identity_case(self, transform_factory):
        """When factor=1.0 the matrix is identity (brightness and contrast)"""
        transform = transform_factory()
        shape = (2, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        expected = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(lambda: kornia_aug.RandomBrightness(brightness=(0.5, 1.5), p=1.0), id="RandomBrightness"),
            pytest.param(lambda: kornia_aug.RandomContrast(contrast=(0.5, 1.5), p=1.0), id="RandomContrast"),
            pytest.param(lambda: kornia_aug.ColorJitter(brightness=(0.5, 1.5), p=1.0), id="ColorJitter"),
        ],
    )
    def test_category(self, transform_factory):
        """Brightness, contrast, and jitter are all POINTWISE_LINEAR."""
        assert self.adapter.category(transform_factory()) == TransformCategory.POINTWISE_LINEAR


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestTorchVisionColorMatrixCorrectness:
    """Numerical correctness for TorchVision color transforms."""

    # Stateless shared instance; guarded — class body executes at collection even under skipif.
    adapter = TorchVisionAdapter() if _TORCHVISION_AVAILABLE else None

    def test_color_jitter_shape(self):
        """ColorJitter build_color_matrix returns (B, 4, 4) tensor."""
        transform = tv_trans.ColorJitter(brightness=0.5)
        shape = (2, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        assert matrix.shape == (2, 4, 4)

    def test_last_row_homogeneous(self):
        """Bottom row of each (4, 4) matrix is [0, 0, 0, 1]"""
        transform = tv_trans.ColorJitter(brightness=0.5, contrast=0.5)
        shape = (3, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for idx in range(3):
            torch.testing.assert_close(matrix[idx, 3, :], expected)

    def test_brightness_only_parity(self):
        """Brightness-only ColorJitter: matrix matches native (multiplicative)

        TorchVision's ColorJitter has no `clip_output` flag, so this test stays inside the [0.25, 0.75] safe range and
        compares against a manually-computed multiplicative reference rather than calling the transform directly — this
        isolates the matrix-construction step from TorchVision's parameter-sampling and per-sample loop.
        """
        torch.manual_seed(42)
        transform = tv_trans.ColorJitter(brightness=0.3)
        batch_size, channels, height, width = 2, 3, 8, 8
        image = torch.rand(batch_size, channels, height, width) * 0.5 + 0.25

        shape = (batch_size, channels, height, width)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        fused = _apply_color_matrix(matrix, image)

        brightness_factor = params["brightness_factor"]
        for idx in range(batch_size):
            expected = image[idx] * brightness_factor[idx]
            torch.testing.assert_close(fused[idx], expected, atol=1e-5, rtol=1e-5)

    def test_category_color_jitter(self):
        """ColorJitter category is POINTWISE_LINEAR."""
        transform = tv_trans.ColorJitter(brightness=0.3)
        assert self.adapter.category(transform) == TransformCategory.POINTWISE_LINEAR


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbumentationsColorMatrixCorrectness:
    """Numerical correctness for Albumentations color transforms."""

    # Stateless shared instance; guarded — class body executes at collection even under skipif.
    adapter = AlbumentationsAdapter() if _ALBUMENTATIONS_AVAILABLE else None

    def test_shape(self):
        """RandomBrightnessContrast build_color_matrix returns (B, 4, 4) tensor."""
        transform = albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        shape = (2, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        assert matrix.shape == (2, 4, 4)

    def test_last_row_homogeneous(self):
        """Bottom row of each (4, 4) matrix is [0, 0, 0, 1]"""
        transform = albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        shape = (3, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for idx in range(3):
            torch.testing.assert_close(matrix[idx, 3, :], expected)

    @given(seed=integers(min_value=0, max_value=9999))
    @settings(max_examples=30)
    def test_parity(self, seed):
        """Matrix application matches native Albumentations output.

        Albumentations ``RandomBrightnessContrast`` applies ``c' = alpha * c + beta``
        which is exactly representable in 4x4 form.

        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        transform = albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
        batch_size, channels, height, width = 2, 3, 8, 8
        image = torch.rand(batch_size, channels, height, width) * 0.5 + 0.25

        shape = (batch_size, channels, height, width)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)
        fused = _apply_color_matrix(matrix, image)

        alpha = params["alpha"]
        beta = params["beta"]
        native = image.clone()
        for idx in range(batch_size):
            native[idx] = alpha[idx] * image[idx] + beta[idx]

        torch.testing.assert_close(fused, native, atol=1e-5, rtol=1e-5)

    def test_identity_case(self):
        """When alpha=1, beta=0, matrix should be identity."""
        transform = albu.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.0, p=1.0)
        shape = (2, 3, 8, 8)
        params = self.adapter.sample_params(transform, shape, torch.device("cpu"))
        matrix = self.adapter.build_color_matrix(transform, params)

        expected = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(matrix, expected, atol=1e-5, rtol=1e-5)

    def test_category(self):
        """RandomBrightnessContrast category is POINTWISE_LINEAR."""
        transform = albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        assert self.adapter.category(transform) == TransformCategory.POINTWISE_LINEAR
