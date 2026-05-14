"""Integration coverage for ProjectiveSegment.

Requires kornia >= 0.6.12 for the Kornia subtests.

These tests verify projective fusion-plan reporting, shape preservation, and saved-warp accounting across supported
backends.

Single-transform projective paths are checked for successful execution and matrix bookkeeping. Multi-transform chains
verify that the fused path composes homographies and reduces the number of warp passes.

"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Kornia tests
# ---------------------------------------------------------------------------


class TestKorniaProjectiveParity:
    """ProjectiveSegment coverage for K.RandomPerspective."""

    @pytest.fixture(autouse=True)
    def require_kornia(self):
        pytest.importorskip("kornia", reason="kornia required")

    def test_single_perspective_shape(self):
        """Single kornia_aug.RandomPerspective produces output with same shape."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations import Compose

        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        image = torch.rand(2, 3, 32, 32)
        image_output = pipe(image)
        assert image_output.shape == image.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([K.RandomPerspective(p=1)]).fusion_plan shows projective(...)."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations import Compose

        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan
        assert "RandomPerspective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two K.RandomPerspective fused into one segment -> n_warps_saved == 1."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations import Compose

        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1

    def test_single_perspective_records_transform_matrix(self):
        """Single fused RandomPerspective records a finite 3x3 transform matrix."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations import Compose

        transform = kornia_aug.RandomPerspective(distortion_scale=0.4, p=1.0)
        image = torch.rand(1, 3, 32, 32)

        # Fused path -- same transform, uses ProjectiveSegment
        pipe = Compose([transform])
        image_output = pipe(image)

        assert image_output.shape == image.shape
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == (1, 3, 3)
        assert torch.isfinite(pipe.transform_matrix).all()

    def test_mixed_rotate_then_perspective_gives_two_segments(self):
        """[Rotation, RandomPerspective] -> fusion_plan has fused(...) -> projective(...)."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations import Compose

        transform_rotation = kornia_aug.RandomRotation(degrees=30, p=1.0)
        transform_perspective = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform_rotation, transform_perspective])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "projective" in plan

    def test_single_parity_with_native(self):
        """perspective_grid + F.grid_sample numerically matches kornia.geometry.warp_perspective.

        Verifies the core mathematical claim: our perspective grid builder and F.grid_sample
        produce the same warped image as Kornia's reference warp_perspective for a known
        homography, confirming that the DLT + perspective_grid pipeline is correct.

        """
        import kornia.geometry
        import torch.nn.functional as F

        from fuse_augmentations.affine._matrix import (
            inv3x3,
            normalize_matrix,
            perspective_from_points,
            perspective_grid,
        )

        # Known 4-point correspondence (slight projective distortion)
        src = torch.tensor([[[0.0, 0.0], [32.0, 0.0], [32.0, 32.0], [0.0, 32.0]]])
        dst = torch.tensor([[[2.0, 1.0], [30.0, 3.0], [31.0, 29.0], [1.0, 28.0]]])
        mtx_forward = perspective_from_points(src, dst)  # (1, 3, 3) src→dst homography

        image = torch.rand(1, 3, 32, 32)

        # Reference: kornia native warp_perspective (forward src→dst homography)
        native_out = kornia.geometry.warp_perspective(image, mtx_forward, (32, 32), align_corners=True)

        # Fused path: perspective_grid + F.grid_sample (ProjectiveSegment internals)
        mtx_inverse = inv3x3(mtx_forward)
        mtx_normalized = normalize_matrix(mtx_inverse, 32, 32)
        grid = perspective_grid(mtx_normalized, 32, 32)
        image_output_fused = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        assert torch.allclose(image_output_fused, native_out, atol=1e-4), (
            "perspective_grid + F.grid_sample must match kornia.geometry.warp_perspective for the same homography"
        )


# ---------------------------------------------------------------------------
# TorchVision tests
# ---------------------------------------------------------------------------


class TestTorchVisionProjectiveParity:
    """Fused ProjectiveSegment vs native T.RandomPerspective."""

    @pytest.fixture(autouse=True)
    def require_torchvision(self):
        pytest.importorskip("torchvision", reason="torchvision required")

    def test_single_perspective_shape(self):
        """Single T.RandomPerspective produces output with same shape."""
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        image = torch.rand(2, 3, 32, 32)
        image_output = pipe(image)
        assert image_output.shape == image.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([T.RandomPerspective]).fusion_plan shows projective(...)."""
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two T.RandomPerspective fused -> n_warps_saved == 1."""
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1

    def test_v2_perspective_shape(self):
        """v2.RandomPerspective works through fused pipeline."""
        import torchvision.transforms.v2 as T

        from fuse_augmentations import Compose

        transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        image = torch.rand(2, 3, 32, 32)
        image_output = pipe(image)
        assert image_output.shape == image.shape


# ---------------------------------------------------------------------------
# Albumentations tests
# ---------------------------------------------------------------------------


class TestAlbumentationsProjectiveParity:
    """Fused AlbuProjectiveSegment vs native albu.Perspective."""

    @pytest.fixture(autouse=True)
    def require_albumentations(self):
        pytest.importorskip("albumentations", reason="albumentations required")

    def test_single_perspective_shape(self):
        """Single albu.Perspective produces output with same shape."""
        import albumentations as albu

        from fuse_augmentations import Compose

        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        image = torch.rand(2, 3, 32, 32)
        image_output = pipe(image)
        assert image_output.shape == image.shape

    def test_single_channel_perspective_shape(self):
        """Single-channel albu.Perspective preserves the grayscale channel axis."""
        import albumentations as albu

        from fuse_augmentations import Compose

        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        image = torch.rand(1, 1, 8, 8)
        image_output = pipe(image)
        assert image_output.shape == image.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([albu.Perspective]).fusion_plan shows projective(...)."""
        import albumentations as albu

        from fuse_augmentations import Compose

        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two albu.Perspective fused -> n_warps_saved == 1."""
        import albumentations as albu

        from fuse_augmentations import Compose

        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1
