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
import torch.nn.functional as F

from fuse_augmentations import Compose
from fuse_augmentations._compat import (
    _ALBUMENTATIONS_AVAILABLE,
    _KORNIA_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)
from fuse_augmentations.affine.matrix import (
    inv3x3,
    normalize_matrix,
    perspective_from_points,
    perspective_grid,
)

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug
    import kornia.geometry

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans
    import torchvision.transforms.v2 as Tv2

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKorniaProjectiveParity:
    """ProjectiveSegment coverage for K.RandomPerspective."""

    def test_single_perspective_shape(self):
        """Single K.RandomPerspective produces output with same shape."""
        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([K.RandomPerspective(p=1)]).fusion_plan shows projective(...)."""
        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan
        assert "RandomPerspective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two K.RandomPerspective fused into one segment -> n_warps_saved == 1."""
        transform = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1

    def test_single_perspective_records_transform_matrix(self):
        """Single fused RandomPerspective records a finite 3x3 transform matrix."""
        transform = kornia_aug.RandomPerspective(distortion_scale=0.4, p=1.0)
        img = torch.rand(1, 3, 32, 32)

        # Fused path -- same transform, uses ProjectiveSegment
        pipe = Compose([transform])
        fused_out = pipe(img)

        assert fused_out.shape == img.shape
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == (1, 3, 3)
        assert torch.isfinite(pipe.transform_matrix).all()

    def test_mixed_rotate_then_perspective_gives_two_segments(self):
        """[Rotation, RandomPerspective] -> fusion_plan has fused(...) -> projective(...)."""
        rot = kornia_aug.RandomRotation(degrees=30, p=1.0)
        persp = kornia_aug.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([rot, persp])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "projective" in plan

    def test_single_parity_with_native(self):
        """perspective_grid + F.grid_sample numerically matches kornia.geometry.warp_perspective.

        Verifies the core mathematical claim: our perspective grid builder and F.grid_sample
        produce the same warped image as Kornia's reference warp_perspective for a known
        homography, confirming that the DLT + perspective_grid pipeline is correct.

        """
        # Known 4-point correspondence (slight projective distortion)
        src = torch.tensor([[[0.0, 0.0], [32.0, 0.0], [32.0, 32.0], [0.0, 32.0]]])
        dst = torch.tensor([[[2.0, 1.0], [30.0, 3.0], [31.0, 29.0], [1.0, 28.0]]])
        mat_fwd = perspective_from_points(src, dst)  # (1, 3, 3) src→dst homography

        img = torch.rand(1, 3, 32, 32)

        # Reference: kornia native warp_perspective (forward src→dst homography)
        native_out = kornia.geometry.warp_perspective(img, mat_fwd, (32, 32), align_corners=True)

        # Fused path: perspective_grid + F.grid_sample (ProjectiveSegment internals)
        mat_inv = inv3x3(mat_fwd)
        mat_norm = normalize_matrix(mat_inv, 32, 32)
        grid = perspective_grid(mat_norm, 32, 32)
        fused_out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        assert torch.allclose(fused_out, native_out, atol=1e-4), (
            "perspective_grid + F.grid_sample must match kornia.geometry.warp_perspective for the same homography"
        )


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestTorchVisionProjectiveParity:
    """Fused ProjectiveSegment vs native tv_trans.RandomPerspective."""

    def test_single_perspective_shape(self):
        """Single tv_trans.RandomPerspective produces output with same shape."""
        transform = tv_trans.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([tv_trans.RandomPerspective]).fusion_plan shows projective(...)."""
        transform = tv_trans.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two tv_trans.RandomPerspective fused -> n_warps_saved == 1."""
        transform = tv_trans.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1

    def test_v2_perspective_shape(self):
        """v2.RandomPerspective works through fused pipeline."""
        transform = Tv2.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([transform])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbumentationsProjectiveParity:
    """Fused AlbuProjectiveSegment vs native A.Perspective."""

    def test_single_perspective_shape(self):
        """Single A.Perspective produces output with same shape."""
        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_channel_perspective_shape(self):
        """Single-channel A.Perspective preserves the grayscale channel axis."""
        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        img = torch.rand(1, 1, 8, 8)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([A.Perspective]).fusion_plan shows projective(...)."""
        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two A.Perspective fused -> n_warps_saved == 1."""
        transform = albu.Perspective(p=1.0)
        pipe = Compose([transform, transform])
        assert pipe.n_warps_saved == 1
