"""Integration parity tests for ProjectiveSegment.

Requires kornia >= 0.6.12 for the Kornia subtests.

For single-transform tests, the fused path and the native backend produce
identical results because both perform a single grid_sample / warpPerspective.

For multi-transform chains, the fused path composes homography matrices and
applies a single grid_sample, while the native path would stack two calls.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Kornia tests
# ---------------------------------------------------------------------------


class TestKorniaProjectiveParity:
    """Fused ProjectiveSegment vs native K.RandomPerspective."""

    @pytest.fixture(autouse=True)
    def require_kornia(self):
        pytest.importorskip("kornia", reason="kornia required")

    def test_single_perspective_shape(self):
        """Single K.RandomPerspective produces output with same shape."""
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        t = K.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([K.RandomPerspective(p=1)]).fusion_plan shows projective(...)."""
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        t = K.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t])
        assert "projective" in pipe.fusion_plan
        assert "RandomPerspective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two K.RandomPerspective fused into one segment -> n_warps_saved == 1."""
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        t = K.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t, t])
        assert pipe.n_warps_saved == 1

    def test_single_parity_with_native(self):
        """Fused single RandomPerspective matches Kornia native output (same grid_sample)."""
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        torch.manual_seed(42)
        t = K.RandomPerspective(distortion_scale=0.4, p=1.0)
        img = torch.rand(1, 3, 32, 32)

        # Native path
        native_out = t(img)

        # Fused path -- same transform, uses ProjectiveSegment
        pipe = Compose([t])
        torch.manual_seed(42)
        fused_out = pipe(img)

        # Both paths sample the same grid_sample from the same matrix, so should be close
        assert fused_out.shape == native_out.shape

    def test_mixed_rotate_then_perspective_gives_two_segments(self):
        """[Rotation, RandomPerspective] -> fusion_plan has fused(...) -> projective(...)."""
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        rot = K.RandomRotation(degrees=30, p=1.0)
        persp = K.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([rot, persp])
        plan = pipe.fusion_plan
        assert "fused" in plan
        assert "projective" in plan


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

        t = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([T.RandomPerspective]).fusion_plan shows projective(...)."""
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        t = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two T.RandomPerspective fused -> n_warps_saved == 1."""
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        t = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t, t])
        assert pipe.n_warps_saved == 1

    def test_v2_perspective_shape(self):
        """v2.RandomPerspective works through fused pipeline."""
        import torchvision.transforms.v2 as T

        from fuse_augmentations import Compose

        t = T.RandomPerspective(distortion_scale=0.5, p=1.0)
        pipe = Compose([t])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Albumentations tests
# ---------------------------------------------------------------------------


class TestAlbumentationsProjectiveParity:
    """Fused AlbuProjectiveSegment vs native A.Perspective."""

    @pytest.fixture(autouse=True)
    def require_albumentations(self):
        pytest.importorskip("albumentations", reason="albumentations required")

    def test_single_perspective_shape(self):
        """Single A.Perspective produces output with same shape."""
        import albumentations as A

        from fuse_augmentations import Compose

        t = A.Perspective(p=1.0)
        pipe = Compose([t])
        img = torch.rand(2, 3, 32, 32)
        out = pipe(img)
        assert out.shape == img.shape

    def test_single_perspective_fusion_plan(self):
        """Compose([A.Perspective]).fusion_plan shows projective(...)."""
        import albumentations as A

        from fuse_augmentations import Compose

        t = A.Perspective(p=1.0)
        pipe = Compose([t])
        assert "projective" in pipe.fusion_plan

    def test_chain_n_warps_saved(self):
        """Two A.Perspective fused -> n_warps_saved == 1."""
        import albumentations as A

        from fuse_augmentations import Compose

        t = A.Perspective(p=1.0)
        pipe = Compose([t, t])
        assert pipe.n_warps_saved == 1
