"""Tests for CROP_RESIZE_FIXED category: adapters, CropResizeSegment, and build_segments routing.

Covers:
- TransformCategory.CROP_RESIZE_FIXED exists
- category() returns CROP_RESIZE_FIXED for RandomResizedCrop in all three adapters
- crop_resize_matrix: identity case and basic scale
- normalize_matrix_io: same-size case equals normalize_matrix; crop-to-half case
- build_segments produces CropResizeSegment for the torch path
- build_segments produces a passthrough for the albumentations numpy path
- CropResizeSegment forward: output shape matches target size
- CropResizeSegment forward: aux_targets passthrough

Run to verify behaviour and guard against regressions:
    pytest tests/test_unit/test_crop_resize_fusion.py -v

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._types import TransformCategory

# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------


def test_crop_resize_fixed_category_exists():
    """TransformCategory.CROP_RESIZE_FIXED is present in the enum."""
    assert hasattr(TransformCategory, "CROP_RESIZE_FIXED")
    assert TransformCategory.CROP_RESIZE_FIXED.value == "crop_resize_fixed"


# ---------------------------------------------------------------------------
# Matrix primitives
# ---------------------------------------------------------------------------


class TestCropResizeMatrix:
    """crop_resize_matrix: pixel-space forward affine for crop+resize."""

    def test_identity_full_image(self):
        """Crop covering the full image with the same target size gives identity."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix

        H, W = 32, 32
        top = torch.zeros(1)
        left = torch.zeros(1)
        crop_h = torch.full((1,), float(H))
        crop_w = torch.full((1,), float(W))
        target_h = torch.full((1,), float(H))
        target_w = torch.full((1,), float(W))

        M = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        assert M.shape == (1, 3, 3)
        assert torch.allclose(M, torch.eye(3).unsqueeze(0))

    def test_scale_halves_output(self):
        """Crop full image but resize to half: scale factor = 0.5."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix

        H, W = 64, 64
        top = torch.zeros(1)
        left = torch.zeros(1)
        crop_h = torch.full((1,), float(H))
        crop_w = torch.full((1,), float(W))
        target_h = torch.full((1,), float(H) / 2)
        target_w = torch.full((1,), float(W) / 2)

        M = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        # align_corners=True endpoint mapping:
        # sx = (W_out-1)/(W_in-1) = 31/63, sy = 31/63
        s = 31.0 / 63.0
        expected = torch.tensor([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        assert torch.allclose(M, expected)

    def test_top_left_offset_translates(self):
        """Crop starting at (top=8, left=8) with same-size output: translation only."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix

        top = torch.tensor([8.0])
        left = torch.tensor([8.0])
        crop_h = torch.tensor([32.0])
        crop_w = torch.tensor([32.0])
        target_h = torch.tensor([32.0])
        target_w = torch.tensor([32.0])

        M = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        # sx = 1, tx = -8; sy = 1, ty = -8
        expected = torch.tensor([[1.0, 0.0, -8.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
        assert torch.allclose(M, expected)

    def test_batch_of_two(self):
        """crop_resize_matrix accepts (B,) tensors and returns (B, 3, 3)."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix

        B = 2
        top = torch.zeros(B)
        left = torch.zeros(B)
        crop_h = torch.full((B,), 32.0)
        crop_w = torch.full((B,), 32.0)
        target_h = torch.full((B,), 32.0)
        target_w = torch.full((B,), 32.0)

        M = crop_resize_matrix(top, left, crop_h, crop_w, target_h, target_w)
        assert M.shape == (B, 3, 3)

    def test_rejects_degenerate_crop_or_target_sizes(self):
        """crop_resize_matrix rejects sizes <= 1 (singular endpoint mapping)."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix

        with pytest.raises(ValueError, match="requires crop and target sizes > 1"):
            crop_resize_matrix(
                top=torch.tensor([0.0]),
                left=torch.tensor([0.0]),
                crop_h=torch.tensor([1.0]),
                crop_w=torch.tensor([16.0]),
                target_h=torch.tensor([8.0]),
                target_w=torch.tensor([8.0]),
            )


class TestNormalizeMatrixIO:
    """normalize_matrix_io: same-size case matches normalize_matrix; IO variant."""

    def test_same_size_matches_normalize_matrix(self):
        """When H_in==H_out and W_in==W_out, normalize_matrix_io == normalize_matrix."""
        from fuse_augmentations.affine._matrix import normalize_matrix, normalize_matrix_io

        H, W = 64, 64
        M = torch.eye(3).unsqueeze(0)
        nm = normalize_matrix(M, H=H, W=W)
        nmio = normalize_matrix_io(M, H_in=H, W_in=W, H_out=H, W_out=W)
        assert torch.allclose(nm, nmio, atol=1e-6)

    def test_crop_to_half_scale_factor(self):
        """normalize_matrix_io for a 2x downscale inverse maps correctly."""
        from fuse_augmentations.affine._matrix import crop_resize_matrix, inv3x3, normalize_matrix_io

        H_in, W_in = 64, 64
        H_out, W_out = 32, 32

        top = torch.zeros(1)
        left = torch.zeros(1)
        M_fwd = crop_resize_matrix(
            top,
            left,
            torch.full((1,), float(H_in)),
            torch.full((1,), float(W_in)),
            torch.full((1,), float(H_out)),
            torch.full((1,), float(W_out)),
        )
        M_inv = inv3x3(M_fwd)
        M_norm = normalize_matrix_io(M_inv, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out)
        assert M_norm.shape == (1, 3, 3)
        # The normalized matrix should have finite values
        assert torch.all(torch.isfinite(M_norm))

    def test_crop_resize_matches_explicit_crop_then_interpolate(self):
        """normalize_matrix_io + grid_sample matches explicit crop + interpolate."""
        import torch.nn.functional as F

        from fuse_augmentations.affine._matrix import crop_resize_matrix, inv3x3, normalize_matrix_io

        H_in, W_in = 64, 80
        H_out, W_out = 32, 32
        top, left = 7, 11
        crop_h, crop_w = 40, 50

        x = torch.linspace(0.0, 1.0, H_in * W_in).reshape(1, 1, H_in, W_in).repeat(1, 3, 1, 1)
        ref = F.interpolate(
            x[:, :, top : top + crop_h, left : left + crop_w],
            size=(H_out, W_out),
            mode="bilinear",
            align_corners=True,
        )

        M_fwd = crop_resize_matrix(
            top=torch.tensor([float(top)]),
            left=torch.tensor([float(left)]),
            crop_h=torch.tensor([float(crop_h)]),
            crop_w=torch.tensor([float(crop_w)]),
            target_h=torch.tensor([float(H_out)]),
            target_w=torch.tensor([float(W_out)]),
        )
        M_inv = inv3x3(M_fwd)
        M_norm = normalize_matrix_io(M_inv, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out)
        grid = F.affine_grid(M_norm[:, :2, :], [1, 3, H_out, W_out], align_corners=True)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Adapter category classification
# ---------------------------------------------------------------------------


class TestKorniaAdapterCategory:
    """KorniaAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_random_resized_crop_category(self):
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        t = kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        assert KorniaAdapter.category(t) == TransformCategory.CROP_RESIZE_FIXED


class TestTorchVisionAdapterCategory:
    """TorchVisionAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_v1_random_resized_crop_category(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        t = tv.transforms.RandomResizedCrop(size=(64, 64))
        assert TorchVisionAdapter.category(t) == TransformCategory.CROP_RESIZE_FIXED

    def test_v2_random_resized_crop_category(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        t = tv.transforms.v2.RandomResizedCrop(size=(64, 64))
        assert TorchVisionAdapter.category(t) == TransformCategory.CROP_RESIZE_FIXED


class TestAlbumentationsAdapterCategory:
    """AlbumentationsAdapter.category() returns CROP_RESIZE_FIXED for RandomResizedCrop."""

    def test_random_resized_crop_category(self):
        pytest.importorskip("albumentations")
        import albumentations as A

        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

        t = A.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        assert AlbumentationsAdapter.category(t) == TransformCategory.CROP_RESIZE_FIXED


# ---------------------------------------------------------------------------
# build_segments routing
# ---------------------------------------------------------------------------


class TestBuildSegmentsCropResize:
    """build_segments emits CropResizeSegment (torch) or passthrough (numpy)."""

    def test_kornia_torch_path_produces_crop_resize_segment(self):
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations.adapters._kornia import KorniaAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment, build_segments

        t = kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        segments = build_segments([t], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    def test_torchvision_v1_torch_path_produces_crop_resize_segment(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment, build_segments

        t = tv.transforms.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        segments = build_segments([t], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    def test_torchvision_v2_torch_path_produces_crop_resize_segment(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment, build_segments

        t = tv.transforms.v2.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        segments = build_segments([t], adapter, use_numpy=False)
        assert len(segments) == 1
        assert isinstance(segments[0], CropResizeSegment)

    def test_albumentations_numpy_path_produces_passthrough(self):
        """Albumentations path emits the raw transform (not CropResizeSegment)."""
        pytest.importorskip("albumentations")
        import albumentations as A

        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment, build_segments

        t = A.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = AlbumentationsAdapter()
        segments = build_segments([t], adapter, use_numpy=True)
        # Should emit passthrough, not CropResizeSegment
        assert len(segments) == 1
        assert not isinstance(segments[0], CropResizeSegment)

    def test_crop_resize_flushes_preceding_geo(self):
        """A preceding GEOMETRIC_INTERP run is flushed before the CropResizeSegment."""
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations.adapters._kornia import KorniaAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment, FusedAffineSegment, build_segments

        t_rotate = kornia.augmentation.RandomRotation(degrees=30.0, p=1.0)
        t_crop = kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        segments = build_segments([t_rotate, t_crop], adapter, use_numpy=False)
        # Two segments: a fused affine segment and a crop-resize segment
        assert len(segments) == 2
        assert isinstance(segments[0], FusedAffineSegment)
        assert isinstance(segments[1], CropResizeSegment)


# ---------------------------------------------------------------------------
# CropResizeSegment forward pass
# ---------------------------------------------------------------------------


class TestCropResizeSegmentForward:
    """CropResizeSegment produces correct output shape."""

    @pytest.fixture
    def kornia_crop_segment(self):
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations.adapters._kornia import KorniaAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment

        t = kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        adapter = KorniaAdapter()
        return CropResizeSegment(t, adapter)

    @pytest.fixture
    def tv_v1_crop_segment(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment

        t = tv.transforms.RandomResizedCrop(size=(64, 64))
        adapter = TorchVisionAdapter()
        return CropResizeSegment(t, adapter)

    def test_kornia_output_shape(self, kornia_crop_segment):
        """Kornia CropResizeSegment outputs (B, C, target_H, target_W)."""
        seg = kornia_crop_segment
        x = torch.rand(2, 3, 128, 128)
        out = seg(x)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_tv_v1_output_shape(self, tv_v1_crop_segment):
        """TorchVision v1 CropResizeSegment outputs (B, C, target_H, target_W)."""
        seg = tv_v1_crop_segment
        x = torch.rand(2, 3, 128, 128)
        out = seg(x)
        assert out.shape == torch.Size([2, 3, 64, 64])

    def test_output_values_bounded(self, kornia_crop_segment):
        """Output values stay within input range (bilinear interpolation, no padding)."""
        seg = kornia_crop_segment
        x = torch.rand(1, 3, 128, 128)
        out = seg(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_aux_targets_passthrough(self, kornia_crop_segment):
        """With aux_targets=None returns a tensor; with dict returns (tensor, dict)."""
        seg = kornia_crop_segment
        x = torch.rand(1, 3, 128, 128)

        # Without aux_targets
        out = seg(x, aux_targets=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([1, 3, 64, 64])

        # With aux_targets (should be passed through unchanged in F.1)
        aux = {"mask": torch.zeros(1, 1, 128, 128)}
        out2, aux_out = seg(x, aux_targets=aux)
        assert isinstance(out2, torch.Tensor)
        assert aux_out is aux  # passed through unchanged

    def test_batch_size_one(self, kornia_crop_segment):
        """Works correctly with batch size 1."""
        seg = kornia_crop_segment
        x = torch.rand(1, 3, 128, 128)
        out = seg(x)
        assert out.shape == torch.Size([1, 3, 64, 64])

    def test_p_zero_is_ignored(self):
        """CropResizeSegment always applies regardless of p; p=0.0 still crops."""
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations.adapters._kornia import KorniaAdapter
        from fuse_augmentations.affine._segment import CropResizeSegment

        t = kornia.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0), p=0.0)
        seg = CropResizeSegment(t, KorniaAdapter())
        x = torch.rand(2, 3, 64, 64)
        out = seg(x)
        # Shape must be target size even though p=0.0
        assert out.shape == torch.Size([2, 3, 32, 32])


# ---------------------------------------------------------------------------
# fusion_plan_descriptors
# ---------------------------------------------------------------------------


class TestFusionPlanDescriptors:
    """fusion_plan_descriptors includes crop_resize kind."""

    def test_crop_resize_descriptor_kind(self):
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        t = kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))
        pipe = FusedCompose([t], adapter=KorniaAdapter())
        descs = pipe.fusion_plan_descriptors
        assert len(descs) == 1
        assert descs[0].kind == "crop_resize"
        assert descs[0].n_warps_saved == 0

    def test_full_pipeline_descriptor_kinds(self):
        """Rotation → CropResize → HFlip gives [fused, crop_resize, exact]."""
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        transforms = [
            kornia.augmentation.RandomRotation(degrees=30.0, p=1.0),
            kornia.augmentation.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),
            kornia.augmentation.RandomHorizontalFlip(p=1.0),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        kinds = [d.kind for d in pipe.fusion_plan_descriptors]
        assert kinds == ["fused", "crop_resize", "exact"]


# ---------------------------------------------------------------------------
# End-to-end forward through FusedCompose
# ---------------------------------------------------------------------------


class TestFusedComposeWithCropResize:
    """FusedCompose.forward with a CropResizeSegment changes output shape."""

    def test_kornia_compose_output_shape(self):
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        t = kornia.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0))
        pipe = FusedCompose([t], adapter=KorniaAdapter())
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_tv_v1_compose_output_shape(self):
        tv = pytest.importorskip("torchvision")
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        t = tv.transforms.RandomResizedCrop(size=(32, 32))
        pipe = FusedCompose([t], adapter=TorchVisionAdapter())
        x = torch.rand(2, 3, 64, 64)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_rotation_then_crop_resize_shape(self):
        """Rotation → CropResize pipeline outputs the crop-resize target size."""
        kornia = pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        transforms = [
            kornia.augmentation.RandomRotation(degrees=15.0, p=1.0),
            kornia.augmentation.RandomResizedCrop(size=(48, 48), scale=(0.5, 1.0)),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        x = torch.rand(2, 3, 96, 96)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 48, 48])
