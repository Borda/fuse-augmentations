"""Integration tests for the Albumentations backend adapter.

Requires albumentations >= 2.0. Tests are skipped gracefully if not installed.

Parity contracts:
- GEOMETRIC_EXACT (flip) transforms: fused ExactAffineSegment vs albumentations cv2 → atol=1e-3
  (tensor.flip is exact; any diff is float32 representation noise)
- GEOMETRIC_INTERP transforms: fused cv2 vs sequential cv2 reference → atol=1e-5
  (same matrix, same cv2.warpAffine call → results are essentially identical)

"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

albu = pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")

from fuse_augmentations import Compose  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL_PIXEL = 1e-3  # flip parity: tensor.flip vs albumentations cv2
ATOL_CV2 = 1e-5  # INTERP parity: fused cv2 vs sequential cv2 reference
HEIGHT, WIDTH, NUM_CHANNELS = 64, 64, 3


def _rand_image(batch_size: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch_size, NUM_CHANNELS, HEIGHT, WIDTH)


def _sequential_albu(transforms: list, img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply transforms sequentially via native albumentations (cv2 backend), return tensor."""
    batch_size = img_tensor.shape[0]
    results = []
    for batch_idx in range(batch_size):
        img_np = img_tensor[batch_idx].permute(1, 2, 0).cpu().numpy()
        for transform in transforms:
            img_np = transform(image=img_np)["image"]
        results.append(torch.as_tensor(img_np.astype(np.float32)).permute(2, 0, 1))
    return torch.stack(results)


def _sequential_cv2(transforms: list, img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply GEOMETRIC_INTERP transforms sequentially via cv2, matching the fused backend.

    Gets each transform's matrix from albumentations' sampler, then applies it using the same cv2.warpAffine call used
    by AlbuFusedAffineSegment. For a single transform this produces output identical to the fused pipeline.

    """
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
    from fuse_augmentations.affine._segment import _CV2_BORDER, _CV2_INTERP, _warp

    adapter = AlbumentationsAdapter()
    interp_flag = _CV2_INTERP["bilinear"]
    border_flag = _CV2_BORDER["zeros"]
    _batch_size, num_channels, _height, _width = img_tensor.shape
    results = []
    for batch_idx in range(_batch_size):
        img_np = img_tensor[batch_idx].permute(1, 2, 0).cpu().numpy()
        for tfm in transforms:
            params = adapter.sample_params(tfm, (1, num_channels, _height, _width), torch.device("cpu"))
            mtx = adapter.build_matrix(tfm, params, _height, _width)[0].double().numpy()
            mtx_inv = np.linalg.inv(mtx)
            if num_channels == 1:
                warped = _warp(img_np[:, :, 0], mtx_inv, _width, _height, interp_flag, border_flag)
                img_np = warped[:, :, np.newaxis]
            else:
                img_np = _warp(img_np, mtx_inv, _width, _height, interp_flag, border_flag)
        results.append(torch.as_tensor(img_np.astype(np.float32)).permute(2, 0, 1))
    return torch.stack(results)


# ---------------------------------------------------------------------------
# Single-transform parity tests (prob=1 to force determinism)
# ---------------------------------------------------------------------------


class TestSingleTransformParity:
    """Fused output with prob=1 must match sequential albumentations output."""

    def test_rotation_parity(self):
        img = _rand_image()

        # Reference: sequential cv2 warp with the same albumentations matrix.
        # Both fused and sequential now use cv2.warpAffine, so results are essentially identical.
        np.random.seed(42)
        seq_out = _sequential_cv2([albu.Rotate(limit=(30, 30), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([albu.Rotate(limit=(30, 30), p=1.0)])(img)

        assert fused_out.shape == img.shape
        assert torch.allclose(fused_out, seq_out, atol=ATOL_CV2), (
            f"Max diff: {(fused_out - seq_out).abs().max().item():.2e}"
        )

    def test_affine_rotation_only_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_cv2([albu.Affine(rotate=(20, 20), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([albu.Affine(rotate=(20, 20), p=1.0)])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_CV2)

    def test_affine_scale_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_cv2([albu.Affine(scale=(0.9, 0.9), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([albu.Affine(scale=(0.9, 0.9), p=1.0)])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_CV2)

    def test_hflip_parity(self):
        img = _rand_image()
        seq_out = _sequential_albu([albu.HorizontalFlip(p=1.0)], img)
        fused_out = Compose([albu.HorizontalFlip(p=1.0)])(img)
        assert torch.allclose(fused_out, seq_out, atol=ATOL_PIXEL)

    def test_vflip_parity(self):
        img = _rand_image()
        seq_out = _sequential_albu([albu.VerticalFlip(p=1.0)], img)
        fused_out = Compose([albu.VerticalFlip(p=1.0)])(img)
        assert torch.allclose(fused_out, seq_out, atol=ATOL_PIXEL)

    def test_safe_rotate_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_cv2([albu.SafeRotate(limit=(30, 30), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([albu.SafeRotate(limit=(30, 30), p=1.0)])(img)

        assert fused_out.shape == img.shape
        assert torch.allclose(fused_out, seq_out, atol=ATOL_CV2), (
            f"Max diff: {(fused_out - seq_out).abs().max().item():.2e}"
        )

    def test_shift_scale_rotate_parity(self):
        img = _rand_image()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_seq = albu.ShiftScaleRotate(rotate_limit=(15, 15), shift_limit=0, scale_limit=0, p=1.0)
            t_fus = albu.ShiftScaleRotate(rotate_limit=(15, 15), shift_limit=0, scale_limit=0, p=1.0)

        np.random.seed(42)
        seq_out = _sequential_cv2([t_seq], img)

        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fused_out = Compose([t_fus])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_CV2)


# ---------------------------------------------------------------------------
# prob=0 identity / prob=1 always-active
# ---------------------------------------------------------------------------


class TestProbabilityEdgeCases:
    def test_p0_output_unchanged(self):
        img = _rand_image()
        out = Compose([albu.Rotate(limit=30, p=0.0)])(img)
        # With prob=0 the composed matrix should be identity → output == input
        assert torch.allclose(out, img, atol=1e-5), "prob=0 should produce identity output"

    def test_p1_output_differs(self):
        img = _rand_image()
        out = Compose([albu.Rotate(limit=(45, 45), p=1.0)])(img)
        assert not torch.allclose(out, img, atol=1e-3), "prob=1 rotate should change the image"

    def test_empty_pipeline_is_identity(self):
        img = _rand_image()
        out = Compose([])(img)
        assert torch.allclose(out, img)


# ---------------------------------------------------------------------------
# Batch correctness
# ---------------------------------------------------------------------------


class TestBatchCorrectness:
    def test_output_shape_preserved(self):
        img = _rand_image(batch_size=4)
        out = Compose([albu.Rotate(limit=30, p=1.0)])(img)
        assert out.shape == img.shape

    def test_batch_size_one(self):
        img = _rand_image(batch_size=1)
        out = Compose([albu.Rotate(limit=30, p=1.0)])(img)
        assert out.shape == (1, NUM_CHANNELS, HEIGHT, WIDTH)

    def test_batch_samples_independent(self):
        """With random prob=1 rotations, batch_size=4 samples should not all be identical."""
        img = _rand_image(batch_size=4)
        out = Compose([albu.Rotate(limit=45, p=1.0)])(img)
        # At minimum, samples should not all be the same (independent per-sample RNG)
        diffs = [(out[sample_idx] - out[0]).abs().max().item() for sample_idx in range(1, 4)]
        assert any(max_diff > 1e-3 for max_diff in diffs), "Expected per-sample independence in outputs"


# ---------------------------------------------------------------------------
# Multi-transform chain
# ---------------------------------------------------------------------------


class TestMultiTransformChain:
    def test_rotate_then_hflip_fusion_plan(self):
        pipe = Compose([albu.Rotate(limit=30, p=1.0), albu.HorizontalFlip(p=1.0)])
        assert "fused" in pipe.fusion_plan
        assert pipe.n_warps_saved == 1

    def test_three_transform_chain(self):
        img = _rand_image(batch_size=2)
        pipe = Compose([
            albu.Rotate(limit=30, p=1.0),
            albu.Affine(scale=(0.8, 0.8), p=1.0),
            albu.HorizontalFlip(p=1.0),
        ])
        out = pipe(img)
        assert out.shape == img.shape
        assert pipe.n_warps_saved == 2

    def test_barrier_breaks_chain(self):
        pipe = Compose([
            albu.Rotate(limit=30, p=1.0),
            albu.GaussianBlur(p=1.0),  # SPATIAL_KERNEL barrier
            albu.HorizontalFlip(p=1.0),
        ])
        # HorizontalFlip after barrier uses ExactAffineSegment (+1); Rotate is standalone
        assert pipe.n_warps_saved == 1
        img = _rand_image()
        out = pipe(img)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# transform_matrix property
# ---------------------------------------------------------------------------


class TestTransformMatrix:
    def test_last_matrix_shape(self):
        img = _rand_image(batch_size=2)
        pipe = Compose([albu.Rotate(limit=30, p=1.0)])
        pipe(img)
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == (2, 3, 3)

    def test_last_matrix_nonzero_determinant(self):
        img = _rand_image(batch_size=3)
        pipe = Compose([albu.Affine(scale=(0.9, 0.9), p=1.0)])
        pipe(img)
        mtx = pipe.transform_matrix
        assert mtx is not None
        dets = torch.linalg.det(mtx)
        assert (dets.abs() > 1e-6).all(), "Composed matrix should be non-singular"

    def test_last_matrix_identity_when_p0(self):
        img = _rand_image(batch_size=2)
        pipe = Compose([albu.Rotate(limit=30, p=0.0)])
        pipe(img)
        mtx = pipe.transform_matrix
        assert mtx is not None
        eye = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(mtx, eye, atol=1e-5)


# ---------------------------------------------------------------------------
# fusion_plan and n_warps_saved
# ---------------------------------------------------------------------------


class TestFusionPlanAndWarps:
    def test_single_transform_zero_warps_saved(self):
        pipe = Compose([albu.Rotate(limit=30)])
        assert pipe.n_warps_saved == 0

    def test_two_transforms_one_warp_saved(self):
        pipe = Compose([albu.Rotate(limit=30), albu.Affine(scale=(0.8, 1.2))])
        assert pipe.n_warps_saved == 1

    def test_fusion_plan_contains_fused(self):
        pipe = Compose([albu.Rotate(limit=30), albu.HorizontalFlip(p=0.5)])
        assert "fused(Rotate, HorizontalFlip)" in pipe.fusion_plan
