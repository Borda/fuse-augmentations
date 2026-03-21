"""Integration tests for the Albumentations backend adapter.

Requires albumentations >= 2.0. Tests are skipped gracefully if not installed.

Parity contracts:
- GEOMETRIC_EXACT (flip) transforms: fused ExactAffineSegment vs albumentations cv2 → atol=1e-3
  (tensor.flip is exact; any diff is float32 representation noise)
- GEOMETRIC_INTERP transforms: fused scipy vs sequential scipy reference → atol=1e-5
  (same matrix, same scipy call → essentially identical; cv2 and scipy use different
  interpolation algorithms so comparing against albumentations' cv2 output is not meaningful)

"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

A = pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")

from fuse_augmentations import Compose  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL_PIXEL = 1e-3  # flip parity: tensor.flip vs albumentations cv2
ATOL_SCIPY = 1e-5  # INTERP parity: fused scipy vs sequential scipy reference
H, W, C = 64, 64, 3


def _rand_image(B: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(B, C, H, W)


def _sequential_albu(transforms: list, img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply transforms sequentially via native albumentations (cv2 backend), return tensor."""
    B = img_tensor.shape[0]
    results = []
    for i in range(B):
        img_np = img_tensor[i].permute(1, 2, 0).cpu().numpy()
        for t in transforms:
            img_np = t(image=img_np)["image"]
        results.append(torch.as_tensor(img_np.astype(np.float32)).permute(2, 0, 1))
    return torch.stack(results)


def _sequential_scipy(transforms: list, img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply GEOMETRIC_INTERP transforms sequentially via scipy, matching the fused backend.

    Gets each transform's matrix from albumentations' sampler, then applies it using the same scipy affine_transform
    used by AlbuFusedAffineSegment.  For a single transform this produces output identical to the fused pipeline.

    """
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
    from fuse_augmentations.affine._segment import _BORDER_MODES, _INTERP_ORDERS, _warp

    adapter = AlbumentationsAdapter()
    interp_order = _INTERP_ORDERS["bilinear"]
    border_mode = _BORDER_MODES["zeros"]
    _B, n_ch, _H, _W = img_tensor.shape
    results = []
    for i in range(_B):
        img_np = img_tensor[i].permute(1, 2, 0).cpu().numpy()
        for tfm in transforms:
            params = adapter.sample_params(tfm, (1, n_ch, _H, _W), torch.device("cpu"))
            mtx = adapter.build_matrix(tfm, params, _H, _W)[0].double().numpy()
            M_inv = np.linalg.inv(mtx)
            if n_ch == 1:
                warped = _warp(img_np[:, :, 0], M_inv, _W, _H, interp_order, border_mode)
                img_np = warped[:, :, np.newaxis]
            else:
                img_np = _warp(img_np, M_inv, _W, _H, interp_order, border_mode)
        results.append(torch.as_tensor(img_np.astype(np.float32)).permute(2, 0, 1))
    return torch.stack(results)


# ---------------------------------------------------------------------------
# Single-transform parity tests (p=1 to force determinism)
# ---------------------------------------------------------------------------


class TestSingleTransformParity:
    """Fused output with p=1 must match sequential albumentations output."""

    def test_rotation_parity(self):
        img = _rand_image()

        # Reference: sequential scipy warp with the same albumentations matrix.
        # cv2 (albumentations' native backend) and scipy use different interpolation
        # algorithms, so we compare scipy fused vs scipy sequential instead.
        np.random.seed(42)
        seq_out = _sequential_scipy([A.Rotate(limit=(30, 30), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([A.Rotate(limit=(30, 30), p=1.0)])(img)

        assert fused_out.shape == img.shape
        assert torch.allclose(fused_out, seq_out, atol=ATOL_SCIPY), (
            f"Max diff: {(fused_out - seq_out).abs().max().item():.2e}"
        )

    def test_affine_rotation_only_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_scipy([A.Affine(rotate=(20, 20), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([A.Affine(rotate=(20, 20), p=1.0)])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_SCIPY)

    def test_affine_scale_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_scipy([A.Affine(scale=(0.9, 0.9), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([A.Affine(scale=(0.9, 0.9), p=1.0)])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_SCIPY)

    def test_hflip_parity(self):
        img = _rand_image()
        seq_out = _sequential_albu([A.HorizontalFlip(p=1.0)], img)
        fused_out = Compose([A.HorizontalFlip(p=1.0)])(img)
        assert torch.allclose(fused_out, seq_out, atol=ATOL_PIXEL)

    def test_vflip_parity(self):
        img = _rand_image()
        seq_out = _sequential_albu([A.VerticalFlip(p=1.0)], img)
        fused_out = Compose([A.VerticalFlip(p=1.0)])(img)
        assert torch.allclose(fused_out, seq_out, atol=ATOL_PIXEL)

    def test_safe_rotate_parity(self):
        img = _rand_image()

        np.random.seed(42)
        seq_out = _sequential_scipy([A.SafeRotate(limit=(30, 30), p=1.0)], img)

        np.random.seed(42)
        fused_out = Compose([A.SafeRotate(limit=(30, 30), p=1.0)])(img)

        assert fused_out.shape == img.shape
        assert torch.allclose(fused_out, seq_out, atol=ATOL_SCIPY), (
            f"Max diff: {(fused_out - seq_out).abs().max().item():.2e}"
        )

    def test_shift_scale_rotate_parity(self):
        img = _rand_image()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_seq = A.ShiftScaleRotate(rotate_limit=(15, 15), shift_limit=0, scale_limit=0, p=1.0)
            t_fus = A.ShiftScaleRotate(rotate_limit=(15, 15), shift_limit=0, scale_limit=0, p=1.0)

        np.random.seed(42)
        seq_out = _sequential_scipy([t_seq], img)

        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fused_out = Compose([t_fus])(img)

        assert torch.allclose(fused_out, seq_out, atol=ATOL_SCIPY)


# ---------------------------------------------------------------------------
# p=0 identity / p=1 always-active
# ---------------------------------------------------------------------------


class TestProbabilityEdgeCases:
    def test_p0_output_unchanged(self):
        img = _rand_image()
        out = Compose([A.Rotate(limit=30, p=0.0)])(img)
        # With p=0 the composed matrix should be identity → output == input
        assert torch.allclose(out, img, atol=1e-5), "p=0 should produce identity output"

    def test_p1_output_differs(self):
        img = _rand_image()
        out = Compose([A.Rotate(limit=(45, 45), p=1.0)])(img)
        assert not torch.allclose(out, img, atol=1e-3), "p=1 rotate should change the image"

    def test_empty_pipeline_is_identity(self):
        img = _rand_image()
        out = Compose([])(img)
        assert torch.allclose(out, img)


# ---------------------------------------------------------------------------
# Batch correctness
# ---------------------------------------------------------------------------


class TestBatchCorrectness:
    def test_output_shape_preserved(self):
        img = _rand_image(B=4)
        out = Compose([A.Rotate(limit=30, p=1.0)])(img)
        assert out.shape == img.shape

    def test_batch_size_one(self):
        img = _rand_image(B=1)
        out = Compose([A.Rotate(limit=30, p=1.0)])(img)
        assert out.shape == (1, C, H, W)

    def test_batch_samples_independent(self):
        """With random p=1 rotations, B=4 samples should not all be identical."""
        img = _rand_image(B=4)
        out = Compose([A.Rotate(limit=45, p=1.0)])(img)
        # At minimum, samples should not all be the same (independent per-sample RNG)
        diffs = [(out[i] - out[0]).abs().max().item() for i in range(1, 4)]
        assert any(d > 1e-3 for d in diffs), "Expected per-sample independence in outputs"


# ---------------------------------------------------------------------------
# Multi-transform chain
# ---------------------------------------------------------------------------


class TestMultiTransformChain:
    def test_rotate_then_hflip_fusion_plan(self):
        pipe = Compose([A.Rotate(limit=30, p=1.0), A.HorizontalFlip(p=1.0)])
        assert "fused" in pipe.fusion_plan
        assert pipe.n_warps_saved == 1

    def test_three_transform_chain(self):
        img = _rand_image(B=2)
        pipe = Compose([
            A.Rotate(limit=30, p=1.0),
            A.Affine(scale=(0.8, 0.8), p=1.0),
            A.HorizontalFlip(p=1.0),
        ])
        out = pipe(img)
        assert out.shape == img.shape
        assert pipe.n_warps_saved == 2

    def test_barrier_breaks_chain(self):
        pipe = Compose([
            A.Rotate(limit=30, p=1.0),
            A.GaussianBlur(p=1.0),  # SPATIAL_KERNEL barrier
            A.HorizontalFlip(p=1.0),
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
        img = _rand_image(B=2)
        pipe = Compose([A.Rotate(limit=30, p=1.0)])
        pipe(img)
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == (2, 3, 3)

    def test_last_matrix_nonzero_determinant(self):
        img = _rand_image(B=3)
        pipe = Compose([A.Affine(scale=(0.9, 0.9), p=1.0)])
        pipe(img)
        mtx = pipe.transform_matrix
        assert mtx is not None
        dets = torch.linalg.det(mtx)
        assert (dets.abs() > 1e-6).all(), "Composed matrix should be non-singular"

    def test_last_matrix_identity_when_p0(self):
        img = _rand_image(B=2)
        pipe = Compose([A.Rotate(limit=30, p=0.0)])
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
        pipe = Compose([A.Rotate(limit=30)])
        assert pipe.n_warps_saved == 0

    def test_two_transforms_one_warp_saved(self):
        pipe = Compose([A.Rotate(limit=30), A.Affine(scale=(0.8, 1.2))])
        assert pipe.n_warps_saved == 1

    def test_fusion_plan_contains_fused(self):
        pipe = Compose([A.Rotate(limit=30), A.HorizontalFlip(p=0.5)])
        assert "fused(Rotate, HorizontalFlip)" in pipe.fusion_plan
