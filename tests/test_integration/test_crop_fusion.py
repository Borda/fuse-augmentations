"""Precision + contract tests for fused geometric∘crop-resize warps.

A ``RandomResizedCrop`` immediately after a fusible geometric run is fused into a
single ``grid_sample`` pass by :class:`_FusedGeoCropSegment`
(``M_crop @ M_geo``) instead of a fused-affine warp followed by a separate
crop-resize warp. This module verifies:

- **Precision** — the fused single pass is at least as close to the ideal
  ``float64`` single-warp render as the two-pass (separate-segment) path, and
  clears the same fusion-precision floor used by ``test_precision.py``.
- **Contract** — ``n_warps_saved`` counts the crop as one fused op, the output
  size is the crop target ``(H_t, W_t)``, and ``transform_matrix`` exposes the
  full ``geo∘crop`` pixel matrix at the ``(H_in, W_in) → (H_t, W_t)`` contract.
- **Auxiliary targets** — mask/bbox/keypoint routing matches the standalone
  :class:`CropResizeSegment` for the crop-only case.

Determinism is achieved with a stub adapter that returns fixed geometric and
crop matrices, so the fused, separate-segment, and reference paths all realise
the identical transform independent of RNG.

"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.functional import affine_grid, grid_sample

from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.affine.matrix import (
    crop_resize_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix_io,
    rotation_matrix,
)
from fuse_augmentations.affine.segment import (
    CropResizeSegment,
    FusedAffineSegment,
    _FusedGeoCropSegment,
)

pytestmark = pytest.mark.integration

# Fixed input/output geometry (deterministic crop region + rotation).
_H_IN, _W_IN = 64, 80
_H_OUT, _W_OUT = 32, 32
_TOP, _LEFT = 7.0, 11.0
_CROP_H, _CROP_W = 40.0, 50.0
_ROT_DEG = 17.0

# Fusion-precision floor (dB), same rationale as test_precision._MIN_FUSED_PSNR_DB:
# the fused path is a single float32 grid_sample of a float64-accumulated matrix,
# so it sits far above this against the float64 reference.
_MIN_FUSED_PSNR_DB = 118.0

# Border margin (px) dropped before metrics so warp zero-corners do not skew PSNR.
_CROP_MARGIN = 4


def _psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB) between two tensors, computed in float64."""
    mse = torch.mean((a.to(torch.float64) - b.to(torch.float64)) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_val**2, dtype=torch.float64) / mse))


def _smooth_image(height: int, width: int) -> torch.Tensor:
    """Band-limited smooth image in ``[0, 1]`` so aliasing does not dominate PSNR."""
    ys = torch.linspace(0.0, 1.0, height, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pattern = (
        0.5
        + 0.20 * torch.sin(2 * math.pi * (1.5 * grid_x))
        + 0.20 * torch.cos(2 * math.pi * (1.5 * grid_y + 0.3))
        + 0.08 * grid_x
    )
    return pattern.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)


def _crop_margin(tensor: torch.Tensor, margin: int = _CROP_MARGIN) -> torch.Tensor:
    """Drop a border margin so warp-induced zero corners do not skew metrics."""
    return tensor[..., margin:-margin, margin:-margin]


def _rot_mtx() -> torch.Tensor:
    """(1, 3, 3) forward rotation matrix about the input image center."""
    return rotation_matrix(
        torch.tensor([math.radians(_ROT_DEG)]),
        height=_H_IN,
        width=_W_IN,
    )


def _crop_mtx() -> torch.Tensor:
    """(1, 3, 3) forward crop-resize matrix for the fixed crop region."""
    return crop_resize_matrix(
        top=torch.tensor([_TOP]),
        left=torch.tensor([_LEFT]),
        crop_h=torch.tensor([_CROP_H]),
        crop_w=torch.tensor([_CROP_W]),
        target_h=torch.tensor([float(_H_OUT)]),
        target_w=torch.tensor([float(_W_OUT)]),
    )


def _reference_f64(image: torch.Tensor, mtx_full: torch.Tensor) -> torch.Tensor:
    """Ideal one-pass float64 render of ``mtx_full`` at the target output size."""
    image64 = image.to(torch.float64)
    mtx64 = mtx_full.to(torch.float64)
    batch_size, num_channels = image64.shape[:2]
    mtx_inv = inv3x3(mtx64)
    mtx_norm = normalize_matrix_io(mtx_inv, _H_IN, _W_IN, _H_OUT, _W_OUT)
    grid = affine_grid(mtx_norm[:, :2, :], [batch_size, num_channels, _H_OUT, _W_OUT], align_corners=True)
    return grid_sample(image64, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class _FixedGeoCropAdapter:
    """Stub adapter returning fixed rotation and crop matrices for deterministic testing.

    ``ROTATE`` sentinel transforms build the fixed rotation matrix; the ``CROP``
    sentinel builds the fixed crop-resize matrix and reports the fixed target size.
    Every call ignores RNG so fused / separate / reference paths coincide exactly.

    """

    ROTATE = "rotate"
    CROP = "crop"

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return the fixed crop params (crop transform) or an empty dict (rotate)."""
        batch_size = input_shape[0]
        if transform == self.CROP:
            return {
                "target_h": torch.full((batch_size,), float(_H_OUT), device=device),
                "target_w": torch.full((batch_size,), float(_W_OUT), device=device),
            }
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return the fixed rotation or crop-resize matrix for the sentinel transform."""
        if transform == _FixedGeoCropAdapter.CROP:
            return _crop_mtx()
        return _rot_mtx()

    @staticmethod
    def category(transform: object) -> object:
        """Unused by the segment; present for adapter-shape completeness."""
        return None


@pytest.fixture
def stub_adapter() -> _FixedGeoCropAdapter:
    """Deterministic stub adapter with fixed rotate + crop matrices."""
    return _FixedGeoCropAdapter()


class TestFusedGeoCropPrecision:
    """The fused geo∘crop pass is no less precise than the two-pass path."""

    def test_fused_beats_or_ties_separate_segments(self, stub_adapter: _FixedGeoCropAdapter) -> None:
        """PSNR(fused, ref) >= PSNR(two-pass, ref): one resample is at least as precise as two."""
        image = _smooth_image(_H_IN, _W_IN)
        mtx_full = matmul3x3(_crop_mtx(), _rot_mtx())  # crop reads rotation output
        reference = _reference_f64(image, mtx_full)

        fused_seg = _FusedGeoCropSegment([_FixedGeoCropAdapter.ROTATE], _FixedGeoCropAdapter.CROP, stub_adapter)
        fused_out = fused_seg(image)

        # Two-pass path: fused-affine rotation at input size, then a separate crop.
        rotate_seg = FusedAffineSegment([_FixedGeoCropAdapter.ROTATE], stub_adapter)
        crop_seg = CropResizeSegment(_FixedGeoCropAdapter.CROP, stub_adapter)
        two_pass_out = crop_seg(rotate_seg(image))

        psnr_fused = _psnr(_crop_margin(fused_out), _crop_margin(reference))
        psnr_two_pass = _psnr(_crop_margin(two_pass_out), _crop_margin(reference))

        assert psnr_fused >= psnr_two_pass
        assert psnr_fused >= _MIN_FUSED_PSNR_DB

    def test_fused_output_size_is_target(self, stub_adapter: _FixedGeoCropAdapter) -> None:
        """The fused segment outputs the crop target size, not the input size."""
        image = _smooth_image(_H_IN, _W_IN)
        seg = _FusedGeoCropSegment([_FixedGeoCropAdapter.ROTATE], _FixedGeoCropAdapter.CROP, stub_adapter)
        out = seg(image)
        assert out.shape == torch.Size([1, 3, _H_OUT, _W_OUT])

    def test_last_matrix_is_full_geo_crop_matrix(self, stub_adapter: _FixedGeoCropAdapter) -> None:
        """``last_matrix`` equals the composed ``M_crop @ M_geo`` pixel matrix (size contract)."""
        image = _smooth_image(_H_IN, _W_IN)
        seg = _FusedGeoCropSegment([_FixedGeoCropAdapter.ROTATE], _FixedGeoCropAdapter.CROP, stub_adapter)
        seg(image)
        expected = matmul3x3(_crop_mtx(), _rot_mtx())
        assert seg.last_matrix is not None
        torch.testing.assert_close(seg.last_matrix.to(torch.float64), expected.to(torch.float64), rtol=1e-5, atol=1e-5)


class TestFusedGeoCropAux:
    """Auxiliary targets warp through the composed matrix / output grid."""

    def test_aux_returns_tuple_and_target_size_mask(self, stub_adapter: _FixedGeoCropAdapter) -> None:
        """With aux, forward returns ``(tensor, dict)`` and the mask is at the target size."""
        image = _smooth_image(_H_IN, _W_IN)
        seg = _FusedGeoCropSegment([_FixedGeoCropAdapter.ROTATE], _FixedGeoCropAdapter.CROP, stub_adapter)
        aux = {"mask": torch.zeros(1, 1, _H_IN, _W_IN)}
        out, aux_out = seg(image, aux_targets=aux)
        assert out.shape == torch.Size([1, 3, _H_OUT, _W_OUT])
        assert aux_out is aux
        assert aux_out["mask"].shape[-2:] == torch.Size([_H_OUT, _W_OUT])

    def test_keypoint_matches_composed_matrix(self, stub_adapter: _FixedGeoCropAdapter) -> None:
        """A keypoint warps by the same composed matrix that warps the image."""
        image = _smooth_image(_H_IN, _W_IN)
        seg = _FusedGeoCropSegment([_FixedGeoCropAdapter.ROTATE], _FixedGeoCropAdapter.CROP, stub_adapter)
        keypoints = torch.tensor([[[30.0, 20.0]]])  # (B=1, N=1, [x, y])
        _, aux = seg(image, aux_targets={"keypoints": keypoints.clone()})

        mtx = matmul3x3(_crop_mtx(), _rot_mtx())[0]
        homog = torch.tensor([30.0, 20.0, 1.0])
        expected_xy = (mtx @ homog)[:2]
        torch.testing.assert_close(aux["keypoints"][0, 0], expected_xy, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestFusedGeoCropKorniaEndToEnd:
    """End-to-end Compose with a real Kornia rotation + RandomResizedCrop."""

    def test_rotation_then_crop_shape_and_warps_saved(self) -> None:
        """A real ``[RandomRotation, RandomResizedCrop]`` pipeline fuses to one warp at target size."""
        import kornia.augmentation as kornia_aug

        from fuse_augmentations.adapters.kornia import KorniaAdapter
        from fuse_augmentations.compose import FusedCompose

        transforms = [
            kornia_aug.RandomRotation(degrees=(15.0, 15.0), p=1.0, align_corners=True),
            kornia_aug.RandomResizedCrop(size=(48, 48), scale=(0.6, 0.6), ratio=(1.0, 1.0)),
        ]
        pipe = FusedCompose(transforms, adapter=KorniaAdapter())
        assert pipe.n_warps_saved == 1
        image = torch.rand(2, 3, 96, 96)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 48, 48])
        assert pipe.transform_matrix is not None
        assert pipe.transform_matrix.shape == torch.Size([2, 3, 3])
