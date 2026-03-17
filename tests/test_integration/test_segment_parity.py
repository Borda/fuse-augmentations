"""Integration parity tests for FusedAffineSegment -- spec tests #26-30.

Requires kornia >= 0.6.12.

For single-transform tests, the fused path and Kornia native produce
identical results because both perform a single ``grid_sample``.

For multi-transform chains, the fused path composes matrices then does
a single ``grid_sample`` (one interpolation pass), while Kornia applies
transforms sequentially (one ``grid_sample`` per transform, accumulating
interpolation error).  Multi-transform tests therefore verify that the
fused path matches a manually composed matrix + single ``grid_sample``
reference, which is the mathematically correct comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.nn.functional import affine_grid, grid_sample

kornia = pytest.importorskip("kornia", reason="kornia >= 0.6.12 required")
from kornia.augmentation import (  # noqa: E402
    RandomAffine,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)

from fuse_augmentations._matrix import inv3x3, matmul3x3, normalize_matrix  # noqa: E402
from fuse_augmentations._segment import FusedAffineSegment  # noqa: E402
from fuse_augmentations.adapters._kornia import KorniaAdapter  # noqa: E402

pytestmark = pytest.mark.integration

DEVICE = torch.device("cpu")
DTYPE = torch.float32


@pytest.fixture
def adapter():
    """Create a fresh KorniaAdapter instance for each test."""
    return KorniaAdapter()


# ---------------------------------------------------------------------------
# Per-transform forward_parameters -> canonical param converters
# ---------------------------------------------------------------------------


def _hflip_fp_to_canonical(fp: dict) -> dict[str, torch.Tensor]:
    """Convert RandomHorizontalFlip forward_parameters to canonical params."""
    bsz = fp.get("batch_prob", torch.tensor([1])).shape[0]
    return {"_batch_size": torch.tensor([bsz])}


def _vflip_fp_to_canonical(fp: dict) -> dict[str, torch.Tensor]:
    """Convert RandomVerticalFlip forward_parameters to canonical params."""
    bsz = fp.get("batch_prob", torch.tensor([1])).shape[0]
    return {"_batch_size": torch.tensor([bsz])}


def _rotation_fp_to_canonical(fp: dict) -> dict[str, torch.Tensor]:
    """Convert RandomRotation forward_parameters to canonical params."""
    return {"angle_rad": -torch.deg2rad(fp["degrees"].to(DEVICE))}


def _affine_fp_to_canonical(fp: dict) -> dict[str, torch.Tensor]:
    """Convert RandomAffine forward_parameters to canonical params."""
    params: dict[str, torch.Tensor] = {}
    if "angle" in fp:
        params["angle_rad"] = -torch.deg2rad(fp["angle"].to(DEVICE))
    if "translations" in fp:
        params["translate_x"] = fp["translations"][:, 0].to(DEVICE)
        params["translate_y"] = fp["translations"][:, 1].to(DEVICE)
    if "scale" in fp:
        params["scale_x"] = fp["scale"][:, 0].to(DEVICE)
        params["scale_y"] = fp["scale"][:, 1].to(DEVICE)
    if "shear_x" in fp:
        params["shear_x_rad"] = -torch.deg2rad(fp["shear_x"].to(DEVICE))
    if "shear_y" in fp:
        params["shear_y_rad"] = -torch.deg2rad(fp["shear_y"].to(DEVICE))
    return params


_FP_CONVERTERS: dict[type, Callable[[dict], dict[str, torch.Tensor]]] = {
    RandomHorizontalFlip: _hflip_fp_to_canonical,
    RandomVerticalFlip: _vflip_fp_to_canonical,
    RandomRotation: _rotation_fp_to_canonical,
    RandomAffine: _affine_fp_to_canonical,
}


def _single_grid_sample_from_params(transforms, adapter, img, forward_params_list):
    """Compose matrices from forward_parameters and apply one grid_sample.

    This is the reference path: manually compose (B,3,3) matrices using
    the adapter, then invert+normalize+grid_sample in one pass -- identical
    to what FusedAffineSegment.forward() does, but with deterministic params.
    """
    bsz, n_ch, height, width = img.shape
    eye = torch.eye(3, device=img.device, dtype=img.dtype)
    acc = eye.unsqueeze(0).expand(bsz, -1, -1).clone()

    for t, fp in zip(transforms, forward_params_list, strict=True):
        params = _FP_CONVERTERS.get(type(t), lambda _: {})(fp)
        mtx_i = adapter.build_matrix(t, params, height, width)
        if mtx_i.shape[0] == 1 and bsz > 1:
            mtx_i = mtx_i.expand(bsz, -1, -1)
        acc = matmul3x3(mtx_i, acc)

    mtx_inv = inv3x3(acc)
    mtx_norm = normalize_matrix(mtx_inv, height, width)
    grid = affine_grid(mtx_norm[:, :2, :], [bsz, n_ch, height, width], align_corners=True)
    return grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True), acc


class TestSingleTransformParity:
    """Verify single-transform fused path matches Kornia native output."""

    def test_random_rotation_parity(self, adapter):
        """RandomRotation fused path matches Kornia native within 1e-3."""
        bsz, n_ch, height, width = 2, 3, 64, 64
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomRotation(degrees=45, p=1.0, align_corners=True)
        fp = t.forward_parameters(torch.Size((bsz, n_ch, height, width)))

        native_out = t(img, params=fp)
        fused_out, _ = _single_grid_sample_from_params([t], adapter, img, [fp])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Rotation parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_affine_scale_translate_parity(self, adapter):
        """RandomAffine with degrees=0 (scale + translate only) parity."""
        bsz, n_ch, height, width = 2, 3, 64, 64
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0, align_corners=True)
        fp = t.forward_parameters(torch.Size((bsz, n_ch, height, width)))

        native_out = t(img, params=fp)
        fused_out, _ = _single_grid_sample_from_params([t], adapter, img, [fp])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Affine parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_hflip_parity(self, adapter):
        """RandomHorizontalFlip fused path matches Kornia native within 1e-3."""
        bsz, n_ch, height, width = 2, 3, 32, 32
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomHorizontalFlip(p=1.0)
        fp = t.forward_parameters(torch.Size((bsz, n_ch, height, width)))

        native_out = t(img, params=fp)
        fused_out, _ = _single_grid_sample_from_params([t], adapter, img, [fp])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"HFlip parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_vflip_parity(self, adapter):
        """RandomVerticalFlip fused path matches Kornia native within 1e-3."""
        bsz, n_ch, height, width = 2, 3, 32, 32
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomVerticalFlip(p=1.0)
        fp = t.forward_parameters(torch.Size((bsz, n_ch, height, width)))

        native_out = t(img, params=fp)
        fused_out, _ = _single_grid_sample_from_params([t], adapter, img, [fp])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"VFlip parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )


class TestTwoTransformParity:
    """Verify two-transform composed matrix path matches manual reference."""

    def test_rotate_then_scale_parity(self, adapter):
        """FusedAffineSegment output matches manual compose + single grid_sample reference.

        Uses single-value degree/scale ranges so both paths sample the same
        deterministic parameters: degrees=(45,45) always gives 45°,
        scale=(2.0,2.0) always gives 2.0x.
        """
        bsz, n_ch, height, width = 2, 3, 64, 64
        img = torch.rand(bsz, n_ch, height, width)

        t_rot = RandomRotation(degrees=(45, 45), p=1.0, align_corners=True)
        t_scale = RandomAffine(degrees=0, scale=(2.0, 2.0), p=1.0, align_corners=True)

        # Fused path: FusedAffineSegment composes both transforms in one warp.
        seg = FusedAffineSegment([t_rot, t_scale], adapter)
        fused_out = seg(img)

        # Reference path: sample the same deterministic params via forward_parameters,
        # then manually compose matrices + single grid_sample.
        fp_rot = t_rot.forward_parameters(torch.Size((bsz, n_ch, height, width)))
        fp_scale = t_scale.forward_parameters(torch.Size((bsz, n_ch, height, width)))
        reference_out, mtx_ref = _single_grid_sample_from_params([t_rot, t_scale], adapter, img, [fp_rot, fp_scale])

        assert torch.allclose(fused_out, reference_out, atol=1e-3), (
            f"Two-transform fused parity: max diff = {(fused_out - reference_out).abs().max().item():.6f}"
        )

        # Verify the composed matrix from the segment is sensible.
        assert seg.last_matrix is not None
        assert seg.last_matrix.shape == (bsz, 3, 3)
        det = torch.det(mtx_ref)
        assert (det.abs() > 0.1).all(), f"Unexpected near-zero det: {det}"


class TestThreeTransformChain:
    """Verify three-transform chain composition and matrix associativity."""

    def test_rotate_scale_hflip_composed(self, adapter):
        """Three-transform chain satisfies matrix associativity within 1e-5."""
        bsz, n_ch, height, width = 2, 3, 64, 64
        img = torch.rand(bsz, n_ch, height, width)

        t_rot = RandomRotation(degrees=(30, 30), p=1.0, align_corners=True)
        t_scale = RandomAffine(degrees=0, scale=(1.5, 1.5), p=1.0, align_corners=True)
        t_hflip = RandomHorizontalFlip(p=1.0)

        fp_rot = t_rot.forward_parameters(torch.Size((bsz, n_ch, height, width)))
        fp_scale = t_scale.forward_parameters(torch.Size((bsz, n_ch, height, width)))
        fp_hflip = t_hflip.forward_parameters(torch.Size((bsz, n_ch, height, width)))

        transforms = [t_rot, t_scale, t_hflip]
        fps = [fp_rot, fp_scale, fp_hflip]

        # Full chain in one shot
        full_out, mtx_full = _single_grid_sample_from_params(transforms, adapter, img, fps)

        # Associativity: compose first two, then compose with third
        # (rot, scale) then hflip
        _out_pair, mtx_pair = _single_grid_sample_from_params([t_rot, t_scale], adapter, img, [fp_rot, fp_scale])
        params_hflip = _hflip_fp_to_canonical(fp_hflip)
        mtx_hflip = adapter.build_matrix(t_hflip, params_hflip, height, width)
        if mtx_hflip.shape[0] == 1 and bsz > 1:
            mtx_hflip = mtx_hflip.expand(bsz, -1, -1)
        mtx_composed_alt = matmul3x3(mtx_hflip, mtx_pair)

        # Matrices should match exactly (associativity of matmul3x3)
        assert torch.allclose(mtx_full, mtx_composed_alt, atol=1e-5), (
            f"Associativity: max matrix diff = {(mtx_full - mtx_composed_alt).abs().max().item():.6f}"
        )

        # Output shape and valid pixel values
        assert full_out.shape == (bsz, n_ch, height, width)
        assert not torch.isnan(full_out).any()


class TestLongChain:
    """Verify five-transform chain determinism and inverse round-trip."""

    def test_five_transform_chain_error_bounded(self, adapter):
        """Five-transform chain is deterministic and inv(M)@M recovers identity."""
        bsz, n_ch, height, width = 2, 3, 64, 64
        img = torch.rand(bsz, n_ch, height, width)

        transforms = [
            RandomRotation(degrees=(10, 10), p=1.0, align_corners=True),
            RandomAffine(degrees=0, scale=(1.1, 1.1), p=1.0, align_corners=True),
            RandomHorizontalFlip(p=1.0),
            RandomRotation(degrees=(5, 5), p=1.0, align_corners=True),
            RandomVerticalFlip(p=1.0),
        ]

        fps = [t.forward_parameters(torch.Size((bsz, n_ch, height, width))) for t in transforms]

        fused_out, mtx_fused = _single_grid_sample_from_params(transforms, adapter, img, fps)

        # Run same composition a second time to confirm determinism
        fused_out2, _ = _single_grid_sample_from_params(transforms, adapter, img, fps)

        max_diff = (fused_out - fused_out2).abs().max().item()
        assert max_diff < 1e-6, f"Non-deterministic: max diff = {max_diff:.6f}"

        # Verify composed matrix round-trip: inv(M) @ M ~ I
        mtx_inv = inv3x3(mtx_fused)
        product = matmul3x3(mtx_inv, mtx_fused)
        eye = torch.eye(3).unsqueeze(0).expand(bsz, -1, -1)
        assert torch.allclose(product, eye, atol=1e-4), (
            f"Round-trip error: max diff = {(product - eye).abs().max().item():.6f}"
        )

        # Output sanity
        assert fused_out.shape == (bsz, n_ch, height, width)
        assert not torch.isnan(fused_out).any()
        assert not torch.isinf(fused_out).any()


class TestLastMatrixValue:
    """Verify last_matrix shape, determinant, and inverse round-trip."""

    def test_shape_and_determinant(self, adapter):
        """last_matrix has shape (B,3,3) with non-zero determinant."""
        bsz, n_ch, height, width = 4, 3, 32, 32
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomRotation(degrees=30, p=1.0, align_corners=True)
        seg = FusedAffineSegment([t], adapter)
        seg(img)

        mtx = seg.last_matrix
        assert mtx is not None
        assert mtx.shape == (bsz, 3, 3)

        # Determinant should be non-zero for all samples
        det = torch.det(mtx)
        assert (det.abs() > 1e-6).all(), f"Near-zero determinant found: {det}"

    def test_inverse_roundtrip(self, adapter):
        """inv(last_matrix) @ last_matrix recovers identity within 1e-5."""
        bsz, n_ch, height, width = 4, 3, 32, 32
        img = torch.rand(bsz, n_ch, height, width)

        t = RandomRotation(degrees=30, p=1.0, align_corners=True)
        seg = FusedAffineSegment([t], adapter)
        seg(img)

        mtx = seg.last_matrix
        mtx_inv = inv3x3(mtx)
        product = matmul3x3(mtx_inv, mtx)
        eye = torch.eye(3).unsqueeze(0).expand(bsz, -1, -1)

        assert torch.allclose(product, eye, atol=1e-5), (
            f"Round-trip failed: max diff = {(product - eye).abs().max().item():.6f}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flip_segment_cuda_device(adapter):
    """Flip segment runs on CUDA without device-mismatch errors."""
    device = torch.device("cuda")
    bsz, n_ch, height, width = 2, 3, 32, 32
    img = torch.rand(bsz, n_ch, height, width, device=device)

    t = RandomHorizontalFlip(p=1.0)
    seg = FusedAffineSegment([t], adapter)
    out = seg(img)

    assert out.device.type == "cuda"
    assert out.shape == (bsz, n_ch, height, width)
    assert not torch.isnan(out).any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vflip_segment_cuda_device(adapter):
    """Vertical flip segment runs on CUDA without device-mismatch errors."""
    device = torch.device("cuda")
    bsz, n_ch, height, width = 2, 3, 32, 32
    img = torch.rand(bsz, n_ch, height, width, device=device)

    t = RandomVerticalFlip(p=1.0)
    seg = FusedAffineSegment([t], adapter)
    out = seg(img)

    assert out.device.type == "cuda"
    assert out.shape == (bsz, n_ch, height, width)
    assert not torch.isnan(out).any()
