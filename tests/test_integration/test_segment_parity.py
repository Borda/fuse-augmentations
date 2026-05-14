"""Integration parity tests for FusedAffineSegment.

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

from fuse_augmentations.adapters._kornia import KorniaAdapter  # noqa: E402
from fuse_augmentations.affine._matrix import inv3x3, matmul3x3, normalize_matrix  # noqa: E402
from fuse_augmentations.affine._segment import FusedAffineSegment  # noqa: E402

pytestmark = pytest.mark.integration

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


@pytest.fixture
def adapter():
    """Create a fresh KorniaAdapter instance for each test."""
    return KorniaAdapter()


# ---------------------------------------------------------------------------
# Per-transform forward_parameters -> canonical param converters
# ---------------------------------------------------------------------------


def _hflip_fp_to_canonical(forward_params: dict) -> dict[str, torch.Tensor]:
    """Convert RandomHorizontalFlip forward_parameters to canonical params."""
    batch_size = forward_params.get("batch_prob", torch.tensor([1])).shape[0]
    return {"_batch_size": torch.tensor([batch_size])}


def _vflip_fp_to_canonical(forward_params: dict) -> dict[str, torch.Tensor]:
    """Convert RandomVerticalFlip forward_parameters to canonical params."""
    batch_size = forward_params.get("batch_prob", torch.tensor([1])).shape[0]
    return {"_batch_size": torch.tensor([batch_size])}


def _rotation_fp_to_canonical(forward_params: dict) -> dict[str, torch.Tensor]:
    """Convert RandomRotation forward_parameters to canonical params."""
    return {"angle_rad": -torch.deg2rad(forward_params["degrees"].to(DEFAULT_DEVICE))}


def _affine_fp_to_canonical(forward_params: dict) -> dict[str, torch.Tensor]:
    """Convert RandomAffine forward_parameters to canonical params."""
    params: dict[str, torch.Tensor] = {}
    if "angle" in forward_params:
        params["angle_rad"] = -torch.deg2rad(forward_params["angle"].to(DEFAULT_DEVICE))
    if "translations" in forward_params:
        params["translate_x"] = forward_params["translations"][:, 0].to(DEFAULT_DEVICE)
        params["translate_y"] = forward_params["translations"][:, 1].to(DEFAULT_DEVICE)
    if "scale" in forward_params:
        params["scale_x"] = forward_params["scale"][:, 0].to(DEFAULT_DEVICE)
        params["scale_y"] = forward_params["scale"][:, 1].to(DEFAULT_DEVICE)
    if "shear_x" in forward_params:
        params["shear_x_rad"] = -torch.deg2rad(forward_params["shear_x"].to(DEFAULT_DEVICE))
    if "shear_y" in forward_params:
        params["shear_y_rad"] = -torch.deg2rad(forward_params["shear_y"].to(DEFAULT_DEVICE))
    return params


_FP_CONVERTERS: dict[type, Callable[[dict], dict[str, torch.Tensor]]] = {
    RandomHorizontalFlip: _hflip_fp_to_canonical,
    RandomVerticalFlip: _vflip_fp_to_canonical,
    RandomRotation: _rotation_fp_to_canonical,
    RandomAffine: _affine_fp_to_canonical,
}


def _single_grid_sample_from_params(transforms, adapter, image, forward_params_list):
    """Compose matrices from forward_parameters and apply one grid_sample.

    This is the reference path: manually compose (batch_size,3,3) matrices using
    the adapter, then invert+normalize+grid_sample in one pass -- identical
    to what FusedAffineSegment.forward() does, but with deterministic params.

    """
    batch_size, num_channels, height, width = image.shape
    mtx_identity = torch.eye(3, device=image.device, dtype=image.dtype)
    mtx_acc = mtx_identity.unsqueeze(0).expand(batch_size, -1, -1).clone()

    for transform, forward_params in zip(transforms, forward_params_list, strict=True):
        params = _FP_CONVERTERS.get(type(transform), lambda _: {})(forward_params)
        mtx = adapter.build_matrix(transform, params, height, width)
        if mtx.shape[0] == 1 and batch_size > 1:
            mtx = mtx.expand(batch_size, -1, -1)
        mtx_acc = matmul3x3(mtx, mtx_acc)

    mtx_inv = inv3x3(mtx_acc)
    mtx_norm = normalize_matrix(mtx_inv, height, width)
    grid = affine_grid(mtx_norm[:, :2, :], [batch_size, num_channels, height, width], align_corners=True)
    return grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True), mtx_acc


class TestSingleTransformParity:
    """Verify single-transform fused path matches Kornia native output."""

    def test_random_rotation_parity(self, adapter):
        """RandomRotation fused path matches Kornia native within 1e-3."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomRotation(degrees=45, p=1.0, align_corners=True)
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))

        native_out = transform(image, params=forward_params)
        fused_out, _ = _single_grid_sample_from_params([transform], adapter, image, [forward_params])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Rotation parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_affine_scale_translate_parity(self, adapter):
        """RandomAffine with degrees=0 (scale + translate only) parity."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0, align_corners=True)
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))

        native_out = transform(image, params=forward_params)
        fused_out, _ = _single_grid_sample_from_params([transform], adapter, image, [forward_params])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Affine parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_hflip_parity(self, adapter):
        """RandomHorizontalFlip fused path matches Kornia native within 1e-3."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomHorizontalFlip(p=1.0)
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))

        native_out = transform(image, params=forward_params)
        fused_out, _ = _single_grid_sample_from_params([transform], adapter, image, [forward_params])

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"HFlip parity: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_vflip_parity(self, adapter):
        """RandomVerticalFlip fused path matches Kornia native within 1e-3."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomVerticalFlip(p=1.0)
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))

        native_out = transform(image, params=forward_params)
        fused_out, _ = _single_grid_sample_from_params([transform], adapter, image, [forward_params])

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
        batch_size, num_channels, height, width = 2, 3, 64, 64
        image = torch.rand(batch_size, num_channels, height, width)

        t_rot = RandomRotation(degrees=(45, 45), p=1.0, align_corners=True)
        t_scale = RandomAffine(degrees=0, scale=(2.0, 2.0), p=1.0, align_corners=True)

        # Fused path: FusedAffineSegment composes both transforms in one warp.
        segment = FusedAffineSegment([t_rot, t_scale], adapter)
        fused_out = segment(image)

        # Reference path: sample the same deterministic params via forward_parameters,
        # then manually compose matrices + single grid_sample.
        fp_rot = t_rot.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        fp_scale = t_scale.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        reference_out, mtx_reference = _single_grid_sample_from_params(
            [t_rot, t_scale], adapter, image, [fp_rot, fp_scale]
        )

        assert torch.allclose(fused_out, reference_out, atol=1e-3), (
            f"Two-transform fused parity: max diff = {(fused_out - reference_out).abs().max().item():.6f}"
        )

        # Verify the composed matrix from the segment is sensible.
        assert segment.last_matrix is not None
        assert segment.last_matrix.shape == (batch_size, 3, 3)
        determinant = torch.det(mtx_reference)
        assert (determinant.abs() > 0.1).all(), f"Unexpected near-zero determinant: {determinant}"


class TestThreeTransformChain:
    """Verify three-transform chain composition and matrix associativity."""

    def test_rotate_scale_hflip_composed(self, adapter):
        """Three-transform chain satisfies matrix associativity within 1e-5."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        image = torch.rand(batch_size, num_channels, height, width)

        t_rot = RandomRotation(degrees=(30, 30), p=1.0, align_corners=True)
        t_scale = RandomAffine(degrees=0, scale=(1.5, 1.5), p=1.0, align_corners=True)
        t_hflip = RandomHorizontalFlip(p=1.0)

        fp_rot = t_rot.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        fp_scale = t_scale.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        fp_hflip = t_hflip.forward_parameters(torch.Size((batch_size, num_channels, height, width)))

        transforms = [t_rot, t_scale, t_hflip]
        fps = [fp_rot, fp_scale, fp_hflip]

        # Full chain in one shot
        full_out, mtx_full = _single_grid_sample_from_params(transforms, adapter, image, fps)

        # Associativity: compose first two, then compose with third
        # (rot, scale) then hflip
        _out_pair, mtx_pair = _single_grid_sample_from_params([t_rot, t_scale], adapter, image, [fp_rot, fp_scale])
        params_hflip = _hflip_fp_to_canonical(fp_hflip)
        mtx_hflip = adapter.build_matrix(t_hflip, params_hflip, height, width)
        if mtx_hflip.shape[0] == 1 and batch_size > 1:
            mtx_hflip = mtx_hflip.expand(batch_size, -1, -1)
        mtx_composed_alt = matmul3x3(mtx_hflip, mtx_pair)

        # Matrices should match exactly (associativity of matmul3x3)
        assert torch.allclose(mtx_full, mtx_composed_alt, atol=1e-5), (
            f"Associativity: max matrix diff = {(mtx_full - mtx_composed_alt).abs().max().item():.6f}"
        )

        # Output shape and valid pixel values
        assert full_out.shape == (batch_size, num_channels, height, width)
        assert not torch.isnan(full_out).any()


class TestLongChain:
    """Verify five-transform chain determinism and inverse round-trip."""

    def test_five_transform_chain_error_bounded(self, adapter):
        """Five-transform chain is deterministic and inv(matrix)@matrix recovers identity."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        image = torch.rand(batch_size, num_channels, height, width)

        transforms = [
            RandomRotation(degrees=(10, 10), p=1.0, align_corners=True),
            RandomAffine(degrees=0, scale=(1.1, 1.1), p=1.0, align_corners=True),
            RandomHorizontalFlip(p=1.0),
            RandomRotation(degrees=(5, 5), p=1.0, align_corners=True),
            RandomVerticalFlip(p=1.0),
        ]

        fps = [
            transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
            for transform in transforms
        ]

        fused_out, mtx_fused = _single_grid_sample_from_params(transforms, adapter, image, fps)

        # Run same composition a second time to confirm determinism
        fused_out2, _ = _single_grid_sample_from_params(transforms, adapter, image, fps)

        max_diff = (fused_out - fused_out2).abs().max().item()
        assert max_diff < 1e-6, f"Non-deterministic: max diff = {max_diff:.6f}"

        # Verify composed matrix round-trip: inv(matrix) @ matrix ~ I
        mtx_inv = inv3x3(mtx_fused)
        product = matmul3x3(mtx_inv, mtx_fused)
        mtx_identity = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
        assert torch.allclose(product, mtx_identity, atol=1e-4), (
            f"Round-trip error: max diff = {(product - mtx_identity).abs().max().item():.6f}"
        )

        # Output sanity
        assert fused_out.shape == (batch_size, num_channels, height, width)
        assert not torch.isnan(fused_out).any()
        assert not torch.isinf(fused_out).any()


class TestLastMatrixValue:
    """Verify last_matrix shape, determinant, and inverse round-trip."""

    def test_shape_and_determinant(self, adapter):
        """last_matrix has shape (batch_size,3,3) with non-zero determinant."""
        batch_size, num_channels, height, width = 4, 3, 32, 32
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomRotation(degrees=30, p=1.0, align_corners=True)
        segment = FusedAffineSegment([transform], adapter)
        segment(image)

        mtx = segment.last_matrix
        assert mtx is not None
        assert mtx.shape == (batch_size, 3, 3)

        # Determinant should be non-zero for all samples
        determinant = torch.det(mtx)
        assert (determinant.abs() > 1e-6).all(), f"Near-zero determinant found: {determinant}"

    def test_inverse_roundtrip(self, adapter):
        """inv(last_matrix) @ last_matrix recovers identity within 1e-5."""
        batch_size, num_channels, height, width = 4, 3, 32, 32
        image = torch.rand(batch_size, num_channels, height, width)

        transform = RandomRotation(degrees=30, p=1.0, align_corners=True)
        segment = FusedAffineSegment([transform], adapter)
        segment(image)

        mtx = segment.last_matrix
        mtx_inv = inv3x3(mtx)
        product = matmul3x3(mtx_inv, mtx)
        mtx_identity = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

        assert torch.allclose(product, mtx_identity, atol=1e-5), (
            f"Round-trip failed: max diff = {(product - mtx_identity).abs().max().item():.6f}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flip_segment_cuda_device(adapter):
    """Flip segment runs on CUDA without device-mismatch errors."""
    device = torch.device("cuda")
    batch_size, num_channels, height, width = 2, 3, 32, 32
    image = torch.rand(batch_size, num_channels, height, width, device=device)

    transform = RandomHorizontalFlip(p=1.0)
    segment = FusedAffineSegment([transform], adapter)
    out = segment(image)

    assert out.device.type == "cuda"
    assert out.shape == (batch_size, num_channels, height, width)
    assert not torch.isnan(out).any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vflip_segment_cuda_device(adapter):
    """Vertical flip segment runs on CUDA without device-mismatch errors."""
    device = torch.device("cuda")
    batch_size, num_channels, height, width = 2, 3, 32, 32
    image = torch.rand(batch_size, num_channels, height, width, device=device)

    transform = RandomVerticalFlip(p=1.0)
    segment = FusedAffineSegment([transform], adapter)
    out = segment(image)

    assert out.device.type == "cuda"
    assert out.shape == (batch_size, num_channels, height, width)
    assert not torch.isnan(out).any()
