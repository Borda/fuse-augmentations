"""Numeric-parity gates for the eager-path matrix micro-optimisations.

Two perf fixes replace a general-but-costly matrix computation with a cheaper
equivalent; each must stay numerically faithful to the form it replaces:

- **PRF-4** ``normalize_matrix_io`` was rewritten from a two-``bmm`` normalization
  sandwich (``N_in @ matrix @ N_out_inv``) to a hand-expanded closed form. The
  closed form must match the sandwich to floating-point round-off.
- **PRF-2** the fused torch warp now skips the ``classify_d4_batch`` host-sync on
  non-CPU devices, relying on a ``grid_sample`` of an exact D4 map being
  near-bit-identical to the lossless ``flip``/``rot90`` shortcut. This pins that
  premise so the skipped shortcut is a precision micro-loss, never a correctness
  regression. (The CPU shortcut itself stays guarded by ``tests/test_d4.py``.)

"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations.affine.matrix import (
    apply_d4_image,
    classify_d4_batch,
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    normalize_matrix_io,
    rotation_matrix,
    vflip_matrix,
)

DTYPE = torch.float64  # tight tolerances for the closed-form parity gate


def _sandwich_normalize_io(
    matrix: torch.Tensor, height_in: int, width_in: int, height_out: int, width_out: int
) -> torch.Tensor:
    """Reference two-bmm ``N_in @ matrix @ N_out_inv`` sandwich the closed form replaces."""
    batch_size = matrix.shape[0]
    n_in = torch.zeros(batch_size, 3, 3, device=matrix.device, dtype=matrix.dtype)
    n_in[:, 0, 0] = 2.0 / (width_in - 1)
    n_in[:, 0, 2] = -1.0
    n_in[:, 1, 1] = 2.0 / (height_in - 1)
    n_in[:, 1, 2] = -1.0
    n_in[:, 2, 2] = 1.0
    n_out_inv = torch.zeros(batch_size, 3, 3, device=matrix.device, dtype=matrix.dtype)
    n_out_inv[:, 0, 0] = (width_out - 1) / 2.0
    n_out_inv[:, 0, 2] = (width_out - 1) / 2.0
    n_out_inv[:, 1, 1] = (height_out - 1) / 2.0
    n_out_inv[:, 1, 2] = (height_out - 1) / 2.0
    n_out_inv[:, 2, 2] = 1.0
    return matmul3x3(matmul3x3(n_in, matrix), n_out_inv)


class TestNormalizeMatrixIoClosedForm:
    """PRF-4: the closed-form expansion matches the two-bmm normalization sandwich."""

    @pytest.mark.parametrize(
        ("height_in", "width_in", "height_out", "width_out"),
        [
            pytest.param(64, 64, 32, 32, id="square-downscale"),
            pytest.param(48, 96, 24, 24, id="anisotropic-in"),
            pytest.param(24, 24, 40, 72, id="anisotropic-out"),
            pytest.param(17, 31, 9, 5, id="odd-sizes"),
        ],
    )
    def test_matches_bmm_sandwich(self, height_in: int, width_in: int, height_out: int, width_out: int) -> None:
        """Closed-form ``normalize_matrix_io`` equals the reference sandwich to round-off."""
        matrix = torch.randn(4, 3, 3, dtype=DTYPE)
        matrix[:, 2, :2] = 0.0  # keep an affine (non-projective) bottom row
        matrix[:, 2, 2] = 1.0
        closed = normalize_matrix_io(matrix, height_in, width_in, height_out, width_out)
        reference = _sandwich_normalize_io(matrix, height_in, width_in, height_out, width_out)
        assert torch.allclose(closed, reference, atol=1e-10, rtol=0.0)

    def test_equal_sizes_match_normalize_matrix(self) -> None:
        """When input and output sizes coincide, the io form equals plain ``normalize_matrix``."""
        matrix = torch.randn(3, 3, 3, dtype=DTYPE)
        matrix[:, 2, :2] = 0.0
        matrix[:, 2, 2] = 1.0
        io_form = normalize_matrix_io(matrix, 40, 40, 40, 40)
        plain = normalize_matrix(matrix, 40, 40)
        assert torch.allclose(io_form, plain, atol=1e-10, rtol=0.0)


def _grid_sample_d4(image: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    """Warp ``image`` by a D4 forward matrix through the affine_grid/grid_sample path."""
    batch_size, channels, height, width = image.shape
    inv = inv3x3(forward)
    norm = normalize_matrix(inv, height, width)
    grid = F.affine_grid(norm[:, :2, :], [batch_size, channels, height, width], align_corners=True)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _hflip_forward(side: int) -> torch.Tensor:
    """Build a batched hflip forward matrix for a ``side x side`` image."""
    return hflip_matrix(width=side, batch_size=1, device=torch.device("cpu"), dtype=DTYPE)


def _vflip_forward(side: int) -> torch.Tensor:
    """Build a batched vflip forward matrix for a ``side x side`` image."""
    return vflip_matrix(height=side, batch_size=1, device=torch.device("cpu"), dtype=DTYPE)


def _rot90_forward(side: int) -> torch.Tensor:
    """Build a batched 90-degree rotation forward matrix for a ``side x side`` image."""
    return rotation_matrix(torch.tensor([torch.pi / 2], dtype=DTYPE), height=side, width=side)


class TestD4GridSampleParity:
    """PRF-2: grid_sample of an exact D4 map ~= the lossless flip/rot90 shortcut."""

    @pytest.mark.parametrize(
        "forward_fn",
        [
            pytest.param(_hflip_forward, id="hflip"),
            pytest.param(_vflip_forward, id="vflip"),
            pytest.param(_rot90_forward, id="rot90"),
        ],
    )
    def test_grid_sample_matches_lossless_shortcut(self, forward_fn: Callable[[int], torch.Tensor]) -> None:
        """The grid warp of a D4 matrix equals its lossless apply within interpolation round-off."""
        side = 16
        image = torch.rand(1, 3, side, side, dtype=DTYPE)
        forward = forward_fn(side)
        name = classify_d4_batch(forward, side, side)
        assert name is not None  # the constructed matrix is a genuine D4 element
        lossless = apply_d4_image(image, name)
        warped = _grid_sample_d4(image, forward)
        assert torch.allclose(warped, lossless, atol=1e-5, rtol=0.0)
