"""Numeric-parity gate for the fused color matmul micro-optimisation (PRF-3).

``FusedColorSegment._matmul_image`` applies a composed ``(B, 4, 4)`` homogeneous
color matrix to a ``(B, 3, H, W)`` image. It was rewritten from an augmented
homogeneous matmul (``cat([pixels, ones])`` + ``bmm`` of the full 4x4) to a single
``baddbmm`` on the 3x3 linear block plus the translation column. The two forms are
mathematically identical up to accumulation order, so this pins the new form to the
homogeneous reference within float round-off.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations.affine.segment import FusedColorSegment


def _homogeneous_matmul_image(image: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    """Reference augmented-homogeneous form the ``baddbmm`` path replaces."""
    batch_size, channels, height, width = image.shape
    pixels = image.reshape(batch_size, channels, height * width)
    ones = torch.ones(batch_size, 1, height * width, device=image.device, dtype=image.dtype)
    pixels_hom = torch.cat([pixels, ones], dim=1)
    transformed = torch.bmm(acc, pixels_hom)
    return transformed[:, :channels, :].reshape(batch_size, channels, height, width)


def _random_color_matrix(batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Build a random homogeneous ``(B, 4, 4)`` color matrix (affine per-channel)."""
    acc = torch.randn(batch_size, 4, 4, dtype=dtype)
    acc[:, 3, :3] = 0.0  # inert homogeneous bottom row
    acc[:, 3, 3] = 1.0
    return acc


class TestMatmulImageParity:
    """The ``baddbmm`` color apply equals the augmented-homogeneous form."""

    @pytest.mark.parametrize(
        ("dtype", "atol"),
        [
            pytest.param(torch.float32, 1e-5, id="float32"),
            pytest.param(torch.float64, 1e-12, id="float64"),
        ],
    )
    def test_matches_homogeneous_form(self, dtype: torch.dtype, atol: float) -> None:
        """New ``_matmul_image`` matches the old homogeneous matmul within round-off."""
        image = torch.rand(2, 3, 5, 7, dtype=dtype)
        acc = _random_color_matrix(2, dtype)
        new = FusedColorSegment._matmul_image(image, acc)
        reference = _homogeneous_matmul_image(image, acc)
        assert torch.allclose(new, reference, atol=atol, rtol=0.0)

    def test_matches_explicit_affine_definition(self) -> None:
        """Output equals the per-channel affine ``c' = A c + b`` computed directly."""
        image = torch.rand(1, 3, 4, 4, dtype=torch.float64)
        acc = _random_color_matrix(1, torch.float64)
        linear = acc[:, :3, :3]
        bias = acc[:, :3, 3:4]
        pixels = image.reshape(1, 3, 16)
        expected = (linear @ pixels + bias).reshape(1, 3, 4, 4)
        assert torch.allclose(FusedColorSegment._matmul_image(image, acc), expected, atol=1e-12, rtol=0.0)
