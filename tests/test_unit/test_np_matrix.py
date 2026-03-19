"""Unit tests for NumPy matrix primitives in _np_matrix.py."""

from __future__ import annotations

import numpy as np
import pytest

from fuse_augmentations._np_matrix import hflip_matrix_np, vflip_matrix_np


class TestHflipMatrixNp:
    def test_shape(self):
        assert hflip_matrix_np(W=64).shape == (3, 3)

    def test_dtype(self):
        assert hflip_matrix_np(W=64).dtype == np.float64

    def test_maps_x_correctly(self):
        W = 10
        M = hflip_matrix_np(W=W)
        # x' = -x + (W-1)
        assert M[0, 0] == -1.0
        assert M[0, 1] == 0.0
        assert M[0, 2] == float(W - 1)

    def test_maps_y_unchanged(self):
        M = hflip_matrix_np(W=10)
        assert M[1, 0] == 0.0
        assert M[1, 1] == 1.0
        assert M[1, 2] == 0.0

    def test_homogeneous_row(self):
        M = hflip_matrix_np(W=10)
        np.testing.assert_array_equal(M[2], [0.0, 0.0, 1.0])

    @pytest.mark.parametrize("W", [1, 4, 64, 256])
    def test_involutory(self, W):
        """hflip applied twice should be identity."""
        M = hflip_matrix_np(W=W)
        assert np.allclose(M @ M, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("W", [4, 64])
    def test_pixel_transform(self, W):
        """Pixel (0, y, 1) maps to (W-1, y, 1)."""
        M = hflip_matrix_np(W=W)
        pt = np.array([0.0, 3.0, 1.0])
        result = M @ pt
        assert result[0] == pytest.approx(W - 1)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(1.0)

    def test_consistent_with_torch_hflip_matrix(self):
        """Matches the torch hflip_matrix from _matrix.py."""
        import torch

        from fuse_augmentations._matrix import hflip_matrix

        W = 32
        torch_M = hflip_matrix(W=W, batch_size=1, device=torch.device("cpu"), dtype=torch.float64)
        np_M = hflip_matrix_np(W=W)
        np.testing.assert_allclose(np_M, torch_M[0].numpy(), atol=1e-10)


class TestVflipMatrixNp:
    def test_shape(self):
        assert vflip_matrix_np(H=64).shape == (3, 3)

    def test_dtype(self):
        assert vflip_matrix_np(H=64).dtype == np.float64

    def test_maps_y_correctly(self):
        H = 10
        M = vflip_matrix_np(H=H)
        assert M[1, 0] == 0.0
        assert M[1, 1] == -1.0
        assert M[1, 2] == float(H - 1)

    def test_maps_x_unchanged(self):
        M = vflip_matrix_np(H=10)
        assert M[0, 0] == 1.0
        assert M[0, 1] == 0.0
        assert M[0, 2] == 0.0

    def test_homogeneous_row(self):
        M = vflip_matrix_np(H=10)
        np.testing.assert_array_equal(M[2], [0.0, 0.0, 1.0])

    @pytest.mark.parametrize("H", [1, 4, 64, 256])
    def test_involutory(self, H):
        """vflip applied twice should be identity."""
        M = vflip_matrix_np(H=H)
        assert np.allclose(M @ M, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("H", [4, 64])
    def test_pixel_transform(self, H):
        """Pixel (x, 0, 1) maps to (x, H-1, 1)."""
        M = vflip_matrix_np(H=H)
        pt = np.array([5.0, 0.0, 1.0])
        result = M @ pt
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(H - 1)
        assert result[2] == pytest.approx(1.0)

    def test_consistent_with_torch_vflip_matrix(self):
        """Matches the torch vflip_matrix from _matrix.py."""
        import torch

        from fuse_augmentations._matrix import vflip_matrix

        H = 32
        torch_M = vflip_matrix(H=H, batch_size=1, device=torch.device("cpu"), dtype=torch.float64)
        np_M = vflip_matrix_np(H=H)
        np.testing.assert_allclose(np_M, torch_M[0].numpy(), atol=1e-10)
