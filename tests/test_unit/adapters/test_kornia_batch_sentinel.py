"""Regression tests: kornia flip/rot90 ``build_matrix`` derives batch from shape, not ``.item()`` (PRF-8).

Flip / rot90 / saturation / normalize transforms carry the batch size in a ``_batch_size`` sentinel.
The sentinel now encodes the count in its *leading dimension* so ``build_matrix`` reads it via
``.shape[0]`` (a metadata read) instead of ``.item()`` (a GPU->host synchronization inside fused GPU
chains). Matrices are numerically identical; these tests pin the batch shape and the flip values.
"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.adapters.kornia import KorniaAdapter, _batch_size_sentinel


class TestBatchSizeSentinel:
    """The sentinel encodes the batch size in its leading dimension (no host sync on read)."""

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_sentinel_shape_encodes_batch_size(self, batch_size):
        """``_batch_size_sentinel(n)`` returns a tensor whose ``shape[0] == n``."""
        sentinel = _batch_size_sentinel(batch_size, torch.device("cpu"))
        assert sentinel.shape[0] == batch_size
        assert sentinel.device.type == "cpu"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia not installed")
class TestKorniaFlipBatchFromShape:
    """Flip/rot90 ``build_matrix`` produces a ``(B, 3, 3)`` matrix for the sampled batch size."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_hflip_build_matrix_batch(self, batch_size):
        """HFlip ``build_matrix`` returns a ``(batch_size, 3, 3)`` matrix derived from the sentinel shape."""
        from kornia.augmentation import RandomHorizontalFlip

        tfm = RandomHorizontalFlip(p=1.0)
        params = KorniaAdapter.sample_params(tfm, (batch_size, 3, 16, 16), torch.device("cpu"))
        mtx = KorniaAdapter.build_matrix(tfm, params, height=16, width=16)
        assert mtx.shape == (batch_size, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_vflip_build_matrix_batch(self, batch_size):
        """VFlip ``build_matrix`` returns a ``(batch_size, 3, 3)`` matrix derived from the sentinel shape."""
        from kornia.augmentation import RandomVerticalFlip

        tfm = RandomVerticalFlip(p=1.0)
        params = KorniaAdapter.sample_params(tfm, (batch_size, 3, 16, 16), torch.device("cpu"))
        mtx = KorniaAdapter.build_matrix(tfm, params, height=16, width=16)
        assert mtx.shape == (batch_size, 3, 3)

    def test_hflip_matrix_numerically_identical(self):
        """The batch-derived hflip matrix equals the canonical width-flip matrix (numerically identical)."""
        from kornia.augmentation import RandomHorizontalFlip

        width = 16
        tfm = RandomHorizontalFlip(p=1.0)
        params = KorniaAdapter.sample_params(tfm, (3, 3, 16, width), torch.device("cpu"))
        mtx = KorniaAdapter.build_matrix(tfm, params, height=16, width=width)
        expected = torch.tensor([[-1.0, 0.0, float(width - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(mtx[0], expected)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_rot90_build_matrix_batch(self, batch_size):
        """``RandomRotation90`` ``build_matrix`` returns a ``(batch_size, 3, 3)`` matrix."""
        from kornia.augmentation import RandomRotation90

        tfm = RandomRotation90(times=(0, 3), p=1.0)
        params = KorniaAdapter.sample_params(tfm, (batch_size, 3, 16, 16), torch.device("cpu"))
        mtx = KorniaAdapter.build_matrix(tfm, params, height=16, width=16)
        assert mtx.shape == (batch_size, 3, 3)
