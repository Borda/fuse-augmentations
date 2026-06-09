"""Albumentations native I/O parity tests.

Verifies that FusedCompose accepts the same dict-input calling convention as albu.Compose (``pipeline(image=ndarray)``),
returning a dict with an ``"image"`` key containing a HWC NumPy array — while leaving the existing BCHW tensor path
completely unchanged.

Parity tests (marked below) MUST FAIL before the implementation and PASS after. Backward-compat tests MUST PASS both
before and after the implementation.

"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbuDictInput:
    """Compose accepts the albu-style ``pipeline(image=ndarray)`` dict-input convention."""

    def test_accepts_hwc_uint8(self):
        """FusedCompose must accept uint8 HWC NumPy dict like albu.Compose."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        transforms = [albu.Rotate(limit=30, p=1.0), albu.HorizontalFlip(p=1.0)]
        fused = Compose(copy.deepcopy(transforms))
        out = fused(image=img)  # must not raise TypeError
        assert isinstance(out, dict), "output must be a dict"
        assert "image" in out
        assert isinstance(out["image"], np.ndarray), "output image must be HWC NumPy"
        assert out["image"].shape == img.shape

    def test_accepts_float32(self):
        """FusedCompose must accept float32 HWC NumPy dict."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        transforms = [albu.Rotate(limit=30, p=1.0)]
        fused = Compose(copy.deepcopy(transforms))
        out = fused(image=img)
        assert isinstance(out["image"], np.ndarray)
        assert out["image"].dtype == np.float32

    def test_tensor_path_unchanged(self):
        """Existing BCHW tensor input must still return BCHW tensor after the fix."""
        transforms = [albu.Rotate(limit=30, p=1.0), albu.HorizontalFlip(p=1.0)]
        fused = Compose(copy.deepcopy(transforms))
        tensor = torch.rand(2, 3, 64, 64)
        out = fused(tensor)
        assert isinstance(out, torch.Tensor), "tensor input must return tensor"
        assert out.shape == tensor.shape

    def test_transform_matrix_populated_with_sampled_rotation(self):
        """Dict-input single-transform fast path records the sampled matrix."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pipe = Compose([albu.Rotate(limit=(30, 30), p=1.0)])
        pipe(image=img)
        matrix = pipe.transform_matrix
        assert matrix is not None
        assert matrix.shape == (1, 3, 3)
        identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device).unsqueeze(0)
        assert not torch.allclose(matrix, identity, rtol=1e-4, atol=1e-6)

    def test_empty_pipeline_passthrough(self):
        """Compose([]) must return input dict unchanged."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        pipe = Compose([])
        out = pipe(image=img)
        np.testing.assert_array_equal(out["image"], img)

    def test_grayscale_hwc1(self):
        """(height, width, 1) single-channel input must be handled."""
        img = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        pipe = Compose([albu.Rotate(limit=30, p=1.0)])
        out = pipe(image=img)
        assert out["image"].shape == img.shape

    def test_fused_plus_passthrough(self):
        """Mixed fused-geometric + passthrough colour op via dict input preserves shape and type.

        Verifies that the dict-input path correctly dispatches across heterogeneous segments:
        albu.Rotate goes through AlbuFusedAffineSegment while albu.GaussianBlur is routed via
        _PassthroughSegment with AlbumentationsAdapter. Both must agree on the HWC NumPy
        round-trip without intermediate tensor conversion errors.

        """
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # albu.Rotate is fused (AlbuFusedAffineSegment); albu.GaussianBlur is a passthrough
        # (_PassthroughSegment with AlbumentationsAdapter).
        pipe = Compose([albu.Rotate(limit=30, p=1.0), albu.GaussianBlur(p=1.0)])
        out = pipe(image=img)
        assert isinstance(out, dict)
        assert isinstance(out["image"], np.ndarray)
        assert out["image"].shape == img.shape
