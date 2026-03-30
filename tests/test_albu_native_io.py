"""Albumentations native I/O parity tests.

Verifies that FusedCompose accepts the same dict-input calling convention as
A.Compose (``pipeline(image=ndarray)``), returning a dict with an ``"image"``
key containing a HWC NumPy array — while leaving the existing BCHW tensor path
completely unchanged.

Parity tests (marked below) MUST FAIL before the implementation and PASS after.
Backward-compat tests MUST PASS both before and after the implementation.

"""

from __future__ import annotations

import copy

import albumentations as A
import numpy as np
import torch

from fuse_augmentations import Compose

# ---------------------------------------------------------------------------
# Parity tests — MUST FAIL before fix (TypeError), MUST PASS after
# ---------------------------------------------------------------------------


def test_fused_compose_albu_accepts_hwc_numpy_dict_uint8():
    """FusedCompose must accept uint8 HWC NumPy dict like A.Compose."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    transforms = [A.Rotate(limit=30, p=1.0), A.HorizontalFlip(p=1.0)]
    fused = Compose(copy.deepcopy(transforms))
    out = fused(image=img)  # must not raise TypeError
    assert isinstance(out, dict), "output must be a dict"
    assert "image" in out
    assert isinstance(out["image"], np.ndarray), "output image must be HWC NumPy"
    assert out["image"].shape == img.shape


def test_fused_compose_albu_accepts_float32_input():
    """FusedCompose must accept float32 HWC NumPy dict."""
    img = np.random.rand(64, 64, 3).astype(np.float32)
    transforms = [A.Rotate(limit=30, p=1.0)]
    fused = Compose(copy.deepcopy(transforms))
    out = fused(image=img)
    assert isinstance(out["image"], np.ndarray)
    assert out["image"].dtype == np.float32


# ---------------------------------------------------------------------------
# Backward-compat test — MUST PASS before AND after the fix
# ---------------------------------------------------------------------------


def test_fused_compose_albu_tensor_path_unchanged():
    """Existing BCHW tensor input must still return BCHW tensor after the fix."""
    transforms = [A.Rotate(limit=30, p=1.0), A.HorizontalFlip(p=1.0)]
    fused = Compose(copy.deepcopy(transforms))
    tensor = torch.rand(2, 3, 64, 64)
    out = fused(tensor)
    assert isinstance(out, torch.Tensor), "tensor input must return tensor"
    assert out.shape == tensor.shape


# ---------------------------------------------------------------------------
# Metadata integrity after dict-input forward pass
# ---------------------------------------------------------------------------


def test_transform_matrix_populated_after_dict_input():
    """pipe.transform_matrix must be (1,3,3) after a dict-input forward pass."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    pipe = Compose([A.Rotate(limit=30, p=1.0)])
    pipe(image=img)
    assert pipe.transform_matrix is not None
    assert pipe.transform_matrix.shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_empty_pipeline_dict_input():
    """Compose([]) must return input dict unchanged."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    pipe = Compose([])
    out = pipe(image=img)
    np.testing.assert_array_equal(out["image"], img)


def test_grayscale_hwc1_dict_input():
    """(H, W, 1) single-channel input must be handled."""
    img = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
    pipe = Compose([A.Rotate(limit=30, p=1.0)])
    out = pipe(image=img)
    assert out["image"].shape == img.shape


def test_fused_plus_passthrough_dict_input():
    """Mixed fused-geometric + passthrough colour op via dict input."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    # A.Rotate is fused (AlbuFusedAffineSegment); A.GaussianBlur is a passthrough
    # (_PassthroughSegment with AlbumentationsAdapter).
    pipe = Compose([A.Rotate(limit=30, p=1.0), A.GaussianBlur(p=1.0)])
    out = pipe(image=img)
    assert isinstance(out, dict)
    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == img.shape
