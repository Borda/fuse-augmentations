"""Auxiliary-target routing for Albumentations ``RandomResizedCrop``.

A ``CROP_RESIZE_FIXED`` op changes the output spatial size. On the Albumentations
(numpy) backend the crop was previously emitted as an image-only passthrough, so a
mask/box/keypoint auxiliary target kept its input size and silently desynced from the
resized image. ``build_segments`` now emits a ``CropResizeSegment`` for the crop when
the pipeline carries any auxiliary target, routing every target to the crop's output
size, for both the ``"cv2"`` (default) and ``"torch"`` execution strategies.

These tests lock:

- a single Albumentations ``RandomResizedCrop`` with a mask resizes both image and
  mask, under both execution strategies;
- a mixed Kornia+Albumentations pipeline whose crop is Albumentations routes the mask;
- the two execution strategies agree bit-for-bit (both build the same segment);
- nearest mask routing introduces no new label values;
- an image-only pipeline (no ``data_keys``) is unaffected — the numpy dict path still
  returns an image-only crop.

Requires albumentations; the mixed-backend test additionally requires kornia.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32
CROP_H, CROP_W = 24, 24


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 32, 32)`` float32 image batch."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH, generator=gen)


def _mask() -> torch.Tensor:
    """Return a deterministic ``(2, 1, 32, 32)`` integer-label mask batch."""
    gen = torch.Generator().manual_seed(1)
    return torch.randint(0, 4, (BATCH, 1, HEIGHT, WIDTH), generator=gen).float()


@pytest.mark.parametrize("execution", [pytest.param("cv2", id="cv2"), pytest.param("torch", id="torch")])
def test_albu_crop_routes_mask_to_output_size(execution):
    """A single Albumentations crop resizes the mask to the crop's output size, not the input size."""
    pipe = Compose(
        [albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)],
        data_keys=["input", "mask"],
        execution=execution,
    )
    out_image, out_mask = pipe(_image(), _mask())
    assert out_image.shape == (BATCH, CHANNELS, CROP_H, CROP_W)
    assert out_mask.shape == (BATCH, 1, CROP_H, CROP_W)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
@pytest.mark.parametrize("execution", [pytest.param("cv2", id="cv2"), pytest.param("torch", id="torch")])
def test_mixed_kornia_albu_crop_routes_mask(execution):
    """A mixed Kornia-rotation + Albumentations-crop pipeline routes the mask through the crop."""
    pipe = Compose(
        [kornia_aug.RandomRotation(degrees=30, p=1.0), albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)],
        data_keys=["input", "mask"],
        execution=execution,
    )
    out_image, out_mask = pipe(_image(), _mask())
    assert out_image.shape == (BATCH, CHANNELS, CROP_H, CROP_W)
    assert out_mask.shape == (BATCH, 1, CROP_H, CROP_W)


def test_albu_crop_aux_matches_across_executions():
    """The ``cv2`` and ``torch`` strategies build the same crop segment, so aux outputs match bit-for-bit.

    Each crop instance is pinned to the same Albumentations RNG seed so both pipelines
    sample the same crop window; any output divergence then reflects an execution-path
    routing difference, which must not exist for the shared ``CropResizeSegment``.

    """
    crop_cv2 = albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)
    crop_torch = albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)
    crop_cv2.set_random_seed(7)
    crop_torch.set_random_seed(7)
    pipe_cv2 = Compose([crop_cv2], data_keys=["input", "mask"], execution="cv2")
    pipe_torch = Compose([crop_torch], data_keys=["input", "mask"], execution="torch")

    img_cv2, mask_cv2 = pipe_cv2(_image(), _mask())
    img_torch, mask_torch = pipe_torch(_image(), _mask())

    torch.testing.assert_close(img_cv2, img_torch)
    torch.testing.assert_close(mask_cv2, mask_torch)


def test_routed_mask_introduces_no_new_labels():
    """Nearest mask routing resamples existing labels only — no interpolation-created values."""
    mask = _mask()
    pipe = Compose(
        [albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)],
        data_keys=["input", "mask"],
        mask_interpolation="nearest",
    )
    _out_image, out_mask = pipe(_image(), mask)
    # Nearest sampling with zero out-of-bounds padding: every output label is an input
    # label or the padding value 0 (which is already an input label here).
    allowed = set(torch.unique(mask).tolist())
    produced = set(torch.unique(out_mask).tolist())
    assert produced <= allowed | {0.0}


def test_image_only_numpy_crop_unaffected_by_routing():
    """Without ``data_keys`` the numpy dict path still applies an image-only crop."""
    pipe = Compose([albu.RandomResizedCrop(size=(CROP_H, CROP_W), p=1.0)])
    image_hwc = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
    out = pipe(image=image_hwc)
    assert isinstance(out, dict)
    assert out["image"].shape == (CROP_H, CROP_W, CHANNELS)
