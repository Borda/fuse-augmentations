"""Integration tests for Albumentations auxiliary-target routing.

Albumentations fused segments route masks, bounding boxes, and keypoints through
the same composed pixel matrix used to warp the image, matching the coordinate
convention (center/``align_corners=True``) of the Kornia/TorchVision torch path.

These tests verify:

- masks, boxes, and keypoints returned by an Albumentations fused pipeline match
  the ground truth obtained by applying the shared ``targets`` helpers with the
  pipeline's own composed matrix (convention parity);
- geometric flips agree with a Kornia-equivalent pipeline within tight tolerance;
- an Albumentations-style keyword call returns a dict keyed by the input names;
- ``output_backend`` conversion is applied per target (types and layouts).

Requires albumentations; the Kornia-equivalence check additionally requires kornia.

"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.targets import (
    transform_bbox_xyxy,
    transform_keypoints,
    transform_mask,
)

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 32, 32


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 32, 32)`` float32 image batch."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH, generator=gen)


def _mask() -> torch.Tensor:
    """Return a deterministic ``(2, 1, 32, 32)`` integer-valued mask batch."""
    gen = torch.Generator().manual_seed(1)
    return torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH), generator=gen).float()


def _boxes() -> torch.Tensor:
    """Return one ``(2, 1, 4)`` xyxy box per sample."""
    return torch.tensor([[[5.0, 8.0, 20.0, 25.0]]] * BATCH)


def _keypoints() -> torch.Tensor:
    """Return two ``(2, 2, 2)`` keypoints per sample."""
    return torch.tensor([[[5.0, 8.0], [20.0, 25.0]]] * BATCH)


def _affine_chain() -> list[object]:
    """Build a fusible Albumentations affine chain (interpolating, prob=1)."""
    return [
        albu.Affine(rotate=(20.0, 20.0), scale=(1.1, 1.1), p=1.0),
        albu.Affine(translate_px=(2, 2), p=1.0),
    ]


@pytest.mark.parametrize("execution", ["cv2", "torch"])
class TestAlbuAuxConventionParity:
    """Aux outputs match the shared ``targets`` helpers applied with the composed matrix."""

    def test_mask_matches_matrix_ground_truth(self, execution: str):
        """The routed mask equals ``transform_mask`` applied with the composed matrix."""
        pipe = Compose(_affine_chain(), data_keys=["input", "mask"], execution=execution)
        image, mask = _image(), _mask()
        np.random.seed(3)
        _, out_mask = pipe(image, mask)

        matrix = pipe.transform_matrix
        assert matrix is not None
        mtx_inv = torch.linalg.inv(matrix.to(torch.float64))
        from fuse_augmentations.affine.matrix import normalize_matrix

        mtx_norm = normalize_matrix(mtx_inv, HEIGHT, WIDTH).to(torch.float32)
        import torch.nn.functional as functional

        grid = functional.affine_grid(mtx_norm[:, :2, :], [BATCH, 1, HEIGHT, WIDTH], align_corners=True)
        expected = transform_mask(mask, grid)
        torch.testing.assert_close(out_mask, expected, atol=0.0, rtol=0.0)

    def test_boxes_match_matrix_ground_truth(self, execution: str):
        """The routed boxes equal ``transform_bbox_xyxy`` applied with the composed matrix."""
        pipe = Compose(_affine_chain(), data_keys=["input", "bbox_xyxy"], execution=execution)
        image, boxes = _image(), _boxes()
        np.random.seed(3)
        _, out_boxes = pipe(image, boxes)

        matrix = pipe.transform_matrix
        assert matrix is not None
        expected = transform_bbox_xyxy(boxes, matrix)
        torch.testing.assert_close(out_boxes, expected, atol=1e-4, rtol=1e-4)

    def test_keypoints_match_matrix_ground_truth(self, execution: str):
        """The routed keypoints equal ``transform_keypoints`` applied with the composed matrix."""
        pipe = Compose(_affine_chain(), data_keys=["input", "keypoints"], execution=execution)
        image, keypoints = _image(), _keypoints()
        np.random.seed(3)
        _, out_kps = pipe(image, keypoints)

        matrix = pipe.transform_matrix
        assert matrix is not None
        expected = transform_keypoints(keypoints, matrix)
        torch.testing.assert_close(out_kps, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
class TestAlbuKorniaEquivalence:
    """Flip geometry agrees with a Kornia-equivalent pipeline (identical mirror math)."""

    def test_hflip_mask_matches_kornia(self):
        """HFlip mask routed by Albumentations matches the Kornia-equivalent pipeline."""
        image, mask = _image(), _mask()
        albu_pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "mask"])
        kornia_pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)], data_keys=["input", "mask"])
        _, albu_mask = albu_pipe(image, mask)
        _, kornia_mask = kornia_pipe(image, mask)
        torch.testing.assert_close(albu_mask, kornia_mask, atol=0.0, rtol=0.0)

    def test_hflip_boxes_match_kornia(self):
        """HFlip boxes routed by Albumentations match the Kornia-equivalent pipeline."""
        image, boxes = _image(), _boxes()
        albu_pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "bbox_xyxy"])
        kornia_pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)], data_keys=["input", "bbox_xyxy"])
        _, albu_boxes = albu_pipe(image, boxes)
        _, kornia_boxes = kornia_pipe(image, boxes)
        torch.testing.assert_close(albu_boxes, kornia_boxes, atol=1e-4, rtol=1e-4)


class TestAlbuDictOutput:
    """Albumentations-style keyword calls return a dict keyed by the input names."""

    def test_kwargs_call_returns_dict(self):
        """A ``pipe(image=..., mask=...)`` call returns a dict with those keys."""
        pipe = Compose(_affine_chain(), data_keys=["input", "mask"])
        np.random.seed(3)
        out = pipe(image=_image(), mask=_mask())
        assert isinstance(out, dict)
        assert set(out) == {"image", "mask"}
        assert out["image"].shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
        assert out["mask"].shape == (BATCH, 1, HEIGHT, WIDTH)

    def test_kwargs_dict_matches_positional_tuple(self):
        """The dict values match the positional tuple API for the same inputs."""
        image, mask = _image(), _mask()
        pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "mask"])
        out_dict = pipe(image=image, mask=mask)
        out_image, out_mask = pipe(image, mask)
        torch.testing.assert_close(out_dict["image"], out_image, atol=0.0, rtol=0.0)
        torch.testing.assert_close(out_dict["mask"], out_mask, atol=0.0, rtol=0.0)

    def test_bboxes_alias_maps_to_declared_box_key(self):
        """The ``bboxes`` keyword maps to the pipeline's declared box data key."""
        pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "bbox_xyxy"])
        out = pipe(image=_image(), bboxes=_boxes())
        assert isinstance(out, dict)
        assert set(out) == {"image", "bboxes"}
        assert out["bboxes"].shape == (BATCH, 1, 4)

    def test_positional_call_still_returns_tuple(self):
        """The positional ``data_keys`` API is unchanged and returns a tuple."""
        pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "mask"])
        out = pipe(_image(), _mask())
        assert isinstance(out, tuple)
        assert len(out) == 2


class TestAlbuMultiTargetOutputBackend:
    """``output_backend`` conversion is applied per target for multi-target outputs."""

    def test_numpy_backend_converts_image_and_mask(self):
        """A numpy backend yields numpy arrays for image and mask; boxes stay tensors."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0)],
            data_keys=["input", "mask", "bbox_xyxy"],
            output_backend="numpy",
        )
        out_image, out_mask, out_boxes = pipe(_image(), _mask(), _boxes())
        assert isinstance(out_image, np.ndarray)
        assert isinstance(out_mask, np.ndarray)
        assert isinstance(out_boxes, torch.Tensor)

    def test_numpy_image_is_channel_last(self):
        """The converted image uses channel-last ``(B, H, W, C)`` layout."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
            output_backend="numpy",
        )
        out_image, _ = pipe(_image(), _mask())
        assert out_image.shape == (BATCH, HEIGHT, WIDTH, CHANNELS)


class TestPassthroughAuxWarning:
    """A geometric passthrough barrier warns that auxiliary targets skip it."""

    def test_passthrough_with_mask_warns(self):
        """A spatial-kernel passthrough with a mask present emits a UserWarning."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0), albu.GaussianBlur(p=1.0)],
            data_keys=["input", "mask"],
        )
        with pytest.warns(UserWarning, match="auxiliary targets"):
            pipe(_image(), _mask())

    def test_image_only_call_does_not_warn(self):
        """The same barrier does not warn for a single-tensor (image-only) call."""
        pipe = Compose([albu.HorizontalFlip(p=1.0), albu.GaussianBlur(p=1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pipe(_image())


class TestKwargsAliasCollision:
    """Ambiguous keyword aliases raise instead of silently dropping an input."""

    def test_bboxes_and_exact_box_key_collide(self):
        """Passing both ``bboxes`` and ``bbox_xyxy`` raises a clear ValueError."""
        pipe = Compose([albu.HorizontalFlip(p=1.0)], data_keys=["input", "bbox_xyxy"])
        with pytest.raises(ValueError, match="both resolve to data key"):
            pipe(image=_image(), bboxes=_boxes(), bbox_xyxy=_boxes())
