"""Integration tests for crop-resize coverage beyond ``RandomResizedCrop``.

``CropResizeSegment`` has always been able to carry a crop-resize and route auxiliary targets through it, but only
``RandomResizedCrop`` was registered as a crop-resize op in the Albumentations adapter. ``RandomSizedCrop`` shares the
same ``_BaseRandomSizedCrop`` parameter surface, so it is now registered too; these tests verify it composes and routes
the same way rather than assuming the shared base class is enough.

The crop window is sampled per call, so the parity checks here are co-location checks — a landmark placed on a bright
pixel still sits on that pixel afterwards, and a box drawn around it still surrounds it — rather than comparisons
against a matrix the pipeline does not expose for a crop-only run.

``RandomSizedBBoxSafeCrop`` is deliberately not registered, and one test pins that: it derives its crop window from the
sample's bounding boxes, which the fused parameter-sampling path never sees.

Requires albumentations.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.affine.segment import CropResizeSegment
from fuse_augmentations.types import TransformCategory

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

    from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]

HEIGHT, WIDTH = 64, 64
TARGET_HEIGHT, TARGET_WIDTH = 32, 32
CROP_SIDE = 48
MARK_Y, MARK_X = 32.0, 32.0
MARK_HALF_SIDE = 3
EXECUTIONS = ("cv2", "torch")


def _sized_crop() -> albu.BasicTransform:
    """Return a ``RandomSizedCrop`` whose crop side is fixed so only its position varies."""
    return albu.RandomSizedCrop(
        min_max_height=(CROP_SIDE, CROP_SIDE),
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        w2h_ratio=1.0,
        p=1.0,
    )


def _resized_crop() -> albu.BasicTransform:
    """Return the already-registered ``RandomResizedCrop`` used as the parity reference."""
    return albu.RandomResizedCrop(size=(TARGET_HEIGHT, TARGET_WIDTH), p=1.0)


def _marked_image() -> torch.Tensor:
    """Return a ``(1, 1, 64, 64)`` image that is zero except for a bright square at the centre.

    The centre survives every sampled window: the crop side is 48 of 64, so the offset is at most 16
    and the marked square is always inside the crop. A square rather than a single pixel because the
    crop downscales by 2/3 and a lone pixel can fall between output samples under nearest
    resampling — which says nothing about routing.

    """
    image = torch.zeros(1, 1, HEIGHT, WIDTH)
    half = MARK_HALF_SIDE
    image[0, 0, int(MARK_Y) - half : int(MARK_Y) + half, int(MARK_X) - half : int(MARK_X) + half] = 1.0
    return image


def _centroid(plane: torch.Tensor) -> tuple[float, float]:
    """Return the intensity-weighted ``(column, row)`` centre of a 2-D plane."""
    rows = torch.arange(plane.shape[0], dtype=plane.dtype)
    columns = torch.arange(plane.shape[1], dtype=plane.dtype)
    total = plane.sum()
    row = float((plane.sum(dim=1) * rows).sum() / total)
    column = float((plane.sum(dim=0) * columns).sum() / total)
    return column, row


class TestRandomSizedCropRegistration:
    """``RandomSizedCrop`` is classified and segmented like the transform it shares a base with."""

    def test_is_categorised_as_crop_resize(self) -> None:
        """The adapter reports ``CROP_RESIZE_FIXED`` rather than treating the op as a barrier.

        An unregistered transform falls back to a passthrough barrier, which stops auxiliary targets from being routed
        through it at all — the failure this registration exists to prevent.

        """
        category = AlbumentationsAdapter.category(_sized_crop())

        assert category == TransformCategory.CROP_RESIZE_FIXED

    def test_standalone_crop_builds_a_crop_resize_segment(self) -> None:
        """On its own the transform becomes a ``CropResizeSegment`` producing the target size.

        The output size is the visible contract: a crop-resize left as a passthrough would return
        the input size, and nothing else in a multi-target run would signal the miss.

        """
        pipe = Compose([_sized_crop()], data_keys=["input", "bbox_xyxy"])
        boxes = torch.tensor([[[8.0, 10.0, 40.0, 44.0]]])

        out_image, _ = pipe(torch.rand(1, 3, HEIGHT, WIDTH), boxes)

        assert [type(segment) for segment in pipe._segments] == [CropResizeSegment]
        assert out_image.shape == (1, 3, TARGET_HEIGHT, TARGET_WIDTH)

    def test_segment_layout_matches_random_resized_crop(self) -> None:
        """After a geometric op the segment layout is the one ``RandomResizedCrop`` already produces.

        Registration parity is the actual deliverable here: the newly registered transform must be
        planned exactly like the transform it shares a parameter surface with, whatever that plan
        happens to be on this adapter.

        """
        data_keys = ["input", "bbox_xyxy"]
        sized = Compose([albu.Affine(rotate=(15.0, 15.0), p=1.0), _sized_crop()], data_keys=data_keys)
        resized = Compose([albu.Affine(rotate=(15.0, 15.0), p=1.0), _resized_crop()], data_keys=data_keys)

        assert [type(segment) for segment in sized._segments] == [type(segment) for segment in resized._segments]


@pytest.mark.parametrize("execution", [pytest.param(mode, id=mode) for mode in EXECUTIONS])
class TestRandomSizedCropTargetParity:
    """Targets routed through the crop stay co-located with the image content they describe."""

    def test_keypoint_follows_the_pixel_it_marks(self, execution: str) -> None:
        """A keypoint on the bright pixel still sits on it after the crop.

        Co-location is the invariant that a routed coordinate target has to preserve; a keypoint that skipped the crop
        matrix, or went through the geometry only, would keep its input coordinates while the image moved under it and
        no shape would change.

        """
        pipe = Compose(
            [_sized_crop()],
            data_keys=["input", "keypoints"],
            execution=execution,
            interpolation="nearest",
        )
        keypoints = torch.tensor([[[MARK_X, MARK_Y]]])

        out_image, out_keypoints = pipe(_marked_image(), keypoints)

        column, row = _centroid(out_image[0, 0])
        assert abs(column - float(out_keypoints[0, 0, 0])) <= 1.0
        assert abs(row - float(out_keypoints[0, 0, 1])) <= 1.0

    def test_box_still_surrounds_the_keypoint(self, execution: str) -> None:
        """A box drawn around the marked pixel still contains the routed keypoint afterwards.

        Boxes and keypoints travel different code paths inside the segment, so agreeing with the image is not enough —
        this pins that the two coordinate targets agree with each other under the same sampled crop window.

        """
        pipe = Compose(
            [_sized_crop()],
            data_keys=["input", "bbox_xyxy", "keypoints"],
            execution=execution,
            interpolation="nearest",
        )
        boxes = torch.tensor([[[MARK_X - 8.0, MARK_Y - 8.0, MARK_X + 8.0, MARK_Y + 8.0]]])
        keypoints = torch.tensor([[[MARK_X, MARK_Y]]])

        _, out_boxes, out_keypoints = pipe(_marked_image(), boxes, keypoints)

        x_min, y_min, x_max, y_max = (float(value) for value in out_boxes[0, 0])
        assert x_min <= float(out_keypoints[0, 0, 0]) <= x_max
        assert y_min <= float(out_keypoints[0, 0, 1]) <= y_max

    def test_mask_marks_the_same_place_as_the_image(self, execution: str) -> None:
        """A mask marking the bright pixel still marks it after the crop.

        Masks resample on the output grid rather than being mapped as coordinates, so they are the third routing path
        and the one where an input-grid resample would return the wrong canvas size without any coordinate looking
        wrong.

        """
        pipe = Compose(
            [_sized_crop()],
            data_keys=["input", "mask"],
            execution=execution,
            interpolation="nearest",
        )

        out_image, out_mask = pipe(_marked_image(), _marked_image())

        assert out_mask.shape == (1, 1, TARGET_HEIGHT, TARGET_WIDTH)
        mask_column, mask_row = _centroid(out_mask[0, 0])
        image_column, image_row = _centroid(out_image[0, 0])
        assert abs(mask_column - image_column) <= 1.0
        assert abs(mask_row - image_row) <= 1.0


class TestBBoxSafeCropStaysUnregistered:
    """``RandomSizedBBoxSafeCrop`` must not be treated as a fusible crop-resize op."""

    def test_is_not_categorised_as_crop_resize(self) -> None:
        """The box-dependent crop stays unregistered, so no window is fused without its boxes.

        Its crop window is chosen from the sample's boxes, but crop parameters are sampled against a dummy image that
        carries none; registering it would produce a window that ignores the instances it exists to keep, and every
        shape would still line up.

        """
        transform = albu.RandomSizedBBoxSafeCrop(height=TARGET_HEIGHT, width=TARGET_WIDTH, p=1.0)

        with pytest.warns(UserWarning, match="Unknown Albumentations transform"):
            category = AlbumentationsAdapter.category(transform)

        assert category != TransformCategory.CROP_RESIZE_FIXED
