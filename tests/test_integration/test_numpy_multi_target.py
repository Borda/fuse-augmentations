"""Integration tests for NumPy input on the multi-target (``data_keys``) path.

The multi-target path operates on ``(batch_size, channels, height, width)`` image tensors, but the Albumentations-facing
call style hands over channel-last NumPy arrays. These tests cover that combination end to end: a raw NumPy image plus
boxes or keypoints, on both execution engines, with the results compared against the equivalent tensor call so the two
entry points cannot drift.

They also cover the mixed call signature ``pipe(image, bboxes=...)`` — the image positionally, auxiliary targets by
keyword — which is accepted only in that one shape.

Requires albumentations.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]

HEIGHT, WIDTH = 32, 24
EXECUTIONS = ("cv2", "torch")


@pytest.fixture
def image_hwc() -> np.ndarray:
    """Return a deterministic ``(32, 24, 3)`` float32 channel-last image."""
    return np.linspace(0.0, 1.0, HEIGHT * WIDTH * 3, dtype=np.float32).reshape(HEIGHT, WIDTH, 3)


@pytest.fixture
def boxes_xyxy() -> np.ndarray:
    """Return three ``(x_min, y_min, x_max, y_max)`` boxes as an unbatched ``(3, 4)`` array."""
    return np.array(
        [[2.0, 3.0, 10.0, 12.0], [0.0, 0.0, 5.0, 5.0], [14.0, 20.0, 23.0, 31.0]],
        dtype=np.float32,
    )


def _flip_pipeline(execution: str, data_keys: list[str]) -> Compose:
    """Build a deterministic horizontal-flip pipeline for the given execution engine."""
    return Compose(
        [albu.HorizontalFlip(p=1.0)],
        data_keys=data_keys,
        execution=execution,
    )


class TestNumpyImageWithBoxes:
    """Raw NumPy HWC image plus bounding boxes through a multi-target pipeline."""

    @pytest.mark.parametrize("execution", [pytest.param(mode, id=mode) for mode in EXECUTIONS])
    def test_matches_the_equivalent_tensor_call(
        self,
        execution: str,
        image_hwc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        """NumPy input produces the same boxes as the tensor call on both execution engines.

        This is the combination that crashed before the inputs were normalised: the dict call cast
        the array to a tensor type without converting it, so a ``(height, width, channels)`` array
        reached a four-way shape unpack. Comparing against the tensor call is what keeps the two
        entry points from drifting once they both work.

        """
        pipe = _flip_pipeline(execution, ["input", "bbox_xyxy"])
        tensor_image = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0)
        tensor_boxes = torch.from_numpy(boxes_xyxy).unsqueeze(0)

        numpy_out = pipe(image=image_hwc, bboxes=boxes_xyxy)
        _, tensor_out_boxes = pipe(tensor_image, tensor_boxes)

        torch.testing.assert_close(
            torch.from_numpy(numpy_out["bboxes"]).unsqueeze(0),
            tensor_out_boxes,
        )

    @pytest.mark.parametrize("execution", [pytest.param(mode, id=mode) for mode in EXECUTIONS])
    def test_returns_arrays_in_the_input_layout(
        self,
        execution: str,
        image_hwc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        """A NumPy call gets NumPy back, with the image channel-last and the boxes unbatched.

        The caller passed channel-last arrays without a batch axis, so returning batched tensors would force every
        caller to unwrap the result by hand — the round-trip is what makes the NumPy entry point usable from an existing
        Albumentations-shaped data pipeline.

        """
        pipe = _flip_pipeline(execution, ["input", "bbox_xyxy"])

        out = pipe(image=image_hwc, bboxes=boxes_xyxy)

        assert isinstance(out["image"], np.ndarray)
        assert isinstance(out["bboxes"], np.ndarray)
        assert out["image"].shape == (HEIGHT, WIDTH, 3)
        assert out["bboxes"].shape == boxes_xyxy.shape

    def test_horizontal_flip_mirrors_box_coordinates(
        self,
        image_hwc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        """A horizontal flip mirrors box x-coordinates about the image centre.

        Shape-only assertions would pass even if the boxes were returned untransformed, which is the exact failure the
        missing routing test would have had to catch; this pins the values. The mirror axis follows the package's
        pixel-centre convention (``align_corners=True``), so the reflection is about ``width - 1``, not ``width``.

        """
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])
        expected = boxes_xyxy.copy()
        expected[:, [0, 2]] = (WIDTH - 1) - boxes_xyxy[:, [2, 0]]

        out = pipe(image=image_hwc, bboxes=boxes_xyxy)

        np.testing.assert_allclose(out["bboxes"], expected, atol=1e-4)


class TestNumpyImageLayouts:
    """Image ranks other than the plain ``(height, width, channels)`` case."""

    def test_grayscale_input_round_trips_without_a_channel_axis(self, boxes_xyxy: np.ndarray) -> None:
        """A ``(height, width)`` image comes back as ``(height, width)``, not ``(height, width, 1)``.

        Grayscale gains a singleton channel axis internally because the pipeline is channel-first throughout; leaking
        that axis back to the caller would silently change the array rank of an existing grayscale pipeline.

        """
        gray = np.linspace(0.0, 1.0, HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])

        out = pipe(image=gray, bboxes=boxes_xyxy)

        assert out["image"].shape == (HEIGHT, WIDTH)

    def test_uint8_input_comes_back_as_uint8(self, boxes_xyxy: np.ndarray) -> None:
        """A ``uint8`` image round-trips as ``uint8``, matching what Albumentations returns.

        Callers coming from Albumentations pass ``uint8`` and expect it back; returning float32 in ``[0, 1]`` instead
        forced a rescale at every call site. It also cost real time -- the normalisation and the four-times-wider warp
        it implies were most of why a detection-shaped step ran at half the speed of native Albumentations.

        """
        image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])

        out = pipe(image=image, bboxes=boxes_xyxy)

        assert out["image"].dtype == np.uint8
        assert out["image"].max() == 255

    def test_uint8_dtype_is_preserved_on_the_tensor_fallback_too(self, boxes_xyxy: np.ndarray) -> None:
        """The dtype contract does not depend on whether the fast NumPy path was eligible.

        ``execution="torch"`` opts out of the NumPy-native path deliberately, so this call normalises to float32
        internally and converts back on the way out. A contract that changed with fast-path eligibility would be the
        worst of both: the same code returning different dtypes depending on which transforms happen to be in the
        chain.

        """
        image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        pipe = _flip_pipeline("torch", ["input", "bbox_xyxy"])

        out = pipe(image=image, bboxes=boxes_xyxy)

        assert out["image"].dtype == np.uint8
        assert out["image"].max() == 255


class TestMixedCallSignature:
    """``pipe(image, bboxes=...)`` — image positionally, auxiliary targets by keyword."""

    def test_accepts_a_single_positional_image_with_auxiliary_keywords(
        self,
        image_hwc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        """The mixed form is accepted and agrees with the all-keyword call.

        This shape is what a caller reaches for first when the image is already in hand and only the targets need
        naming; it used to raise, so the agreement check is what makes it a real alternative spelling rather than a
        second code path.

        """
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])

        mixed = pipe(image_hwc, bboxes=boxes_xyxy)
        keyword = pipe(image=image_hwc, bboxes=boxes_xyxy)

        np.testing.assert_allclose(mixed["bboxes"], keyword["bboxes"])

    def test_accepts_a_tensor_image_positionally(self, boxes_xyxy: np.ndarray) -> None:
        """The mixed form works for tensor input too, not only for NumPy arrays.

        The widening is about the call signature, not the input layout, so a tensor caller that only wants to name its
        auxiliary targets must be served by the same path.

        """
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])
        tensor_image = torch.rand(1, 3, HEIGHT, WIDTH)
        tensor_boxes = torch.from_numpy(boxes_xyxy).unsqueeze(0)

        out = pipe(tensor_image, bboxes=tensor_boxes)

        assert isinstance(out["image"], torch.Tensor)
        assert out["bboxes"].shape == tensor_boxes.shape

    def test_rejects_several_positional_arguments_mixed_with_keywords(
        self,
        image_hwc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> None:
        """Two positional arguments alongside a keyword still raise ``TypeError``.

        Only the single-positional-image shape has an unambiguous mapping onto ``data_keys``; the guard is deliberately
        kept for everything else so a mis-ordered call fails loudly.

        """
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0)],
            data_keys=["input", "bbox_xyxy", "keypoints"],
            execution="cv2",
        )
        tensor_image = torch.rand(1, 3, HEIGHT, WIDTH)
        tensor_boxes = torch.from_numpy(boxes_xyxy).unsqueeze(0)

        with pytest.raises(TypeError, match="single positional argument"):
            pipe(tensor_image, tensor_boxes, keypoints=torch.zeros(1, 2, 2))


class TestInputLayoutMixing:
    """Guards against combining tensor and array layouts in one call."""

    def test_numpy_auxiliary_targets_with_a_tensor_image_raise(self, boxes_xyxy: np.ndarray) -> None:
        """NumPy boxes alongside a tensor image raise ``TypeError`` instead of guessing.

        A tensor image carries an explicit batch axis and an unbatched array does not, so silently accepting the mix
        would have to invent a batch dimension — better to name the problem.

        """
        pipe = _flip_pipeline("cv2", ["input", "bbox_xyxy"])
        tensor_image = torch.rand(1, 3, HEIGHT, WIDTH)

        with pytest.raises(TypeError, match="same layout"):
            pipe(tensor_image, boxes_xyxy)
