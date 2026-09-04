"""The NumPy-native multi-target path: same results as the tensor path, without the tensor round-trip.

A multi-target call used to leave NumPy the moment it arrived -- normalise to float32 ``(batch, channel, height,
width)``, warp, convert back -- which cost three full-image layout copies and a four-times-wider warp that native
Albumentations never pays. Nothing about auxiliary targets required it: coordinate tables are tiny, and the NumPy image
path already composes the same fused matrix the tensor path routes them through.

These tests pin the two things that make that shortcut safe to take: every target modality lands exactly where the
tensor path puts it, and a pipeline the shortcut cannot serve falls back silently rather than producing something
different.

"""

from __future__ import annotations

import albumentations as albu
import numpy as np
import pytest

from fuse_augmentations import Compose

pytestmark = pytest.mark.integration

HEIGHT = WIDTH = 48
ALL_KEYS = ["input", "bbox_xyxy", "keypoints", "mask", "rboxes"]

#: A single affine with fixed parameters: the two paths draw from the NumPy RNG a different number of
#: times (the NumPy path skips the Bernoulli draw for a ``p=1.0`` transform), so a comparison between
#: them is only meaningful when the sampled geometry cannot vary.
FIXED_AFFINE = [albu.Affine(rotate=(13.0, 13.0), scale=(1.07, 1.07), p=1.0)]


@pytest.fixture
def targets() -> dict[str, np.ndarray]:
    """Return one deterministic sample carrying every supported target modality."""
    rng = np.random.default_rng(0)
    return {
        "image": rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8),
        "bbox_xyxy": np.array([[4.0, 6.0, 20.0, 30.0], [10.0, 10.0, 40.0, 44.0]], dtype=np.float32),
        "keypoints": np.array([[5.0, 7.0], [22.0, 31.0]], dtype=np.float32),
        "mask": rng.integers(0, 3, size=(HEIGHT, WIDTH)).astype(np.uint8),
        "rboxes": np.array([[20.0, 20.0, 10.0, 6.0, 0.3]], dtype=np.float32),
    }


class TestTargetParityWithTensorPath:
    """Every auxiliary modality lands where the tensor path puts it."""

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("bbox_xyxy", id="boxes"),
            pytest.param("keypoints", id="keypoints"),
            pytest.param("mask", id="mask"),
            pytest.param("rboxes", id="rboxes"),
        ],
    )
    def test_auxiliary_target_matches_exactly(self, targets: dict[str, np.ndarray], key: str) -> None:
        """A target routed through the NumPy path is bit-identical to the same target routed as a tensor.

        Both paths route through the same composed matrix and the same ``targets`` builders, so anything other than
        exact equality would mean the NumPy path had grown its own convention -- an off-by-one origin or a different
        mask sampler -- which is precisely the drift this shortcut must not introduce.

        """
        native = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="cv2")
        tensor = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="torch")

        out_native = native(**targets)
        out_tensor = tensor(**targets)

        np.testing.assert_array_equal(out_native[key], out_tensor[key])

    def test_image_matches_within_one_intensity_level(self, targets: dict[str, np.ndarray]) -> None:
        """The warped image agrees with the tensor path to within one ``uint8`` level.

        The two are not required to be identical: the NumPy path warps the ``uint8`` array directly, as Albumentations
        does, while the tensor path warps a float32 copy and quantises afterwards. One level is the rounding, and a
        larger gap would mean a genuinely different resample.

        """
        native = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="cv2")
        tensor = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="torch")

        image_native = native(**targets)["image"].astype(np.int16)
        image_tensor = tensor(**targets)["image"].astype(np.int16)

        assert np.abs(image_native - image_tensor).max() <= 1

    def test_reported_matrix_matches(self, targets: dict[str, np.ndarray]) -> None:
        """``transform_matrix`` is populated on the NumPy path and equals the tensor path's matrix."""
        native = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="cv2")
        tensor = Compose(FIXED_AFFINE, data_keys=ALL_KEYS, execution="torch")

        native(**targets)
        tensor(**targets)

        np.testing.assert_allclose(
            native.transform_matrix.numpy(),
            tensor.transform_matrix.numpy(),
            atol=1e-6,
        )


class TestEligibility:
    """Which pipelines take the shortcut, and what happens to the ones that cannot."""

    def test_cv2_multi_target_pipeline_is_eligible(self) -> None:
        """A cv2 multi-target pipeline of fused affines takes the NumPy path."""
        pipe = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="cv2")
        assert pipe._native_multi_ok is True

    def test_torch_execution_opts_out(self) -> None:
        """``execution="torch"`` keeps the tensor path even though NumPy would be faster on CPU.

        Execution is a documented reproducibility axis fixed at construction. A caller who asked for ``grid_sample`` did
        so to match something; handing them a cv2 render because their input happened to be an array would make the axis
        depend on the input type.

        """
        pipe = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="torch")
        assert pipe._native_multi_ok is False

    def test_crop_in_the_chain_opts_out(self) -> None:
        """A crop moves coordinates without publishing a matrix, so the pipeline is not eligible."""
        pipe = Compose(
            [albu.RandomResizedCrop(size=(32, 32), p=1.0), *FIXED_AFFINE],
            data_keys=["input", "bbox_xyxy"],
            execution="cv2",
        )
        assert pipe._native_multi_ok is False

    def test_explicit_output_backend_opts_out(self) -> None:
        """An explicit ``output_backend`` is the caller naming what they want back, and it wins."""
        pipe = Compose(
            FIXED_AFFINE,
            data_keys=["input", "bbox_xyxy"],
            execution="cv2",
            output_backend="torch",
        )
        assert pipe._native_multi_ok is False

    def test_ineligible_pipeline_still_preserves_dtype(self, targets: dict[str, np.ndarray]) -> None:
        """Falling back to the tensor path does not change the dtype the caller gets.

        This is the property that makes the shortcut invisible: adding a crop to a chain changes which path runs, and
        must not also change what the call returns.

        """
        pipe = Compose(
            [albu.RandomResizedCrop(size=(32, 32), p=1.0), *FIXED_AFFINE],
            data_keys=["input", "bbox_xyxy"],
            execution="cv2",
        )

        out = pipe(image=targets["image"], bboxes=targets["bbox_xyxy"])

        assert out["image"].dtype == np.uint8
        assert out["image"].shape == (32, 32, 3)


class TestCallForms:
    """Both multi-target call shapes reach the NumPy path."""

    def test_mixed_positional_image_takes_the_same_path(self, targets: dict[str, np.ndarray]) -> None:
        """``pipe(image, bboxes=...)`` agrees with the all-keyword call.

        The mixed form resolves through the same keyword dispatch, so a divergence here would mean the shortcut was
        wired into one call shape and not the other -- the kind of gap that only shows up in someone else's training
        loop.

        """
        pipe = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="cv2")

        out_keyword = pipe(image=targets["image"], bboxes=targets["bbox_xyxy"])
        out_mixed = pipe(targets["image"], bboxes=targets["bbox_xyxy"])

        np.testing.assert_array_equal(out_mixed["image"], out_keyword["image"])
        np.testing.assert_array_equal(out_mixed["bboxes"], out_keyword["bboxes"])

    def test_tensor_inputs_still_take_the_tensor_path(self, targets: dict[str, np.ndarray]) -> None:
        """An eligible pipeline called with tensors returns tensors, unchanged by the NumPy shortcut."""
        import torch

        pipe = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="cv2")
        image = torch.from_numpy(targets["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        boxes = torch.from_numpy(targets["bbox_xyxy"]).unsqueeze(0)

        out_image, out_boxes = pipe(image, boxes)

        assert isinstance(out_image, torch.Tensor)
        assert isinstance(out_boxes, torch.Tensor)
        assert out_image.shape == image.shape
