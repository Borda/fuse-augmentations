"""``execution="auto"``: an opt-in routing rule, and the engine it resolved to made readable.

The engine used to be fixed at construction, full stop. That is the right default -- it is what makes a recorded
configuration determine the output -- but it leaves a caller who genuinely does not know where their data will live
writing the branch themselves. ``"auto"`` is the opt-in: host data warps in OpenCV, accelerator data warps in
``grid_sample``, the rule is fixed rather than measured at runtime, and ``resolved_execution`` says which one ran.

These tests pin the rule, the readability, and the fact that ``"auto"`` never leaks into a pipeline that did not ask for
it.

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

HEIGHT = WIDTH = 32

#: Constructed at import, so it needs the same guard the import does: a module-level ``skipif`` marks
#: the tests skipped but does not stop the module body from running, and the empty fallback is never
#: reached by a test that the marker lets execute.
FIXED_AFFINE = [albu.Affine(rotate=(13.0, 13.0), scale=(1.07, 1.07), p=1.0)] if _ALBUMENTATIONS_AVAILABLE else []


@pytest.fixture
def image_hwc() -> np.ndarray:
    """Return one deterministic channel-last ``uint8`` image."""
    return np.random.default_rng(0).integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


class TestRoutingRule:
    """Which engine ``"auto"`` picks, and what it reports afterwards."""

    def test_host_tensor_routes_to_cv2(self, image_hwc: np.ndarray) -> None:
        """A CPU tensor resolves to the OpenCV engine.

        This is the case the rule exists for: on the host, ``grid_sample`` measured five to sixteen times
        slower than ``warpAffine`` on a detection-shaped step, and cv2 is where the data already is.

        """
        pipe = Compose(FIXED_AFFINE, execution="auto")

        pipe(torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0)

        assert pipe.resolved_execution == "cv2"

    def test_numpy_input_routes_to_cv2(self, image_hwc: np.ndarray) -> None:
        """An array is host data, so it resolves to OpenCV like a CPU tensor does."""
        pipe = Compose(FIXED_AFFINE, execution="auto")

        pipe(image=image_hwc)

        assert pipe.resolved_execution == "cv2"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_routes_to_torch(self, image_hwc: np.ndarray) -> None:
        """An accelerator tensor resolves to ``grid_sample``, which is the only engine that can run there.

        ``cv2.warpAffine`` needs a host array, so routing a CUDA batch to it would mean a round trip that costs more
        than the warp.

        """
        pipe = Compose(FIXED_AFFINE, execution="auto")
        image = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

        pipe(image)

        assert pipe.resolved_execution == "torch"

    def test_auto_matches_the_engine_it_resolved_to(self, image_hwc: np.ndarray) -> None:
        """On the host, ``"auto"`` renders exactly what an explicit ``"cv2"`` pipeline renders.

        Resolving to an engine has to mean running that engine, not running something close to it -- a rule that routes
        correctly but renders differently would be worse than no rule.

        """
        image = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        auto = Compose(FIXED_AFFINE, execution="auto")
        explicit = Compose(FIXED_AFFINE, execution="cv2")

        torch.testing.assert_close(auto(image), explicit(image))


class TestExplicitSettingsAreUntouched:
    """A caller who pinned an engine is unaffected by the new value existing."""

    @pytest.mark.parametrize(
        "execution",
        [pytest.param("cv2", id="cv2"), pytest.param("torch", id="torch")],
    )
    def test_pinned_engine_is_reported_verbatim(self, image_hwc: np.ndarray, execution: str) -> None:
        """An explicit engine resolves to itself, so ``resolved_execution`` reads back the pinned value."""
        pipe = Compose(FIXED_AFFINE, execution=execution)

        pipe(torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0)

        assert pipe.resolved_execution == execution

    def test_resolved_execution_is_none_before_the_first_call(self) -> None:
        """Nothing has been warped yet, so there is no engine to report.

        Reporting a guess here would be worse than reporting nothing: it would look like a fact about a
        call that never happened.

        """
        assert Compose(FIXED_AFFINE, execution="auto").resolved_execution is None

    def test_unknown_execution_still_raises(self) -> None:
        """A misspelled engine is rejected at construction, and the message names all three values."""
        with pytest.raises(ValueError, match="'cv2', 'torch' or 'auto'"):
            Compose(FIXED_AFFINE, execution="opencv")


class TestAutoOnTheMultiTargetPath:
    """``"auto"`` participates in the NumPy-native multi-target shortcut."""

    def test_numpy_multi_target_call_is_eligible(self) -> None:
        """An ``"auto"`` pipeline takes the NumPy path on array input, as an explicit cv2 one does.

        Array input is host input, which is exactly the case ``"auto"`` routes to cv2 -- so excluding it from the
        shortcut would mean paying the tensor round-trip to reach the engine it had already chosen.

        """
        pipe = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="auto")
        assert pipe._native_multi_ok is True

    def test_numpy_multi_target_matches_the_explicit_cv2_pipeline(self, image_hwc: np.ndarray) -> None:
        """Targets and image agree with the explicitly-cv2 pipeline on the same input."""
        boxes = np.array([[4.0, 6.0, 20.0, 30.0]], dtype=np.float32)
        auto = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="auto")
        explicit = Compose(FIXED_AFFINE, data_keys=["input", "bbox_xyxy"], execution="cv2")

        out_auto = auto(image=image_hwc, bboxes=boxes)
        out_explicit = explicit(image=image_hwc, bboxes=boxes)

        np.testing.assert_array_equal(out_auto["image"], out_explicit["image"])
        np.testing.assert_array_equal(out_auto["bboxes"], out_explicit["bboxes"])
