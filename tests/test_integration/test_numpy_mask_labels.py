"""NumPy mask layout conversion preserves label values rather than image intensities."""

import numpy as np
import pytest

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as A


@pytest.mark.parametrize("dtype,label", [(np.uint8, 255), (np.int16, -1), (np.bool_, True)])
def test_numpy_integer_mask_rejects_soft_sampling(dtype, label):
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = np.full((5, 7), label, dtype=dtype)
    pipeline = Compose.from_params(rotation=(10.0, 10.0), data_keys=["input", "mask"], mask_interpolation="bilinear")
    with pytest.raises(TypeError, match="floating"):
        pipeline(image, mask)


@pytest.mark.parametrize("dtype,label", [(np.uint8, 255), (np.int16, -1), (np.bool_, True)])
def test_numpy_mask_keeps_labels_on_roundtrip(dtype, label):
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = np.full((5, 7), label, dtype=dtype)
    _, actual = Compose([], data_keys=["input", "mask"])(image, mask)
    np.testing.assert_array_equal(actual, mask)
    assert actual.dtype == mask.dtype


@pytest.mark.parametrize("execution", ["native", "torch"])
@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="Albumentations is optional")
def test_numpy_uint8_fill_uses_label_units(execution):
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = np.full((5, 7), 17, dtype=np.uint8)
    pipeline = Compose(
        [A.Affine(translate_px={"x": 2, "y": 0}, p=1.0)],
        data_keys=["input", "mask"],
        mask_fill=1,
        execution="cv2" if execution == "native" else "torch",
    )
    result = pipeline(image=image, mask=mask) if execution == "native" else pipeline(image, mask)
    actual = result["mask"] if execution == "native" else result[1]
    expected = np.full_like(mask, 17)
    expected[:, :2] = 1
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
