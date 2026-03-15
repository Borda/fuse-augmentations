"""Tests for _backend.py — detect_backend."""

import warnings

import pytest

from fuse_augmentations._backend import detect_backend


def _make_mock(module_path: str):
    """Create a mock object whose type has a specific __module__."""
    ns = {"__module__": module_path, "__qualname__": "MockTransform"}
    cls = type("MockTransform", (), ns)
    return cls()


# --- Test #25: Mixed backends -> ValueError ---


def test_mixed_backends_raises():
    kornia_mock = _make_mock("kornia.augmentation")
    tv_mock = _make_mock("torchvision.transforms")
    with pytest.raises(ValueError, match="Mixed backends"):
        detect_backend([kornia_mock, tv_mock])


# --- Empty list -> "unknown" ---


def test_empty_list_returns_unknown():
    assert detect_backend([]) == "unknown"


# --- All unknown -> "unknown" + warning ---


def test_all_unknown_returns_unknown_with_warning():
    unknown_mock = _make_mock("my_custom_lib.transforms")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = detect_backend([unknown_mock])
    assert result == "unknown"
    assert len(w) == 1
    assert "Unrecognized transform" in str(w[0].message)


# --- Single known backends ---


def test_single_kornia_backend():
    mock = _make_mock("kornia.augmentation._2d")
    assert detect_backend([mock]) == "kornia"


def test_single_albumentations_backend():
    mock = _make_mock("albumentations.augmentations.transforms")
    assert detect_backend([mock]) == "albumentations"


def test_single_torchvision_backend():
    mock = _make_mock("torchvision.transforms.v2")
    assert detect_backend([mock]) == "torchvision"


# --- Multiple same-backend transforms ---


def test_multiple_kornia_transforms():
    m1 = _make_mock("kornia.augmentation._2d.geometric")
    m2 = _make_mock("kornia.augmentation._2d.intensity")
    assert detect_backend([m1, m2]) == "kornia"


# --- Mix of known + unknown -> known backend + warning ---


def test_known_plus_unknown_emits_warning_returns_known():
    kornia_mock = _make_mock("kornia.augmentation")
    unknown_mock = _make_mock("custom_lib.stuff")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = detect_backend([kornia_mock, unknown_mock])
    assert result == "kornia"
    assert len(w) == 1
    assert "Unrecognized transform" in str(w[0].message)
