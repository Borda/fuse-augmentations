"""Tests for _backend.py -- detect_backend."""

import warnings

import pytest

from fuse_augmentations._backend import Backend, detect_backend


def _make_mock(module_path: str):
    """Create a mock object whose type has a specific __module__."""
    namespace = {"__module__": module_path, "__qualname__": "MockTransform"}
    mock_class = type("MockTransform", (), namespace)
    return mock_class()


class TestMixedBackends:
    """detect_backend raises on mixed backends."""

    def test_raises_value_error(self):
        """Mixing kornia + torchvision raises ValueError."""
        kornia_mock = _make_mock("kornia.augmentation")
        tv_mock = _make_mock("torchvision.transforms")
        with pytest.raises(ValueError, match="Mixed backends"):
            detect_backend([kornia_mock, tv_mock])


class TestUnknownBackends:
    """detect_backend behaviour with empty or unrecognized transforms."""

    def test_empty_list_returns_unknown(self):
        """Empty transform list returns Backend.UNKNOWN."""
        assert detect_backend([]) == Backend.UNKNOWN

    def test_all_unknown_returns_unknown_with_warning(self):
        """All-unknown transforms return Backend.UNKNOWN and emit a warning."""
        unknown_mock = _make_mock("my_custom_lib.transforms")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = detect_backend([unknown_mock])
        assert result == Backend.UNKNOWN
        assert len(record) == 1
        assert "Unrecognized transform" in str(record[0].message)

    def test_known_plus_unknown_emits_warning_returns_known(self):
        """Known + unknown mix returns the known Backend enum member with a warning."""
        kornia_mock = _make_mock("kornia.augmentation")
        unknown_mock = _make_mock("custom_lib.stuff")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = detect_backend([kornia_mock, unknown_mock])
        assert result == Backend.KORNIA
        assert len(record) == 1
        assert "Unrecognized transform" in str(record[0].message)


class TestSingleKnownBackends:
    """detect_backend identifies each supported single backend."""

    @pytest.mark.parametrize(
        "module_path, expected",
        [
            ("kornia.augmentation._2d", Backend.KORNIA),
            ("albumentations.augmentations.transforms", Backend.ALBUMENTATIONS),
            ("torchvision.transforms.v2", Backend.TORCHVISION),
        ],
    )
    def test_single_backend(self, module_path, expected):
        """Single transform from a known backend returns the corresponding Backend member."""
        mock = _make_mock(module_path)
        assert detect_backend([mock]) == expected

    def test_multiple_kornia_transforms(self):
        """Multiple kornia transforms still detected as Backend.KORNIA."""
        mock1 = _make_mock("kornia.augmentation._2d.geometric")
        mock2 = _make_mock("kornia.augmentation._2d.intensity")
        assert detect_backend([mock1, mock2]) == Backend.KORNIA
