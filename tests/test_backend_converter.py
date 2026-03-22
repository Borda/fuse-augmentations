"""Tests for the BackendConverter protocol (D.1)."""

from __future__ import annotations

from typing import Any

import fuse_augmentations
from fuse_augmentations import BackendConverter


class _ValidConverter:
    """Concrete class satisfying BackendConverter."""

    def convert(self, tensor: Any) -> Any:
        return tensor

    @property
    def target_backend(self) -> str:
        return "test"


class _MissingConvert:
    """Class missing convert() — should NOT satisfy BackendConverter."""

    @property
    def target_backend(self) -> str:
        return "test"


class TestBackendConverterProtocol:
    """Verify the BackendConverter protocol definition and runtime checks."""

    def test_importable_from_package(self) -> None:
        assert hasattr(fuse_augmentations, "BackendConverter")

    def test_in_all(self) -> None:
        assert "BackendConverter" in fuse_augmentations.__all__

    def test_isinstance_valid(self) -> None:
        converter = _ValidConverter()
        assert isinstance(converter, BackendConverter)

    def test_isinstance_missing_convert_fails(self) -> None:
        obj = _MissingConvert()
        assert not isinstance(obj, BackendConverter)

    def test_importable_from_fuse_aug(self) -> None:
        from fuse_aug import BackendConverter as BC

        assert BC is BackendConverter
