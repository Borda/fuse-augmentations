"""Tests for the BackendConverter protocol (D.1)."""

from __future__ import annotations

from typing import Any

import torch

import fuse_augmentations
from fuse_augmentations import BackendConverter, FusedCompose


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


class _IdentityConverter:
    """Custom BackendConverter that returns the tensor unchanged (identity)."""

    def convert(self, tensor: Any) -> Any:
        return tensor

    @property
    def target_backend(self) -> str:
        return "torch"


class TestBackendConverterExamples:
    """Protocol-compliant converters remain usable outside FusedCompose internals."""

    def test_custom_converter_is_backend_converter(self) -> None:
        assert isinstance(_IdentityConverter(), BackendConverter)

    def test_custom_converter_round_trips_tensor(self) -> None:
        """A custom identity converter leaves the tensor unchanged."""
        x = torch.rand(1, 3, 8, 8)
        result = _IdentityConverter().convert(x)
        assert result is x

    def test_fused_compose_torch_backend_uses_identity_path(self) -> None:
        """FusedCompose with output_backend='torch' keeps the native tensor output."""
        pipe = FusedCompose([], output_backend="torch")
        x = torch.rand(1, 3, 8, 8)
        result = pipe(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
