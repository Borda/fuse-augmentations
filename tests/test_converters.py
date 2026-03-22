"""Tests for NumpyToTorchConverter and TorchToNumpyConverter (D.2)."""

from __future__ import annotations

import torch

np = __import__("pytest").importorskip("numpy")

from fuse_augmentations import BackendConverter  # noqa: E402
from fuse_augmentations._converters import NumpyToTorchConverter, TorchToNumpyConverter  # noqa: E402


class TestNumpyToTorchConverter:
    """Verify NumpyToTorchConverter layout and dtype conversion."""

    def test_hwc_to_1chw(self) -> None:
        arr = np.random.rand(16, 24, 3).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 16, 24)
        assert result.dtype == torch.float32

    def test_bhwc_to_bchw(self) -> None:
        arr = np.random.rand(4, 16, 24, 3).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3, 16, 24)

    def test_uint8_normalised(self) -> None:
        arr = np.full((8, 8, 3), 255, dtype=np.uint8)
        converter = NumpyToTorchConverter()
        result = converter.convert(arr)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.ones(1, 3, 8, 8))

    def test_isinstance_backend_converter(self) -> None:
        assert isinstance(NumpyToTorchConverter(), BackendConverter)

    def test_target_backend(self) -> None:
        assert NumpyToTorchConverter().target_backend == "torch"


class TestTorchToNumpyConverter:
    """Verify TorchToNumpyConverter layout conversion."""

    def test_1chw_to_hwc(self) -> None:
        tensor = torch.rand(1, 3, 16, 24)
        converter = TorchToNumpyConverter()
        result = converter.convert(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 24, 3)
        assert result.dtype == np.float32

    def test_bchw_to_bhwc(self) -> None:
        tensor = torch.rand(4, 3, 16, 24)
        converter = TorchToNumpyConverter()
        result = converter.convert(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 16, 24, 3)

    def test_isinstance_backend_converter(self) -> None:
        assert isinstance(TorchToNumpyConverter(), BackendConverter)

    def test_target_backend(self) -> None:
        assert TorchToNumpyConverter().target_backend == "numpy"
