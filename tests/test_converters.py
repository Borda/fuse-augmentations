"""Tests for NumpyToTorchConverter and TorchToNumpyConverter."""

from __future__ import annotations

import pytest
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

    def test_hwc_with_non_rgb_channel_count_round_trips(self) -> None:
        arr = np.random.rand(8, 8, 5).astype(np.float32)
        converter = NumpyToTorchConverter()
        result = converter.convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 5, 8, 8)

    def test_zero_channel_axis_raises(self) -> None:
        arr = np.empty((8, 8, 0), dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="non-empty channel axis"):
            converter.convert(arr)

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

    def test_multichannel_round_trip(self) -> None:
        tensor = torch.rand(1, 5, 8, 8)
        numpy_result = TorchToNumpyConverter().convert(tensor)
        torch_result = NumpyToTorchConverter().convert(numpy_result)
        torch.testing.assert_close(torch_result, tensor)

    def test_isinstance_backend_converter(self) -> None:
        assert isinstance(TorchToNumpyConverter(), BackendConverter)

    def test_target_backend(self) -> None:
        assert TorchToNumpyConverter().target_backend == "numpy"

    def test_3d_chw_input_raises_valueerror(self) -> None:
        """3-D tensor (C, H, W) raises ValueError — converter expects 4-D (B, C, H, W)."""
        converter = TorchToNumpyConverter()
        tensor_3d = torch.rand(3, 16, 16)
        with pytest.raises(ValueError, match="Expected 4-D tensor"):
            converter.convert(tensor_3d)


class TestNumpyToTorchConverterEdgeCases:
    """Edge-case dimensionality checks for NumpyToTorchConverter."""

    def test_1d_input_raises_valueerror(self) -> None:
        """1-D array raises ValueError — only 2-D/3-D/4-D are accepted."""
        arr = np.zeros(8, dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="Expected 2-D/3-D/4-D"):
            converter.convert(arr)

    def test_5d_input_raises_valueerror(self) -> None:
        """5-D array raises ValueError — only 2-D/3-D/4-D are accepted."""
        arr = np.zeros((2, 2, 8, 8, 3), dtype=np.float32)
        converter = NumpyToTorchConverter()
        with pytest.raises(ValueError, match="Expected 2-D/3-D/4-D"):
            converter.convert(arr)
