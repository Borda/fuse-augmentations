"""Tests for output_backend parameter on FusedCompose."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import FusedCompose


class TestOutputBackend:
    """Verify output_backend parameter converts pipeline output correctly."""

    def test_default_returns_tensor(self) -> None:
        """output_backend=None (default) returns torch.Tensor — regression test."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0))
        image = torch.rand(1, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, torch.Tensor)

    def test_numpy_returns_ndarray(self) -> None:
        """output_backend='numpy' returns ndarray."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="numpy")
        image = torch.rand(1, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, np.ndarray)

    def test_numpy_hwc_alias_returns_ndarray(self) -> None:
        """output_backend='numpy_hwc' uses the same NumPy channel-last conversion."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="numpy_hwc")
        image = torch.rand(1, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16, 3)

    def test_torch_returns_tensor(self) -> None:
        """output_backend='torch' returns torch.Tensor (identity)."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="torch")
        image = torch.rand(1, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, torch.Tensor)

    def test_unknown_raises_valueerror(self) -> None:
        """Unknown output_backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output_backend"):
            FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="jax")

    def test_numpy_output_shape_bhwc(self) -> None:
        """output_backend='numpy' produces (batch_size, height, width, channels) for batch > 1."""
        pipe = FusedCompose.from_params(rotation=(-5.0, 5.0), output_backend="numpy")
        image = torch.rand(4, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 16, 16, 3)

    def test_numpy_output_shape_hwc_single(self) -> None:
        """output_backend='numpy' produces (height, width, channels) for batch == 1."""
        pipe = FusedCompose.from_params(rotation=(-5.0, 5.0), output_backend="numpy")
        image = torch.rand(1, 3, 16, 16)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16, 3)

    def test_from_params_with_output_backend(self) -> None:
        """Real pipeline: from_params with output_backend='numpy' returns ndarray."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="numpy")
        image = torch.rand(2, 3, 32, 32)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 32, 32, 3)

    def test_empty_pipeline_numpy(self) -> None:
        """Empty pipeline with output_backend='numpy' still converts."""
        pipe = FusedCompose([], output_backend="numpy")
        image = torch.rand(1, 3, 8, 8)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8, 3)

    def test_empty_pipeline_numpy_from_chw_input(self) -> None:
        """Unbatched CHW tensors convert to HWC when output_backend='numpy'."""
        pipe = FusedCompose([], output_backend="numpy")
        image = torch.rand(3, 8, 8)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8, 3)

    def test_empty_pipeline_default(self) -> None:
        """Empty pipeline with default output_backend returns Tensor."""
        pipe = FusedCompose([])
        image = torch.rand(1, 3, 8, 8)
        result = pipe(image)
        assert isinstance(result, torch.Tensor)

    def test_single_data_key_converts_without_warning(self) -> None:
        """output_backend='numpy' + data_keys=['input'] (single key) converts and does NOT warn."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pipe = FusedCompose([], output_backend="numpy", data_keys=["input"])

        image = torch.rand(1, 3, 8, 8)
        result = pipe(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8, 3)

    def test_output_backend_with_data_keys_warns(self) -> None:
        """output_backend + data_keys emits UserWarning at construction time."""
        with pytest.warns(UserWarning, match="output_backend.*data_keys"):
            FusedCompose([], output_backend="numpy", data_keys=["input", "mask"])

    def test_output_backend_with_data_keys_no_conversion(self) -> None:
        """When data_keys is set, output_backend conversion is NOT applied (no-op)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipe = FusedCompose([], output_backend="numpy", data_keys=["input", "mask"])

        image = torch.rand(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        result = pipe(image, mask)
        # Multi-key pipeline returns tuple of raw tensors regardless of output_backend.
        assert isinstance(result, tuple)
        assert all(isinstance(tensor, torch.Tensor) for tensor in result)
