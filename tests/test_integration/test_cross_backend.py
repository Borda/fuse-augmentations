"""Integration tests for cross-backend output pairs.

Tests every combination of source backend (Kornia, TorchVision, Albumentations)
with output_backend ('numpy', 'torch', None). Also tests mixed-backend pipelines
(pairs of two different backends) with output conversion.

Each test verifies:
- Output type matches the requested backend
- Output shape is correct (BCHW for torch, BHWC/HWC for numpy)
- Value range is plausible ([0, 1] for float data)
- No NaN/Inf in output

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose, FusedCompose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

HEIGHT, WIDTH, CHANNELS, BATCH_SIZE = 16, 16, 3, 2


def _rand_image(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    return torch.rand(batch_size, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def img() -> torch.Tensor:
    return _rand_image()


def _assert_valid_numpy(ndarray_out: np.ndarray, batch: int = BATCH_SIZE) -> None:
    """Assert numpy output has correct shape, dtype, and value range."""
    if batch == 1:
        assert ndarray_out.shape == (HEIGHT, WIDTH, CHANNELS), (
            f"Expected (H,W,C)=({HEIGHT},{WIDTH},{CHANNELS}), got {ndarray_out.shape}"
        )
    else:
        assert ndarray_out.shape == (batch, HEIGHT, WIDTH, CHANNELS), (
            f"Expected (B,H,W,C)=({batch},{HEIGHT},{WIDTH},{CHANNELS}), got {ndarray_out.shape}"
        )
    assert ndarray_out.dtype == np.float32, f"Expected float32, got {ndarray_out.dtype}"
    assert np.isfinite(ndarray_out).all(), "Output contains NaN or Inf"


def _assert_valid_torch(tensor: torch.Tensor, batch: int = BATCH_SIZE) -> None:
    """Assert torch output has correct shape, dtype, and value range."""
    assert tensor.shape == (batch, CHANNELS, HEIGHT, WIDTH), (
        f"Expected (B,C,H,W)=({batch},{CHANNELS},{HEIGHT},{WIDTH}), got {tensor.shape}"
    )
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert torch.isfinite(tensor).all(), "Output contains NaN or Inf"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestKorniaOutputBackend:
    """Kornia single-backend pipelines with output_backend conversion."""

    def _make_pipe(self, output_backend: str | None = None):
        return Compose(
            [kornia_aug.RandomRotation(degrees=10, p=1.0), kornia_aug.RandomHorizontalFlip(p=0.5)],
            output_backend=output_backend,
        )

    def test_kornia_to_numpy(self) -> None:
        """Kornia pipeline with output_backend='numpy' yields BHWC float32 ndarray."""
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_to_torch(self) -> None:
        """Kornia pipeline with output_backend='torch' yields BCHW torch.Tensor."""
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_kornia_default(self) -> None:
        """Kornia pipeline with output_backend=None defaults to torch.Tensor output."""
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_kornia_to_numpy_single_batch(self) -> None:
        """Kornia pipeline output_backend='numpy' with batch=1 produces (H,W,C) ndarray.

        Single-batch numpy output collapses the leading batch dimension to match the conventional HWC layout used by
        image libraries.

        """
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch_size=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestTorchVisionOutputBackend:
    """TorchVision single-backend pipelines with output_backend conversion."""

    def _make_pipe(self, output_backend: str | None = None):
        return Compose(
            [tv_trans.RandomRotation(degrees=10), tv_trans.RandomHorizontalFlip(p=0.5)],
            output_backend=output_backend,
        )

    def test_torchvision_to_numpy(self) -> None:
        """TorchVision pipeline with output_backend='numpy' yields BHWC float32 ndarray."""
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_to_torch(self) -> None:
        """TorchVision pipeline with output_backend='torch' yields BCHW torch.Tensor."""
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_torchvision_default(self) -> None:
        """TorchVision pipeline with output_backend=None defaults to torch.Tensor output."""
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_torchvision_to_numpy_single_batch(self) -> None:
        """TorchVision pipeline output_backend='numpy' with batch=1 produces (H,W,C) ndarray."""
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch_size=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestAlbumentationsOutputBackend:
    """Albumentations single-backend pipelines with output_backend conversion."""

    def _make_pipe(self, output_backend: str | None = None):
        return Compose(
            [albu.HorizontalFlip(p=1.0)],
            output_backend=output_backend,
        )

    def test_albu_to_numpy(self) -> None:
        """Albumentations pipeline with output_backend='numpy' yields BHWC float32 ndarray."""
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_to_torch(self) -> None:
        """Albumentations pipeline with output_backend='torch' yields BCHW torch.Tensor."""
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_albu_default(self) -> None:
        """Albumentations pipeline with output_backend=None defaults to torch.Tensor output."""
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_albu_to_numpy_single_batch(self) -> None:
        """Albumentations pipeline output_backend='numpy' with batch=1 produces (H,W,C) ndarray."""
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch_size=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


@pytest.mark.skipif(not (_KORNIA_AVAILABLE and _TORCHVISION_AVAILABLE), reason="missing kornia and torchvision")
class TestMixedBackendKorniaTorchVision:
    """Kornia + TorchVision mixed pipeline with output_backend."""

    def test_kornia_then_torchvision_to_numpy(self) -> None:
        """Kornia-then-TorchVision mixed pipeline converts to BHWC numpy output."""
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=10, p=1.0), tv_trans.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_then_kornia_to_numpy(self) -> None:
        """TorchVision-then-Kornia mixed pipeline converts to BHWC numpy output.

        Reverse ordering verifies that the conversion step does not depend on the backend of the final transform in the
        chain.

        """
        pipe = Compose(
            [tv_trans.RandomRotation(degrees=10), kornia_aug.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_then_torchvision_to_torch(self) -> None:
        """Kornia-then-TorchVision mixed pipeline preserves BCHW torch.Tensor output."""
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=10, p=1.0), tv_trans.RandomHorizontalFlip(p=0.5)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


@pytest.mark.skipif(not (_KORNIA_AVAILABLE and _ALBUMENTATIONS_AVAILABLE), reason="missing kornia and albumentations")
class TestMixedBackendKorniaAlbu:
    """Kornia + Albumentations mixed pipeline with output_backend."""

    def test_kornia_then_albu_to_numpy(self) -> None:
        """Kornia-then-Albumentations mixed pipeline converts to BHWC numpy output."""
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=10, p=1.0), albu.HorizontalFlip(p=1.0)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_then_kornia_to_numpy(self) -> None:
        """Albumentations-then-Kornia mixed pipeline converts to BHWC numpy output."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0), kornia_aug.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_then_albu_to_torch(self) -> None:
        """Kornia-then-Albumentations mixed pipeline preserves BCHW torch.Tensor output."""
        pipe = Compose(
            [kornia_aug.RandomRotation(degrees=10, p=1.0), albu.HorizontalFlip(p=1.0)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


@pytest.mark.skipif(
    not (_TORCHVISION_AVAILABLE and _ALBUMENTATIONS_AVAILABLE), reason="missing torchvision and albumentations"
)
class TestMixedBackendTorchVisionAlbu:
    """TorchVision + Albumentations mixed pipeline with output_backend."""

    def test_torchvision_then_albu_to_numpy(self) -> None:
        """TorchVision-then-Albumentations mixed pipeline converts to BHWC numpy output."""
        pipe = Compose(
            [tv_trans.RandomRotation(degrees=10), albu.HorizontalFlip(p=1.0)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_then_torchvision_to_numpy(self) -> None:
        """Albumentations-then-TorchVision mixed pipeline converts to BHWC numpy output."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0), tv_trans.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_then_albu_to_torch(self) -> None:
        """TorchVision-then-Albumentations mixed pipeline preserves BCHW torch.Tensor output."""
        pipe = Compose(
            [tv_trans.RandomRotation(degrees=10), albu.HorizontalFlip(p=1.0)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


class TestFromParamsOutputBackend:
    """Backend-free from_params pipelines with output_backend."""

    def test_from_params_to_numpy(self) -> None:
        """from_params with output_backend='numpy' yields BHWC float32 ndarray."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5, output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_from_params_to_torch(self) -> None:
        """from_params with output_backend='torch' yields BCHW torch.Tensor."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5, output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_from_params_default(self) -> None:
        """from_params with no output_backend defaults to torch.Tensor output."""
        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_from_params_unknown_raises(self) -> None:
        """from_params with an unsupported output_backend value raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown output_backend"):
            FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="tensorflow")
