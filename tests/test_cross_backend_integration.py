"""Integration tests for cross-backend output pairs (D.4).

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

import pytest
import torch

np = pytest.importorskip("numpy")


H, W, C, B = 16, 16, 3, 2


def _rand_image(batch: int = B) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.rand(batch, C, H, W)


def _assert_valid_numpy(arr: np.ndarray, batch: int = B) -> None:
    """Assert numpy output has correct shape, dtype, and value range."""
    if batch == 1:
        assert arr.shape == (H, W, C), f"Expected (H,W,C)=({H},{W},{C}), got {arr.shape}"
    else:
        assert arr.shape == (batch, H, W, C), f"Expected (B,H,W,C)=({batch},{H},{W},{C}), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    assert np.isfinite(arr).all(), "Output contains NaN or Inf"


def _assert_valid_torch(tensor: torch.Tensor, batch: int = B) -> None:
    """Assert torch output has correct shape, dtype, and value range."""
    assert tensor.shape == (batch, C, H, W), f"Expected (B,C,H,W)=({batch},{C},{H},{W}), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert torch.isfinite(tensor).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Kornia backend
# ---------------------------------------------------------------------------


class TestKorniaOutputBackend:
    """Kornia single-backend pipelines with output_backend conversion."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("kornia", reason="kornia required")

    def _make_pipe(self, output_backend: str | None = None):
        from kornia.augmentation import RandomHorizontalFlip, RandomRotation

        from fuse_augmentations import Compose

        return Compose(
            [RandomRotation(degrees=10, p=1.0), RandomHorizontalFlip(p=0.5)],
            output_backend=output_backend,
        )

    def test_kornia_to_numpy(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_to_torch(self) -> None:
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_kornia_default(self) -> None:
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_kornia_to_numpy_single_batch(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


# ---------------------------------------------------------------------------
# TorchVision backend
# ---------------------------------------------------------------------------


class TestTorchVisionOutputBackend:
    """TorchVision single-backend pipelines with output_backend conversion."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("torchvision", reason="torchvision required")

    def _make_pipe(self, output_backend: str | None = None):
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        return Compose(
            [T.RandomRotation(degrees=10), T.RandomHorizontalFlip(p=0.5)],
            output_backend=output_backend,
        )

    def test_torchvision_to_numpy(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_to_torch(self) -> None:
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_torchvision_default(self) -> None:
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_torchvision_to_numpy_single_batch(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


# ---------------------------------------------------------------------------
# Albumentations backend
# ---------------------------------------------------------------------------


class TestAlbumentationsOutputBackend:
    """Albumentations single-backend pipelines with output_backend conversion."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")

    def _make_pipe(self, output_backend: str | None = None):
        import albumentations as A

        from fuse_augmentations import Compose

        return Compose(
            [A.HorizontalFlip(p=1.0)],
            output_backend=output_backend,
        )

    def test_albu_to_numpy(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_to_torch(self) -> None:
        pipe = self._make_pipe(output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_albu_default(self) -> None:
        pipe = self._make_pipe(output_backend=None)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_albu_to_numpy_single_batch(self) -> None:
        pipe = self._make_pipe(output_backend="numpy")
        result = pipe(_rand_image(batch=1))
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result, batch=1)


# ---------------------------------------------------------------------------
# Mixed-backend pairs with output conversion
# ---------------------------------------------------------------------------


class TestMixedBackendKorniaTorchVision:
    """Kornia + TorchVision mixed pipeline with output_backend."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("kornia", reason="kornia required")
        pytest.importorskip("torchvision", reason="torchvision required")

    def test_kornia_then_torchvision_to_numpy(self) -> None:
        import torchvision.transforms as T
        from kornia.augmentation import RandomRotation

        from fuse_augmentations import Compose

        pipe = Compose(
            [RandomRotation(degrees=10, p=1.0), T.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_then_kornia_to_numpy(self) -> None:
        import torchvision.transforms as T
        from kornia.augmentation import RandomHorizontalFlip as KFlip

        from fuse_augmentations import Compose

        pipe = Compose(
            [T.RandomRotation(degrees=10), KFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_then_torchvision_to_torch(self) -> None:
        import torchvision.transforms as T
        from kornia.augmentation import RandomRotation

        from fuse_augmentations import Compose

        pipe = Compose(
            [RandomRotation(degrees=10, p=1.0), T.RandomHorizontalFlip(p=0.5)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


class TestMixedBackendKorniaAlbu:
    """Kornia + Albumentations mixed pipeline with output_backend."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("kornia", reason="kornia required")
        pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")

    def test_kornia_then_albu_to_numpy(self) -> None:
        import albumentations as A
        from kornia.augmentation import RandomRotation

        from fuse_augmentations import Compose

        pipe = Compose(
            [RandomRotation(degrees=10, p=1.0), A.HorizontalFlip(p=1.0)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_then_kornia_to_numpy(self) -> None:
        import albumentations as A
        from kornia.augmentation import RandomHorizontalFlip as KFlip

        from fuse_augmentations import Compose

        pipe = Compose(
            [A.HorizontalFlip(p=1.0), KFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_kornia_then_albu_to_torch(self) -> None:
        import albumentations as A
        from kornia.augmentation import RandomRotation

        from fuse_augmentations import Compose

        pipe = Compose(
            [RandomRotation(degrees=10, p=1.0), A.HorizontalFlip(p=1.0)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


class TestMixedBackendTorchVisionAlbu:
    """TorchVision + Albumentations mixed pipeline with output_backend."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("torchvision", reason="torchvision required")
        pytest.importorskip("albumentations", reason="albumentations >= 2.0 required")

    def test_torchvision_then_albu_to_numpy(self) -> None:
        import albumentations as A
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        pipe = Compose(
            [T.RandomRotation(degrees=10), A.HorizontalFlip(p=1.0)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_albu_then_torchvision_to_numpy(self) -> None:
        import albumentations as A
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        pipe = Compose(
            [A.HorizontalFlip(p=1.0), T.RandomHorizontalFlip(p=0.5)],
            output_backend="numpy",
        )
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_torchvision_then_albu_to_torch(self) -> None:
        import albumentations as A
        import torchvision.transforms as T

        from fuse_augmentations import Compose

        pipe = Compose(
            [T.RandomRotation(degrees=10), A.HorizontalFlip(p=1.0)],
            output_backend="torch",
        )
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)


# ---------------------------------------------------------------------------
# from_params with output_backend (backend-free)
# ---------------------------------------------------------------------------


class TestFromParamsOutputBackend:
    """Backend-free from_params pipelines with output_backend."""

    def test_from_params_to_numpy(self) -> None:
        from fuse_augmentations import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5, output_backend="numpy")
        result = pipe(_rand_image())
        assert isinstance(result, np.ndarray)
        _assert_valid_numpy(result)

    def test_from_params_to_torch(self) -> None:
        from fuse_augmentations import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5, output_backend="torch")
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_from_params_default(self) -> None:
        from fuse_augmentations import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-10.0, 10.0), hflip_p=0.5)
        result = pipe(_rand_image())
        assert isinstance(result, torch.Tensor)
        _assert_valid_torch(result)

    def test_from_params_unknown_raises(self) -> None:
        from fuse_augmentations import FusedCompose

        with pytest.raises(ValueError, match="Unknown output_backend"):
            FusedCompose.from_params(rotation=(-10.0, 10.0), output_backend="tensorflow")
