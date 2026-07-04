"""Smoke tests pinning the private backend internals the adapters depend on.

The adapters bind to undocumented backend APIs (Albumentations ``get_params_dependent_on_data`` result keys,
Kornia ``transform._params``, TorchVision ``transforms.v2`` module layout). These can silently change across
backend minor versions; each test here fails fast in CI on a version bump instead of at user runtime.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations not installed")
class TestAlbumentationsInternals:
    """Pin the get_params / update_*_params / get_params_dependent_on_data keys used by _sample_* helpers."""

    @staticmethod
    def _dependent_params(transform, height: int = 8, width: int = 8) -> dict:
        """Replicate the adapter's internal-param invocation sequence for one draw."""
        data = {"image": np.zeros((height, width, 1), dtype=np.float32)}
        base = transform.get_params()
        if hasattr(transform, "update_transform_params"):
            base = transform.update_transform_params(base, data)
        else:
            base = transform.update_params(base, **data)
        return transform.get_params_dependent_on_data(base, data)

    def test_affine_returns_matrix_key(self):
        """Affine's dependent params contain the 'matrix' key consumed by _sample_matrices."""
        from albumentations import Affine

        params = self._dependent_params(Affine(rotate=(-30, 30), p=1.0))
        assert "matrix" in params
        assert np.asarray(params["matrix"]).shape == (3, 3)

    def test_random_resized_crop_returns_crop_coords_key(self):
        """RandomResizedCrop's dependent params contain 'crop_coords' consumed by _sample_crop_resize_params."""
        from albumentations import RandomResizedCrop

        params = self._dependent_params(RandomResizedCrop(size=(4, 4), p=1.0))
        assert "crop_coords" in params

    def test_brightness_contrast_returns_alpha_beta_keys(self):
        """RandomBrightnessContrast's dependent params contain 'alpha' and 'beta' used by _sample_color_params."""
        from albumentations import RandomBrightnessContrast

        params = self._dependent_params(RandomBrightnessContrast(p=1.0), height=4, width=4)
        assert "alpha" in params
        assert "beta" in params

    def test_rotate90_returns_factor_key(self):
        """RandomRotate90.get_params returns the 'factor' key used for k90 sampling."""
        from albumentations import RandomRotate90

        assert "factor" in RandomRotate90(p=1.0).get_params()

    def test_d4_returns_group_element_key(self):
        """D4.get_params returns the 'group_element' key used for d4_code sampling."""
        from albumentations import D4

        assert "group_element" in D4(p=1.0).get_params()


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia not installed")
class TestKorniaInternals:
    """Pin transform._params population that convert_native_params reads after a native forward."""

    def test_params_populated_after_forward(self):
        """RandomRotation populates _params with a 'degrees' tensor after a forward pass."""
        from kornia.augmentation import RandomRotation

        transform = RandomRotation(degrees=30.0, p=1.0)
        transform(torch.rand(2, 3, 8, 8))
        assert hasattr(transform, "_params")
        assert "degrees" in transform._params


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision not installed")
class TestTorchVisionInternals:
    """Pin the v2 detection contract used by is_torchvision_v2_transform."""

    def test_v2_module_prefix_detection(self):
        """V2 transforms are detected via the torchvision.transforms.v2 module prefix (primary check)."""
        from torchvision.transforms.v2 import RandomRotation

        from fuse_augmentations.adapters.torchvision import is_torchvision_v2_transform

        assert is_torchvision_v2_transform(RandomRotation(degrees=30))

    def test_v1_not_detected_as_v2(self):
        """V1 transforms are not classified as v2."""
        from torchvision.transforms import RandomRotation

        from fuse_augmentations.adapters.torchvision import is_torchvision_v2_transform

        assert not is_torchvision_v2_transform(RandomRotation(degrees=30))
