"""Regression tests: ``KorniaAdapter.category`` matches subclasses via ``isinstance`` (ADP-3).

The kornia adapter previously looked up categories with an exact ``type(...)`` lookup, so a user
subclass of a registered kornia transform fell through to the ``SPATIAL_KERNEL`` barrier (with a
spurious warning). It now uses an ``isinstance`` loop over the registry -- mirroring the TorchVision
and Albumentations adapters -- so subclasses inherit their parent's category.
"""

from __future__ import annotations

import warnings

import pytest

from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.adapters.kornia import KorniaAdapter
from fuse_augmentations.types import TransformCategory


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia not installed")
class TestKorniaCategorySubclass:
    """A subclass of a registered kornia transform inherits its parent's category."""

    def test_subclass_of_random_rotation_is_geometric_interp(self):
        """A subclass of ``RandomRotation`` categorizes as ``GEOMETRIC_INTERP`` without a barrier warning."""
        from kornia.augmentation import RandomRotation

        class MyRotation(RandomRotation):
            pass

        instance = MyRotation(degrees=30.0, p=1.0)
        with warnings.catch_warnings():
            # A barrier fall-through would emit a UserWarning; turn it into an error to assert none fires.
            warnings.simplefilter("error", UserWarning)
            cat = KorniaAdapter.category(instance)
        assert cat == TransformCategory.GEOMETRIC_INTERP

    def test_subclass_of_horizontal_flip_is_geometric_exact(self):
        """A subclass of ``RandomHorizontalFlip`` categorizes as ``GEOMETRIC_EXACT``."""
        from kornia.augmentation import RandomHorizontalFlip

        class MyHFlip(RandomHorizontalFlip):
            pass

        assert KorniaAdapter.category(MyHFlip(p=1.0)) == TransformCategory.GEOMETRIC_EXACT
