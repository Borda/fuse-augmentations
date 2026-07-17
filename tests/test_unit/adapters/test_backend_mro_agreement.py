"""Regression tests: ``detect_backend`` and ``detect_backends_per_transform`` agree via MRO (CORE-5).

``detect_backend`` previously matched only a transform's direct ``__module__`` prefix, while
``detect_backends_per_transform`` additionally walked the MRO. A user subclass of a backend
transform (defined outside the backend package) therefore resolved differently between the two.
Both now share the same module-plus-MRO classification helper and agree.
"""

from __future__ import annotations

import pytest

from fuse_augmentations._backend import Backend, detect_backend, detect_backends_per_transform
from fuse_augmentations._compat import _KORNIA_AVAILABLE


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia not installed")
class TestDetectBackendSubclassMRO:
    """A user subclass of a kornia transform resolves to ``KORNIA`` in both detect functions."""

    @staticmethod
    def _make_subclass() -> object:
        """Return an instance of a locally-defined subclass of a kornia transform.

        The subclass's own ``__module__`` is this test module (not ``kornia.*``), so a correct
        resolution must come from walking its MRO to the kornia ancestor.

        """
        from kornia.augmentation import RandomRotation

        class MyRotation(RandomRotation):
            pass

        return MyRotation(degrees=30.0, p=1.0)

    def test_detect_backend_resolves_subclass_via_mro(self):
        """``detect_backend`` resolves a kornia subclass to ``KORNIA`` via its MRO."""
        sub = self._make_subclass()
        assert detect_backend([sub]) == Backend.KORNIA

    def test_both_detectors_agree_on_subclass(self):
        """``detect_backend`` and ``detect_backends_per_transform`` return the same backend for a subclass."""
        sub = self._make_subclass()
        assert detect_backend([sub]) == detect_backends_per_transform([sub])[0]
