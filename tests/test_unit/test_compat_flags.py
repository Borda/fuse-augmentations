"""Unit tests for the optional-backend availability flags in ``_compat``.

Each flag (``_KORNIA_AVAILABLE``, ``_TORCHVISION_AVAILABLE``, ``_ALBUMENTATIONS_AVAILABLE``) is set once, at module
import time, by a narrow ``except ImportError`` clause. In a normal dev/CI environment every backend is installed,
so that branch never runs and is invisible to coverage. These tests force the branch by making the backend's
``import`` statement raise ``ImportError`` (via ``sys.modules[name] = None``, the standard technique for simulating
a missing module) and reloading ``_compat``, then restore the real module state so later tests are unaffected.

"""

from __future__ import annotations

import importlib
import sys

import pytest

from fuse_augmentations import _compat


@pytest.mark.parametrize(
    ("module_name", "flag_name"),
    [
        pytest.param("kornia", "_KORNIA_AVAILABLE", id="kornia"),
        pytest.param("torchvision", "_TORCHVISION_AVAILABLE", id="torchvision"),
        pytest.param("albumentations", "_ALBUMENTATIONS_AVAILABLE", id="albumentations"),
    ],
)
def test_import_error_flips_flag_false_without_raising(module_name: str, flag_name: str) -> None:
    """Forcing ``ImportError`` on an optional backend flips its flag to False and does not raise.

    ``importlib.reload`` completing without an exception is itself the "package still imports/degrades" assertion:
    the narrow ``except ImportError`` catches exactly the forced failure and the module falls through to
    ``<flag> = False`` instead of propagating. Restoration happens manually (not via a fixture) so the reload back
    to real state runs before this test returns, independent of any other fixture's teardown order.

    """
    original = sys.modules.get(module_name)
    sys.modules[module_name] = None
    try:
        importlib.reload(_compat)
        assert getattr(_compat, flag_name) is False
    finally:
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original
        importlib.reload(_compat)
