"""Backward-compatibility tests for the deprecated `fuse_augmentations.data.shapes` import path."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from fuse_augmentations.data import geometry

_MODULE = "fuse_augmentations.data.shapes"

#: The names the released `fuse_augmentations.data.shapes` published, which the shim must keep
#: reachable. `animal_keypoints` is not among them: it moved to `fuse_augmentations.data.animals`,
#: not to the geometry module this shim forwards to.
_REEXPORTED = (
    "CIRCLE_POINTS",
    "GEOMETRIC_SHAPES",
    "RECT_ASPECT",
    "bbox_iou",
    "polygon_to_bbox_xyxy",
    "polygon_to_obb",
    "rotate_polygon",
    "shape_polygon",
)


@pytest.fixture(autouse=True)
def _restore_module_cache() -> object:
    """Put back whatever `sys.modules[_MODULE]` held before the test, once it's done.

    `_fresh_import` pops the cached entry to force a real re-import; left unrestored, every test in this file after the
    first leaves a different module object cached than what any earlier importer of the shim is holding a reference to.

    """
    original = sys.modules.get(_MODULE)
    yield
    if original is not None:
        sys.modules[_MODULE] = original
    else:
        sys.modules.pop(_MODULE, None)


def _fresh_import() -> object:
    """Import the shim from scratch so its import-time warning fires again.

    A module warns only on first import; without dropping the cached entry, every test after the first would see no
    warning and silently pass.

    """
    sys.modules.pop(_MODULE, None)
    return importlib.import_module(_MODULE)


def _import_quietly() -> object:
    """Import the shim fresh with its deprecation suppressed, for tests about what it re-exports."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _fresh_import()


def test_importing_the_shim_warns_and_points_at_geometry() -> None:
    """Importing the old path still works and says where the code moved to.

    The module was renamed to `geometry` after it had already shipped, so downstream code importing
    `fuse_augmentations.data.shapes` would break outright without this shim. The warning is what turns a silent alias
    into a migration signal.

    """
    with pytest.warns(DeprecationWarning, match="fuse_augmentations.data.geometry"):
        module = _fresh_import()

    assert module.__name__ == _MODULE


@pytest.mark.parametrize("name", [pytest.param(name, id=name) for name in _REEXPORTED])
def test_reexports_the_released_public_name(name: str) -> None:
    """Each name the released module published resolves to the very same object in `geometry`.

    Re-exporting a copy — or a renamed near-equivalent — would let the shim drift from the module it forwards to;
    identity is what proves the old path and the new one are one implementation.

    """
    module = _import_quietly()

    assert getattr(module, name) is getattr(geometry, name)


def test_from_import_of_a_moved_function_still_works() -> None:
    """`from fuse_augmentations.data.shapes import shape_polygon` keeps working and still warns.

    This is the exact statement downstream geometric workflows are written with — the reason the removal was a breaking
    change rather than dead-code cleanup — so it is asserted in its own form rather than only through `getattr`.

    """
    sys.modules.pop(_MODULE, None)

    with pytest.warns(DeprecationWarning, match="is deprecated"):
        from fuse_augmentations.data.shapes import shape_polygon

    assert shape_polygon is geometry.shape_polygon


def test_reexported_function_still_computes() -> None:
    """A re-exported function is callable through the old path and returns the geometry result.

    Name resolution alone would pass even if the shim forwarded something unusable, so the shim is exercised end to end
    on a real call.

    """
    module = _import_quietly()

    polygon = module.shape_polygon("square", center=(5.0, 5.0), size=4.0)

    assert module.polygon_to_bbox_xyxy(polygon) == (3.0, 3.0, 7.0, 7.0)


def test_declares_its_public_surface() -> None:
    """`__all__` lists exactly the re-exported names, so `import *` from the old path is unchanged.

    The shim's whole job is surface preservation; an `__all__` that drifted from what it imports would silently drop
    names for star-importers while attribute access kept working.

    """
    module = _import_quietly()

    assert sorted(module.__all__) == sorted(_REEXPORTED)
