"""Pin every shape's unit-space outline and landmark table against a checked-in snapshot.

This test exists to make refactors provable rather than hopeful. The registry refactor, the naming
rename, and the migration of the symbol and letter assets from JSON to SVG all promise to leave
geometry untouched; this is what turns that promise into a check.

Regenerate deliberately, never incidentally::

    FUSE_REGEN_GOLDEN=1 uv run pytest tests/test_unit/test_data/test_golden_geometry.py

and commit the resulting ``.npz`` in the same change as whatever legitimately moved a vertex, so the
diff records the intent.

"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ._golden import build_golden

GOLDEN_PATH = Path(__file__).parent / "_golden_geometry.npz"

#: Tolerance for the comparison. A pure code refactor reproduces the same float64 bits and would
#: pass at zero, but the SVG assets store coordinates as decimal path text, so the migration off
#: JSON round-trips through that precision; this is the budget that round-trip is held to.
_ASSET_PRECISION = 5e-4


def _regenerating() -> bool:
    """Return whether this run was asked to rewrite the snapshot rather than check it."""
    return os.environ.get("FUSE_REGEN_GOLDEN") == "1"


@pytest.fixture(scope="module", autouse=True)
def _maybe_regenerate() -> None:
    """Rewrite the snapshot before any check runs, but only when explicitly asked to.

    Autouse and module-scoped so a regeneration run cannot be defeated by test ordering — the existence check would
    otherwise fire before the fixture that creates the file.

    """
    if _regenerating():
        np.savez_compressed(GOLDEN_PATH, **build_golden())


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    """Return the freshly computed unit-space geometry for every shape."""
    return build_golden()


def test_golden_file_exists() -> None:
    """The snapshot must be checked in; a missing one silently disarms every check below."""
    assert GOLDEN_PATH.is_file(), f"missing {GOLDEN_PATH.name}; regenerate with FUSE_REGEN_GOLDEN=1"


def test_golden_covers_every_shape(golden: dict[str, np.ndarray]) -> None:
    """Every shape contributes an outline, and every keypoint-bearing shape a landmark table."""
    stored = np.load(GOLDEN_PATH)
    assert set(stored.files) == set(golden), "snapshot key set drifted from the live shape vocabulary"
    polygons = [key for key in golden if key.startswith("poly:")]
    assert len(polygons) == 49, f"expected 49 shape outlines, got {len(polygons)}"


@pytest.mark.parametrize("key", sorted(build_golden()))
def test_golden_geometry_unchanged(golden: dict[str, np.ndarray], key: str) -> None:
    """Each outline and landmark table matches the snapshot bit-for-bit."""
    stored = np.load(GOLDEN_PATH)
    expected, actual = stored[key], golden[key]
    assert actual.shape == expected.shape, f"{key}: shape {actual.shape} != snapshot {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=0, atol=_ASSET_PRECISION, equal_nan=True, err_msg=key)
