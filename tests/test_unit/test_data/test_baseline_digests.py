"""Hold the generator's emitted pixels and labels against a checked-in digest.

The snapshot in ``_baseline_digests.json`` was taken before any background, degradation, distractor
or occluder knob existed. Every one of those knobs claims to be inert when unset; this is the file
that can refuse the claim, because it compares against bytes committed from code that predates the
knob rather than against a second configuration built from the same tree.

Regenerate deliberately, never incidentally::

    FUSE_REGEN_BASELINE=1 uv run pytest tests/test_unit/test_data/test_baseline_digests.py

and commit the rewritten JSON in the same change as whatever legitimately moved a byte, so the diff
records the intent. A regeneration that lands on its own is indistinguishable from a silent
regression being papered over.

"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fuse_augmentations.data.config import SyntheticConfig

from ._baseline import _FEATURE_MATRIX, _MATRIX, baseline_names, build_baseline

BASELINE_PATH = Path(__file__).parent / "_baseline_digests.json"

#: Relative skew applied to `np.cos`/`np.sin` by the portability check below. The whole ladder, in
#: one place: drift between two floating-point environments is around `1e-16`; this skew is `1e-12`,
#: four orders above it; the measured flip threshold for this matrix sits two orders higher again,
#: between `1e-10` (no digest moves) and `1e-9` (two of twenty-five move); and no skew moves a pixel
#: at all, up to `1e-6` where the sweep stopped.
_PERTURBATION = 1e-12


def _regenerating() -> bool:
    """Return whether this run was asked to rewrite the snapshot rather than check it."""
    return os.environ.get("FUSE_REGEN_BASELINE") == "1"


@pytest.fixture(scope="module", autouse=True)
def _maybe_regenerate() -> None:
    """Rewrite the snapshot before any check runs, but only when explicitly asked to.

    Autouse and module-scoped for the same reason as the geometry snapshot's equivalent: the existence check would
    otherwise fire before the fixture that creates the file.

    """
    if _regenerating():
        BASELINE_PATH.write_text(json.dumps(build_baseline(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def live() -> dict[str, str]:
    """Return the digests the current code produces for the pinned configuration matrix."""
    return build_baseline()


@pytest.fixture(scope="module")
def stored() -> dict[str, str]:
    """Return the digests committed alongside this test."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_baseline_file_exists() -> None:
    """The snapshot must be checked in; a missing one silently disarms every check below."""
    assert BASELINE_PATH.is_file(), f"missing {BASELINE_PATH.name}; regenerate with FUSE_REGEN_BASELINE=1"


def test_the_two_matrices_stay_separate() -> None:
    """No configuration in the pre-feature matrix may set a knob added after it was snapshotted.

    The two halves of this file carry different guarantees and only stay distinguishable while that holds. `_MATRIX` was
    snapshotted before backgrounds, degradations, distractors and occluders existed, so its digests prove those knobs
    are *inert when unset* — a claim no entry generated from the current code can make. `_FEATURE_MATRIX` was generated
    from the code that implements them, so its digests prove *stability*: one moving means a rendering change, and a
    reviewer decides whether it was intended.

    Adding a knob-setting configuration to `_MATRIX` would quietly downgrade it to the weaker guarantee while it still
    read as the stronger one.

    """
    untouched = SyntheticConfig(img_size=8)
    for name, (config, _seed) in _MATRIX.items():
        assert config.background == untouched.background, name
        assert config.degrade == (), name
        assert config.distractors == 0, name
        assert config.occluders == 0, name

    assert set(_MATRIX) & set(_FEATURE_MATRIX) == set()


def test_baseline_covers_the_whole_matrix(live: dict[str, str], stored: dict[str, str]) -> None:
    """The snapshot names exactly the configurations the matrix builds, no more and no fewer."""
    assert sorted(stored) == sorted(live), "snapshot key set drifted from the configuration matrix"


def test_a_floating_point_perturbation_moves_no_digest(monkeypatch: pytest.MonkeyPatch, live: dict[str, str]) -> None:
    """Skewing the transcendentals the geometry is built from must leave every digest where it was.

    This is the test that keeps the snapshot portable, and it exists because the first version of this file was not.
    Shape outlines come from `np.cos`/`np.sin` — `primitives.py` builds every vertex from them and `geometry.py` builds
    every placement rotation from them — and numpy promises no bitwise-identical result for either across SIMD paths and
    library builds. A snapshot hashing label floats at full `float64` precision therefore pinned the floating-point
    environment that wrote it: taken on one developer machine, it failed on all 24 CI jobs at once — ubuntu 3.10 through
    3.14, macOS, Windows and the oldest-dependency job alike — with 19 of 25 digests moved in each.

    Quantizing the labels buys headroom rather than a proof: a value sitting near a grid boundary still crosses it if
    pushed hard enough, and with 47306 label values in the matrix a few always sit close. The headroom is measured, and
    `_PERTURBATION` states the ladder. If this test starts failing right after entries are added to the matrix, the
    first suspicion is one new label value sitting near a grid boundary rather than a portability regression — at this
    skew the expected number of crossings is around `0.03` and it grows with the matrix, so the remedy is dropping
    `_PERTURBATION` one order, not loosening the grid.

    Pixels need none of that: a `1e-6` skew, three orders larger again, moves no pixel anywhere in the matrix, because
    the rasterizer snaps a vertex to a grid far coarser than any drift. That is why the image half of the digest stays
    exact while the label half cannot.

    """
    real_cos, real_sin = np.cos, np.sin

    def skewed_cos(values: object) -> object:
        return real_cos(values) * (1.0 + _PERTURBATION)

    def skewed_sin(values: object) -> object:
        return real_sin(values) * (1.0 + _PERTURBATION)

    monkeypatch.setattr(np, "cos", skewed_cos)
    monkeypatch.setattr(np, "sin", skewed_sin)

    assert build_baseline() == live


@pytest.mark.parametrize("name", sorted(baseline_names()))
def test_generated_bytes_are_unchanged(name: str, live: dict[str, str], stored: dict[str, str]) -> None:
    """Each configuration still renders and labels exactly what it rendered and labelled before.

    A failure here is a finding, not a number to refresh: read what moved first, decide whether the move was intended,
    and only then regenerate.

    """
    assert live[name] == stored[name], f"{name}: generated bytes or labels moved against the pinned baseline"
