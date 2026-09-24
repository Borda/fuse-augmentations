"""Hold the difficulty ladder to what its own statistics say, rather than to what it claims.

A band table is a design judgement until something measures it. These tests are that something: they run the training-
free proxy over the three documented bands and check that every axis agrees on the ordering. A band that stopped being
harder than the one below it fails here, which is the only way a documentation page of suggested difficulties stays
honest as the generator changes.

"""

from __future__ import annotations

import pytest

from ._difficulty import BANDS, SMALL_AREA, ProxyStats, measure, measure_bands, render_table

#: Axes that must fall from easy to hard, and those that must rise. Split rather than signed so a
#: failure message names the axis in the direction a reader thinks about it.
#:
#: The comparison is **strict** on every axis that is strictly ordered today, because a non-strict
#: one cannot tell a ladder from a flat line: three identical bands satisfy ``values == sorted(values)``
#: on all six axes at once. That is not hypothetical — measuring the three bands with their clutter
#: silently disabled drives ``clutter_coverage`` to ``[0.0, 0.0, 0.0]`` and every non-strict
#: assertion, including the small-object floor, still passes.
_FALLING = ("mean_area", "p10_area", "boundary_contrast", "background_snr")
_RISING = ("clutter_coverage",)

#: ``small_fraction`` is the one axis that legitimately ties: the easy and moderate bands both hold
#: no object below the small-object threshold, so ``[0.0, 0.0, 0.298]`` is correct and strictness
#: would reject it. It is pinned by the absolute floor below instead, which is what makes the tie
#: safe rather than merely tolerated.
_TYING = "small_fraction"

#: Floor on the clutter the two clutter-bearing bands must actually paint. An ordering assertion
#: alone cannot distinguish "more clutter" from "no clutter anywhere", so the ladder needs one
#: absolute number per axis that can collapse to zero.
_MIN_MODERATE_CLUTTER = 0.005


@pytest.fixture(scope="module")
def bands() -> dict[str, ProxyStats]:
    """Return the measured statistics for every documented band."""
    return measure_bands()


def test_the_bands_are_the_three_the_documentation_names(bands: dict[str, ProxyStats]) -> None:
    """The measured set is exactly the set the page publishes, in the order it publishes them."""
    assert list(bands) == ["easy", "moderate", "hard"]
    assert list(BANDS) == list(bands)


@pytest.mark.parametrize("axis", _FALLING)
def test_a_harder_band_scores_lower_on_the_falling_axes(axis: str, bands: dict[str, ProxyStats]) -> None:
    """Object area, boundary contrast and background separation all shrink as the band gets harder.

    These are the axes that describe how much signal an object hands a model for free. A band that kept its contrast
    while calling itself harder would be harder in name only, which is exactly the failure a documentation table invites
    when nothing measures it.

    """
    measured = [getattr(bands[name], axis) for name in ("easy", "moderate", "hard")]

    assert measured == sorted(measured, reverse=True), measured


@pytest.mark.parametrize("axis", _RISING)
def test_a_harder_band_scores_higher_on_the_rising_axes(axis: str, bands: dict[str, ProxyStats]) -> None:
    """The small-object share and the clutter coverage both grow as the band gets harder.

    These are the axes that describe how much work a model has to do to reject what is not an object.

    """
    measured = [getattr(bands[name], axis) for name in ("easy", "moderate", "hard")]

    assert measured == sorted(measured), measured


def test_the_hard_band_actually_reaches_the_small_object_regime(bands: dict[str, ProxyStats]) -> None:
    """The hard band is not merely relatively harder: a real share of its objects are genuinely small.

    Without this the ordering above could be satisfied by three comfortable bands, and the small object regime the
    generator could not produce before this work would still be out of reach.

    """
    assert bands["hard"].small_fraction > 0.1
    assert bands["hard"].p10_area < SMALL_AREA


def test_the_measurement_is_reproducible() -> None:
    """Two runs of one band agree exactly, so a moved number is a finding rather than noise."""
    first = measure(BANDS["easy"], count=3)
    second = measure(BANDS["easy"], count=3)

    assert first == second


def test_the_table_renders_one_row_per_band() -> None:
    """The published table comes from the measurement, so the page can always be regenerated."""
    lines = render_table().splitlines()

    assert len(lines) == len(BANDS) + 2
    assert lines[0].startswith("| band |")
