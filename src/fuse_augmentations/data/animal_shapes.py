"""Animal-silhouette outline and landmark tables, loaded from the packaged ``zoo`` files.

Eight side-profile silhouettes — duck, snail, elephant, giraffe, fish, turtle, snake, rabbit —
traced from public-domain reference art (see :data:`ANIMAL_SOURCES`) rather than guessed by hand,
so each one is recognizable at preview size. Every animal ships as one JSON document,
``fuse_augmentations/data/zoo/<animal>.json``, holding its outline polygon, its five landmarks and
its provenance; the artwork can therefore be inspected, corrected or extended without touching
Python. This module is only the loader that turns those documents into NumPy tables.

Each outline is a **simple** (non-self-intersecting) polygon in the unit space
:func:`~fuse_augmentations.data.shapes._base_polygon` uses for the geometric shapes: vertex centroid
at the origin and the larger of the two extents scaled to ``1``, so multiplying by a pixel ``size``
yields a shape bounded by ``size`` pixels. Coordinates are in screen orientation — ``+x`` right,
``+y`` **down** — matching Pillow's raster axes, so every animal renders upright and faces left.
The stored values are already normalized; they are passed through :func:`_normalized` again at load
time, which makes the invariant impossible to break by editing a JSON file.

:data:`ANIMAL_KEYPOINTS` carries the five landmarks — ``head``, ``eye``, ``back``, ``tail``,
``foot`` (``config.KEYPOINT_NAMES`` order) — that
:attr:`~fuse_augmentations.data.config.Task.KEYPOINTS` annotates. Landmarks are hand-placed (they
are not part of the source art) and mapped through *their outline's* transform, so the two tables
live in one frame: a landmark sits **inside or on** its silhouette (an eye is a point in the head
region, not an outline vertex) and, unlike an outline table, is neither centred on the origin nor
scaled to unit extent by itself.

Unlike the geometric shapes these outlines are asymmetric and each belongs to a distinct silhouette
archetype, so the classes stay separable at a glance and every outline point keeps an unambiguous
identity under rotation.

Pure NumPy — no Pillow, no torch, no import from :mod:`fuse_augmentations.data.config` (tables are
keyed by the plain ``Shape`` *values*, keeping the geometry layer independent of the configuration
layer).

Examples:
    ```pycon
    >>> from fuse_augmentations.data.animal_shapes import ANIMAL_KEYPOINTS, ANIMAL_POLYGONS
    >>> sorted(ANIMAL_POLYGONS)
    ['duck', 'elephant', 'fish', 'giraffe', 'rabbit', 'snail', 'snake', 'turtle']
    >>> duck = ANIMAL_POLYGONS["duck"]
    >>> duck.shape[1]
    2
    >>> bool(abs(duck.mean(axis=0)).max() < 1e-9)
    True
    >>> ANIMAL_KEYPOINTS["duck"].shape
    (5, 2)
    >>> sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)
    True

    ```

"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

#: Animal names, in :class:`~fuse_augmentations.data.config.Shape` declaration order. Each one has a
#: ``<name>.json`` document in the packaged ``zoo`` directory.
ANIMAL_NAMES: tuple[str, ...] = ("duck", "snail", "elephant", "giraffe", "fish", "turtle", "snake", "rabbit")

#: Landmark order inside a zoo document, mirroring ``config.KEYPOINT_NAMES``. Duplicated as plain
#: strings because this module deliberately does not import the configuration layer; a test pins the
#: two together so the schemas cannot drift apart.
_KEYPOINT_ORDER: tuple[str, ...] = ("head", "eye", "back", "tail", "foot")

#: Directory holding the packaged animal documents, resolved through :mod:`importlib.resources` so it
#: works from a source checkout and from an installed wheel alike.
_ZOO = files("fuse_augmentations.data") / "zoo"


def _frame(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return the ``(offset, extent)`` that normalize an outline into unit space.

    Args:
        points: ``(num_points, 2)`` raw outline array.

    Returns:
        The vertex mean to subtract and the larger extent to divide by.

    Raises:
        ValueError: If the outline collapses to a point.

    """
    offset: NDArray[np.float64] = points.mean(axis=0)
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if extent <= 0.0:
        raise ValueError("outline has zero extent; every vertex is identical")
    return offset, extent


def _normalized(vertices: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    """Center a raw outline on its vertex mean and scale its larger extent to ``1``.

    Args:
        vertices: ``(x, y)`` outline points in any convenient scale, ordered along the outline
            (winding direction is irrelevant).

    Returns:
        Read-only ``(num_points, 2)`` float array with zero vertex mean and a maximum
        extent of exactly ``1``. The array is frozen because it is shared by every caller;
        consumers scale it into a fresh array rather than mutating the table.

    Raises:
        ValueError: If the outline has fewer than three points or collapses to a point.

    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"an outline needs at least 3 (x, y) points, got array of shape {points.shape}")
    offset, extent = _frame(points)
    scaled: NDArray[np.float64] = (points - offset) / extent
    scaled.setflags(write=False)
    return scaled


def _normalized_pair(
    outline: Sequence[tuple[float, float]], landmarks: Sequence[tuple[float, float]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize an outline and map its landmarks through the *outline's* transform.

    Landmarks are authored in the same coordinates as the outline they annotate, so they must be
    centred and scaled by the outline's own mean and extent — normalizing them independently would
    re-centre the five points on their own mean and detach them from the silhouette. Pairing the two
    tables in one call is what makes that impossible to get wrong.

    Args:
        outline: ``(x, y)`` outline points; see :func:`_normalized`.
        landmarks: The five ``(x, y)`` landmarks in :data:`_KEYPOINT_ORDER` order, in the same
            coordinates as ``outline``.

    Returns:
        The read-only normalized ``(num_points, 2)`` outline and the read-only ``(5, 2)`` landmark
        table. Unlike the outline, the landmark table is **not** self-normalized: its mean is not the
        origin and its extent is not ``1``.

    Raises:
        ValueError: If the outline is degenerate (see :func:`_normalized`) or the landmark table does
            not hold exactly ``len(_KEYPOINT_ORDER)`` ``(x, y)`` points.

    """
    polygon = _normalized(outline)
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (len(_KEYPOINT_ORDER), 2):
        raise ValueError(
            f"a keypoint table needs exactly {len(_KEYPOINT_ORDER)} (x, y) landmarks, got array of shape {points.shape}"
        )
    offset, extent = _frame(np.asarray(outline, dtype=np.float64))
    mapped: NDArray[np.float64] = (points - offset) / extent
    mapped.setflags(write=False)
    return polygon, mapped


def _read_document(name: str) -> dict[str, Any]:
    """Read and validate one zoo document.

    Args:
        name: Animal name, i.e. the ``<name>.json`` stem in the packaged ``zoo`` directory.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the file is missing a required key or a landmark. A malformed document is
            rejected here, at import, rather than surfacing mid-generation as a lookup failure.

    """
    document: dict[str, Any] = json.loads((_ZOO / f"{name}.json").read_text(encoding="utf-8"))
    missing = [key for key in ("name", "archetype", "outline", "keypoints", "source") if key not in document]
    if missing:
        raise ValueError(f"zoo document {name}.json is missing the key(s) {missing}")
    absent = [key for key in _KEYPOINT_ORDER if key not in document["keypoints"]]
    if absent:
        raise ValueError(f"zoo document {name}.json is missing the landmark(s) {absent}")
    return document


def _load() -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
    dict[str, dict[str, str]],
]:
    """Load every packaged animal into the outline, landmark and provenance tables."""
    polygons: dict[str, NDArray[np.float64]] = {}
    keypoints: dict[str, NDArray[np.float64]] = {}
    sources: dict[str, dict[str, str]] = {}
    for name in ANIMAL_NAMES:
        document = _read_document(name)
        table = [document["keypoints"][key] for key in _KEYPOINT_ORDER]
        polygons[name], keypoints[name] = _normalized_pair(document["outline"], table)
        sources[name] = document["source"]
    return polygons, keypoints, sources


_POLYGONS, _KEYPOINTS, _SOURCES = _load()

#: Outline table per animal :class:`~fuse_augmentations.data.config.Shape` *value*.
#: Every entry is unit-normalized and read-only; scale a copy rather than mutating it.
ANIMAL_POLYGONS: dict[str, NDArray[np.float64]] = _POLYGONS

#: Landmark table per animal :class:`~fuse_augmentations.data.config.Shape` *value*, in
#: ``config.KEYPOINT_NAMES`` order. Every entry is a read-only ``(5, 2)`` array in its outline's unit
#: frame, so landmarks lie inside or on the silhouette; scale a copy rather than mutating it.
ANIMAL_KEYPOINTS: dict[str, NDArray[np.float64]] = _KEYPOINTS

#: Provenance per animal: ``origin`` (source page), ``title`` (what the art depicts), ``license``,
#: ``attribution`` (credit, not required by CC0/PDM) and a ``note`` on how the art was processed.
ANIMAL_SOURCES: dict[str, dict[str, str]] = _SOURCES
