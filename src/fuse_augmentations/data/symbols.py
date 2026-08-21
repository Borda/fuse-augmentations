"""Analytic symbol-outline and landmark tables: a third, keypoint-capable shape family.

Seven simple, rotation-unambiguous 2D symbols — a kite, a trapezoid, a house, an arrow, a Latin
cross, a teardrop, and an anchor — each hand-authored as a straight-edge polygon rather than
traced from source art (unlike
:mod:`~fuse_augmentations.data.animals`, these have no artwork to attribute). Three of them (arrow,
cross, anchor) are concave, so their segmentation polygon and oriented bounding box carry real
information an axis-aligned box alone does not.

Every symbol is drawn **mirror-symmetric about its vertical axis** in canonical (unrotated)
orientation. That invariant is what makes :data:`SYMBOL_KEYPOINT_FLIP_IDX` a genuine (non-identity)
permutation rather than the animals' viewer-relative identity mapping: a horizontal flip of a
bilaterally symmetric silhouette yields a rotation-congruent shape with its left/right landmarks
exchanged, and rotation preserves winding, so left and right stay distinguishable under the
generator's random in-plane rotation. An eighth symbol must keep this invariant to reuse the schema.

Vertices are authored directly as unit-space literals (raw coordinates in a nominal
``[-1.3, 1.3]`` box, screen orientation — ``+x`` right, ``+y`` **down**, matching
:mod:`~fuse_augmentations.data.animals`) and pushed through
:func:`~fuse_augmentations.data.landmarks._normalized_pair` at import time, so the same unit-space
invariant the zoo loader enforces on traced art (center of mass at the origin, larger extent
scaled to ``1``) is enforced here too rather than merely assumed of the hand-picked numbers.

Every outline point keeps an unambiguous identity under rotation for the same reason the animal
silhouettes do — see :class:`~fuse_augmentations.data.animals.AnimalShape` — but the identity comes
from each symbol's own distinct archetype rather than anatomical structure, which is why the
landmark schema below uses seven generic structural slots (``center``, ``apex``, ``tail``,
``flank_left/right``, ``base_left/right``) instead of a shared semantic vocabulary: a "flank" means
a kite's side corner on one shape and an arrow's barb on another, and only ``center`` — the
outline's own area centroid (center of mass), the same point every polygon is already normalized
to sit at the origin around, so ``center`` is exactly ``(0, 0)`` before placement — is mandatory.
Absent slots are ``(nan, nan)``, the same row-level contract :mod:`~fuse_augmentations.data.animals`
uses.

Per-shape slot meaning (``—`` = absent, i.e. a ``(nan, nan)`` row; every ``center`` is that shape's
own area centroid):

=================== ========= ========= ========= ================== ==================
shape               center    apex      tail      flank_left/right   base_left/right
=================== ========= ========= ========= ================== ==================
kite                 centroid  top       bottom    left / right       —
trapezoid            centroid  —         —         top corners        bottom corners
house                centroid  roof apex —         eaves               base corners
arrow                centroid  tip       tail mid  barbs              —
cross                centroid  top       bottom    left/right arm     —
teardrop             centroid  round top point     waist              —
anchor               centroid  ring      crux      flukes             —
=================== ========= ========= ========= ================== ==================

The skeleton is a **star** from ``center`` to each of the other six slots: 6 edges over 7 nodes, so
every optional slot is a leaf and an absent one drops exactly its own edge, orphaning nothing — the
same property :data:`~fuse_augmentations.data.animals.ANIMAL_KEYPOINT_SKELETON` relies on.

Pure NumPy, no image-library dependency, and no import from
:mod:`~fuse_augmentations.data.animals`: the two families are siblings under
:mod:`~fuse_augmentations.data.landmarks`, neither importing the other. Tables are keyed by the
plain :class:`SymbolShape` *values*.

Examples:
    ```pycon
    >>> from fuse_augmentations.data.symbols import SYMBOL_KEYPOINTS, SYMBOL_POLYGONS
    >>> len(SYMBOL_POLYGONS)
    7
    >>> sorted(SYMBOL_POLYGONS)[:3]
    ['anchor', 'arrow', 'cross']
    >>> kite = SYMBOL_POLYGONS["kite"]
    >>> round(float((kite.max(axis=0) - kite.min(axis=0)).max()), 9)
    1.0
    >>> SYMBOL_KEYPOINTS["kite"].shape
    (7, 2)
    >>> sorted(SYMBOL_KEYPOINTS) == sorted(SYMBOL_POLYGONS)
    True

    ```

"""

from __future__ import annotations

import json
from enum import Enum
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from fuse_augmentations.data.landmarks import KeypointSchema, _normalized_pair

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable

    from numpy.typing import NDArray

_NAN = float("nan")
#: Placeholder for an absent landmark row — a slot a given symbol has no vertex for.
_ABSENT: tuple[float, float] = (_NAN, _NAN)


class SymbolShape(str, Enum):
    """Analytic symbol vocabulary (definition order is the symbol class order).

    Seven straight-edge 2D symbols, each mirror-symmetric about its own vertical axis and belonging
    to a distinct silhouette archetype, so — like :class:`~fuse_augmentations.data.animals.AnimalShape`
    — every outline point keeps an unambiguous identity under rotation. There is no plain
    ``TRIANGLE``/``ISOSCELES_TRIANGLE`` member here: an isosceles (and, being acute, tied) triangle
    doubled up on both the naming collision with :attr:`~fuse_augmentations.data.geometry.GeomShape.TRIANGLE`
    and the "minimum-area OBB has no unique answer" problem that motivated redesigning that
    geometric shape (see its docstring) — not worth solving twice for a shape this family does not
    need to keep.

    Attributes:
        KITE: Diamond quadrilateral with unequal top/bottom diagonal lengths.
        TRAPEZOID: Isosceles trapezoid, short parallel side up.
        HOUSE: Square body with a triangular roof (five-sided, convex).
        ARROW: Up-pointing arrow with two barbs (concave).
        CROSS: Latin cross with an elongated lower arm (concave).
        TEARDROP: Rounded top tapering to a bottom point.
        ANCHOR: Ring, stock, shaft and two flukes (concave).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.symbols import SymbolShape
        >>> len(SymbolShape)
        7
        >>> SymbolShape("kite")
        <SymbolShape.KITE: 'kite'>

        ```

    """

    KITE = "kite"
    TRAPEZOID = "trapezoid"
    HOUSE = "house"
    ARROW = "arrow"
    CROSS = "cross"
    TEARDROP = "teardrop"
    ANCHOR = "anchor"


#: Symbol names in :class:`SymbolShape` declaration order.
SYMBOL_NAMES: tuple[str, ...] = tuple(shape.value for shape in SymbolShape)

#: Landmark names for :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS` under the symbol
#: family, in the order every keypoint table, annotation, and label row uses. See the module
#: docstring's per-shape table for what each slot means on a given symbol; only ``center`` is
#: mandatory.
SYMBOL_KEYPOINT_NAMES: tuple[str, ...] = (
    "center",
    "apex",
    "tail",
    "flank_left",
    "flank_right",
    "base_left",
    "base_right",
)

#: Star skeleton: ``center`` (index 0) to each other slot. Visualization-only, like
#: :data:`~fuse_augmentations.data.animals.ANIMAL_KEYPOINT_SKELETON`.
SYMBOL_KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6))

#: Horizontal-flip permutation: ``flank_left``/``flank_right`` (indices 3, 4) and ``base_left``/
#: ``base_right`` (indices 5, 6) swap; ``center``, ``apex``, and ``tail`` sit on the symmetry axis
#: and map to themselves. A genuine (non-identity) permutation, valid because every symbol is
#: bilaterally symmetric about its canonical vertical axis — see the module docstring.
SYMBOL_KEYPOINT_FLIP_IDX: tuple[int, ...] = (0, 1, 2, 4, 3, 6, 5)

#: The complete keypoint schema for every :class:`SymbolShape` — the one artifact
#: :func:`~fuse_augmentations.data.config.keypoint_schema_for` and the writers need to describe a
#: symbol ``Task.KEYPOINTS`` run.
SYMBOL_KEYPOINT_SCHEMA = KeypointSchema(
    names=SYMBOL_KEYPOINT_NAMES,
    skeleton=SYMBOL_KEYPOINT_SKELETON,
    flip_idx=SYMBOL_KEYPOINT_FLIP_IDX,
    shape_values=SYMBOL_NAMES,
)

#: Packaged asset holding the raw ``(outline, landmarks)`` literal per symbol, in a nominal
#: ``[-1.3, 1.3]`` unit-ish box, screen orientation (``+y`` down). ``landmarks`` follows
#: :data:`SYMBOL_KEYPOINT_NAMES` order, ``null`` for an absent slot (see :func:`_read_raw`). Every
#: entry is mirror-symmetric about ``x == 0`` (so its center-of-mass ``x`` is exactly ``0`` before
#: normalization), which is what keeps :data:`SYMBOL_KEYPOINT_FLIP_IDX` correct. Each ``center`` is
#: set to its own outline's raw-space area centroid (shoelace formula), computed and verified (in
#: ``test_symbols.py``) to lie inside the outline for every shape here — a future, more exotic
#: concave outline whose true centroid falls outside its own silhouette would need a hand-placed
#: override instead. Every other present landmark sits three-quarters of the way from ``center`` to
#: its named outline vertex — not *on* that vertex — because a landmark placed exactly on a polygon
#: boundary is fragile under rotation: a rasterizer's scanline fill can exclude the exact boundary
#: pixel from whichever direction the fill sweeps, and which vertex that is changes with the
#: rotation angle. Pulling every point solidly into the interior, the way the animals' anatomical
#: landmarks already are, makes the placement robust at any angle. ``arrow``'s ``flank_left/right``
#: are the one exception to the three-quarters rule: they are the centroid of each barb's own
#: triangle (tip, barb tip, notch), because the straight path from ``center`` to a barb tip crosses
#: the concave notch and exits the polygon.
_ASSET: Traversable = files("fuse_augmentations.data") / "symbols.json"


def _read_raw() -> dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]]:
    """Load and validate :data:`_ASSET` into the raw ``(outline, landmarks)`` literal per symbol.

    Raises:
        ValueError: If the asset is missing a registered :class:`SymbolShape`'s entry, or a
            ``landmarks`` list does not hold exactly ``len(SYMBOL_KEYPOINT_NAMES)`` entries.

    """
    with _ASSET.open("rb") as handle:
        payload = json.load(handle)
    raw: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]] = {}
    for name in SYMBOL_NAMES:
        entry = payload.get(name)
        if entry is None:
            raise ValueError(f"symbols.json is missing the shape {name!r}")
        landmarks = entry["landmarks"]
        if len(landmarks) != len(SYMBOL_KEYPOINT_NAMES):
            raise ValueError(
                f"symbols.json shape {name!r} needs exactly {len(SYMBOL_KEYPOINT_NAMES)} landmarks, "
                f"got {len(landmarks)}"
            )
        outline = [tuple(point) for point in entry["outline"]]
        points = [_ABSENT if point is None else tuple(point) for point in landmarks]
        raw[name] = (outline, points)
    return raw


def _load() -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Normalize every raw ``(outline, landmarks)`` literal into the unit-space table pair."""
    raw = _read_raw()
    polygons: dict[str, NDArray[np.float64]] = {}
    keypoints: dict[str, NDArray[np.float64]] = {}
    for name in SYMBOL_NAMES:
        outline, landmarks = raw[name]
        polygons[name], keypoints[name] = _normalized_pair(outline, landmarks, SYMBOL_KEYPOINT_NAMES)
    return polygons, keypoints


_POLYGONS, _KEYPOINTS = _load()

#: Outline table per symbol :class:`~fuse_augmentations.data.config.Shape` *value*. Every entry is
#: unit-normalized and read-only; scale a copy rather than mutating it. Like
#: :data:`~fuse_augmentations.data.animals.ANIMAL_POLYGONS`, the table itself is a read-only view.
SYMBOL_POLYGONS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_POLYGONS)

#: Landmark table per symbol :class:`~fuse_augmentations.data.config.Shape` *value*, in
#: :data:`SYMBOL_KEYPOINT_NAMES` order. Every entry is a read-only ``(7, 2)`` array in its outline's
#: unit frame. A row is ``(nan, nan)`` for a slot the symbol does not use — see the module
#: docstring's per-shape table.
SYMBOL_KEYPOINTS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_KEYPOINTS)


def symbol_shapes(count: int | None = None) -> tuple[SymbolShape, ...]:
    """Return the symbol shapes, optionally just the first ``count`` of them.

    A convenience selector for :attr:`~fuse_augmentations.data.config.SyntheticConfig.shapes`,
    mirroring :func:`~fuse_augmentations.data.animals.animal_shapes`. "First ``count``" means
    :class:`SymbolShape` declaration order, the same order the class-id vocabulary uses, so
    ``symbol_shapes(3)`` names the same three symbols on every call and across releases.

    Args:
        count: How many symbols to take, from the start of :class:`SymbolShape`. ``None`` (the
            default) returns every symbol.

    Returns:
        The selected :class:`SymbolShape` members, in declaration order.

    Raises:
        ValueError: If ``count`` is negative or exceeds the number of symbols.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.symbols import symbol_shapes
        >>> symbol_shapes(2)
        (<SymbolShape.KITE: 'kite'>, <SymbolShape.TRAPEZOID: 'trapezoid'>)
        >>> len(symbol_shapes())
        7

        ```

    """
    every = tuple(SymbolShape)
    if count is None:
        return every
    if not 0 <= count <= len(every):
        raise ValueError(f"count must be within [0, {len(every)}], got {count}")
    return every[:count]


def symbol_keypoints(
    shape: SymbolShape, center: tuple[float, float], size: float, angle: float = 0.0, skew: float = 0.0
) -> NDArray[np.float64]:
    """Place one symbol's landmark table into image coordinates.

    The table is looked up in :data:`SYMBOL_KEYPOINTS`, scaled, skewed, rotated, and translated
    exactly as :func:`~fuse_augmentations.data.geometry.shape_polygon` treats the matching outline,
    so passing the same ``center``, ``size``, ``angle``, and ``skew`` to both puts every landmark on
    the silhouette that was drawn — mirroring :func:`~fuse_augmentations.data.animals.animal_keypoints`.

    Args:
        shape: A :class:`SymbolShape` member.
        center: Target center ``(x, y)`` in pixels — the same value passed to ``shape_polygon``.
        size: Bounding size in pixels — the same value passed to ``shape_polygon``.
        angle: Rotation in radians about the shape center — likewise.
        skew: Signed fraction narrowing one pre-rotation half — likewise; see
            :attr:`~fuse_augmentations.data.config.SyntheticConfig.asymmetry_jitter`.

    Returns:
        ``(7, 2)`` float array of landmark coordinates in image pixels, ordered by
        :data:`SYMBOL_KEYPOINT_NAMES`. Points may fall outside the canvas; clipping is the caller's
        decision. A row is ``(nan, nan)`` for a slot the symbol does not use.

    Raises:
        ValueError: If ``shape`` has no keypoint table.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.symbols import SymbolShape, symbol_keypoints
        >>> points = symbol_keypoints(SymbolShape.KITE, center=(50.0, 50.0), size=20.0)
        >>> points.shape
        (7, 2)

        ```

    """
    # deferred: geometry imports this module's tables, so a module-level import here would cycle
    from fuse_augmentations.data.geometry import _placed

    table = SYMBOL_KEYPOINTS.get(shape.value)
    if table is None:
        known = ", ".join(SYMBOL_KEYPOINTS)
        raise ValueError(f"shape {shape.value!r} has no keypoint table; expected one of {known}")
    # The stored table is frozen, so multiplying returns a fresh writable array, never an alias.
    return _placed(table * size, center, angle, skew)
