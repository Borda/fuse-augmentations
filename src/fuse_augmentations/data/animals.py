"""Animal-silhouette outline and landmark tables, loaded from the packaged ``zoo`` files.

Twelve side-profile silhouettes — duck, elephant, giraffe, fish, rabbit, camel, eagle, penguin,
whale, kangaroo, flamingo, crocodile — traced from public-domain reference art (see
:data:`ANIMAL_SOURCES`) rather than guessed by hand, so each one is recognizable at preview size.
Every animal ships as one SVG document,
``fuse_augmentations/data/zoo/<animal>.svg``, holding its outline path, its sixteen landmarks (as a
``<g id="keypoints">`` of ``<circle>`` elements) and its provenance (as ``zoo:``-namespaced root
attributes); the artwork can therefore be opened, inspected and corrected in any SVG editor without
touching Python. This module is only the loader that turns those documents into NumPy tables.

Each outline is a **simple** (non-self-intersecting) polygon in the unit space
:func:`~fuse_augmentations.data.geometry._base_polygon` uses for the geometric shapes: vertex centroid
at the origin and the larger of the two extents scaled to ``1``, so multiplying by a pixel ``size``
yields a shape bounded by ``size`` pixels. Coordinates are in screen orientation — ``+x`` right,
``+y`` **down** — matching Pillow's raster axes, so every animal renders upright and faces left.
The stored values are already normalized; they are passed through :func:`_normalized` again at load
time, which makes the invariant impossible to break by editing an SVG file.

:data:`ANIMAL_KEYPOINTS` carries the sixteen landmarks in :data:`ANIMAL_KEYPOINT_NAMES` order — an
anatomical ``mouth``/``eye``/``ear``/``head``/``neck``/``body_top``/``body_bottom``/``tail`` chain
plus two-segment front limbs (``front_elbow_*`` then ``front_limb_*``: paws, wings, or
fins/flippers) and the optional two-segment hind legs (``hind_knee_*`` then ``hind_limb_*``) — that
:attr:`~fuse_augmentations.data.config.Task.KEYPOINTS` annotates.
Landmarks are mapped through *their outline's* transform, so the two tables live in one frame: a
landmark sits **inside or on** its silhouette and, unlike an outline table, is neither centred on the
origin nor scaled to unit extent by itself.

**Absent keypoints.** The four hind-leg names (``hind_knee_*``, ``hind_limb_*``) are the only ones
an animal may omit (a whale's or a fish's silhouette shows pectoral fins/flippers as its front
limbs but no hind legs): an omitted ``<circle>`` in the source SVG becomes a
``NaN`` row — ``(nan, nan)`` — in the returned table rather than a faked point. Every other name is
mandatory; a missing one is a load-time :class:`ValueError`, not a silent gap. The NaN encoding was
chosen because the rest of the pipeline already handles it for free: a NaN coordinate compares
``False`` against every bound check downstream (canvas-visibility, placement clipping), so it falls
through to "not labeled" exactly like a point clipped off-canvas — see
:func:`~fuse_augmentations.data.generator._visible_keypoints` and
:func:`~fuse_augmentations.data.writers._keypoint_triples`. This module's own :func:`_frame` only
ever measures the **outline**, never the landmark table, so a NaN landmark can never poison the
outline's centring/scaling math — that invariant is what makes the encoding safe.

Unlike the geometric shapes these outlines are asymmetric and each belongs to a distinct silhouette
archetype, so the classes stay separable at a glance and every outline point keeps an unambiguous
identity under rotation.

Pure NumPy plus the stdlib XML parser — no Pillow, no torch, and no import from
:mod:`fuse_augmentations.data.config`: the configuration layer imports *this* module for the animal
half of its shape vocabulary, never the other way round. Tables are keyed by the plain
:class:`AnimalShape` *values*.

Examples:
    ```pycon
    >>> from fuse_augmentations.data.animals import ANIMAL_KEYPOINTS, ANIMAL_POLYGONS
    >>> len(ANIMAL_POLYGONS)
    12
    >>> sorted(ANIMAL_POLYGONS)[:6]
    ['camel', 'crocodile', 'duck', 'eagle', 'elephant', 'fish']
    >>> duck = ANIMAL_POLYGONS["duck"]
    >>> duck.shape[1]
    2
    >>> bool(abs(duck.mean(axis=0)).max() < 1e-9)
    True
    >>> ANIMAL_KEYPOINTS["duck"].shape
    (16, 2)
    >>> sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)
    True

    ```

"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from enum import Enum
from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.resources.abc import Traversable

    from numpy.typing import NDArray


class AnimalShape(str, Enum):
    """Animal silhouette vocabulary (definition order is the animal class order).

    Twelve fixed side-profile silhouettes traced from public-domain reference art. Each is
    asymmetric and belongs to a distinct silhouette archetype, so the classes stay separable at a
    glance and every outline point keeps an unambiguous identity under rotation — the property a
    landmark needs and a square or circle cannot offer. Every member has a ``<value>.svg`` document
    in the packaged ``zoo`` directory and therefore a landmark table, which is what makes
    :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS` well-defined for exactly this enum.

    Attributes:
        DUCK: Compact duck silhouette with an S-curved neck and a beak.
        ELEPHANT: Bulky elephant silhouette with a trunk, a large ear, and thick legs.
        GIRAFFE: Tall, thin giraffe silhouette with a very long neck and thin legs.
        FISH: Streamlined fish silhouette with a forked tail fin.
        RABBIT: Compact rabbit silhouette with long upright ears.
        CAMEL: Humped camel silhouette on four long legs.
        EAGLE: Perched eagle silhouette with a hooked beak and a long tail.
        PENGUIN: Upright penguin silhouette with flippers and webbed feet.
        WHALE: Streamlined whale silhouette with a pectoral flipper and a tail fluke.
        KANGAROO: Hopping kangaroo silhouette with a heavy tail and one large hind foot.
        FLAMINGO: Long-legged flamingo silhouette with an S-curved neck.
        CROCODILE: Low, elongated crocodile silhouette with a long snout and sprawled legs.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import AnimalShape
        >>> len(AnimalShape)
        12
        >>> AnimalShape("duck")
        <AnimalShape.DUCK: 'duck'>

        ```

    """

    DUCK = "duck"
    ELEPHANT = "elephant"
    GIRAFFE = "giraffe"
    FISH = "fish"
    RABBIT = "rabbit"
    CAMEL = "camel"
    EAGLE = "eagle"
    PENGUIN = "penguin"
    WHALE = "whale"
    KANGAROO = "kangaroo"
    FLAMINGO = "flamingo"
    CROCODILE = "crocodile"


#: Animal names in :class:`AnimalShape` declaration order — the enum's values, kept as a plain tuple
#: because the loader and its tables are keyed by name rather than by member.
ANIMAL_NAMES: tuple[str, ...] = tuple(shape.value for shape in AnimalShape)

#: Landmark names for :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS`, in the order every
#: keypoint table, annotation, and label row uses — and the order landmarks are read out of a zoo
#: document. One shared anatomical schema across all animals: Ultralytics' YOLO pose format carries
#: a single dataset-wide ``kpt_shape``, so a per-class name list is not representable.
#: The ``front_limb_*`` pair covers whatever the animal actually has at that slot — paws, wings, or
#: flippers/fins; ``left`` is the limb nearer the viewer (fully visible), ``right`` the far one — a
#: documented convention, since a side-profile silhouette cannot truly tell left from right. Every
#: limb is articulated in two points, proximal before distal: ``front_elbow_*`` (elbow, wing wrist,
#: or flipper bend) then ``front_limb_*`` (the paw/wing tip/fin tip), and ``hind_knee_*`` (the
#: knee/hock bend) then ``hind_limb_*`` (the foot) — a limb's bend is the most visible pose cue on a
#: silhouette. All four hind points are optional — see :data:`_OPTIONAL_KEYPOINTS`.
ANIMAL_KEYPOINT_NAMES: tuple[str, ...] = (
    "mouth",
    "eye",
    "ear",
    "head",
    "neck",
    "body_top",
    "body_bottom",
    "tail",
    "front_elbow_left",
    "front_elbow_right",
    "front_limb_left",
    "front_limb_right",
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
)

#: The only landmark names a document may omit; every other name in :data:`ANIMAL_KEYPOINT_NAMES` is
#: mandatory and a missing one is a load-time :class:`ValueError`.
_OPTIONAL_KEYPOINTS: frozenset[str] = frozenset({
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
})

#: Skeleton edges as index pairs into :data:`ANIMAL_KEYPOINT_NAMES`: ``mouth-head``, ``eye-head``,
#: ``ear-head``, the ``head-neck-body_top-body_bottom-tail`` chain, a two-segment
#: ``body_top-front_elbow-front_limb`` chain per front limb, and a two-segment
#: ``body_bottom-hind_knee-hind_limb`` chain per hind leg. 15 edges over 16 nodes is a spanning tree
#: whose optional hind points hang off the end of their own chain, so an absent hind leg drops
#: exactly its own two edges and orphans nothing. Purely a visualization aid (COCO viewers connect
#: the dots with it); nothing in generation, writing, or validation depends on it.
ANIMAL_KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (5, 8),
    (5, 9),
    (8, 10),
    (9, 11),
    (6, 12),
    (6, 13),
    (12, 14),
    (13, 15),
)

#: Provenance attributes every document must carry (as ``zoo:``-namespaced root attributes).
#: ``attribution`` is deliberately excluded — CC0/PDM art carries no attribution obligation.
_REQUIRED_PROVENANCE: tuple[str, ...] = ("origin", "title", "license", "note")

#: SVG commands accepted by :func:`_parse_path_d`, absolute and relative.
_SUPPORTED_PATH_COMMANDS = "MmLlHhVvZz"

#: Curve commands rejected with an authoring hint — anything with an SVG letter that isn't a straight
#: line, a move, or a close belongs to this set.
_CURVE_PATH_COMMANDS = "CcSsQqTtAa"

_SVG_NS = "http://www.w3.org/2000/svg"
_ZOO_NS = "https://github.com/Borda/fuse-augmentations/ns/zoo"

#: Directory holding the packaged animal documents, resolved through :mod:`importlib.resources` so it
#: works from a source checkout and from an installed wheel alike.
_ZOO: Traversable = files("fuse_augmentations.data") / "zoo"


def _svg_tag(tag: str) -> str:
    """Return ``tag`` fully qualified with the SVG namespace, for ``ElementTree`` lookups."""
    return f"{{{_SVG_NS}}}{tag}"


def _zoo_attr(name: str) -> str:
    """Return ``name`` fully qualified with the ``zoo:`` namespace, for ``ElementTree`` lookups."""
    return f"{{{_ZOO_NS}}}{name}"


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
    re-centre them on their own mean and detach them from the silhouette. Pairing the two tables in
    one call is what makes that impossible to get wrong.

    Args:
        outline: ``(x, y)`` outline points; see :func:`_normalized`.
        landmarks: The sixteen ``(x, y)`` landmarks in :data:`ANIMAL_KEYPOINT_NAMES` order, in the same
            coordinates as ``outline``. An absent optional landmark is ``(nan, nan)``.

    Returns:
        The read-only normalized ``(num_points, 2)`` outline and the read-only ``(16, 2)`` landmark
        table. Unlike the outline, the landmark table is **not** self-normalized: its mean is not the
        origin and its extent is not ``1``. NaN rows pass through unchanged: this call only ever
        measures the *outline* (via :func:`_frame`), never the landmark table, so a NaN landmark can
        never poison the offset/extent it and every other landmark are mapped through.

    Raises:
        ValueError: If the outline is degenerate (see :func:`_normalized`), the landmark table does
            not hold exactly ``len(ANIMAL_KEYPOINT_NAMES)`` ``(x, y)`` points, or a row has exactly one NaN
            coordinate (a parser bug — a real absence is NaN in both).

    """
    polygon = _normalized(outline)
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (len(ANIMAL_KEYPOINT_NAMES), 2):
        raise ValueError(
            f"a keypoint table needs exactly {len(ANIMAL_KEYPOINT_NAMES)} (x, y) landmarks, "
            f"got array of shape {points.shape}"
        )
    nan_mask = np.isnan(points)
    half_nan = nan_mask.any(axis=1) & ~nan_mask.all(axis=1)
    if half_nan.any():
        bad = [ANIMAL_KEYPOINT_NAMES[i] for i in np.nonzero(half_nan)[0]]
        raise ValueError(f"keypoint(s) {bad} have exactly one NaN coordinate; an absent landmark must be NaN in both")
    offset, extent = _frame(np.asarray(outline, dtype=np.float64))
    mapped: NDArray[np.float64] = (points - offset) / extent
    mapped.setflags(write=False)
    return polygon, mapped


def _parse_path_d(d: str, name: str) -> list[tuple[float, float]]:
    """Parse an SVG path ``d`` attribute into a closed polygon's vertex list.

    Tolerant of both absolute and relative ``M``/``L``/``H``/``V``/``Z`` and of SVG's implicit
    command repetition (a bare coordinate pair following ``M``/``L`` repeats the last command) —
    that tolerance is what lets a re-saved Inkscape path (relative, ``H``/``V``-heavy) still parse,
    rather than only the absolute ``M``/``L``/``Z`` this package's own authoring script emits.

    Args:
        d: The ``<path>`` element's ``d`` attribute.
        name: Animal name, for error messages.

    Returns:
        The outline vertices in path order.

    Raises:
        ValueError: If the path uses a curve command, an unrecognised command, a second subpath
            (a second ``M``/``m``), or is not closed with a trailing ``Z``/``z``.

    """
    raw_tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    cmd: str | None = None
    started = False
    closed = False
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if tok[:1].isalpha():
            cmd, started, closed, i = _consume_command(tok, name, started, i)
            continue
        if cmd is None:
            raise ValueError(f"zoo document {name}.svg path data must start with a moveto command")
        current, cmd, consumed = _consume_coordinate(cmd, raw_tokens, i, current)
        points.append(current)
        i += consumed
    if not closed:
        raise ValueError(f"zoo document {name}.svg path is not closed; append a trailing Z")
    return points


def _consume_command(tok: str, name: str, started: bool, i: int) -> tuple[str, bool, bool, int]:
    """Validate one command letter token and return ``(cmd, started, closed, next_index)``."""
    if tok in _CURVE_PATH_COMMANDS:
        raise ValueError(
            f"zoo document {name}.svg uses a curve command {tok!r}; in Inkscape use Path ▸ Flatten "
            "to straight segments before saving"
        )
    if tok not in _SUPPORTED_PATH_COMMANDS:
        raise ValueError(f"zoo document {name}.svg path data has an unsupported command {tok!r}")
    if tok in "Mm" and started:
        raise ValueError(f"zoo document {name}.svg path has a second subpath (a second M/m); expected exactly one")
    if tok in "Zz":
        return tok, started, True, i + 1
    return tok, started or tok in "Mm", False, i + 1


def _consume_coordinate(
    cmd: str, tokens: list[str], i: int, current: tuple[float, float]
) -> tuple[tuple[float, float], str, int]:
    """Apply one numeric argument (or coordinate pair) to ``current``; returns the new command.

    A bare coordinate pair following ``M``/``m`` is an implicit lineto for every pair after the first, per the SVG path
    grammar — the returned ``cmd`` reflects that so the next bare pair (if any) is interpreted correctly too.

    """
    if cmd in "Hh":
        x = float(tokens[i])
        return (current[0] + x if cmd == "h" else x, current[1]), cmd, 1
    if cmd in "Vv":
        y = float(tokens[i])
        return (current[0], current[1] + y if cmd == "v" else y), cmd, 1
    x, y = float(tokens[i]), float(tokens[i + 1])
    if cmd == "M":
        return (x, y), "L", 2
    if cmd == "m":
        return (current[0] + x, current[1] + y), "l", 2
    if cmd == "L":
        return (x, y), "L", 2
    return (current[0] + x, current[1] + y), "l", 2  # cmd == "l"


def _reject_transforms(root: ET.Element, name: str) -> None:
    """Raise if any element in the document carries a ``transform`` attribute."""
    for element in root.iter():
        if "transform" in element.attrib:
            tag = element.tag.rsplit("}", 1)[-1]
            raise ValueError(
                f"zoo document {name}.svg has a transform on <{tag}>; in Inkscape set "
                "Preferences ▸ Behavior ▸ Transforms ▸ Store transformation = Optimized"
            )


def _read_keypoints(root: ET.Element, name: str) -> dict[str, tuple[float, float]]:
    """Parse and validate the ``<g id="keypoints">`` circles of one zoo document."""
    group = root.find(f"{_svg_tag('g')}[@id='keypoints']")
    if group is None:
        raise ValueError(f'zoo document {name}.svg is missing the <g id="keypoints"> group')
    seen: dict[str, tuple[float, float]] = {}
    for circle in group.findall(_svg_tag("circle")):
        kp_name = circle.get(_zoo_attr("name"))
        if kp_name not in ANIMAL_KEYPOINT_NAMES:
            raise ValueError(
                f"zoo document {name}.svg has an unknown or missing zoo:name {kp_name!r}; "
                f"expected one of {ANIMAL_KEYPOINT_NAMES}"
            )
        if kp_name in seen:
            raise ValueError(f"zoo document {name}.svg has a duplicate zoo:name {kp_name!r}")
        seen[kp_name] = (float(circle.get("cx", "nan")), float(circle.get("cy", "nan")))
    missing = [key for key in ANIMAL_KEYPOINT_NAMES if key not in seen and key not in _OPTIONAL_KEYPOINTS]
    if missing:
        raise ValueError(f"zoo document {name}.svg is missing the landmark(s) {missing}")
    return seen


def _read_svg(
    name: str,
) -> tuple[list[tuple[float, float]], dict[str, tuple[float, float]], dict[str, str]]:
    """Read, parse and validate one zoo SVG document.

    Args:
        name: Animal name, i.e. the ``<name>.svg`` stem in the packaged ``zoo`` directory.

    Returns:
        The outline vertices, the present keypoints (by name), and the provenance attributes.

    Raises:
        ValueError: If the document uses a curve or a transform, does not have exactly one closed
            simple path, or is missing a mandatory landmark or provenance attribute. A malformed
            document is rejected here, at import, rather than surfacing mid-generation as a lookup
            failure.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - see module-level noqa above
    _reject_transforms(root, name)
    paths = root.findall(_svg_tag("path"))
    if len(paths) != 1:
        raise ValueError(f"zoo document {name}.svg must have exactly one <path>, found {len(paths)}")
    outline = _parse_path_d(paths[0].get("d", ""), name)
    missing_provenance = [key for key in _REQUIRED_PROVENANCE if not root.get(_zoo_attr(key))]
    if missing_provenance:
        raise ValueError(f"zoo document {name}.svg is missing the key(s) {missing_provenance}")
    source = {
        key: value
        for key in ("origin", "title", "license", "attribution", "note")
        if (value := root.get(_zoo_attr(key)))
    }
    keypoints = _read_keypoints(root, name)
    return outline, keypoints, source


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
        outline, present, source = _read_svg(name)
        table = [present.get(key, (np.nan, np.nan)) for key in ANIMAL_KEYPOINT_NAMES]
        polygons[name], keypoints[name] = _normalized_pair(outline, table)
        sources[name] = source
    return polygons, keypoints, sources


_POLYGONS, _KEYPOINTS, _SOURCES = _load()

#: Outline table per animal :class:`~fuse_augmentations.data.config.Shape` *value*.
#: Every entry is unit-normalized and read-only; scale a copy rather than mutating it.
ANIMAL_POLYGONS: dict[str, NDArray[np.float64]] = _POLYGONS

#: Landmark table per animal :class:`~fuse_augmentations.data.config.Shape` *value*, in
#: ``config.KEYPOINT_NAMES`` order. Every entry is a read-only ``(16, 2)`` array in its outline's
#: unit frame, so landmarks lie inside or on the silhouette; scale a copy rather than mutating it. A
#: row is ``(nan, nan)`` for an animal that has no hind legs (all four hind rows).
ANIMAL_KEYPOINTS: dict[str, NDArray[np.float64]] = _KEYPOINTS

#: Provenance per animal: ``origin`` (source page), ``title`` (what the art depicts), ``license``,
#: ``attribution`` (credit, not required by CC0/PDM) and a ``note`` on how the art was processed.
ANIMAL_SOURCES: dict[str, dict[str, str]] = _SOURCES


def animal_shapes(count: int | None = None) -> tuple[AnimalShape, ...]:
    """Return the animal silhouettes, optionally just the first ``count`` of them.

    A convenience selector for :attr:`~fuse_augmentations.data.config.SyntheticConfig.shapes`, which
    still takes (and stores) a plain tuple — naming members explicitly stays equally valid. "First
    ``count``" means :class:`AnimalShape` declaration order, the same order the class-id vocabulary
    uses, so ``animal_shapes(3)`` names the same three animals on every call and across releases;
    appending a thirteenth animal can only extend the tail of that list.

    Args:
        count: How many animals to take, from the start of :class:`AnimalShape`. ``None`` (the
            default) returns every animal.

    Returns:
        The selected :class:`AnimalShape` members, in declaration order.

    Raises:
        ValueError: If ``count`` is negative or exceeds the number of animals.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import animal_shapes
        >>> animal_shapes(3)
        (<AnimalShape.DUCK: 'duck'>, <AnimalShape.ELEPHANT: 'elephant'>, <AnimalShape.GIRAFFE: 'giraffe'>)
        >>> len(animal_shapes())
        12

        ```

    """
    every = tuple(AnimalShape)
    if count is None:
        return every
    if not 0 <= count <= len(every):
        raise ValueError(f"count must be within [0, {len(every)}], got {count}")
    return every[:count]


def animal_keypoints(
    shape: AnimalShape, center: tuple[float, float], size: float, angle: float = 0.0
) -> NDArray[np.float64]:
    """Place one animal's landmark table into image coordinates.

    The table is looked up in :data:`ANIMAL_KEYPOINTS`, scaled, rotated, and translated exactly as
    :func:`~fuse_augmentations.data.geometry.shape_polygon` treats the matching outline, so passing
    the same ``center``, ``size``, and ``angle`` to both puts every landmark on the silhouette that
    was drawn. No randomness is involved: the result is a pure function of the placement the
    generator already sampled.

    Args:
        shape: An :class:`AnimalShape` member. The geometric shapes have no landmark table at all
            (a square's 4-fold symmetry gives a fixed landmark no stable identity), which is why
            this signature names the animal enum rather than the shape union.
        center: Target center ``(x, y)`` in pixels — the same value passed to ``shape_polygon``.
        size: Bounding size in pixels — the same value passed to ``shape_polygon``.
        angle: Rotation in radians about the shape center — likewise.

    Returns:
        ``(16, 2)`` float array of landmark coordinates in image pixels, ordered by
        :data:`ANIMAL_KEYPOINT_NAMES`. Points may fall outside the canvas; clipping is the caller's
        decision. A row is ``(nan, nan)`` for an animal with no hind legs; NaN propagates through
        unchanged since scaling and translation are row-independent arithmetic.

    Raises:
        ValueError: If ``shape`` has no keypoint table.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import AnimalShape, animal_keypoints
        >>> points = animal_keypoints(AnimalShape.DUCK, center=(50.0, 50.0), size=20.0)
        >>> points.shape
        (16, 2)

        ```

    """
    # deferred: geometry imports this module's tables, so a module-level import here would cycle
    from fuse_augmentations.data.geometry import _placed

    table = ANIMAL_KEYPOINTS.get(shape.value)
    if table is None:
        known = ", ".join(ANIMAL_KEYPOINTS)
        raise ValueError(f"shape {shape.value!r} has no keypoint table; expected one of {known}")
    # The stored table is frozen, so multiplying returns a fresh writable array, never an alias.
    return _placed(table * size, center, angle)
