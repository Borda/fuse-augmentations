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
:func:`~fuse_augmentations.data.geometry._base_polygon` uses for the geometric shapes: center of mass
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

**Absent keypoints.** ``ear`` and the four hind-leg names (``hind_knee_*``, ``hind_limb_*``) are the
only ones an animal may omit — a whale's or a fish's silhouette shows pectoral fins/flippers as its
front limbs but no hind legs, and neither taxon has an external ear to annotate at all: an omitted
``<circle>`` in the source SVG becomes a ``NaN`` row — ``(nan, nan)`` — in the returned table rather
than a faked point. Every other name is mandatory; a missing one is a load-time
:class:`ValueError`, not a silent gap. The NaN encoding was
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
    >>> round(float((duck.max(axis=0) - duck.min(axis=0)).max()), 9)
    1.0
    >>> ANIMAL_KEYPOINTS["duck"].shape
    (16, 2)
    >>> sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)
    True

    ```

"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from enum import Enum
from importlib.resources import files
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from fuse_augmentations.data.landmarks import KeypointSchema, _normalized_pair

if TYPE_CHECKING:
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
#: silhouette. ``ear`` and all four hind points are optional — see :data:`_OPTIONAL_KEYPOINTS`.
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
#: mandatory and a missing one is a load-time :class:`ValueError`. ``ear`` earns its place next to the
#: hind legs on the same anatomical ground: a fish and a whale have no external ear to point at, so the
#: alternative is inventing a spot on the head and labelling it visible.
_OPTIONAL_KEYPOINTS: frozenset[str] = frozenset({
    "ear",
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
})

#: Skeleton edges as index pairs into :data:`ANIMAL_KEYPOINT_NAMES`: ``mouth-head``, ``eye-head``,
#: ``ear-head``, the ``head-neck-body_top-body_bottom-tail`` chain, a two-segment
#: ``body_top-front_elbow-front_limb`` chain per front limb, and a two-segment
#: ``body_bottom-hind_knee-hind_limb`` chain per hind leg. 15 edges over 16 nodes is a spanning tree
#: whose optional points sit at the ends of their chains — ``ear`` is a leaf on ``head``, the hind
#: points hang off ``body_bottom`` — so an absent ear drops exactly its one edge, an absent hind leg
#: exactly its own two, and neither orphans anything. Purely a visualization aid (COCO viewers connect
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
            (a second ``M``/``m``), a moveto without a coordinate pair, ends part-way through a
            coordinate pair, puts a command letter where the second half of a pair belongs, carries
            coordinates after its closing ``Z``/``z``, or is not closed with a trailing ``Z``/``z``.

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
            # A moveto that consumed a pair leaves ``cmd`` as the implicit lineto it decays to, so
            # ``cmd`` still being ``M``/``m`` here means the moveto never got its coordinates — the
            # first vertex would otherwise be dropped without a word.
            if cmd in {"M", "m"}:
                raise ValueError(f"zoo document {name}.svg has a moveto without a coordinate pair")
            cmd, started, closed, i = _consume_command(tok, name, started, i)
            continue
        if cmd is None:
            raise ValueError(f"zoo document {name}.svg path data must start with a moveto command")
        if cmd in "Zz":
            # ``Z``/``z`` takes no arguments, so a numeric token here is malformed path data. Without
            # this guard it fell through every branch of _consume_coordinate to the relative-lineto
            # default and was silently absorbed as a phantom vertex.
            raise ValueError(
                f"zoo document {name}.svg has coordinates after the closing Z; a close command takes no arguments"
            )
        current, cmd, consumed = _consume_coordinate(cmd, name, raw_tokens, i, current)
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
    cmd: str, name: str, tokens: list[str], i: int, current: tuple[float, float]
) -> tuple[tuple[float, float], str, int]:
    """Apply one numeric argument (or coordinate pair) to ``current``; returns the new command.

    A bare coordinate pair following ``M``/``m`` is an implicit lineto for every pair after the first, per the SVG path
    grammar — the returned ``cmd`` reflects that so the next bare pair (if any) is interpreted correctly too.

    ``name`` is threaded in for the same reason :func:`_consume_command` takes it: a two-token read has to be validated
    before it happens, and the resulting error belongs to the named-``ValueError`` contract :func:`_parse_path_d`
    documents — not to a raw ``IndexError`` off the end of the token list, nor to :func:`float`'s own message about a
    command letter, neither of which names the document that failed to load.

    Args:
        cmd: The command in effect for this argument, absolute or relative.
        name: Animal name, for error messages.
        tokens: The whole token list, so a coordinate *pair* can be read (and bounds-checked) as one unit.
        i: Index of the first argument token to consume.
        current: The point the path is standing on, which relative commands offset from.

    Returns:
        The new current point, the command in effect for the next argument, and how many tokens were consumed.

    Raises:
        ValueError: If a pair is cut short by the end of the path data, or if its second half is a command letter.

    """
    if cmd in "Hh":
        x = float(tokens[i])
        return (current[0] + x if cmd == "h" else x, current[1]), cmd, 1
    if cmd in "Vv":
        y = float(tokens[i])
        return (current[0], current[1] + y if cmd == "v" else y), cmd, 1
    # Every command left takes an (x, y) pair, so the second token must exist and must be a number. The tokenizer emits
    # only command letters and numerals, which is what makes a leading alpha the exact test for "this is not a
    # coordinate".
    if i + 1 >= len(tokens):
        raise ValueError(f"zoo document {name}.svg path data ends mid-coordinate-pair after {tokens[i]!r}")
    if tokens[i + 1][:1].isalpha():
        raise ValueError(
            f"zoo document {name}.svg has the command {tokens[i + 1]!r} inside a coordinate pair, "
            f"where the second half of the pair opened by {tokens[i]!r} belongs"
        )
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
        # A present-but-coordinate-less circle is malformed whether the landmark is mandatory or
        # optional, so it is rejected here rather than defaulted to NaN: the mandatory-landmark
        # guard below only tests key presence, so a NaN sentinel would sail straight through it.
        cx, cy = circle.get("cx"), circle.get("cy")
        if cx is None or cy is None:
            raise ValueError(f"zoo document {name}.svg landmark {kp_name!r} is missing cx/cy")
        seen[kp_name] = (float(cx), float(cy))
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
            simple path, or is missing a mandatory landmark, a landmark's coordinates, or a
            provenance attribute. A malformed document is rejected here, at import, rather than
            surfacing mid-generation as a lookup failure.

    """
    # ``str()`` on a Traversable is not guaranteed to yield an openable path (a zipimport or frozen
    # loader has no real file behind it), so the document is read through the resource's own opener.
    # Binary, not text: that leaves any XML declaration's encoding for ElementTree to honor.
    with (_ZOO / f"{name}.svg").open("rb") as handle:
        root = ET.parse(handle).getroot()  # noqa: S314 - packaged asset addressed by a fixed enum name, never untrusted input
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
        polygons[name], keypoints[name] = _normalized_pair(outline, table, ANIMAL_KEYPOINT_NAMES)
        sources[name] = source
    return polygons, keypoints, sources


_POLYGONS, _KEYPOINTS, _SOURCES = _load()

#: Outline table per animal :class:`~fuse_augmentations.data.config.Shape` *value*.
#: Every entry is unit-normalized and read-only; scale a copy rather than mutating it. The table
#: itself is a read-only view as well, so ``ANIMAL_POLYGONS["duck"] = other`` raises instead of
#: silently repointing an outline every consumer in the process shares.
ANIMAL_POLYGONS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_POLYGONS)

#: Landmark table per animal :class:`~fuse_augmentations.data.config.Shape` *value*, in
#: ``config.KEYPOINT_NAMES`` order. Every entry is a read-only ``(16, 2)`` array in its outline's
#: unit frame, so landmarks lie inside or on the silhouette; scale a copy rather than mutating it. A
#: row is ``(nan, nan)`` for a landmark the animal does not have — the ``ear`` row and all four hind
#: rows for a fish or a whale. Like :data:`ANIMAL_POLYGONS`, the table itself is a read-only view,
#: not just the arrays in it.
ANIMAL_KEYPOINTS: Mapping[str, NDArray[np.float64]] = MappingProxyType(_KEYPOINTS)

#: Provenance per animal: ``origin`` (source page), ``title`` (what the art depicts), ``license``,
#: ``attribution`` (credit, not required by CC0/PDM) and a ``note`` on how the art was processed.
#: Frozen at both levels — neither the animal-to-record mapping nor an individual record can be
#: rewritten, because a licence field edited in place would misdescribe artwork already in the wheel.
ANIMAL_SOURCES: Mapping[str, Mapping[str, str]] = MappingProxyType({
    animal: MappingProxyType(source) for animal, source in _SOURCES.items()
})

#: Identity permutation: this schema's ``left``/``right`` are viewer-relative, not the animal's own
#: left/right (a side-profile silhouette cannot truly tell them apart — see
#: :data:`ANIMAL_KEYPOINT_NAMES`), so mirroring a side profile never turns a near limb into a far
#: one and every landmark maps to itself under a horizontal flip.
ANIMAL_KEYPOINT_FLIP_IDX: tuple[int, ...] = tuple(range(len(ANIMAL_KEYPOINT_NAMES)))

#: The complete keypoint schema for every :class:`AnimalShape` — the one artifact
#: :func:`~fuse_augmentations.data.config.keypoint_schema_for` and the writers need to describe an
#: animal ``Task.KEYPOINTS`` run.
ANIMAL_KEYPOINT_SCHEMA = KeypointSchema(
    names=ANIMAL_KEYPOINT_NAMES,
    skeleton=ANIMAL_KEYPOINT_SKELETON,
    flip_idx=ANIMAL_KEYPOINT_FLIP_IDX,
    shape_values=ANIMAL_NAMES,
)


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
    shape: AnimalShape, center: tuple[float, float], size: float, angle: float = 0.0, skew: float = 0.0
) -> NDArray[np.float64]:
    """Place one animal's landmark table into image coordinates.

    The table is looked up in :data:`ANIMAL_KEYPOINTS`, scaled, skewed, rotated, and translated
    exactly as :func:`~fuse_augmentations.data.geometry.shape_polygon` treats the matching outline,
    so passing the same ``center``, ``size``, ``angle``, and ``skew`` to both puts every landmark on
    the silhouette that was drawn. No randomness is involved: the result is a pure function of the
    placement the generator already sampled.

    Args:
        shape: An :class:`AnimalShape` member. The geometric shapes have no landmark table at all
            (a square's 4-fold symmetry gives a fixed landmark no stable identity), which is why
            this signature names the animal enum rather than the shape union.
        center: Target center ``(x, y)`` in pixels — the same value passed to ``shape_polygon``.
        size: Bounding size in pixels — the same value passed to ``shape_polygon``.
        angle: Rotation in radians about the shape center — likewise.
        skew: Signed fraction narrowing one pre-rotation half — likewise; see
            :attr:`~fuse_augmentations.data.config.SyntheticConfig.asymmetry_jitter`.

    Returns:
        ``(16, 2)`` float array of landmark coordinates in image pixels, ordered by
        :data:`ANIMAL_KEYPOINT_NAMES`. Points may fall outside the canvas; clipping is the caller's
        decision. A row is ``(nan, nan)`` for a landmark the animal does not have (a fish's or a
        whale's ear and hind legs); NaN propagates through unchanged since scaling and translation
        are row-independent arithmetic.

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
    return _placed(table * size, center, angle, skew)
