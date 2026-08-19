"""Animal table validity: simplicity, unit convention, archetype, landmarks, provenance."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fuse_augmentations.data.animals import (
    _OPTIONAL_KEYPOINTS,
    _SVG_NS,
    _ZOO,
    _ZOO_NS,
    ANIMAL_KEYPOINT_NAMES,
    ANIMAL_KEYPOINTS,
    ANIMAL_NAMES,
    ANIMAL_POLYGONS,
    ANIMAL_SOURCES,
    AnimalShape,
    _normalized,
    _normalized_pair,
    _parse_path_d,
    _read_keypoints,
    _reject_transforms,
    _svg_tag,
    _zoo_attr,
    animal_keypoints,
)
from fuse_augmentations.data.geometry import GeomShape, polygon_to_bbox_xyxy, shape_polygon

ZOO_ANIMAL_NAMES = ANIMAL_NAMES

#: Per-animal set of landmark names this specific silhouette lacks (a NaN row in
#: :data:`ANIMAL_KEYPOINTS`), pinned as a literal so a change here is a deliberate edit, not an
#: incidental side effect of re-authoring an SVG.
ABSENT_KEYPOINTS = {
    "duck": frozenset(),
    "elephant": frozenset(),
    "giraffe": frozenset(),
    "fish": frozenset({"hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}),
    "rabbit": frozenset(),
    "camel": frozenset(),
    "eagle": frozenset(),
    "penguin": frozenset(),
    "whale": frozenset({"hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}),
    "kangaroo": frozenset(),
    "flamingo": frozenset(),
    "crocodile": frozenset(),
}

# Pillow includes boundary pixels when filling polygons. At the test scale the whale's and the
# crocodile's long, narrow bodies carry the largest rasterization-vs-shoelace area bias; the other
# silhouettes remain within the general 5% bound. The bias shrinks as the same outline is
# rasterized at higher scale.
RASTER_AREA_REL_TOLERANCE = {"whale": 0.08, "crocodile": 0.08}

# CC0 1.0 and the Public Domain Mark are the only provenances the packaged art may carry: both
# place the work in the public domain worldwide, so redistributing it inside a wheel is unencumbered.
PUBLIC_DOMAIN_LICENSES = {
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
}

# Width-to-height bands for the documented silhouette archetypes, bracketing the proportions of the
# public-domain reference art each outline was traced from. They are wide enough to absorb a
# re-trace but narrow enough that an animal losing its archetype (e.g. the giraffe no longer being
# tall-thin) fails instead of silently degrading.
ARCHETYPE_ASPECT_BANDS = {
    "duck": (0.58, 0.78),  # upright-bird
    "elephant": (1.65, 2.15),  # bulky-quadruped
    "giraffe": (0.66, 0.90),  # tall-thin
    "fish": (2.27, 2.95),  # streamlined
    "rabbit": (1.03, 1.35),  # compact-eared
    "camel": (1.10, 1.45),  # bulky-quadruped-humped
    "eagle": (0.39, 0.52),  # upright-bird
    "penguin": (0.55, 0.74),  # upright-bird
    "whale": (4.00, 5.25),  # streamlined-aquatic-large
    "kangaroo": (1.47, 1.92),  # hopping-marsupial
    "flamingo": (0.50, 0.67),  # long-legged-wader
    "crocodile": (3.42, 4.48),  # sprawling-reptile
}


def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the z-component of ``(a - origin) x (b - origin)``."""
    return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """Whether collinear point ``r`` lies within the bounding box of segment ``p``-``q``."""
    return bool(min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Whether closed segments ``p1p2`` and ``p3p4`` share at least one point (CLRS predicate)."""
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 < 0 and d3 * d4 < 0:
        return True
    collinear_hits = (
        (d1 == 0 and _on_segment(p3, p4, p1)),
        (d2 == 0 and _on_segment(p3, p4, p2)),
        (d3 == 0 and _on_segment(p1, p2, p3)),
        (d4 == 0 and _on_segment(p1, p2, p4)),
    )
    return any(collinear_hits)


def _self_intersections(points: np.ndarray) -> list[tuple[int, int]]:
    """Return every pair of non-adjacent closed-polygon edge indices that touch or cross."""
    count = len(points)
    edges = [(points[i], points[(i + 1) % count]) for i in range(count)]
    # Skip each edge's own neighbours (they legitimately share a vertex); trimming the last
    # index for i == 0 keeps the closing edge from being compared against edge 0.
    pairs = [(i, j) for i in range(count) for j in range(i + 2, count - (1 if i == 0 else 0))]
    return [(i, j) for i, j in pairs if _segments_intersect(*edges[i], *edges[j])]


def _shoelace_area(points: np.ndarray) -> float:
    """Return the absolute polygon area via the shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rasterize(name: str, size: float, canvas: int, angle: float = 0.0) -> np.ndarray:
    """Fill one animal outline into a boolean canvas exactly as the generator would."""
    image = Image.new("L", (canvas, canvas), 0)
    poly = shape_polygon(name, center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    return np.asarray(image) > 0


def test_every_animal_member_has_a_table():
    """`AnimalShape` and the loaded outline tables are the same set, in both directions.

    Guards the two halves of the vocabulary drifting apart: a member without a document would raise only at draw time,
    deep inside a generation run, and a document without a member would ship dead weight in the wheel.

    """
    assert {shape.value for shape in AnimalShape} == set(ANIMAL_POLYGONS)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_is_a_simple_polygon(name):
    """No two non-adjacent outline edges touch or cross.

    A self-intersecting outline renders as an unpredictable bow-tie under Pillow's fill and makes the segmentation mask
    disagree with the rasterized pixels, so simplicity is the load-bearing invariant of a hand-authored table.

    """
    assert _self_intersections(ANIMAL_POLYGONS[name]) == []


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_has_no_repeated_vertex(name):
    """Every outline vertex is distinct, so the polygon has no pinch point.

    A duplicated vertex is the degenerate case the intersection predicate cannot flag as a crossing yet still produces a
    zero-width neck in the filled shape.

    """
    unique = np.unique(ANIMAL_POLYGONS[name], axis=0)
    assert len(unique) == len(ANIMAL_POLYGONS[name])


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_vertex_mean_is_the_origin(name):
    """The outline is centred on its vertex mean, matching `_base_polygon`'s convention.

    `shape_polygon` translates the base outline by the requested centre without re-centring, so a table whose mean
    drifts would place objects off their annotated centre.

    """
    assert ANIMAL_POLYGONS[name].mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_larger_extent_is_unit(name):
    """The larger of the two extents is exactly 1, so a `size` argument bounds the shape.

    Placement rejection and the size-ratio knobs both assume `size` is the true bounding extent; a table scaled
    differently would silently break `min_size_ratio`/`max_size_ratio`.

    """
    poly = ANIMAL_POLYGONS[name]
    extents = poly.max(axis=0) - poly.min(axis=0)
    assert max(extents) == pytest.approx(1.0)
    assert min(extents) > 0.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_vertex_count_is_in_the_authoring_band(name):
    """Each outline carries enough vertices to read as an animal without bloating labels.

    Segmentation labels emit every vertex, so an over-detailed table inflates label files while an under-detailed one
    stops being recognizable — this pins both ends of that trade-off.

    """
    assert 150 <= len(ANIMAL_POLYGONS[name]) <= 300


@pytest.mark.parametrize(("name", "band"), sorted(ARCHETYPE_ASPECT_BANDS.items()))
def test_table_matches_its_documented_archetype(name, band):
    """Each silhouette keeps the width-to-height ratio its archetype is documented with.

    The eight animals are chosen to be mutually distinguishable by gross proportion; coordinate tuning that pushes one
    into another's band would erode that separability.

    """
    poly = ANIMAL_POLYGONS[name]
    width, height = poly.max(axis=0) - poly.min(axis=0)
    low, high = band
    assert low <= width / height <= high


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_table_is_frozen_against_mutation(name):
    """The shared table rejects in-place writes.

    Tables are module-level constants handed to every caller; a consumer mutating one would corrupt every later sample
    in the process, which is exactly the failure this prevents.

    """
    with pytest.raises(ValueError, match="read-only"):
        ANIMAL_POLYGONS[name][0, 0] = 99.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_scaled_polygon_does_not_alias_the_table(name):
    """`shape_polygon` returns a fresh writable array rather than a view of the constant.

    Downstream code freely mutates the returned polygon (rotation, translation); aliasing the frozen table would either
    raise or silently corrupt the vocabulary.

    """
    poly = shape_polygon(name, center=(0.0, 0.0), size=10.0)
    poly[0, 0] += 1.0  # must not raise
    assert poly.base is not ANIMAL_POLYGONS[name]


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_polygon_scales_and_translates_like_a_geometric_shape(name):
    """`shape_polygon` bounds an animal by `size` and centres it on the requested point.

    This is the contract the placement loop depends on: it derives the candidate box from the
    polygon and rejects on that box, so a mis-scaled animal would break boundary handling.

    """
    poly = shape_polygon(name, center=(120.0, 80.0), size=40.0)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert max(x2 - x1, y2 - y1) == pytest.approx(40.0)
    assert poly.mean(axis=0) == pytest.approx([120.0, 80.0], abs=1e-9)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_rasterized_fill_matches_the_analytic_area(name):
    """Pillow's filled pixel count agrees with the outline's shoelace area.

    A self-intersecting or degenerate outline fills to a different area than its signed area predicts, so this is an
    end-to-end check that the annotation matches the drawn pixels.

    """
    size, canvas = 240.0, 320
    filled = int(_rasterize(name, size, canvas).sum())
    expected = _shoelace_area(ANIMAL_POLYGONS[name] * size)
    tolerance = RASTER_AREA_REL_TOLERANCE.get(name, 0.05)
    assert filled == pytest.approx(expected, rel=tolerance)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_silhouette_survives_a_small_rotated_draw(name):
    """A rotated animal at the smallest realistic size still rasterizes to visible pixels.

    `min_size_ratio` defaults to 0.1, so on a 320 px canvas the generator routinely draws 32 px animals at arbitrary
    angles; a thin outline that vanished there would emit empty labels.

    """
    image = Image.new("L", (64, 64), 0)
    poly = shape_polygon(name, center=(32.0, 32.0), size=32.0, angle=0.7)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    assert int((np.asarray(image) > 0).sum()) > 0


def test_normalized_rejects_a_degenerate_outline():
    """Fewer than three points cannot describe an outline and is refused up front.

    The helper runs at import time, so a malformed table must fail loudly at module load rather than produce an unusable
    constant that only breaks during generation.

    """
    with pytest.raises(ValueError, match="at least 3"):
        _normalized([(0.0, 0.0), (1.0, 1.0)])


def test_normalized_rejects_a_zero_extent_outline():
    """An outline whose vertices coincide is refused instead of dividing by zero.

    Guards the scaling step: a zero extent would otherwise yield NaN coordinates that propagate
    silently into every annotation derived from the polygon.

    """
    with pytest.raises(ValueError, match="zero extent"):
        _normalized([(2.0, 2.0), (2.0, 2.0), (2.0, 2.0)])


def test_normalized_centres_and_scales_a_raw_outline():
    """A raw authoring-scale outline comes back centred with its larger extent equal to 1.

    This is the guarantee that lets the tables be tuned visually in convenient coordinates without any hand-maintained
    normalization, which is where drift would otherwise creep in.

    """
    poly = _normalized([(10.0, 10.0), (30.0, 10.0), (30.0, 20.0), (10.0, 20.0)])
    assert poly.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert (poly.max(axis=0) - poly.min(axis=0)) == pytest.approx([1.0, 0.5])


def test_zoo_directory_holds_exactly_the_declared_animals():
    """The packaged JSON files and the loader's animal list are the same set.

    The files are data, not code, so a rename or a missed addition would otherwise surface only when a generation run
    reached that animal; comparing the directory listing to the declared names catches it at test time.

    """
    stems = {entry.name.removesuffix(".svg") for entry in _ZOO.iterdir() if entry.name.endswith(".svg")}
    assert stems == set(ZOO_ANIMAL_NAMES)


def test_zoo_animal_order_matches_the_enum():
    """The loader lists animals in `AnimalShape` declaration order, not alphabetically.

    Class ids come from the enum, so keeping the zoo in that order is what lets a reader line the two up; an
    alphabetical drift here would make every side-by-side table misleading.

    """
    assert tuple(shape.value for shape in AnimalShape) == ZOO_ANIMAL_NAMES


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_document_records_public_domain_provenance(name):
    """Every packaged silhouette carries a source URL and a public-domain license.

    The artwork is redistributed inside the wheel, so provenance is a licensing obligation rather than documentation
    polish: an entry that lost its license field, or gained a share-alike one, must fail the build.

    """
    source = ANIMAL_SOURCES[name]
    assert set(source) >= {"origin", "title", "license", "note"}
    assert source["origin"].startswith("https://")
    assert source["license"] in PUBLIC_DOMAIN_LICENSES


def test_every_animal_has_a_keypoint_table():
    """The outline and landmark vocabularies cover exactly the same animals.

    `Task.KEYPOINTS` looks a landmark table up by the shape it just drew, so an animal with an outline but no table
    would fail mid-generation rather than at configuration time.

    """
    assert sorted(ANIMAL_KEYPOINTS) == sorted(ANIMAL_POLYGONS)


def test_keypoint_shapes_config_matches_the_tables():
    """`config.tuple(AnimalShape)` names exactly the shapes that ship a landmark table.

    That constant is what `SyntheticConfig` validates against, and it is hand-maintained so the configuration layer
    stays free of NumPy; this is the check that keeps the two halves honest.

    """
    assert {shape.value for shape in tuple(AnimalShape)} == set(ANIMAL_KEYPOINTS)


def test_keypoint_colors_are_consistent_across_every_document():
    """One fill per landmark name, the same in all twelve documents, and never shared between names.

    The colors are a navigation aid: knowing that blue is always the neck only works if no document disagrees and no
    two landmarks look alike. Derived from the packaged assets rather than pinned against a constant, because the
    library itself never reads a fill — only the authoring editor does.

    """
    fills: dict[str, set[str]] = {}
    for name in ANIMAL_NAMES:
        root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - our own packaged asset, not untrusted input
        circles = root.find(_svg_tag("g[@id='keypoints']")).findall(_svg_tag("circle"))
        assert circles
        for circle in circles:
            fills.setdefault(circle.get(_zoo_attr("name")), set()).add(circle.get("fill"))

    inconsistent = {key: sorted(value) for key, value in fills.items() if len(value) != 1}
    assert inconsistent == {}
    # fish and whale omit the four hind landmarks, so only the mandatory names are guaranteed present
    assert set(fills) >= set(ANIMAL_KEYPOINT_NAMES) - _OPTIONAL_KEYPOINTS
    used = [next(iter(value)) for value in fills.values()]
    assert len(set(used)) == len(used)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoint_table_holds_sixteen_points(name):
    """Each animal carries exactly one `(x, y)` per schema name.

    YOLO's pose format declares a single dataset-wide `kpt_shape`, so a table of a different length cannot be
    represented at all — it has to be caught here.

    """
    assert ANIMAL_KEYPOINTS[name].shape == (len(ANIMAL_KEYPOINT_NAMES), 2)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoints_are_distinct_positions(name):
    """No two *present* landmarks of an animal sit on the same point.

    Two coincident landmarks would train a model on contradictory targets for two different names; it is also the exact
    symptom of a table authored by copying a neighbouring row. Restricted to present rows: `hash(nan)` is identity-based
    since Python 3.10, so an absent (all-NaN) row would otherwise count as trivially "distinct" without this filter, and
    the test would check one row fewer than it appears to for an animal with no hind limb.

    """
    table = ANIMAL_KEYPOINTS[name]
    present = [(round(float(x), 6), round(float(y), 6)) for x, y in table if not np.isnan(x)]
    assert len(set(present)) == len(present)


@pytest.mark.parametrize("angle", [0.0, 0.9])
@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoints_lie_inside_the_rasterized_silhouette(name, angle):
    """Every landmark falls on a filled pixel of the animal it annotates, rotated or not.

    This is the load-bearing property of a hand-placed table: a landmark that drifts off the silhouette teaches a model
    a point that is not on the object. Rasterizing rather than testing the polygon analytically also proves the
    landmark pipeline applies the same scale/rotate/translate as the outline pipeline.

    """
    size, canvas = 240.0, 320
    mask = _rasterize(name, size, canvas, angle=angle)
    points = animal_keypoints(AnimalShape(name), center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    outside = [
        key
        for key, (x, y) in zip(ANIMAL_KEYPOINT_NAMES, points, strict=False)
        if not np.isnan(x) and not mask[round(float(y)), round(float(x))]
    ]
    assert outside == []


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoint_table_is_frozen_against_mutation(name):
    """The shared landmark table rejects in-place writes.

    Like the outlines, these tables are module-level constants handed to every caller; a consumer mutating one would
    corrupt every later sample in the process.

    """
    with pytest.raises(ValueError, match="read-only"):
        ANIMAL_KEYPOINTS[name][0, 0] = 99.0


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_placed_keypoints_do_not_alias_the_table(name):
    """`animal_keypoints` returns a fresh writable array rather than a view of the constant.

    Callers are free to translate or clip the returned points; aliasing the frozen table would either raise or silently
    corrupt the vocabulary for the rest of the process.

    """
    points = animal_keypoints(AnimalShape(name), center=(0.0, 0.0), size=10.0)
    points[0, 0] += 1.0  # must not raise
    assert points.base is not ANIMAL_KEYPOINTS[name]


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_placed_keypoints_stay_within_the_placed_polygon(name):
    """Landmarks land inside the bounding box of the outline placed with the same arguments.

    The generator derives the annotation box from the polygon and the landmarks from the table; if the two pipelines
    disagreed about centre, scale or rotation, the points would sit outside the very box they are exported with.

    """
    center, size, angle = (120.0, 80.0), 40.0, 0.9
    poly = shape_polygon(name, center=center, size=size, angle=angle)
    points = animal_keypoints(AnimalShape(name), center=center, size=size, angle=angle)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    present = points[~np.isnan(points[:, 0])]
    # A hand-placed extremity landmark (`nose`, `tail`, either limb) sits close to the silhouette
    # boundary, so integer authoring coordinates can leave it a hair proud of the outline. The
    # tolerance scales with `size` to stay a fixed fraction of the shape: it absorbs that
    # quantization while still failing loudly on the thing this test is for, a centre/scale/rotation
    # disagreement between the two pipelines, which would put points whole pixels outside the box.
    tol = 1e-3 * size
    assert np.all(present[:, 0] >= x1 - tol)
    assert np.all(present[:, 0] <= x2 + tol)
    assert np.all(present[:, 1] >= y1 - tol)
    assert np.all(present[:, 1] <= y2 + tol)


def test_animal_keypoints_rejects_a_shape_without_a_table():
    """A geometric shape has no landmark table and is refused with a listing of the ones that do.

    A square is four-fold symmetric, so a fixed landmark on it carries no stable identity; failing loudly here is what
    turns that modelling fact into an actionable error instead of a `KeyError`.

    """
    with pytest.raises(ValueError, match="no keypoint table"):
        animal_keypoints(GeomShape.SQUARE, center=(0.0, 0.0), size=10.0)  # type: ignore[arg-type]


def test_normalized_pair_rejects_a_wrong_sized_landmark_table():
    """A document whose landmark table is not exactly sixteen points is refused at load time.

    The loader runs at import, so a malformed zoo file must fail loudly on import rather than produce an animal that
    breaks only once a keypoints run reaches it.

    """
    outline = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="exactly 16"):
        _normalized_pair(outline, [(0.5, 0.5), (1.0, 0.5)])


def test_normalized_pair_maps_landmarks_through_the_outline_frame():
    """Landmarks are centred and scaled by the *outline's* mean and extent, not their own.

    Normalizing the sixteen points independently would re-centre them on their own mean and detach them from the
    silhouette; this pins the shared-frame contract that keeps a landmark on the animal.

    """
    outline = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    corners = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    landmarks_in = [(10.0, 5.0), *corners, *corners, *corners, *corners[:3]]  # 1 centre + 15 corner repeats = 16
    polygon, landmarks = _normalized_pair(outline, landmarks_in)
    assert landmarks[0] == pytest.approx([0.0, 0.0])  # outline centre maps to the origin
    assert landmarks[1:5] == pytest.approx(polygon)  # corners map onto the normalized corners


def test_normalized_pair_rejects_a_half_nan_landmark_row():
    """A landmark with exactly one NaN coordinate is refused rather than silently treated as absent.

    A real absence (no hind limbs on a whale) is NaN in *both* coordinates; a single NaN is a parser bug — reading only
    one of `cx`/`cy` off a malformed circle, say — and must fail loudly rather than propagate as a half-broken point.

    """
    outline = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    landmarks = [(float("nan"), 5.0), *([(1.0, 1.0)] * 15)]
    with pytest.raises(ValueError, match="NaN"):
        _normalized_pair(outline, landmarks)


def test_absent_keypoints_match_the_pinned_matrix():
    """Each animal's absent-landmark set matches a literal, hand-reviewed pin, not just "whatever came out".

    The absent-slot matrix is decided per animal when its SVG is authored — pinning it as a literal means an edit that
    silently changes which animal has a hind limb is a diff a reviewer sees, not a value that just drifts.

    """
    for name in ANIMAL_NAMES:
        table = ANIMAL_KEYPOINTS[name]
        absent = {ANIMAL_KEYPOINT_NAMES[i] for i in range(len(ANIMAL_KEYPOINT_NAMES)) if np.isnan(table[i, 0])}
        assert absent == ABSENT_KEYPOINTS[name]
        assert absent <= _OPTIONAL_KEYPOINTS  # only the four hind-leg points may ever be absent


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_is_well_formed(name):
    """Every packaged SVG parses without error and has the SVG root tag.

    A hand-edit in Inkscape (or a bug in the authoring script) that produces invalid XML must fail here rather than
    surface as an opaque `ElementTree.ParseError` deep inside module import.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    assert root.tag == _svg_tag("svg")


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_vertices_lie_inside_the_viewbox(name):
    """Every outline vertex and every keypoint circle sits within the declared `0 0 1000 1000` viewBox.

    A vertex outside the viewBox would still parse and load fine but renders as a clipped or invisible artifact in any
    SVG viewer — an authoring-time regression this only-at-load-time loader cannot otherwise catch.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    (path_el,) = root.findall(_svg_tag("path"))
    points = _parse_path_d(path_el.get("d"), name)
    for x, y in points:
        assert 0 <= x <= 1000
        assert 0 <= y <= 1000
    group = root.find(f"{_svg_tag('g')}[@id='keypoints']")
    for circle in group.findall(_svg_tag("circle")):
        assert 0 <= float(circle.get("cx")) <= 1000
        assert 0 <= float(circle.get("cy")) <= 1000


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_zoo_svg_title_matches_zoo_title(name):
    """The visible `<title>` element and the machine-read `zoo:title` attribute agree.

    Both carry the same provenance text by design (`<title>` for a human hovering in a viewer, `zoo:title` for the
    loader) — letting them drift would mean the file lies to whichever reader looks at the "wrong" one.

    """
    root = ET.parse(str(_ZOO / f"{name}.svg")).getroot()  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    title_el = root.find(_svg_tag("title"))
    assert title_el is not None
    assert title_el.text == root.get(_zoo_attr("title"))


def test_parse_path_d_rejects_a_curve_command():
    """A `C`/`S`/`Q`/`T`/`A` command is refused with the Inkscape-flatten hint.

    The authoring script and this loader only ever speak straight lines; a curve reaching the parser means someone drew
    with the pen tool instead of flattening first, and the message should say so.

    """
    with pytest.raises(ValueError, match="curve command"):
        _parse_path_d("M 0 0 C 1 1 2 2 3 3 Z", "test")


def test_parse_path_d_rejects_a_second_subpath():
    """A second `M`/`m` mid-path (a second subpath) is refused; a zoo outline is exactly one closed path."""
    with pytest.raises(ValueError, match="second subpath"):
        _parse_path_d("M 0 0 L 1 1 Z M 5 5 L 6 6 Z", "test")


def test_parse_path_d_rejects_an_unclosed_path():
    """A path missing its trailing `Z`/`z` is refused rather than silently treated as closed."""
    with pytest.raises(ValueError, match="not closed"):
        _parse_path_d("M 0 0 L 1 1 L 2 2", "test")


def test_parse_path_d_rejects_an_unsupported_command():
    """A recognised-but-unsupported letter (not curve, not M/L/H/V/Z) is refused, not silently skipped."""
    with pytest.raises(ValueError, match="unsupported command"):
        _parse_path_d("M 0 0 L 1 1 B 2 2 Z", "test")


@pytest.mark.parametrize(
    "d",
    [
        pytest.param("M 10 10 L 20 10 L 20 20 L 10 20 Z", id="absolute"),
        pytest.param("m 10 10 l 10 0 l 0 10 l -10 0 z", id="relative"),
        pytest.param("M 10 10 H 20 V 20 H 10 Z", id="horizontal-vertical"),
        pytest.param("M 10 10 20 10 20 20 10 20 Z", id="implicit-lineto"),
    ],
)
def test_parse_path_d_tolerant_forms_agree(d):
    """Absolute, relative, H/V, and implicit-lineto forms of the same square parse to the same vertices.

    Inkscape's default save is relative with H/V shorthand; the authoring script always writes absolute M/L/Z. Both must
    round-trip to the same geometry, or a save-in-Inkscape-then-reload cycle silently reshapes the outline.

    """
    expected = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    assert _parse_path_d(d, "test") == expected


def test_reject_transforms_flags_the_offending_element():
    """A `transform` attribute anywhere in the document is refused, naming the element and the Inkscape fix."""
    root = ET.fromstring(f'<svg xmlns="{_SVG_NS}"><path transform="translate(1,1)" d="M 0 0 Z"/></svg>')  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    with pytest.raises(ValueError, match="transform"):
        _reject_transforms(root, "test")


def test_read_keypoints_rejects_an_unknown_zoo_name():
    """A `zoo:name` outside the nine-name schema is refused rather than silently ignored."""
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}">'
        f'<g id="keypoints"><circle zoo:name="wing" cx="1" cy="1"/></g></svg>'
    )
    with pytest.raises(ValueError, match="unknown"):
        _read_keypoints(root, "test")


def test_read_keypoints_rejects_a_duplicate_zoo_name():
    """Two circles claiming the same `zoo:name` are refused; only one landmark per name is meaningful."""
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">'
        f'<circle zoo:name="mouth" cx="1" cy="1"/><circle zoo:name="mouth" cx="2" cy="2"/>'
        "</g></svg>"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _read_keypoints(root, "test")


def test_read_keypoints_rejects_a_missing_mandatory_landmark():
    """A document missing a mandatory (non-optional) landmark is refused, listing what is missing."""
    root = ET.fromstring(  # noqa: S314 - fixed literal XML, not untrusted input
        f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">'
        f'<circle zoo:name="mouth" cx="1" cy="1"/></g></svg>'
    )
    with pytest.raises(ValueError, match="missing the landmark"):
        _read_keypoints(root, "test")


def test_read_keypoints_allows_missing_hind_legs():
    """A document with every mandatory landmark but no hind-leg points loads (the only optional names)."""
    mandatory = [name for name in ANIMAL_KEYPOINT_NAMES if name not in _OPTIONAL_KEYPOINTS]
    circles = "".join(f'<circle zoo:name="{name}" cx="1" cy="1"/>' for name in mandatory)
    root = ET.fromstring(f'<svg xmlns="{_SVG_NS}" xmlns:zoo="{_ZOO_NS}"><g id="keypoints">{circles}</g></svg>')  # noqa: S314 - fixed literal XML or our own packaged asset, not untrusted input
    present = _read_keypoints(root, "test")
    assert set(present) == set(mandatory)
