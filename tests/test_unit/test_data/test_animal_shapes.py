"""Animal table validity: simplicity, unit convention, archetype, landmarks, provenance."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fuse_augmentations.data.animal_shapes import (
    _KEYPOINT_ORDER,
    _ZOO,
    ANIMAL_KEYPOINTS,
    ANIMAL_POLYGONS,
    ANIMAL_SOURCES,
    _normalized,
    _normalized_pair,
)
from fuse_augmentations.data.animal_shapes import ANIMAL_NAMES as ZOO_ANIMAL_NAMES
from fuse_augmentations.data.config import DEFAULT_SHAPES, KEYPOINT_NAMES, KEYPOINT_SHAPES, Shape
from fuse_augmentations.data.shapes import animal_keypoints, polygon_to_bbox_xyxy, shape_polygon

ANIMAL_NAMES = sorted(ANIMAL_POLYGONS)

# Pillow includes boundary pixels when filling polygons. At the test scale the snake's long,
# narrow, wavy body has a 7.3% rasterization-vs-shoelace area bias; the other silhouettes remain
# within the general 5% bound. The bias shrinks as the same outline is rasterized at higher scale.
RASTER_AREA_REL_TOLERANCE = {"snake": 0.08}

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
    "snail": (1.95, 2.50),  # shelled-crawler
    "elephant": (1.70, 2.20),  # bulky-quadruped
    "giraffe": (0.66, 0.90),  # tall-thin
    "fish": (2.30, 2.90),  # streamlined
    "turtle": (1.85, 2.35),  # domed-shell
    "snake": (4.40, 5.60),  # elongated-legless
    "rabbit": (1.05, 1.35),  # compact-eared
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


def test_every_non_default_shape_has_a_table():
    """The eight animal enum members and the eight outline tables are the same set.

    Guards the two halves of the vocabulary drifting apart: adding a `Shape` member without a
    table would raise only at draw time, deep inside a generation run.

    """
    animal_members = {shape.value for shape in Shape if shape not in DEFAULT_SHAPES}
    assert animal_members == set(ANIMAL_POLYGONS)


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
    assert 15 <= len(ANIMAL_POLYGONS[name]) <= 48


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
    stems = {entry.name.removesuffix(".json") for entry in _ZOO.iterdir() if entry.name.endswith(".json")}
    assert stems == set(ZOO_ANIMAL_NAMES)


def test_zoo_animal_order_matches_the_shape_enum():
    """The loader lists animals in `Shape` declaration order, not alphabetically.

    Class ids come from the enum, so keeping the zoo in that order is what lets a reader line the two up; an
    alphabetical drift here would make every side-by-side table misleading.

    """
    assert tuple(shape.value for shape in Shape if shape not in DEFAULT_SHAPES) == ZOO_ANIMAL_NAMES


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
    """`config.KEYPOINT_SHAPES` names exactly the shapes that ship a landmark table.

    That constant is what `SyntheticConfig` validates against, and it is hand-maintained so the configuration layer
    stays free of NumPy; this is the check that keeps the two halves honest.

    """
    assert {shape.value for shape in KEYPOINT_SHAPES} == set(ANIMAL_KEYPOINTS)


def test_keypoint_order_matches_the_published_schema():
    """The loader's landmark order is the order `config.KEYPOINT_NAMES` publishes.

    Landmarks are written positionally into COCO and YOLO records, so a divergence here would silently relabel every
    exported point rather than raise.

    """
    assert _KEYPOINT_ORDER == KEYPOINT_NAMES


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoint_table_holds_five_points(name):
    """Each animal carries exactly one `(x, y)` per schema name.

    YOLO's pose format declares a single dataset-wide `kpt_shape`, so a table of a different length cannot be
    represented at all — it has to be caught here.

    """
    assert ANIMAL_KEYPOINTS[name].shape == (len(KEYPOINT_NAMES), 2)


@pytest.mark.parametrize("name", ANIMAL_NAMES)
def test_keypoints_are_distinct_positions(name):
    """No two landmarks of an animal sit on the same point.

    Two coincident landmarks would train a model on contradictory targets for two different names; it is also the exact
    symptom of a table authored by copying a neighbouring row.

    """
    table = ANIMAL_KEYPOINTS[name]
    assert len({(round(float(x), 6), round(float(y), 6)) for x, y in table}) == len(KEYPOINT_NAMES)


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
    points = animal_keypoints(Shape(name), center=(canvas / 2.0, canvas / 2.0), size=size, angle=angle)
    outside = [
        key for key, (x, y) in zip(KEYPOINT_NAMES, points, strict=False) if not mask[round(float(y)), round(float(x))]
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
    points = animal_keypoints(Shape(name), center=(0.0, 0.0), size=10.0)
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
    points = animal_keypoints(Shape(name), center=center, size=size, angle=angle)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert np.all(points[:, 0] >= x1 - 1e-9)
    assert np.all(points[:, 0] <= x2 + 1e-9)
    assert np.all(points[:, 1] >= y1 - 1e-9)
    assert np.all(points[:, 1] <= y2 + 1e-9)


def test_animal_keypoints_rejects_a_shape_without_a_table():
    """A geometric shape has no landmark table and is refused with a listing of the ones that do.

    A square is four-fold symmetric, so a fixed landmark on it carries no stable identity; failing loudly here is what
    turns that modelling fact into an actionable error instead of a `KeyError`.

    """
    with pytest.raises(ValueError, match="no keypoint table"):
        animal_keypoints(Shape.SQUARE, center=(0.0, 0.0), size=10.0)


def test_normalized_pair_rejects_a_wrong_sized_landmark_table():
    """A document whose landmark table is not exactly five points is refused at load time.

    The loader runs at import, so a malformed zoo file must fail loudly on import rather than produce an animal that
    breaks only once a keypoints run reaches it.

    """
    outline = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="exactly 5"):
        _normalized_pair(outline, [(0.5, 0.5), (1.0, 0.5)])


def test_normalized_pair_maps_landmarks_through_the_outline_frame():
    """Landmarks are centred and scaled by the *outline's* mean and extent, not their own.

    Normalizing the five points independently would re-centre them on their own mean and detach them from the
    silhouette; this pins the shared-frame contract that keeps a landmark on the animal.

    """
    outline = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
    polygon, landmarks = _normalized_pair(outline, [(10.0, 5.0), (0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)])
    assert landmarks[0] == pytest.approx([0.0, 0.0])  # outline centre maps to the origin
    assert landmarks[1:] == pytest.approx(polygon)  # corners map onto the normalized corners
