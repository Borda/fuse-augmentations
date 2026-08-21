"""Geometry tests: polygon construction, rotation, and box derivation."""

from __future__ import annotations

import numpy as np
import pytest

from fuse_augmentations.data.animals import AnimalShape
from fuse_augmentations.data.geometry import (
    GeomShape,
    bbox_iou,
    polygon_to_bbox_xyxy,
    polygon_to_obb,
    rotate_polygon,
    shape_polygon,
)
from fuse_augmentations.data.symbols import SymbolShape


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _area_centroid(points: np.ndarray) -> np.ndarray:
    """Independent shoelace-formula centroid, kept separate from `landmarks._polygon_centroid`."""
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def _rect_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


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
    pairs = [(i, j) for i in range(count) for j in range(i + 2, count - (1 if i == 0 else 0))]
    return [(i, j) for i, j in pairs if _segments_intersect(*edges[i], *edges[j])]


@pytest.mark.parametrize("shape", [s.value for s in (*GeomShape, *AnimalShape, *SymbolShape)])
def test_shape_polygon_is_centered(shape: str) -> None:
    """Polygon shapes are centered at the specified point, by area centroid (not vertex mean).

    A traced or hand-authored outline can have vertices bunched unevenly along its edges (an arrow's barbs, an animal's
    long neck), so the vertex mean of a placed polygon need not sit on the requested center — only its area centroid
    (center of mass) is guaranteed to.

    """
    poly = shape_polygon(shape, center=(50.0, 50.0), size=20.0)
    assert poly.ndim == 2
    assert poly.shape[1] == 2
    centroid = _area_centroid(poly)
    assert np.allclose(centroid, [50.0, 50.0], atol=1e-6)


def test_square_bbox_exact() -> None:
    """Square bbox computation returns exact coordinate tuple."""
    poly = shape_polygon("square", center=(5.0, 5.0), size=4.0)
    assert polygon_to_bbox_xyxy(poly) == (3.0, 3.0, 7.0, 7.0)


def test_rectangle_is_non_square() -> None:
    """Rectangle bounding box has different width and height."""
    poly = shape_polygon("rectangle", center=(0.0, 0.0), size=10.0)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert (x2 - x1) != pytest.approx(y2 - y1)


@pytest.mark.parametrize("shape", ["square", "rectangle", "triangle"])
def test_rotation_preserves_area(shape: str) -> None:
    """Polygon rotation preserves area to floating point precision."""
    poly = shape_polygon(shape, center=(30.0, 30.0), size=12.0)
    rotated = rotate_polygon(poly, np.pi / 5, center=(30.0, 30.0))
    assert _polygon_area(rotated) == pytest.approx(_polygon_area(poly), rel=1e-6)


def test_zero_skew_matches_the_unskewed_polygon() -> None:
    """`skew=0.0` is a true no-op, byte-identical to omitting the argument.

    This is the property that keeps every pre-existing seeded configuration's output unchanged:
    `SyntheticConfig.asymmetry_jitter` defaults to `0.0`, so the generator must never observe a difference from before
    the knob existed.

    """
    plain = shape_polygon("square", center=(10.0, 10.0), size=8.0, angle=0.3)
    skewed = shape_polygon("square", center=(10.0, 10.0), size=8.0, angle=0.3, skew=0.0)
    assert np.array_equal(plain, skewed)


def test_positive_skew_narrows_only_the_right_half_of_a_square() -> None:
    """A positive `skew` scales the `x > 0` half toward the axis, leaving the `x < 0` half fixed."""
    poly = shape_polygon("square", center=(0.0, 0.0), size=10.0, skew=0.3)
    xs = sorted(poly[:, 0])
    assert xs[:2] == pytest.approx([-5.0, -5.0])  # left edge untouched
    assert xs[2:] == pytest.approx([3.5, 3.5])  # right edge narrowed by 30%


def test_negative_skew_narrows_only_the_left_half_of_a_square() -> None:
    """A negative `skew` narrows the `x < 0` half instead, mirroring the positive case."""
    poly = shape_polygon("square", center=(0.0, 0.0), size=10.0, skew=-0.3)
    xs = sorted(poly[:, 0])
    assert xs[:2] == pytest.approx([-3.5, -3.5])  # left edge narrowed by 30%
    assert xs[2:] == pytest.approx([5.0, 5.0])  # right edge untouched


@pytest.mark.parametrize("shape", [s.value for s in (*GeomShape, *AnimalShape, *SymbolShape)])
@pytest.mark.parametrize("skew", [0.49, -0.49])
def test_skew_near_its_upper_bound_keeps_every_shape_simple(shape: str, skew: float) -> None:
    """Every shape stays a simple (non-self-intersecting) polygon at the edge of the allowed skew range.

    `SyntheticConfig.asymmetry_jitter` caps at just under `0.5`; this pins that the cap is actually safe for every shape
    this package draws, not just the ones exercised by example configs.

    """
    poly = shape_polygon(shape, center=(0.0, 0.0), size=10.0, skew=skew)
    assert _self_intersections(poly) == []


def test_obb_returns_four_corners_within_aabb() -> None:
    """OBB area is smaller or equal to AABB area."""
    poly = shape_polygon("rectangle", center=(40.0, 40.0), size=16.0, angle=0.6)
    corners = polygon_to_obb(poly)
    assert corners.shape == (4, 2)
    obb_area = _polygon_area(corners)
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area <= aabb_area * (1.0 + 1e-6)


def test_obb_tight_for_axis_aligned_rectangle() -> None:
    """OBB equals AABB when rectangle is axis-aligned."""
    poly = shape_polygon("rectangle", center=(0.0, 0.0), size=10.0, angle=0.0)
    obb_area = _polygon_area(polygon_to_obb(poly))
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area == pytest.approx(aabb_area, rel=1e-6)


def test_circle_obb_approximates_aabb() -> None:
    """OBB for circle approximates AABB within 5% relative tolerance."""
    poly = shape_polygon("circle", center=(20.0, 20.0), size=10.0)
    obb_area = _polygon_area(polygon_to_obb(poly))
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area == pytest.approx(aabb_area, rel=0.05)


def _best_corner_set_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Max corner distance between two (4, 2) corner sets, treated as unordered point sets.

    Both winding direction and starting corner are unconstrained — `polygon_to_obb` promises the four *corners* of a
    box, not a canonical starting index — so this sorts each set by angle around its own centroid, then tries every
    cyclic shift and both windings of ``b`` against ``a``, returning the best (smallest) achievable max-corner-distance.

    """
    a_sorted = a[np.argsort(np.arctan2(*(a - a.mean(axis=0)).T[::-1]))]
    b_sorted = b[np.argsort(np.arctan2(*(b - b.mean(axis=0)).T[::-1]))]
    best = np.inf
    for candidate in (b_sorted, b_sorted[::-1]):
        for shift in range(4):
            distances = np.linalg.norm(a_sorted - np.roll(candidate, shift, axis=0), axis=1)
            best = min(best, float(distances.max()))
    return best


@pytest.mark.parametrize("shape", [s.value for s in (*GeomShape, *AnimalShape, *SymbolShape)])
def test_obb_is_rotation_equivariant(shape: str) -> None:
    """The chosen OBB rotates rigidly with the shape — it never flips to a different tied candidate.

    A shape with reflective symmetry (every animal and symbol here, plus the geometric square and rectangle) can have
    more than one candidate box achieving the true minimum area — `kite` and `teardrop` were the two shapes that first
    exposed this bug (see `polygon_to_obb`'s docstring for why a right or acute triangle would tie too, even though the
    shipped `GeomShape.TRIANGLE` is deliberately obtuse-scalene and does not). Before this was fixed, which tied
    candidate won depended on `_convex_hull`'s rotated-frame lexicographic sort rather than the shape's own geometry, so
    the box orientation "wobbled" between the tied candidates as an otherwise-identical shape rotated — even though
    nothing about the shape changed relative to its own frame.

    """
    center, size = (0.0, 0.0), 100.0
    reference = polygon_to_obb(shape_polygon(shape, center=center, size=size, angle=0.0))
    for angle_deg in (10, 47, 91, 133, 179, 250, 311):
        angle = np.radians(float(angle_deg))
        actual = polygon_to_obb(shape_polygon(shape, center=center, size=size, angle=angle))
        expected = rotate_polygon(reference, angle, center=center)
        assert _best_corner_set_distance(expected, actual) < 1e-3, f"{shape} at {angle_deg} degrees"


def test_unknown_shape_raises() -> None:
    """Unknown shape name raises ValueError."""
    with pytest.raises(ValueError, match="unknown shape"):
        shape_polygon("hexagon", center=(0.0, 0.0), size=4.0)


def test_bbox_iou_known_value() -> None:
    """Bbox IOU for two overlapping boxes matches expected value."""
    assert bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_bbox_iou_disjoint_is_zero() -> None:
    """Bbox IOU for disjoint boxes is zero."""
    assert bbox_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
