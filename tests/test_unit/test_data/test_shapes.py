"""Geometry tests: polygon construction, rotation, and box derivation."""

from __future__ import annotations

import numpy as np
import pytest

from fuse_augmentations.data.config import Shape
from fuse_augmentations.data.shapes import (
    bbox_iou,
    polygon_to_bbox_xyxy,
    polygon_to_obb,
    rotate_polygon,
    shape_polygon,
)


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rect_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


@pytest.mark.parametrize("shape", [s.value for s in Shape])
def test_shape_polygon_is_centered(shape):
    poly = shape_polygon(shape, center=(50.0, 50.0), size=20.0)
    assert poly.ndim == 2
    assert poly.shape[1] == 2
    centroid = poly.mean(axis=0)
    assert np.allclose(centroid, [50.0, 50.0], atol=1.0)


def test_square_bbox_exact():
    poly = shape_polygon("square", center=(5.0, 5.0), size=4.0)
    assert polygon_to_bbox_xyxy(poly) == (3.0, 3.0, 7.0, 7.0)


def test_rectangle_is_non_square():
    poly = shape_polygon("rectangle", center=(0.0, 0.0), size=10.0)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    assert (x2 - x1) != pytest.approx(y2 - y1)


@pytest.mark.parametrize("shape", ["square", "rectangle", "triangle"])
def test_rotation_preserves_area(shape):
    poly = shape_polygon(shape, center=(30.0, 30.0), size=12.0)
    rotated = rotate_polygon(poly, np.pi / 5, center=(30.0, 30.0))
    assert _polygon_area(rotated) == pytest.approx(_polygon_area(poly), rel=1e-6)


def test_obb_returns_four_corners_within_aabb():
    poly = shape_polygon("rectangle", center=(40.0, 40.0), size=16.0, angle=0.6)
    corners = polygon_to_obb(poly)
    assert corners.shape == (4, 2)
    obb_area = _polygon_area(corners)
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area <= aabb_area * (1.0 + 1e-6)


def test_obb_tight_for_axis_aligned_rectangle():
    poly = shape_polygon("rectangle", center=(0.0, 0.0), size=10.0, angle=0.0)
    obb_area = _polygon_area(polygon_to_obb(poly))
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area == pytest.approx(aabb_area, rel=1e-6)


def test_circle_obb_approximates_aabb():
    poly = shape_polygon("circle", center=(20.0, 20.0), size=10.0)
    obb_area = _polygon_area(polygon_to_obb(poly))
    aabb_area = _rect_area(polygon_to_bbox_xyxy(poly))
    assert obb_area == pytest.approx(aabb_area, rel=0.05)


def test_unknown_shape_raises():
    with pytest.raises(ValueError, match="unknown shape"):
        shape_polygon("hexagon", center=(0.0, 0.0), size=4.0)


def test_bbox_iou_known_value():
    assert bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_bbox_iou_disjoint_is_zero():
    assert bbox_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
