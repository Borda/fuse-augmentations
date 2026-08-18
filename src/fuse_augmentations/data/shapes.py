"""Analytic shape geometry: polygons, keypoints, rotation, and box derivation.

Pure NumPy, no image-library dependency. Polygons are ``(num_points, 2)`` float
arrays of ``(x, y)`` pixel coordinates. Boxes are derived from the polygon so
annotations always match the rasterized pixels. :func:`animal_keypoints` places an
animal's landmark table through the very same scale/rotate/translate pipeline as
:func:`shape_polygon`, so keypoints always land on the silhouette that was drawn.

Examples:
    ```pycon
    >>> from fuse_augmentations.data.shapes import shape_polygon, polygon_to_bbox_xyxy
    >>> poly = shape_polygon("square", center=(5.0, 5.0), size=4.0)
    >>> polygon_to_bbox_xyxy(poly)
    (3.0, 3.0, 7.0, 7.0)

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fuse_augmentations.data.animal_shapes import ANIMAL_KEYPOINTS, ANIMAL_POLYGONS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # Type-only: at runtime this module reads a shape's ``.value`` and never imports the
    # configuration layer, keeping the geometry layer free of that dependency.
    from fuse_augmentations.data.config import Shape

#: Number of vertices used to approximate a circle outline.
CIRCLE_POINTS = 32

#: Height-to-width ratio for the ``rectangle`` shape (non-square, so its OBB is oriented).
RECT_ASPECT = 0.5

#: Analytically computed shape names, in :class:`~fuse_augmentations.data.config.Shape` order.
GEOMETRIC_SHAPES: tuple[str, ...] = ("square", "rectangle", "triangle", "circle")


def _base_polygon(shape: str, size: float) -> NDArray[np.float64]:
    """Return an origin-centered polygon for ``shape`` spanning ``size`` pixels.

    Geometric shapes are computed analytically; animal shapes are looked up in
    :data:`~fuse_augmentations.data.animal_shapes.ANIMAL_POLYGONS`, whose tables share this
    function's unit convention (vertex centroid at the origin, larger extent equal to ``1``)
    and therefore need only a scale by ``size``.

    Args:
        shape: A :class:`~fuse_augmentations.data.config.Shape` value — one of the four
            geometric names (``"square"``, ``"rectangle"``, ``"triangle"``, ``"circle"``)
            or one of the eight animal names (``"duck"``, ``"snail"``, ``"elephant"``,
            ``"giraffe"``, ``"fish"``, ``"turtle"``, ``"snake"``, ``"rabbit"``).
        size: Bounding size (side / diameter / larger extent) in pixels.

    Returns:
        ``(num_points, 2)`` float array centered at the origin.

    Raises:
        ValueError: If ``shape`` is not recognised.

    """
    half = size / 2.0
    if shape == "square":
        return np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float64)
    if shape == "rectangle":
        h = half * RECT_ASPECT
        return np.array([[-half, -h], [half, -h], [half, h], [-half, h]], dtype=np.float64)
    if shape == "triangle":
        height = size * np.sqrt(3.0) / 2.0
        return np.array([[0.0, -2.0 * height / 3.0], [half, height / 3.0], [-half, height / 3.0]], dtype=np.float64)
    if shape == "circle":
        angles = np.linspace(0.0, 2.0 * np.pi, CIRCLE_POINTS, endpoint=False)
        return np.stack([half * np.cos(angles), half * np.sin(angles)], axis=1).astype(np.float64)
    animal = ANIMAL_POLYGONS.get(shape)
    if animal is not None:
        # One lookup rather than eight near-identical branches; the stored table is frozen,
        # so multiplying returns a fresh writable array and never aliases the constant.
        return animal * size
    known = ", ".join((*GEOMETRIC_SHAPES, *ANIMAL_POLYGONS))
    raise ValueError(f"unknown shape {shape!r}; expected one of {known}")


def _rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Return the 2x2 counter-clockwise rotation matrix for ``angle`` radians."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def rotate_polygon(
    points: NDArray[np.float64], angle: float, center: tuple[float, float] = (0.0, 0.0)
) -> NDArray[np.float64]:
    """Rotate ``points`` by ``angle`` radians about ``center``.

    Args:
        points: ``(num_points, 2)`` array of ``(x, y)`` coordinates.
        angle: Rotation angle in radians (counter-clockwise).
        center: Pivot point ``(x, y)``.

    Returns:
        Rotated ``(num_points, 2)`` array.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.shapes import rotate_polygon
        >>> pts = np.array([[1.0, 0.0]])
        >>> out = rotate_polygon(pts, np.pi / 2)
        >>> bool(np.allclose(out, [[0.0, 1.0]]))
        True

        ```

    """
    pivot = np.asarray(center, dtype=np.float64)
    rotated: NDArray[np.float64] = (points - pivot) @ _rotation_matrix(angle).T + pivot
    return rotated


def _placed(points: NDArray[np.float64], center: tuple[float, float], angle: float) -> NDArray[np.float64]:
    """Rotate origin-centered ``points`` by ``angle`` and translate them onto ``center``.

    Shared by :func:`shape_polygon` and :func:`animal_keypoints` so an outline and its landmarks
    can never drift apart: both are placed by this one implementation.

    Args:
        points: ``(num_points, 2)`` array already scaled and centered on the origin.
        center: Target center ``(x, y)`` in pixels.
        angle: Rotation in radians applied about the origin (i.e. about the shape center).

    Returns:
        ``(num_points, 2)`` float array in image coordinates.

    """
    if angle:
        points = points @ _rotation_matrix(angle).T
    return points + np.asarray(center, dtype=np.float64)


def shape_polygon(shape: str, center: tuple[float, float], size: float, angle: float = 0.0) -> NDArray[np.float64]:
    """Build a rotated, translated polygon for a shape.

    Args:
        shape: Shape name (see :func:`_base_polygon`).
        center: Target center ``(x, y)`` in pixels.
        size: Bounding size in pixels.
        angle: Rotation in radians applied about the shape center.

    Returns:
        ``(num_points, 2)`` float array in image coordinates.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.shapes import shape_polygon
        >>> poly = shape_polygon("triangle", center=(10.0, 10.0), size=6.0)
        >>> poly.shape
        (3, 2)

        ```

    """
    return _placed(_base_polygon(shape, size), center, angle)


def animal_keypoints(shape: Shape, center: tuple[float, float], size: float, angle: float = 0.0) -> NDArray[np.float64]:
    """Place one animal's landmark table into image coordinates.

    The table is looked up in :data:`~fuse_augmentations.data.animal_shapes.ANIMAL_KEYPOINTS`,
    scaled, rotated, and translated exactly as :func:`shape_polygon` treats the matching outline,
    so passing the same ``center``, ``size``, and ``angle`` to both puts every landmark on the
    silhouette that was drawn. No randomness is involved: the result is a pure function of the
    placement the generator already sampled.

    Args:
        shape: An animal :class:`~fuse_augmentations.data.config.Shape` member; the geometric
            shapes have no landmark table (a square's 4-fold symmetry gives a fixed landmark no
            stable identity).
        center: Target center ``(x, y)`` in pixels — the same value passed to :func:`shape_polygon`.
        size: Bounding size in pixels — the same value passed to :func:`shape_polygon`.
        angle: Rotation in radians about the shape center — likewise.

    Returns:
        ``(5, 2)`` float array of landmark coordinates in image pixels, ordered by
        :data:`~fuse_augmentations.data.config.KEYPOINT_NAMES`. Points may fall outside the canvas;
        clipping is the caller's decision.

    Raises:
        ValueError: If ``shape`` has no keypoint table.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Shape
        >>> from fuse_augmentations.data.shapes import animal_keypoints
        >>> points = animal_keypoints(Shape.DUCK, center=(50.0, 50.0), size=20.0)
        >>> points.shape
        (5, 2)

        ```

    """
    table = ANIMAL_KEYPOINTS.get(shape.value)
    if table is None:
        known = ", ".join(ANIMAL_KEYPOINTS)
        raise ValueError(f"shape {shape.value!r} has no keypoint table; expected one of {known}")
    # The stored table is frozen, so multiplying returns a fresh writable array, never an alias.
    return _placed(table * size, center, angle)


def polygon_to_bbox_xyxy(points: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box ``(x_min, y_min, x_max, y_max)`` of a polygon.

    Args:
        points: ``(num_points, 2)`` array of coordinates.

    Returns:
        Bounding box tuple in pixels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.shapes import polygon_to_bbox_xyxy
        >>> polygon_to_bbox_xyxy(np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 4.0]]))
        (0.0, 2.0, 3.0, 5.0)

        ```

    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))


def _convex_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the convex hull of ``points`` (counter-clockwise) via monotone chain.

    Args:
        points: ``(num_points, 2)`` array.

    Returns:
        ``(hull_points, 2)`` array of hull vertices.

    """
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _cross(o: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[NDArray[np.float64]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[NDArray[np.float64]] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_to_obb(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the minimum-area oriented bounding box of a polygon as four corners.

    Uses the rotating-calipers property that a minimum-area rectangle shares an edge
    with the convex hull. For a rotation-invariant outline (e.g. a circle) the result
    is effectively the axis-aligned box.

    Args:
        points: ``(num_points, 2)`` array of polygon coordinates.

    Returns:
        ``(4, 2)`` array of corner coordinates in order.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.shapes import polygon_to_obb
        >>> corners = polygon_to_obb(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]))
        >>> corners.shape
        (4, 2)

        ```

    """
    hull = _convex_hull(points)
    if len(hull) < 3:
        x1, y1, x2, y2 = polygon_to_bbox_xyxy(points)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

    best_corners: NDArray[np.float64] | None = None
    best_area = np.inf
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        angle = np.arctan2(edge[1], edge[0])
        rot = hull @ _rotation_matrix(-angle).T
        min_xy = rot.min(axis=0)
        max_xy = rot.max(axis=0)
        area = float((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))
        if area < best_area:
            best_area = area
            aligned = np.array(
                [
                    [min_xy[0], min_xy[1]],
                    [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]],
                    [min_xy[0], max_xy[1]],
                ],
                dtype=np.float64,
            )
            best_corners = aligned @ _rotation_matrix(angle).T
    assert best_corners is not None  # noqa: S101 - loop over >=3 hull edges always assigns
    return best_corners


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Return the intersection-over-union of two axis-aligned xyxy boxes.

    Args:
        box_a: First box ``(x_min, y_min, x_max, y_max)``.
        box_b: Second box ``(x_min, y_min, x_max, y_max)``.

    Returns:
        IoU in ``[0, 1]``; ``0.0`` when the boxes do not overlap.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.shapes import bbox_iou
        >>> bbox_iou((0, 0, 2, 2), (1, 1, 3, 3))
        0.14285714285714285

        ```

    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
