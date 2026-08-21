"""Family-agnostic polygon math: placement, rotation, and box derivation.

Pure NumPy, no image-library dependency and no knowledge of which shape families exist. Polygons are
``(num_points, 2)`` float arrays of ``(x, y)`` pixel coordinates. Boxes are derived from the polygon
so annotations always match the rasterized pixels.

The analytic shape family that used to live here moved to
:mod:`~fuse_augmentations.data.primitives`, and the per-family outline dispatch moved to
:mod:`~fuse_augmentations.data.families`. What is left is the math every family shares — which is
what lets each family module import this one without the import cycle the old arrangement dodged
with deferred imports.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from fuse_augmentations.data.geometry import polygon_to_bbox_xyxy
    >>> square = np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]])
    >>> polygon_to_bbox_xyxy(square)
    (3.0, 3.0, 7.0, 7.0)

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        >>> from fuse_augmentations.data.geometry import rotate_polygon
        >>> pts = np.array([[1.0, 0.0]])
        >>> out = rotate_polygon(pts, np.pi / 2)
        >>> bool(np.allclose(out, [[0.0, 1.0]]))
        True

        ```

    """
    pivot = np.asarray(center, dtype=np.float64)
    rotated: NDArray[np.float64] = (points - pivot) @ _rotation_matrix(angle).T + pivot
    return rotated


def _skewed(points: NDArray[np.float64], skew: float) -> NDArray[np.float64]:
    """Narrow one half of origin-centered ``points`` toward the vertical axis.

    Every shape in this package but :attr:`PrimitiveShape.CIRCLE` is drawn mirror-symmetric about its
    own local vertical axis (before rotation), so its oriented bounding box would otherwise show
    identical margins on both sides of that axis — see
    :attr:`~fuse_augmentations.data.config.SyntheticConfig.asymmetry_jitter`. This breaks that
    symmetry per placed instance instead.

    Args:
        points: ``(num_points, 2)`` array already centered on the origin, in the shape's own
            pre-rotation frame — "left" and "right" are only meaningful there, which is why this
            runs before :func:`place_points` applies ``angle``. NaN rows (absent landmarks) pass through
            unchanged: a comparison against NaN is always false, so the selection mask below never
            selects one.
        skew: Signed fraction narrowing one half. Positive narrows the ``x > 0`` half, negative the
            ``x < 0`` half; magnitude is the fractional narrowing (``0.2`` == 20% narrower).

    Returns:
        A new ``(num_points, 2)`` array; ``points`` itself is never mutated.

    """
    narrowed: NDArray[np.float64] = points.copy()
    on_narrowed_side = (points[:, 0] * np.sign(skew)) > 0.0
    narrowed[on_narrowed_side, 0] *= 1.0 - abs(skew)
    return narrowed


def place_points(
    points: NDArray[np.float64], center: tuple[float, float], angle: float, skew: float = 0.0
) -> NDArray[np.float64]:
    """Skew, rotate, and translate origin-centered ``points`` onto ``center``, in that order.

    The single placement implementation every family shares — outlines via
    :func:`~fuse_augmentations.data.families.shape_outline`, landmarks via each family's own
    ``*_keypoints`` — so an outline and its landmarks can never drift apart. Skew runs first because
    "left"/"right" are only meaningful in the shape's own pre-rotation frame.

    Args:
        points: ``(num_points, 2)`` array already scaled and centered on the origin.
        center: Target center ``(x, y)`` in pixels.
        angle: Rotation in radians applied about the origin (i.e. about the shape center).
        skew: Signed fraction (see :func:`_skewed`) narrowing one pre-rotation half. ``0.0`` (the
            default) skips the step entirely, so existing callers are unaffected.

    Returns:
        ``(num_points, 2)`` float array in image coordinates.

    """
    if skew:
        points = _skewed(points, skew)
    if angle:
        points = points @ _rotation_matrix(angle).T
    return points + np.asarray(center, dtype=np.float64)


def polygon_to_bbox_xyxy(points: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box ``(x_min, y_min, x_max, y_max)`` of a polygon.

    Args:
        points: ``(num_points, 2)`` array of coordinates.

    Returns:
        Bounding box tuple in pixels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.geometry import polygon_to_bbox_xyxy
        >>> polygon_to_bbox_xyxy(np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 4.0]]))
        (0.0, 2.0, 3.0, 5.0)

        ```

    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))


def _convex_hull(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return the convex hull of ``points`` (counter-clockwise) via monotone chain.

    Args:
        points: ``(num_points, 2)`` array, in the shape's own fixed vertex order (index ``i`` names
            the same authored vertex at every rotation, since :func:`_placed` rotates the whole
            array rigidly).

    Returns:
        The ``(hull_points, 2)`` hull vertex array, paired with each hull vertex's index into
        ``points``. The index is what makes a hull edge identifiable independent of the polygon's
        absolute rotation: hull *position* depends on ``np.unique``'s lexicographic sort of the
        coordinates actually passed in, which shifts with rotation, but a vertex's original index
        does not — see :func:`polygon_to_obb`, which needs that stability to break ties between
        equal-area candidate boxes consistently.

    """
    unique_points, unique_index = np.unique(points, axis=0, return_index=True)
    if len(unique_points) <= 2:
        return unique_points, unique_index
    order = np.lexsort((unique_points[:, 1], unique_points[:, 0]))
    pts = unique_points[order]
    idx = unique_index[order]

    def _cross(o: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower_pts: list[NDArray[np.float64]] = []
    lower_idx: list[np.intp] = []
    for p, i in zip(pts, idx, strict=True):
        while len(lower_pts) >= 2 and _cross(lower_pts[-2], lower_pts[-1], p) <= 0:
            lower_pts.pop()
            lower_idx.pop()
        lower_pts.append(p)
        lower_idx.append(i)
    upper_pts: list[NDArray[np.float64]] = []
    upper_idx: list[np.intp] = []
    for p, i in zip(pts[::-1], idx[::-1], strict=True):
        while len(upper_pts) >= 2 and _cross(upper_pts[-2], upper_pts[-1], p) <= 0:
            upper_pts.pop()
            upper_idx.pop()
        upper_pts.append(p)
        upper_idx.append(i)
    hull_pts = np.array(lower_pts[:-1] + upper_pts[:-1], dtype=np.float64)
    hull_idx = np.array(lower_idx[:-1] + upper_idx[:-1], dtype=np.intp)
    return hull_pts, hull_idx


#: Relative tolerance for treating two candidate boxes' areas as tied in :func:`polygon_to_obb`.
#: Genuine float64 noise between equal-area candidates is ~1e-13 relative; a real area gap between
#: distinct candidates is always many orders of magnitude larger, so this threshold cannot conflate
#: the two.
_OBB_AREA_TIE_RTOL = 1e-9


class _ObbCandidate(NamedTuple):
    """One rotating-calipers candidate box for :func:`polygon_to_obb`, flush to one hull edge."""

    area: float
    #: The candidate edge's original vertex-index pair (sorted), used to break ties consistently —
    #: see :func:`_convex_hull`.
    key: tuple[int, int]
    min_xy: NDArray[np.float64]
    max_xy: NDArray[np.float64]
    #: Radians the hull was rotated by to make this candidate's edge horizontal; rotating the
    #: axis-aligned box built from ``min_xy``/``max_xy`` back by this angle places it in image space.
    angle: float


def polygon_to_obb(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the minimum-area oriented bounding box of a polygon as four corners.

    Uses the rotating-calipers property that a minimum-area rectangle shares an edge
    with the convex hull. For a rotation-invariant outline (e.g. a circle) the result
    is effectively the axis-aligned box.

    A polygon with reflective symmetry (every symbol and animal in this package; see
    :mod:`~fuse_augmentations.data.symbols`) can have **more than one** candidate edge achieve the
    true minimum area — so can a triangle with no symmetry at all: for a right *or* acute triangle,
    the altitude from every vertex lands inside the opposite side, so the longest side's
    hypotenuse/base-flush candidate always ties (exactly) at least one leg-flush candidate. Only an
    *obtuse* triangle avoids this — its altitude from the obtuse vertex falls outside the opposite
    side, leaving the longest side's candidate strictly smaller than the others (see
    :attr:`PrimitiveShape.TRIANGLE`). Ties are broken by each candidate edge's
    original vertex-index pair (see :func:`_convex_hull`) — a value fixed to the shape's own
    geometry — rather than by which candidate a plain ``<`` comparison happens to visit first,
    which would depend on the polygon's absolute rotation and make the chosen box orientation
    flip inconsistently as an otherwise-identical shape spins.

    Args:
        points: ``(num_points, 2)`` array of polygon coordinates, in the shape's own fixed vertex
            order (see :func:`_convex_hull`).

    Returns:
        ``(4, 2)`` array of corner coordinates in order.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.geometry import polygon_to_obb
        >>> corners = polygon_to_obb(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]))
        >>> corners.shape
        (4, 2)

        ```

    """
    hull, hull_index = _convex_hull(points)
    if len(hull) < 3:
        x1, y1, x2, y2 = polygon_to_bbox_xyxy(points)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

    # Two passes, not one: computing every candidate up front and comparing each against a single
    # fixed `min_area` (rather than an incrementally-updated running best) rules out tolerance
    # drifting across a chain of near-but-not-exactly-equal candidates.
    count = len(hull)
    candidates: list[_ObbCandidate] = []
    for i in range(count):
        j = (i + 1) % count
        edge = hull[j] - hull[i]
        angle = np.arctan2(edge[1], edge[0])
        rot = hull @ _rotation_matrix(-angle).T
        min_xy = rot.min(axis=0)
        max_xy = rot.max(axis=0)
        area = float((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))
        key = (int(min(hull_index[i], hull_index[j])), int(max(hull_index[i], hull_index[j])))
        candidates.append(_ObbCandidate(area, key, min_xy, max_xy, angle))

    min_area = min(c.area for c in candidates)
    tolerance = _OBB_AREA_TIE_RTOL * max(min_area, 1.0)
    # Ties are broken by each candidate edge's original vertex-index pair (see _convex_hull) — a
    # value fixed to the shape's own geometry — rather than by which tied candidate happens to be
    # smallest-by-a-strict-`<`-comparison, which would depend on the polygon's absolute rotation
    # and make the chosen box orientation flip inconsistently as an otherwise-identical shape spins.
    winner = min((c for c in candidates if c.area <= min_area + tolerance), key=lambda c: c.key)

    aligned = np.array(
        [
            [winner.min_xy[0], winner.min_xy[1]],
            [winner.max_xy[0], winner.min_xy[1]],
            [winner.max_xy[0], winner.max_xy[1]],
            [winner.min_xy[0], winner.max_xy[1]],
        ],
        dtype=np.float64,
    )
    corners: NDArray[np.float64] = aligned @ _rotation_matrix(winner.angle).T
    return corners


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Return the intersection-over-union of two axis-aligned xyxy boxes.

    Args:
        box_a: First box ``(x_min, y_min, x_max, y_max)``.
        box_b: Second box ``(x_min, y_min, x_max, y_max)``.

    Returns:
        IoU in ``[0, 1]``; ``0.0`` when the boxes do not overlap.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.geometry import bbox_iou
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
