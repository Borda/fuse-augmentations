"""Canonical animal-silhouette outline tables for the synthetic generator.

Eight hand-authored side-profile silhouettes — duck, snail, elephant, giraffe, fish,
turtle, snake, rabbit — kept here so :mod:`fuse_augmentations.data.shapes` stays free of
large coordinate literals. Each table is a **simple** (non-self-intersecting) polygon in
the same unit space :func:`~fuse_augmentations.data.shapes._base_polygon` uses for the
geometric shapes: vertex centroid at the origin and the larger of the two extents scaled
to ``1``, so multiplying by a pixel ``size`` yields a shape bounded by ``size`` pixels.

Coordinates are authored in screen orientation — ``+x`` right, ``+y`` **down** — matching
Pillow's raster axes, so every animal renders upright and faces left. Raw literals are
passed through :func:`_normalized` at import time, which means visual tuning of a table
can never silently break the centring or scaling invariant.

Unlike the geometric shapes these outlines are asymmetric and each belongs to a distinct
silhouette archetype (compact, bulky, tall-thin, streamlined, elongated), so the classes
stay separable at a glance and every outline point keeps an unambiguous identity under
rotation.

Pure NumPy — no Pillow, no torch, no import from :mod:`fuse_augmentations.data.config`
(tables are keyed by the plain ``Shape`` *values*, keeping the geometry layer independent
of the configuration layer).

Examples:
    ```pycon
    >>> from fuse_augmentations.data.animal_shapes import ANIMAL_POLYGONS
    >>> sorted(ANIMAL_POLYGONS)
    ['duck', 'elephant', 'fish', 'giraffe', 'rabbit', 'snail', 'snake', 'turtle']
    >>> duck = ANIMAL_POLYGONS["duck"]
    >>> duck.shape[1]
    2
    >>> bool(abs(duck.mean(axis=0)).max() < 1e-9)
    True

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _normalized(vertices: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    """Center a raw outline on its vertex mean and scale its larger extent to ``1``.

    Args:
        vertices: Hand-authored ``(x, y)`` outline points in any convenient scale,
            ordered along the outline (winding direction is irrelevant).

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
    points = points - points.mean(axis=0)
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if extent <= 0.0:
        raise ValueError("outline has zero extent; every vertex is identical")
    scaled: NDArray[np.float64] = points / extent
    scaled.setflags(write=False)
    return scaled


#: Compact duck: S-curved neck, beak, rounded body, short upswept tail (~1.3:1).
_DUCK_POLYGON: NDArray[np.float64] = _normalized([
    (0, 30),  # beak tip
    (12, 22),  # beak upper edge
    (20, 12),  # forehead
    (30, 6),  # crown
    (40, 12),  # back of head
    (44, 24),  # nape
    (40, 38),  # neck rear, tucked in by the S-curve
    (44, 52),  # neck meets the back
    (58, 46),  # back, front
    (76, 42),  # back, mid
    (96, 48),  # back, rear
    (112, 58),  # rump
    (126, 54),  # tail tip
    (118, 70),  # tail underside
    (106, 84),  # rear underside
    (88, 96),  # belly, rear
    (66, 100),  # belly, mid
    (46, 96),  # belly, front
    (34, 82),  # breast, lower
    (28, 68),  # breast
    (26, 54),  # neck front, lower
    (22, 42),  # neck front, mid
    (16, 34),  # chin
    (8, 34),  # beak lower edge
])

#: Snail: flat creeping foot, stalked head, big spiral-tucked shell bump (~1:1).
_SNAIL_POLYGON: NDArray[np.float64] = _normalized([
    (2, 92),  # snout tip
    (2, 80),  # head, front
    (8, 70),  # head, top front
    (15, 60),  # left eye stalk tip
    (20, 70),  # valley between the stalks
    (27, 58),  # right eye stalk tip
    (32, 72),  # right stalk, rear
    (38, 60),  # shell, whorl tuck (front lower left)
    (28, 40),  # shell, left
    (34, 22),  # shell, upper left
    (50, 10),  # shell, upper
    (70, 8),  # shell, top right
    (82, 12),  # shell, upper right
    (94, 24),  # shell, right upper
    (98, 40),  # shell, right
    (94, 58),  # shell, right lower
    (84, 78),  # shell, bottom right
    (88, 96),  # foot, rear tip
    (62, 102),  # foot, rear
    (32, 103),  # foot, mid
])

#: Elephant: trunk, notched ear lobe, bulky barrel body, four thick legs (~1.5:1).
_ELEPHANT_POLYGON: NDArray[np.float64] = _normalized([
    (22, 44),  # brow, front
    (28, 26),  # forehead
    (44, 14),  # crown
    (62, 14),  # ear, top
    (80, 36),  # ear, rear edge (shallow saddle down to the withers)
    (104, 24),  # back, front
    (130, 26),  # back, rear
    (148, 42),  # rump
    (152, 62),  # tail base
    (154, 80),  # tail tip
    (144, 78),  # tail, front edge
    (142, 88),  # rear leg, rear edge
    (140, 100),  # rear foot, rear
    (122, 100),  # rear foot, front
    (124, 82),  # rear leg, front edge
    (106, 78),  # belly, rear
    (88, 80),  # belly, mid
    (84, 100),  # front leg, rear edge
    (66, 100),  # front foot, front
    (62, 78),  # chest, lower
    (54, 60),  # chest
    (46, 48),  # cheek, behind the trunk
    (42, 66),  # trunk, rear edge upper
    (34, 84),  # trunk, rear edge lower
    (26, 98),  # trunk, rear lower
    (14, 104),  # trunk tip, rear
    (8, 96),  # trunk tip
    (18, 84),  # trunk, front edge lower
    (22, 66),  # trunk, front edge upper
])

#: Giraffe: very long neck, small deep body, four thin legs, tall aspect (~0.6:1).
_GIRAFFE_POLYGON: NDArray[np.float64] = _normalized([
    (2, 12),  # muzzle tip
    (8, 4),  # head, top front
    (17, 5),  # head, top rear (ossicone line)
    (21, 13),  # back of the head
    (29, 29),  # neck, rear upper
    (37, 49),  # neck, rear lower
    (45, 60),  # withers
    (53, 68),  # back
    (58, 80),  # rump
    (56, 90),  # rear leg, rear edge
    (53, 102),  # rear hoof, rear
    (48, 102),  # rear hoof, front
    (49, 82),  # rear leg, front edge
    (35, 80),  # belly
    (33, 102),  # front leg, rear edge
    (28, 102),  # front hoof, front
    (28, 74),  # front leg, front edge
    (24, 62),  # chest
    (18, 50),  # neck, front lower
    (12, 34),  # neck, front mid
    (8, 20),  # throat
])

#: Fish: streamlined lens body, dorsal and anal fins, forked caudal fin (~2:1).
_FISH_POLYGON: NDArray[np.float64] = _normalized([
    (0, 38),  # snout tip
    (10, 26),  # head, top
    (26, 18),  # nape
    (40, 14),  # dorsal fin, front base
    (52, 2),  # dorsal fin, peak
    (66, 14),  # dorsal fin, rear base
    (86, 20),  # back, rear
    (104, 30),  # caudal peduncle, top
    (122, 8),  # tail, upper lobe tip
    (138, 4),  # tail, upper trailing edge
    (124, 36),  # tail fork notch
    (138, 68),  # tail, lower trailing edge
    (122, 64),  # tail, lower lobe tip
    (104, 44),  # caudal peduncle, bottom
    (86, 54),  # belly, rear
    (66, 60),  # anal fin, rear base
    (56, 70),  # anal fin, tip
    (46, 58),  # anal fin, front base
    (30, 54),  # belly, front
    (12, 46),  # lower jaw
])

#: Turtle: high domed shell over a flat plastron, small head, stumpy legs (~1.2:1).
_TURTLE_POLYGON: NDArray[np.float64] = _normalized([
    (0, 50),  # snout tip
    (2, 38),  # head, top front
    (12, 30),  # head, top
    (24, 38),  # neck, top
    (34, 30),  # shell, front edge
    (48, 16),  # shell dome, front
    (66, 8),  # shell dome, top
    (86, 16),  # shell dome, rear
    (100, 30),  # shell, rear edge
    (110, 46),  # shell, rear lower
    (116, 60),  # tail
    (104, 64),  # under the shell, rear
    (98, 76),  # rear leg, rear edge
    (100, 98),  # rear foot, rear
    (84, 98),  # rear foot, front
    (80, 76),  # rear leg, front edge
    (56, 78),  # plastron, rear
    (44, 78),  # plastron, front
    (40, 98),  # front leg, rear edge
    (24, 98),  # front foot, front
    (22, 74),  # front leg, front edge / shoulder
    (14, 68),  # throat
    (2, 60),  # jaw
])

#: Snake: legless wavy band tapering to a tail, wider at the head (~4:1).
_SNAKE_POLYGON: NDArray[np.float64] = _normalized([
    (0, 30),  # snout tip
    (8, 16),  # head, top
    (24, 12),  # head, rear top
    (44, 6),  # first crest
    (66, 6),  # crest, rear
    (88, 16),  # descending
    (108, 30),  # mid body
    (128, 42),  # through, front
    (148, 44),  # through
    (168, 38),  # rising
    (188, 30),  # pre-tail
    (200, 22),  # tail tip
    (186, 44),  # underside, rising back
    (166, 52),  # underside, through rear
    (146, 58),  # underside, through
    (126, 56),  # underside, through front
    (106, 44),  # underside, mid body
    (86, 30),  # underside, ascending
    (66, 22),  # underside, crest rear
    (46, 22),  # underside, crest
    (26, 30),  # underside, neck
    (13, 42),  # jaw, rear
    (2, 44),  # jaw, front
])

#: Rabbit: two long upright ears, round haunched body, compact tall aspect (~0.77:1).
_RABBIT_POLYGON: NDArray[np.float64] = _normalized([
    (20, 34),  # head, top front
    (16, 14),  # front ear, front edge
    (22, 0),  # front ear tip
    (30, 16),  # front ear, rear edge
    (34, 30),  # valley between the ears
    (38, 12),  # rear ear, front edge
    (46, 4),  # rear ear tip
    (50, 20),  # rear ear, rear edge
    (46, 36),  # back of the head
    (50, 46),  # nape dip
    (62, 54),  # shoulder
    (74, 70),  # back
    (80, 86),  # rump
    (72, 98),  # haunch, rear
    (64, 104),  # rear foot, rear
    (42, 104),  # rear foot, front
    (40, 92),  # haunch, front
    (28, 94),  # belly
    (20, 102),  # front foot, rear
    (8, 100),  # front foot, front
    (10, 86),  # front leg, front edge
    (6, 68),  # chest
    (2, 54),  # throat
    (6, 42),  # chin
    (12, 36),  # muzzle
])

#: Outline table per animal :class:`~fuse_augmentations.data.config.Shape` *value*.
#: Every entry is unit-normalized and read-only; scale a copy rather than mutating it.
ANIMAL_POLYGONS: dict[str, NDArray[np.float64]] = {
    "duck": _DUCK_POLYGON,
    "snail": _SNAIL_POLYGON,
    "elephant": _ELEPHANT_POLYGON,
    "giraffe": _GIRAFFE_POLYGON,
    "fish": _FISH_POLYGON,
    "turtle": _TURTLE_POLYGON,
    "snake": _SNAKE_POLYGON,
    "rabbit": _RABBIT_POLYGON,
}
