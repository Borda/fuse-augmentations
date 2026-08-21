"""Render one static, monochrome reference image per shape, with its detection box and keypoints.

Unlike ``animate_synthetic_dataset.py``'s colored, randomly placed, animated previews, these are
plain single-shape images: upright, at each shape's own authored orientation (no rotation, no
skew) — a field-guide-style lookup of what each shape and its keypoint schema actually look like,
not a sample of what the generator draws in practice. Every symbol and animal is authored
mirror-symmetric about its own vertical axis (see :mod:`~fuse_augmentations.data.symbols`), so
rendering at that authored angle is what keeps a reference recognizable: an ``arrow`` pointing up,
a ``house`` with its roof up, a ``kite`` on its long axis.

The box drawn in blue is the shape's plain axis-aligned **detection** box (``bbox``) at this
reference orientation — deliberately *not* :func:`~fuse_augmentations.data.geometry.polygon_to_obb`'s
minimum-area **OBB**, which the generator's actually-rotated samples use (see the animated OBB
preview and the "Tasks" section for that box, which is not axis-aligned in general — e.g. the
``arrow``'s true minimum-area OBB is a diamond tilted off its own vertical axis even when the
arrow itself is drawn upright). Showing the true rotated OBB here, at a fixed unrotated pose,
produced a tilted box for concave/asymmetric outlines and an arbitrary tied choice for a
symmetric one (an isosceles triangle's minimum-area OBB has no unique answer — see
:attr:`~fuse_augmentations.data.geometry.GeomShape.TRIANGLE`'s docstring for why a right *or*
acute triangle ties), neither of which says anything useful about a shape drawn in its own
reference position; the plain detection box is simpler and unambiguous for every shape. The two
keypoint-bearing families (animals, symbols) get their landmark dots and skeleton drawn in the
same orange ``animate_synthetic_dataset.py`` uses for a visible-but-occluded keypoint. The
geometric family has no keypoint schema, so its images carry only the outline and the box. There
is no filename caption baked into the image — one shape per file already names it.

Each shape is scaled independently — not to one shared size — to the largest ``size`` that fits
its own outline inside the canvas margin, so a thin shape like the geometric triangle or a tall
one like the giraffe each fill as much of their own frame as their own proportions allow, rather
than leaving the fixed-size margin of a shared scale empty.

Files are named ``<prefix><shape>.png`` with the same prefix convention
``animate_synthetic_dataset.py`` uses for its animated previews (``geometry-``, ``animals-``,
``symbols-``), so e.g. ``symbols-arrow.png`` and ``animals-duck.png``.

Render every shape in every family:
    python examples/render_shape_reference.py

Render one family:
    python examples/render_shape_reference.py --families symbols

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_SKELETON, AnimalShape, animal_keypoints
from fuse_augmentations.data.geometry import GeomShape, polygon_to_bbox_xyxy, shape_polygon
from fuse_augmentations.data.symbols import SYMBOL_KEYPOINT_SKELETON, SymbolShape, symbol_keypoints

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_CANVAS = 200
#: Pixels reserved on every side of the canvas so a fitted shape's stroke width and keypoint dot
#: radius never clip at the edge.
_MARGIN = 12
_DOT_RADIUS = 3
_INK = (0, 0, 0)
_FILL = (222, 222, 222)
_PAPER = (255, 255, 255)
#: Drawn for the plain axis-aligned detection box (not the rotated OBB — see the module docstring).
_BBOX_RGB = (0, 102, 255)
#: Matches ``animate_synthetic_dataset.py``'s ``_KEYPOINT_COLORS[1]`` — drawn for landmark dots and
#: the skeleton; more visible than yellow at this dot/line size.
_KEYPOINT_RGB = (255, 165, 0)
_SHAPE_CENTER = (_CANVAS / 2.0, _CANVAS / 2.0)

#: A family's file prefix (matching ``animate_synthetic_dataset.py``'s ``PREFIXES``), roster,
#: keypoint placer (``None`` for the schema-less geometric family), and skeleton edges — everything
#: :func:`_render_shape` needs, keyed by ``--families`` choice.
_FAMILIES: dict[
    str,
    tuple[str, tuple[str, ...], Callable[..., NDArray[np.float64]] | None, tuple[tuple[int, int], ...]],
] = {
    "geometric": ("geometry-", tuple(s.value for s in GeomShape), None, ()),
    "symbols": ("symbols-", tuple(s.value for s in SymbolShape), symbol_keypoints, SYMBOL_KEYPOINT_SKELETON),
    "animals": ("animals-", tuple(s.value for s in AnimalShape), animal_keypoints, ANIMAL_KEYPOINT_SKELETON),
}


def _fit_size(name: str) -> float:
    """Return the largest ``size`` keeping ``name``'s outline within the canvas margin.

    Computed from a unit-size (``size=1.0``) render at the shape's authored (unrotated)
    orientation: the drawn detection box always equals the outline's own bounding box at this
    orientation (see :func:`_draw_shape`), so fitting the outline alone is sufficient.

    """
    poly = shape_polygon(name, center=(0.0, 0.0), size=1.0)
    half_extent = float(np.abs(poly).max())
    half_available = (_CANVAS - 2 * _MARGIN) / 2.0
    return half_available / half_extent


def _draw_shape(draw: ImageDraw.ImageDraw, name: str, size: float) -> None:
    """Draw one shape's outline (filled light gray, black edge) and its blue detection box.

    The box is the plain axis-aligned bounding box at this reference orientation — not
    :func:`~fuse_augmentations.data.geometry.polygon_to_obb`'s minimum-area OBB, which the
    generator's rotated samples use instead (see the module docstring for why).

    """
    poly = shape_polygon(name, center=_SHAPE_CENTER, size=size)
    draw.polygon([(float(x), float(y)) for x, y in poly], outline=_INK, fill=_FILL, width=2)
    x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly)
    draw.rectangle((x1, y1, x2, y2), outline=_BBOX_RGB, width=1)


def _draw_keypoints(
    draw: ImageDraw.ImageDraw,
    keypoints_fn: Callable[..., NDArray[np.float64]],
    shape: AnimalShape | SymbolShape,
    skeleton: tuple[tuple[int, int], ...],
    size: float,
) -> None:
    """Draw one shape's skeleton edges then its landmark dots, skipping absent (NaN) rows."""
    points = keypoints_fn(shape, _SHAPE_CENTER, size)
    visible = {i: (float(x), float(y)) for i, (x, y) in enumerate(points) if x == x}  # x == x excludes NaN
    for first, second in skeleton:
        if first in visible and second in visible:
            draw.line([visible[first], visible[second]], fill=_KEYPOINT_RGB, width=1)
    for x, y in visible.values():
        draw.ellipse((x - _DOT_RADIUS, y - _DOT_RADIUS, x + _DOT_RADIUS, y + _DOT_RADIUS), fill=_KEYPOINT_RGB)


def _render_shape(
    name: str,
    shape_cls: type[AnimalShape | SymbolShape] | None,
    keypoints_fn: Callable[..., NDArray[np.float64]] | None,
    skeleton: tuple[tuple[int, int], ...],
    output_path: Path,
) -> Path:
    """Render one shape's reference image and write it to ``output_path``."""
    size = _fit_size(name)
    canvas = Image.new("RGB", (_CANVAS, _CANVAS), _PAPER)
    draw = ImageDraw.Draw(canvas)
    _draw_shape(draw, name, size)
    if keypoints_fn is not None and shape_cls is not None:
        _draw_keypoints(draw, keypoints_fn, shape_cls(name), skeleton, size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main(output_dir: str = "docs/assets/shape-references", families: str = "all") -> None:
    """Render one monochrome, canonical-orientation reference image per shape.

    Args:
        output_dir: Directory the ``<prefix><shape>.png`` images are written into.
        families: ``all``, or a comma-separated subset of ``geometric``, ``symbols``, ``animals``.

    Examples:
        >>> callable(main)
        True

    """
    chosen = tuple(_FAMILIES) if families == "all" else tuple(f.strip() for f in families.split(","))
    assert all(f in _FAMILIES for f in chosen), f"--families must be a subset of {tuple(_FAMILIES)}, got {families!r}"
    out = Path(output_dir)
    shape_classes: dict[str, type[AnimalShape | SymbolShape] | None] = {
        "geometric": None,
        "symbols": SymbolShape,
        "animals": AnimalShape,
    }
    for family in chosen:
        prefix, names, keypoints_fn, skeleton = _FAMILIES[family]
        for name in names:
            path = _render_shape(name, shape_classes[family], keypoints_fn, skeleton, out / f"{prefix}{name}.png")
            print(f"wrote {path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
