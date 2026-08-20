"""Seeded synthetic-image generator producing format-agnostic samples.

:class:`SyntheticGenerator` draws colored shapes on a gray canvas, rejecting
placements that fall off the canvas or overlap existing objects, and returns a
:class:`~fuse_augmentations.data.sample.Sample` carrying every annotation
representation (polygon, axis-aligned box, oriented box). All randomness flows
through a caller-supplied :class:`numpy.random.Generator`, so a fixed seed yields
byte-identical output.

Under :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS` each annotation also carries the
animal's landmarks, derived from the placement that was already sampled — no extra random draw —
so a seed produces the same scene whatever the configured task.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from fuse_augmentations.data.config import SyntheticConfig
    >>> from fuse_augmentations.data.generator import SyntheticGenerator
    >>> gen = SyntheticGenerator(SyntheticConfig(img_size=64, min_objects=2, max_objects=2))
    >>> sample = gen.sample(np.random.default_rng(0))
    >>> sample.image.shape
    (64, 64, 3)
    >>> len(sample.annotations)
    2

    ```

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from fuse_augmentations.data.animals import AnimalShape, animal_keypoints
from fuse_augmentations.data.config import Color, Shape, Task, class_id_of, class_names
from fuse_augmentations.data.geometry import (
    GeomShape,
    bbox_iou,
    polygon_to_bbox_xyxy,
    polygon_to_obb,
    shape_polygon,
)
from fuse_augmentations.data.sample import Annotation, Sample

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fuse_augmentations.data.config import SyntheticConfig

_BBox = tuple[float, float, float, float]

#: COCO visibility flags emitted for a landmark: ``2`` is "labeled and visible", ``0`` is
#: "not labeled". The intermediate ``1`` ("labeled but not visible") never occurs here — the
#: placement loop rejects overlapping objects, so only the canvas frame or an absent
#: hind limb (a NaN row from :mod:`~fuse_augmentations.data.animals`) can hide a
#: landmark.
_KEYPOINT_VISIBLE = 2
_KEYPOINT_HIDDEN = 0


def _visible_keypoints(points: NDArray[np.float64], img_size: int) -> tuple[tuple[float, float, int], ...]:
    """Tag each landmark with its COCO visibility flag, zeroing the ones off the canvas.

    Args:
        points: ``(num_keypoints, 2)`` landmark coordinates in image pixels.
        img_size: Canvas side length in pixels.

    Returns:
        One ``(x, y, visibility)`` triple per landmark, in input order: the real coordinates with
        :data:`_KEYPOINT_VISIBLE` while the point lies inside ``[0, img_size)`` on both axes, and
        ``(0.0, 0.0)`` with :data:`_KEYPOINT_HIDDEN` otherwise. Zeroing a clipped point (rather than
        keeping its off-canvas coordinates) is COCO's "not labeled" convention. An absent
        hind limb lands here as ``(nan, nan)``; both comparisons in
        ``0.0 <= nan < img_size`` are false, so it falls to the hidden branch with no special-casing.

    """
    triples: list[tuple[float, float, int]] = []
    for x, y in points:
        inside = 0.0 <= x < img_size and 0.0 <= y < img_size
        triples.append((float(x), float(y), _KEYPOINT_VISIBLE) if inside else (0.0, 0.0, _KEYPOINT_HIDDEN))
    return tuple(triples)


def _boundary_overlap(bbox: _BBox, img_size: int) -> float:
    """Return the fraction of ``bbox`` area lying outside a square canvas.

    Args:
        bbox: Candidate box ``(x_min, y_min, x_max, y_max)``.
        img_size: Canvas side length in pixels.

    Returns:
        ``0.0`` when fully inside, up to ``1.0`` when fully outside.

    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 1.0
    inside_w = max(0.0, min(img_size, x2) - max(0.0, x1))
    inside_h = max(0.0, min(img_size, y2) - max(0.0, y1))
    return 1.0 - (inside_w * inside_h) / area


class SyntheticGenerator:
    """Draw colored shapes into reproducible :class:`Sample` objects.

    Args:
        config: Generation knobs; see :class:`~fuse_augmentations.data.config.SyntheticConfig`.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.config import SyntheticConfig
        >>> from fuse_augmentations.data.generator import SyntheticGenerator
        >>> gen = SyntheticGenerator(SyntheticConfig(img_size=32))
        >>> a = gen.sample(np.random.default_rng(1))
        >>> b = gen.sample(np.random.default_rng(1))
        >>> bool(np.array_equal(a.image, b.image))
        True

        ```

    """

    def __init__(self, config: SyntheticConfig) -> None:
        """Store config and precompute the class-name vocabulary."""
        self.config = config
        self.class_names = class_names(config.class_mode)

    def _attempt_placement(
        self, rng: np.random.Generator, kept: list[_BBox]
    ) -> tuple[Shape, Color, NDArray[np.float64], NDArray[np.float64] | None] | None:
        """Draw one candidate shape; return it if in-bounds and non-overlapping, else ``None``.

        Exactly one candidate is sampled per call (fixed RNG draw order: shape, color, size, centre x, centre y, then
        angle when rotation applies), so the caller controls the retry budget and the RNG consumption stays
        deterministic for a given seed.

        The shape is drawn from ``cfg.shapes`` and the color from ``cfg.colors`` rather than the full :class:`Shape` and
        :class:`Color` vocabularies, so widening either enum never changes what an existing seeded configuration
        produces.

        Landmarks (last tuple element, ``None`` off the keypoints task) are a pure function of the placement that was
        just sampled, so computing them consumes no further randomness and leaves every seeded stream unchanged.

        """
        cfg = self.config
        shapes, colors = cfg.shapes, cfg.colors
        shape = shapes[int(rng.integers(len(shapes)))]
        color = colors[int(rng.integers(len(colors)))]
        size_px = float(rng.uniform(cfg.min_size_ratio, cfg.max_size_ratio)) * cfg.img_size
        center = (float(rng.uniform(0, cfg.img_size)), float(rng.uniform(0, cfg.img_size)))
        angle = float(rng.uniform(0, 2 * np.pi)) if cfg.rotate and shape is not GeomShape.CIRCLE else 0.0
        poly = shape_polygon(shape.value, center, size_px, angle)
        bbox = polygon_to_bbox_xyxy(poly)
        if _boundary_overlap(bbox, cfg.img_size) > cfg.boundary_tolerance:
            return None
        if any(bbox_iou(bbox, other) > cfg.overlap_iou for other in kept):
            return None
        # only an AnimalShape has a landmark table; the config validator already rejects any other
        # shape under Task.KEYPOINTS, so this narrowing can never silently drop a labeled object
        keypoints = None
        if cfg.task is Task.KEYPOINTS and isinstance(shape, AnimalShape):
            keypoints = animal_keypoints(shape, center, size_px, angle)
        return shape, color, poly, keypoints

    def sample(self, rng: np.random.Generator) -> Sample:
        """Generate one image and its annotations.

        Args:
            rng: Random generator driving object count, shapes, colors, and placement.

        Returns:
            A :class:`Sample` with an RGB ``uint8`` image and one annotation per drawn shape.

        Raises:
            RuntimeError: If fewer than ``min_objects`` shapes could be placed within the overall
                attempt budget (``num_objects * max_placement_attempts``); relax ``overlap_iou`` or
                ``boundary_tolerance``, lower ``min_objects``, or raise ``max_placement_attempts``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> from fuse_augmentations.data.config import SyntheticConfig
            >>> from fuse_augmentations.data.generator import SyntheticGenerator
            >>> gen = SyntheticGenerator(SyntheticConfig(img_size=48, min_objects=1, max_objects=3))
            >>> s = gen.sample(np.random.default_rng(7))
            >>> 1 <= len(s.annotations) <= 3
            True

            ```

        """
        cfg = self.config
        canvas = Image.new("RGB", (cfg.img_size, cfg.img_size), cfg.background)
        draw = ImageDraw.Draw(canvas)
        num_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))

        annotations: list[Annotation] = []
        kept: list[_BBox] = []
        # Retry failed placements against a shared budget so we reach num_objects when feasible
        # instead of silently dropping objects; the budget bounds RNG draws deterministically.
        budget = num_objects * cfg.max_placement_attempts
        for _ in range(budget):
            if len(annotations) >= num_objects:
                break
            placed = self._attempt_placement(rng, kept)
            if placed is None:
                continue
            shape, color, poly, points = placed
            draw.polygon([(float(x), float(y)) for x, y in poly], fill=color.rgb)
            bbox = polygon_to_bbox_xyxy(poly)
            kept.append(bbox)
            class_id = class_id_of(shape, color, cfg.class_mode)
            annotations.append(
                Annotation(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    polygon=[float(v) for v in poly.reshape(-1)],
                    bbox_xyxy=bbox,
                    obb_corners=[float(v) for v in polygon_to_obb(poly).reshape(-1)],
                    keypoints=None if points is None else _visible_keypoints(points, cfg.img_size),
                )
            )
        if len(annotations) < cfg.min_objects:
            raise RuntimeError(
                f"could not place the required min_objects={cfg.min_objects} shapes within the "
                f"placement budget of {budget} attempts (placed {len(annotations)}); relax overlap_iou/"
                f"boundary_tolerance, lower min_objects, or raise max_placement_attempts"
            )
        return Sample(image=np.asarray(canvas), annotations=annotations, width=cfg.img_size, height=cfg.img_size)

    def generate(self, num_images: int, seed: int | None = None) -> Iterator[Sample]:
        """Lazily yield ``num_images`` samples from a fresh seeded generator.

        This is the streaming primitive: only one :class:`Sample` is materialized at
        a time, so both in-memory training feeds and writing very large datasets stay
        within a bounded memory footprint. All samples draw from a single
        :class:`numpy.random.Generator`, so a fixed ``seed`` yields a reproducible stream.

        Args:
            num_images: Number of samples to yield.
            seed: Seed for the internal generator; ``None`` uses fresh entropy.

        Yields:
            One :class:`Sample` per iteration.

        Examples:
            ```pycon
            >>> from fuse_augmentations.data.config import SyntheticConfig
            >>> from fuse_augmentations.data.generator import SyntheticGenerator
            >>> gen = SyntheticGenerator(SyntheticConfig(img_size=32))
            >>> samples = list(gen.generate(3, seed=0))
            >>> len(samples)
            3

            ```

        """
        rng = np.random.default_rng(seed)
        for _ in range(num_images):
            yield self.sample(rng)
