"""Format-agnostic in-memory representation of one synthetic image.

A :class:`Sample` carries the rendered image plus every annotation field a writer
could need (polygon, axis-aligned box, oriented box, and — for the keypoints task —
landmarks). Writers pick the subset that their target task requires, so the generator
never needs to know the output format.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from fuse_augmentations.data.geometry import polygon_to_obb

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fuse_augmentations.data.keypoints import KeypointSchema

#: The COCO landmark visibility flags a triple may carry: ``0`` "not labeled", ``1`` "labeled but not
#: visible", ``2`` "labeled and visible". Spelled as the accepted set rather than a ``0 <= v <= 2``
#: range test so a value that is numerically in range but not a flag (``1.5``) is rejected too.
_KEYPOINT_VISIBILITIES: frozenset[int] = frozenset({0, 1, 2})


@dataclass(frozen=True)
class Annotation:
    """One object instance with all task representations precomputed.

    Coordinates are absolute pixel values in the image frame. Polygons and OBB corners are flat
    ``[x1, y1, x2, y2, ...]`` lists. The oriented box is *derived* from the polygon on first access
    (see :attr:`obb_corners`) rather than stored, so the tasks that never read it never pay for it.

    A landmark table is validated against its own schema on construction (see
    :meth:`__post_init__`), so every consumer — the writers,
    :class:`~fuse_augmentations.data.datasets.SyntheticIterableDataset`, and any third-party code
    reading a :class:`Sample` — can rely on the width without re-checking it. Validating here rather
    than in a writer is what makes that guarantee hold for consumers that never touch a writer.

    Args:
        class_id: Zero-based class index (see :func:`~fuse_augmentations.data.config.class_names`).
        class_name: Human-readable class label.
        polygon: Filled-shape outline as a flat pixel-coordinate list.
        bbox_xyxy: Axis-aligned box ``(x_min, y_min, x_max, y_max)`` in pixels.
        keypoints: Landmarks as ``(x, y, visibility)`` triples in ``keypoint_schema`` order, or
            ``None`` for any task other than
            :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS`. Visibility follows COCO: ``2``
            for a point inside the canvas, ``0`` for one clipped away by the frame — a ``0`` point
            carries ``(0.0, 0.0)`` rather than its off-canvas coordinates.
        keypoint_schema: The keypoint-bearing family ``keypoints`` was drawn from, which names and
            sizes the table. Required whenever ``keypoints`` is given, ``None`` otherwise. Carrying
            it here is what lets any consumer — the writers,
            :class:`~fuse_augmentations.data.datasets.SyntheticIterableDataset`, third-party code —
            interpret a table without being told separately which family produced it.

    Raises:
        ValueError: If ``keypoints`` is given without a ``keypoint_schema``, holds a number of
            triples other than the schema's ``kpt_shape``, or carries a visibility outside COCO's
            ``{0, 1, 2}``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_SCHEMA
        >>> from fuse_augmentations.data.sample import Annotation
        >>> ann = Annotation(0, "square", [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ...                  (0.0, 0.0, 2.0, 2.0))
        >>> ann.class_name
        'square'
        >>> ann.keypoints is None
        True
        >>> table = tuple((1.0, 2.0, 2) for _ in ANIMAL_KEYPOINT_SCHEMA.names)
        >>> duck = Annotation(4, "duck", [], (0.0, 0.0, 2.0, 2.0), keypoints=table,
        ...                   keypoint_schema=ANIMAL_KEYPOINT_SCHEMA)
        >>> duck.keypoints[0]
        (1.0, 2.0, 2)

        ```

    """

    class_id: int
    class_name: str
    polygon: list[float]
    bbox_xyxy: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...] | None = None
    keypoint_schema: KeypointSchema | None = None

    @cached_property
    def obb_corners(self) -> list[float]:
        """Return the minimum-area oriented box as four corners, flat ``[x1, y1, ..., x4, y4]``.

        Derived from :attr:`polygon` on first access rather than stored. It used to be computed for
        every object at generation time, which meant every detection, segmentation and keypoint run
        paid for a convex hull plus a rotating-calipers scan per object and then never read the
        result — measured at **75% of generation time** on a mixed-family run. Deriving it here costs
        exactly the same for an OBB run, and nothing for the three tasks that do not want it.

        Returns:
            The eight corner coordinates, or an empty list when :attr:`polygon` holds fewer than
            three points and has no oriented box to speak of.

        Examples:
            ```pycon
            >>> from fuse_augmentations.data.sample import Annotation
            >>> square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]
            >>> ann = Annotation(0, "square", square, (0.0, 0.0, 2.0, 2.0))
            >>> len(ann.obb_corners)
            8

            ```

        """
        points = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 3:
            return []
        return [float(value) for value in polygon_to_obb(points).reshape(-1)]

    def __post_init__(self) -> None:
        """Reject a landmark table that does not match the schema it claims to follow.

        A short, long, or mis-flagged table is silently lossy downstream rather than loud: a COCO
        ``keypoints`` array of the wrong length still parses, and a YOLO pose row of the wrong width
        is read positionally, so both mislabel every landmark after the first missing one instead of
        failing. Catching it at construction turns that into an error at the point the bad table was
        built.

        The schema is carried rather than inferred. It used to be looked up by *table length* —
        which worked only because the registered families happened to have distinct landmark counts,
        and left an annotation unable to say which family it belonged to, so a writer had to be told
        separately and the two could disagree.

        Raises:
            ValueError: If ``keypoints`` is given without a ``keypoint_schema``, if it holds a
                number of triples other than the schema's ``kpt_shape``, or if a triple's visibility
                is not a COCO flag.

        """
        if self.keypoints is None:
            return
        if self.keypoint_schema is None:
            raise ValueError(
                f"keypoints were given without a keypoint_schema; pass the family's schema so the "
                f"{len(self.keypoints)} triples can be named and validated"
            )
        names = self.keypoint_schema.names
        if len(self.keypoints) != len(names):
            raise ValueError(
                f"keypoints must hold exactly {len(names)} (x, y, visibility) triples to match the "
                f"schema, got {len(self.keypoints)}"
            )
        for name, (_x, _y, visibility) in zip(names, self.keypoints, strict=True):
            if type(visibility) is not int or visibility not in _KEYPOINT_VISIBILITIES:
                raise ValueError(
                    f"keypoint {name!r} has visibility {visibility!r}, but COCO allows only "
                    f"{sorted(_KEYPOINT_VISIBILITIES)}: 0 not labeled, 1 labeled but hidden, 2 labeled and visible"
                )


@dataclass(frozen=True)
class Sample:
    """A rendered image and its annotations.

    Args:
        image: RGB image, shape ``(height, width, 3)``, dtype ``uint8``.
        annotations: Object annotations, one per drawn shape.
        width: Image width in pixels.
        height: Image height in pixels.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.sample import Sample
        >>> img = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> Sample(img, [], width=4, height=4).width
        4

        ```

    """

    image: NDArray[Any]
    annotations: list[Annotation]
    width: int
    height: int
