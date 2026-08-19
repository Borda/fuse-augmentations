"""Format-agnostic in-memory representation of one synthetic image.

A :class:`Sample` carries the rendered image plus every annotation field a writer
could need (polygon, axis-aligned box, oriented box, and — for the keypoints task —
landmarks). Writers pick the subset that their target task requires, so the generator
never needs to know the output format.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_NAMES

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: The COCO landmark visibility flags a triple may carry: ``0`` "not labeled", ``1`` "labeled but not
#: visible", ``2`` "labeled and visible". Spelled as the accepted set rather than a ``0 <= v <= 2``
#: range test so a value that is numerically in range but not a flag (``1.5``) is rejected too.
_KEYPOINT_VISIBILITIES: frozenset[int] = frozenset({0, 1, 2})


@dataclass(frozen=True)
class Annotation:
    """One object instance with all task representations precomputed.

    Coordinates are absolute pixel values in the image frame. Polygons and OBB
    corners are flat ``[x1, y1, x2, y2, ...]`` lists.

    A landmark table is validated on construction (see :meth:`__post_init__`), so every consumer —
    the writers, :class:`~fuse_augmentations.data.datasets.SyntheticIterableDataset`, and any
    third-party code reading a :class:`Sample` — can rely on the fixed-width schema without
    re-checking it. Validating here rather than in a writer is what makes that guarantee hold for
    consumers that never touch a writer at all.

    Args:
        class_id: Zero-based class index (see :func:`~fuse_augmentations.data.config.class_names`).
        class_name: Human-readable class label.
        polygon: Filled-shape outline as a flat pixel-coordinate list.
        bbox_xyxy: Axis-aligned box ``(x_min, y_min, x_max, y_max)`` in pixels.
        obb_corners: Oriented box as four corners, flat ``[x1, y1, x2, y2, x3, y3, x4, y4]``.
        keypoints: Landmarks as ``(x, y, visibility)`` triples in
            :data:`~fuse_augmentations.data.animals.ANIMAL_KEYPOINT_NAMES` order, or ``None`` for any task
            other than :attr:`~fuse_augmentations.data.config.Task.KEYPOINTS`. Visibility follows
            COCO: ``2`` for a point inside the canvas, ``0`` for one clipped away by the frame — a
            ``0`` point carries ``(0.0, 0.0)`` rather than its off-canvas coordinates.

    Raises:
        ValueError: If ``keypoints`` is not ``None`` and either holds a number of triples other than
            ``len(ANIMAL_KEYPOINT_NAMES)`` or carries a visibility outside COCO's ``{0, 1, 2}``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_NAMES
        >>> from fuse_augmentations.data.sample import Annotation
        >>> ann = Annotation(0, "square", [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ...                  (0.0, 0.0, 2.0, 2.0), [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
        >>> ann.class_name
        'square'
        >>> ann.keypoints is None
        True
        >>> table = tuple((1.0, 2.0, 2) for _ in ANIMAL_KEYPOINT_NAMES)
        >>> Annotation(4, "duck", [], (0.0, 0.0, 2.0, 2.0), [], keypoints=table).keypoints[0]
        (1.0, 2.0, 2)

        ```

    """

    class_id: int
    class_name: str
    polygon: list[float]
    bbox_xyxy: tuple[float, float, float, float]
    obb_corners: list[float]
    keypoints: tuple[tuple[float, float, int], ...] | None = None

    def __post_init__(self) -> None:
        """Reject a landmark table that does not match the animal keypoint schema.

        A short, long, or mis-flagged table is silently lossy downstream rather than loud: a COCO
        ``keypoints`` array of the wrong length still parses, and a YOLO pose row of the wrong width
        is read positionally, so both mislabel every landmark after the first missing one instead of
        failing. Catching it at construction turns that into an error at the point the bad table was
        built.

        Raises:
            ValueError: If ``keypoints`` holds a number of triples other than
                ``len(ANIMAL_KEYPOINT_NAMES)``, or a triple's visibility is not a COCO flag.

        """
        if self.keypoints is None:
            return
        if len(self.keypoints) != len(ANIMAL_KEYPOINT_NAMES):
            raise ValueError(
                f"keypoints must hold exactly {len(ANIMAL_KEYPOINT_NAMES)} (x, y, visibility) triples, one per name "
                f"in ANIMAL_KEYPOINT_NAMES, got {len(self.keypoints)}"
            )
        for name, (_x, _y, visibility) in zip(ANIMAL_KEYPOINT_NAMES, self.keypoints, strict=True):
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
