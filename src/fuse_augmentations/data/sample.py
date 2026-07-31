"""Format-agnostic in-memory representation of one synthetic image.

A :class:`Sample` carries the rendered image plus every annotation field a writer
could need (polygon, axis-aligned box, oriented box). Writers pick the subset that
their target task requires, so the generator never needs to know the output format.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Annotation:
    """One object instance with all task representations precomputed.

    Coordinates are absolute pixel values in the image frame. Polygons and OBB
    corners are flat ``[x1, y1, x2, y2, ...]`` lists.

    Args:
        class_id: Zero-based class index (see :func:`~fuse_augmentations.data.config.class_names`).
        class_name: Human-readable class label.
        polygon: Filled-shape outline as a flat pixel-coordinate list.
        bbox_xyxy: Axis-aligned box ``(x_min, y_min, x_max, y_max)`` in pixels.
        obb_corners: Oriented box as four corners, flat ``[x1, y1, x2, y2, x3, y3, x4, y4]``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.sample import Annotation
        >>> ann = Annotation(0, "square", [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ...                  (0.0, 0.0, 2.0, 2.0), [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
        >>> ann.class_name
        'square'

        ```

    """

    class_id: int
    class_name: str
    polygon: list[float]
    bbox_xyxy: tuple[float, float, float, float]
    obb_corners: list[float]


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
