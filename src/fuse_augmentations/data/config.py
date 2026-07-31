"""Configuration primitives for synthetic dataset generation.

Defines the task/format/class-mode enums, the class-name vocabulary derived from
shapes and colors, the train/val/test split ratios, and the :class:`SyntheticConfig`
knob bundle consumed by :class:`~fuse_augmentations.data.generator.SyntheticGenerator`.

All values are plain Python (no torch/Pillow) so this module imports without any
optional dependency installed.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_SPLIT_SUM_TOL = 1e-6


class Shape(str, Enum):
    """Drawable shape vocabulary (definition order is the class order).

    ``RECTANGLE`` (non-square) plus per-shape rotation give oriented bounding boxes
    real orientation variety; ``CIRCLE`` is rotation-invariant so its OBB collapses
    to the axis-aligned box.

    Attributes:
        SQUARE: Axis-aligned equal-sided quadrilateral.
        RECTANGLE: Non-square quadrilateral.
        TRIANGLE: Equilateral triangle.
        CIRCLE: Polygon-approximated circle.

    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"


class Color(str, Enum):
    """Fill-color vocabulary; :attr:`rgb` yields the 8-bit RGB tuple.

    Attributes:
        RED: ``(255, 0, 0)``.
        GREEN: ``(0, 128, 0)``.
        BLUE: ``(0, 0, 255)``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Color
        >>> Color.GREEN.rgb
        (0, 128, 0)

        ```

    """

    RED = "red"
    GREEN = "green"
    BLUE = "blue"

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the 8-bit RGB fill tuple for this color."""
        return {
            Color.RED: (255, 0, 0),
            Color.GREEN: (0, 128, 0),
            Color.BLUE: (0, 0, 255),
        }[self]


class Task(str, Enum):
    """Annotation task the dataset targets.

    Attributes:
        DETECTION: Axis-aligned bounding boxes only.
        SEGMENTATION: Bounding boxes plus filled polygon masks.
        OBB: Oriented (rotated) bounding boxes as four corner points.

    The canonical ``OBB`` value is ``"oriented_bounding_boxes"``; the short alias
    ``"obb"`` is also accepted (case-insensitive).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Task
        >>> Task("obb") is Task.OBB
        True
        >>> Task("oriented_bounding_boxes") is Task.OBB
        True

        ```

    """

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    OBB = "oriented_bounding_boxes"

    @classmethod
    def _missing_(cls, value: object) -> Task | None:
        """Accept the short ``"obb"`` alias (case-insensitive) for :attr:`OBB`."""
        if isinstance(value, str) and value.lower() in {"obb", "oriented_bounding_box"}:
            return cls.OBB
        return None


class OutputFormat(str, Enum):
    """On-disk dataset layout.

    Attributes:
        COCO: Roboflow-style ``<split>/_annotations.coco.json`` plus images.
        YOLO: ``images/<split>`` + ``labels/<split>`` + ``data.yaml``.

    """

    COCO = "coco"
    YOLO = "yolo"


class ClassMode(str, Enum):
    """How object classes are derived from shape and color.

    Attributes:
        SHAPE: One class per shape (square, rectangle, triangle, circle).
        COLOR: One class per color (red, green, blue).
        SHAPE_COLOR: Cartesian product, named ``"<color>_<shape>"``.

    """

    SHAPE = "shape"
    COLOR = "color"
    SHAPE_COLOR = "shape_color"


def class_names(class_mode: ClassMode) -> list[str]:
    """Return the ordered class-name vocabulary for a class mode.

    Args:
        class_mode: The selected :class:`ClassMode`.

    Returns:
        Class names in a stable order; list index is the class id.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import ClassMode, class_names
        >>> class_names(ClassMode.SHAPE)
        ['square', 'rectangle', 'triangle', 'circle']
        >>> class_names(ClassMode.COLOR)
        ['red', 'green', 'blue']
        >>> class_names(ClassMode.SHAPE_COLOR)[:2]
        ['red_square', 'green_square']

        ```

    """
    if class_mode is ClassMode.SHAPE:
        return [shape.value for shape in Shape]
    if class_mode is ClassMode.COLOR:
        return [color.value for color in Color]
    return [f"{color.value}_{shape.value}" for shape in Shape for color in Color]


def class_id_of(shape: Shape, color: Color, class_mode: ClassMode) -> int:
    """Return the class id for a (shape, color) pair under a class mode.

    Args:
        shape: The :class:`Shape` member.
        color: The :class:`Color` member.
        class_mode: The selected :class:`ClassMode`.

    Returns:
        Zero-based class id indexing into :func:`class_names`.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import ClassMode, Color, Shape, class_id_of
        >>> class_id_of(Shape.TRIANGLE, Color.RED, ClassMode.SHAPE)
        2
        >>> class_id_of(Shape.TRIANGLE, Color.RED, ClassMode.COLOR)
        0

        ```

    """
    return class_names(class_mode).index(_class_key(shape, color, class_mode))


def _class_key(shape: Shape, color: Color, class_mode: ClassMode) -> str:
    """Return the class-name key for a (shape, color) pair under a class mode."""
    if class_mode is ClassMode.SHAPE:
        return shape.value
    if class_mode is ClassMode.COLOR:
        return color.value
    return f"{color.value}_{shape.value}"


@dataclass(frozen=True)
class SplitRatios:
    """Train/validation/test fractions; must be non-negative and sum to ~1.

    Args:
        train: Training fraction.
        val: Validation fraction.
        test: Test fraction.

    Raises:
        ValueError: If any fraction is negative or the three do not sum to 1.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import SplitRatios
        >>> SplitRatios().to_dict()
        {'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> SplitRatios(0.8, 0.2, 0.0).to_dict()
        {'train': 0.8, 'val': 0.2}

        ```

    """

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self) -> None:
        """Validate non-negativity and unit sum."""
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value < 0:
                raise ValueError(f"split ratio {name!r} must be non-negative, got {value}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _SPLIT_SUM_TOL:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        """Return the non-zero splits as an ordered ``name -> fraction`` mapping."""
        return {
            name: value for name, value in (("train", self.train), ("val", self.val), ("test", self.test)) if value > 0
        }


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs controlling one synthetic image's content.

    Args:
        img_size: Square canvas side length in pixels.
        min_objects: Minimum objects drawn per image (inclusive).
        max_objects: Maximum objects drawn per image (inclusive).
        min_size_ratio: Minimum object size as a fraction of ``img_size``.
        max_size_ratio: Maximum object size as a fraction of ``img_size``.
        overlap_iou: Reject a candidate whose IoU with any kept box exceeds this.
        boundary_tolerance: Max fraction of a box allowed outside the canvas.
        max_placement_attempts: Retry cap per object before giving up.
        background: RGB background fill.
        rotate: Apply a random rotation to each polygonal shape.
        class_mode: How classes are derived (see :class:`ClassMode`).

    Raises:
        ValueError: On non-positive sizes or inverted min/max ranges.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import SyntheticConfig
        >>> SyntheticConfig(img_size=128).img_size
        128

        ```

    """

    img_size: int = 640
    min_objects: int = 1
    max_objects: int = 10
    min_size_ratio: float = 0.1
    max_size_ratio: float = 0.3
    overlap_iou: float = 0.1
    boundary_tolerance: float = 0.05
    max_placement_attempts: int = 100
    background: tuple[int, int, int] = (128, 128, 128)
    rotate: bool = True
    class_mode: ClassMode = ClassMode.SHAPE

    def __post_init__(self) -> None:
        """Validate size, object-count, and ratio ranges."""
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if not 1 <= self.min_objects <= self.max_objects:
            raise ValueError(f"require 1 <= min_objects <= max_objects, got {self.min_objects}, {self.max_objects}")
        if not 0 < self.min_size_ratio <= self.max_size_ratio <= 1:
            raise ValueError(
                f"require 0 < min_size_ratio <= max_size_ratio <= 1, got {self.min_size_ratio}, {self.max_size_ratio}"
            )
