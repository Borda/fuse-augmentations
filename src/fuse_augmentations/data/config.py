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

    Two families share one vocabulary. The four *geometric* shapes are computed
    analytically: ``RECTANGLE`` (non-square) plus per-shape rotation give oriented
    bounding boxes real orientation variety, while ``CIRCLE`` is rotation-invariant so
    its OBB collapses to the axis-aligned box. The eight *animal* shapes are fixed,
    hand-authored side-profile silhouettes (see
    :mod:`fuse_augmentations.data.animal_shapes`); each is asymmetric and belongs to a
    distinct silhouette archetype, so they stay separable at a glance and every outline
    point keeps an unambiguous identity under rotation.

    Only the shapes listed in :attr:`SyntheticConfig.shapes` are drawn; that field
    defaults to :data:`DEFAULT_SHAPES` (the four geometric shapes), so appending the
    animals left every existing caller's seeded output unchanged.

    Attributes:
        SQUARE: Axis-aligned equal-sided quadrilateral.
        RECTANGLE: Non-square quadrilateral.
        TRIANGLE: Equilateral triangle.
        CIRCLE: Polygon-approximated circle.
        DUCK: Compact duck silhouette with an S-curved neck and a beak.
        SNAIL: Round snail silhouette with a spiral shell over a flat foot.
        ELEPHANT: Bulky elephant silhouette with a trunk, a large ear, and thick legs.
        GIRAFFE: Tall, thin giraffe silhouette with a very long neck and thin legs.
        FISH: Streamlined fish silhouette with a forked tail fin.
        TURTLE: Flat-bottomed turtle silhouette with a domed shell.
        SNAKE: Elongated, legless snake silhouette following a wavy S-curve.
        RABBIT: Compact rabbit silhouette with long upright ears.

    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    DUCK = "duck"
    SNAIL = "snail"
    ELEPHANT = "elephant"
    GIRAFFE = "giraffe"
    FISH = "fish"
    TURTLE = "turtle"
    SNAKE = "snake"
    RABBIT = "rabbit"


#: Shapes drawn when :attr:`SyntheticConfig.shapes` is not overridden — the four
#: geometric shapes, i.e. the vocabulary that predates the animal silhouettes.
DEFAULT_SHAPES: tuple[Shape, ...] = (Shape.SQUARE, Shape.RECTANGLE, Shape.TRIANGLE, Shape.CIRCLE)

#: Shapes carrying a landmark table and therefore usable with :attr:`Task.KEYPOINTS` — the eight
#: animal silhouettes. The geometric shapes are excluded on purpose: a square is 4-fold symmetric
#: and a circle rotation-invariant, so a fixed landmark on them has no identity a model could learn.
#: Listed explicitly rather than imported from
#: :data:`~fuse_augmentations.data.animal_shapes.ANIMAL_KEYPOINTS` so this module stays plain Python;
#: a test pins the two against each other.
KEYPOINT_SHAPES: tuple[Shape, ...] = (
    Shape.DUCK,
    Shape.SNAIL,
    Shape.ELEPHANT,
    Shape.GIRAFFE,
    Shape.FISH,
    Shape.TURTLE,
    Shape.SNAKE,
    Shape.RABBIT,
)


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
        KEYPOINTS: Bounding boxes plus the five named landmarks of
            :data:`KEYPOINT_NAMES`; restricted to :data:`KEYPOINT_SHAPES`.

    The canonical ``OBB`` value is ``"oriented_bounding_boxes"``; the short alias
    ``"obb"`` is also accepted (case-insensitive).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Task
        >>> Task("obb") is Task.OBB
        True
        >>> Task("oriented_bounding_boxes") is Task.OBB
        True
        >>> Task("keypoints") is Task.KEYPOINTS
        True

        ```

    """

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    OBB = "oriented_bounding_boxes"
    KEYPOINTS = "keypoints"

    @classmethod
    def _missing_(cls, value: object) -> Task | None:
        """Accept the short ``"obb"`` alias (case-insensitive) for :attr:`OBB`."""
        if isinstance(value, str) and value.lower() in {"obb", "oriented_bounding_box"}:
            return cls.OBB
        return None


#: Landmark names for :attr:`Task.KEYPOINTS`, in the order every keypoint table, annotation, and
#: label row uses. One shared schema across all animals: Ultralytics' YOLO pose format carries a
#: single dataset-wide ``kpt_shape``, so a per-class name list is not representable.
KEYPOINT_NAMES: tuple[str, ...] = ("head", "eye", "back", "tail", "foot")

#: Skeleton edges as index pairs into :data:`KEYPOINT_NAMES` — ``head-eye``, ``head-back``,
#: ``back-tail``, ``back-foot``. Purely a visualization aid (COCO viewers connect the dots with it);
#: nothing in generation, writing, or validation depends on it.
KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (2, 3), (2, 4))


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

    The vocabulary always spans the **full** :class:`Shape` enum, independently of
    :attr:`SyntheticConfig.shapes`. That keeps a class id meaning the same thing across every
    configuration — a dataset restricted to ``(Shape.GIRAFFE,)`` uses giraffe's id from this list
    rather than renumbering it to ``0`` — at the cost of declaring classes that a restricted run
    never draws.

    Args:
        class_mode: The selected :class:`ClassMode`.

    Returns:
        Class names in a stable order; list index is the class id.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import ClassMode, class_names
        >>> class_names(ClassMode.SHAPE)[:4]
        ['square', 'rectangle', 'triangle', 'circle']
        >>> class_names(ClassMode.SHAPE)[4:]
        ['duck', 'snail', 'elephant', 'giraffe', 'fish', 'turtle', 'snake', 'rabbit']
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
        shapes: Shapes the generator may draw, sampled uniformly. Defaults to
            :data:`DEFAULT_SHAPES`; pass e.g. ``(Shape.DUCK, Shape.GIRAFFE)`` to draw
            animal silhouettes instead. Restricting this does **not** renumber classes:
            class ids always index the full :class:`Shape` vocabulary.
        task: Annotation task the generated samples target. Only :attr:`Task.KEYPOINTS`
            changes what the generator computes (it adds the landmark table); the other
            tasks all read the same polygon/box fields, so they differ at write time only.

    Raises:
        ValueError: On non-positive sizes, inverted min/max ranges, an ``overlap_iou`` or
            ``boundary_tolerance`` outside ``[0, 1]``, ``max_placement_attempts`` below 1,
            a ``shapes`` tuple that is empty or holds a non-:class:`Shape` element, a ``task``
            that is not a :class:`Task` member, or a :attr:`Task.KEYPOINTS` task combined with
            a shape that has no keypoint table (see :data:`KEYPOINT_SHAPES`).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Shape, SyntheticConfig, Task
        >>> SyntheticConfig(img_size=128).img_size
        128
        >>> SyntheticConfig(shapes=(Shape.DUCK, Shape.SNAIL)).shapes
        (<Shape.DUCK: 'duck'>, <Shape.SNAIL: 'snail'>)
        >>> SyntheticConfig(task=Task.KEYPOINTS, shapes=(Shape.DUCK,)).task
        <Task.KEYPOINTS: 'keypoints'>

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
    shapes: tuple[Shape, ...] = DEFAULT_SHAPES
    task: Task = Task.DETECTION

    def __post_init__(self) -> None:
        """Validate size, object-count, ratio, and placement-knob ranges, then the vocabulary."""
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if not 1 <= self.min_objects <= self.max_objects:
            raise ValueError(f"require 1 <= min_objects <= max_objects, got {self.min_objects}, {self.max_objects}")
        if not 0 < self.min_size_ratio <= self.max_size_ratio <= 1:
            raise ValueError(
                f"require 0 < min_size_ratio <= max_size_ratio <= 1, got {self.min_size_ratio}, {self.max_size_ratio}"
            )
        if not 0 <= self.overlap_iou <= 1:
            raise ValueError(f"overlap_iou must be within [0, 1], got {self.overlap_iou}")
        if not 0 <= self.boundary_tolerance <= 1:
            raise ValueError(f"boundary_tolerance must be within [0, 1], got {self.boundary_tolerance}")
        if self.max_placement_attempts < 1:
            raise ValueError(f"max_placement_attempts must be >= 1, got {self.max_placement_attempts}")
        self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Reject an unusable shape tuple, a non-:class:`Task` task, or an unannotatable pairing.

        Split out of :meth:`__post_init__` so neither routine outgrows the project's complexity
        budget; it carries every check that reads the enum-valued fields rather than the numbers.

        Raises:
            ValueError: If ``shapes`` is empty or holds a non-:class:`Shape` element, ``task`` is
                not a :class:`Task` member, or a :attr:`Task.KEYPOINTS` task is paired with a shape
                that has no keypoint table.

        """
        if not self.shapes:
            raise ValueError("shapes must name at least one Shape, got an empty sequence")
        # ``Shape`` is a str-Enum, so a bare "duck" compares equal to Shape.DUCK yet is not an
        # instance -- reject it here rather than let it surface as an opaque lookup failure.
        invalid = [value for value in self.shapes if not isinstance(value, Shape)]
        if invalid:
            raise ValueError(f"shapes must contain only Shape members, got {invalid!r}")
        # Same trap for the task: a bare "keypoints" equals Task.KEYPOINTS but fails every identity
        # test downstream, which would silently emit a detection dataset instead of raising.
        if not isinstance(self.task, Task):
            raise ValueError(f"task must be a Task member, got {self.task!r}; pass e.g. Task.KEYPOINTS")
        if self.task is Task.KEYPOINTS:
            unsupported = [shape.value for shape in self.shapes if shape not in KEYPOINT_SHAPES]
            if unsupported:
                supported = ", ".join(shape.value for shape in KEYPOINT_SHAPES)
                raise ValueError(
                    f"task {Task.KEYPOINTS.value!r} needs a keypoint table for every drawable shape, but "
                    f"{unsupported} have none; restrict shapes to the animal silhouettes: {supported}"
                )
