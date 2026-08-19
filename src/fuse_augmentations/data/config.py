"""Configuration primitives for synthetic dataset generation.

Defines the task/format/class-mode enums, the class-name vocabulary derived from
shapes and colors, the train/val/test split ratios, and the :class:`SyntheticConfig`
knob bundle consumed by :class:`~fuse_augmentations.data.generator.SyntheticGenerator`.

No torch and no Pillow: this module imports with only the base dependencies installed. It does
import :mod:`fuse_augmentations.data.animals` for the animal half of the shape vocabulary, which
parses the packaged zoo documents at import time (NumPy plus the stdlib XML parser).

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fuse_augmentations.data.animals import AnimalShape
from fuse_augmentations.data.geometry import GeomShape

_SPLIT_SUM_TOL = 1e-6


#: The drawable-shape vocabulary: the analytic geometric shapes plus the animal silhouettes, kept as
#: a union rather than one blended enum so each family owns its own definition (and so an animal-only
#: API such as :func:`~fuse_augmentations.data.animals.animal_keypoints` can say so in its
#: signature). It is a real runtime union: ``isinstance(value, Shape)`` accepts either member type
#: and still rejects a bare ``"duck"`` string, which is what :meth:`SyntheticConfig._validate_vocabulary`
#: relies on. It is not iterable — build a full vocabulary as ``[*GeomShape, *AnimalShape]``.
Shape = GeomShape | AnimalShape

#: Shapes drawn when :attr:`SyntheticConfig.shapes` is not overridden — the four geometric shapes,
#: i.e. the vocabulary that predates the animal silhouettes.
DEFAULT_SHAPES: tuple[GeomShape, ...] = tuple(GeomShape)


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
        KEYPOINTS: Bounding boxes plus the sixteen named landmarks of
            :data:`~fuse_augmentations.data.animals.ANIMAL_KEYPOINT_NAMES`; restricted to
            :class:`~fuse_augmentations.data.animals.AnimalShape`, the only shapes with a landmark
            table. Points an animal does not have (e.g. a whale's hind limbs) are absent rather
            than faked — see :mod:`fuse_augmentations.data.animals` for the NaN-row contract.

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
    configuration — a dataset restricted to ``(AnimalShape.GIRAFFE,)`` uses giraffe's id from this list
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
        >>> class_names(ClassMode.SHAPE)[4:8]
        ['duck', 'elephant', 'giraffe', 'fish']
        >>> class_names(ClassMode.SHAPE)[8:]
        ['rabbit', 'camel', 'eagle', 'penguin', 'whale', 'kangaroo', 'flamingo', 'crocodile']
        >>> class_names(ClassMode.COLOR)
        ['red', 'green', 'blue']
        >>> class_names(ClassMode.SHAPE_COLOR)[:2]
        ['red_square', 'green_square']

        ```

    """
    if class_mode is ClassMode.SHAPE:
        return [shape.value for shape in (*GeomShape, *AnimalShape)]
    if class_mode is ClassMode.COLOR:
        return [color.value for color in Color]
    return [f"{color.value}_{shape.value}" for shape in (*GeomShape, *AnimalShape) for color in Color]


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
        >>> from fuse_augmentations.data.config import ClassMode, Color, class_id_of
        >>> from fuse_augmentations.data.geometry import GeomShape
        >>> class_id_of(GeomShape.TRIANGLE, Color.RED, ClassMode.SHAPE)
        2
        >>> class_id_of(GeomShape.TRIANGLE, Color.RED, ClassMode.COLOR)
        0

        ```

    """
    return class_names(class_mode).index(_class_key(shape, color, class_mode))


def _class_key(shape: Shape, color: Color, class_mode: ClassMode) -> str:
    """Return the class-name key for a (shape, color) pair under a class mode."""
    if class_mode is ClassMode.SHAPE:
        return str(shape.value)
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
            :data:`DEFAULT_SHAPES`; pass e.g. ``(AnimalShape.DUCK, AnimalShape.GIRAFFE)`` to draw
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
            a shape that has no keypoint table (anything but an
                :class:`~fuse_augmentations.data.animals.AnimalShape`).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import Shape, SyntheticConfig, Task
        >>> SyntheticConfig(img_size=128).img_size
        128
        >>> SyntheticConfig(shapes=(AnimalShape.DUCK, AnimalShape.CAMEL)).shapes
        (<AnimalShape.DUCK: 'duck'>, <AnimalShape.CAMEL: 'camel'>)
        >>> SyntheticConfig(task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK,)).task
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
        # ``Shape`` is a str-Enum, so a bare "duck" compares equal to AnimalShape.DUCK yet is not an
        # instance -- reject it here rather than let it surface as an opaque lookup failure.
        invalid = [value for value in self.shapes if not isinstance(value, Shape)]
        if invalid:
            raise ValueError(f"shapes must contain only Shape members, got {invalid!r}")
        # Same trap for the task: a bare "keypoints" equals Task.KEYPOINTS but fails every identity
        # test downstream, which would silently emit a detection dataset instead of raising.
        if not isinstance(self.task, Task):
            raise ValueError(f"task must be a Task member, got {self.task!r}; pass e.g. Task.KEYPOINTS")
        if self.task is Task.KEYPOINTS:
            unsupported = [shape.value for shape in self.shapes if not isinstance(shape, AnimalShape)]
            if unsupported:
                supported = ", ".join(shape.value for shape in AnimalShape)
                raise ValueError(
                    f"task {Task.KEYPOINTS.value!r} needs a keypoint table for every drawable shape, but "
                    f"{unsupported} have none; restrict shapes to the animal silhouettes: {supported}"
                )
