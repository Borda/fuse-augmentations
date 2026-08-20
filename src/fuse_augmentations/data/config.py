"""Configuration primitives for synthetic dataset generation.

Defines the task/format/class-mode enums, the class-name vocabulary derived from
shapes and colors, the train/val/test split ratios, and the :class:`SyntheticConfig`
knob bundle consumed by :class:`~fuse_augmentations.data.generator.SyntheticGenerator`.

No torch and no Pillow: this module imports with only the base dependencies installed. It does
import :mod:`fuse_augmentations.data.animals` for the animal half of the shape vocabulary, which
parses the packaged zoo documents at import time (NumPy plus the stdlib XML parser).

"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache

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


#: Colors drawn when :attr:`SyntheticConfig.colors` is not overridden — the full :class:`Color`
#: vocabulary, i.e. the behavior that predated the field existing.
DEFAULT_COLORS: tuple[Color, ...] = tuple(Color)


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


def class_names(class_mode: ClassMode, shapes: Iterable[Shape] | None = None) -> list[str]:
    """Return the ordered class-name vocabulary for a class mode.

    By default (``shapes=None``) the vocabulary always spans the full range for the selected
    ``class_mode``, independently of :attr:`SyntheticConfig.shapes`. ``ClassMode.SHAPE`` spans the
    full **16**-member :class:`Shape` enum; ``ClassMode.SHAPE_COLOR`` spans all 16 shapes times the 3
    colors (48 combined classes); and ``ClassMode.COLOR`` spans only the 3 colors, independently of
    the shape vocabulary entirely. That keeps a class id meaning the same thing across every
    configuration — a dataset restricted to ``(AnimalShape.GIRAFFE,)`` still uses giraffe's ``SHAPE``
    id from this list rather than renumbering it to ``0`` — at the cost of declaring classes that a
    restricted run never draws. This is what :func:`class_id_of` relies on, so it never passes
    ``shapes``.

    Pass ``shapes`` to narrow ``ClassMode.SHAPE``/``ClassMode.SHAPE_COLOR`` to a specific shape
    family instead — e.g. a golden-fixture assertion that wants a stable 4-category list for
    :data:`DEFAULT_SHAPES` regardless of how many animal shapes the full vocabulary later grows to.
    ``ClassMode.COLOR`` ignores ``shapes`` since it never depends on the shape vocabulary.

    Args:
        class_mode: The selected :class:`ClassMode`.
        shapes: Shape family to narrow the vocabulary to, in the given order. ``None`` (the default)
            keeps the full-vocabulary behavior described above.

    Returns:
        Class names in a stable order; list index is the class id when ``shapes`` is ``None``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import ClassMode, DEFAULT_SHAPES, class_names
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
        >>> class_names(ClassMode.SHAPE, shapes=DEFAULT_SHAPES)
        ['square', 'rectangle', 'triangle', 'circle']

        ```

    """
    universe = (*GeomShape, *AnimalShape) if shapes is None else tuple(shapes)
    if class_mode is ClassMode.SHAPE:
        return [shape.value for shape in universe]
    if class_mode is ClassMode.COLOR:
        return [color.value for color in Color]
    return [f"{color.value}_{shape.value}" for shape in universe for color in Color]


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
    return _class_ids(class_mode)[_class_key(shape, color, class_mode)]


@cache
def _class_ids(class_mode: ClassMode) -> dict[str, int]:
    """Return the cached ``class name -> class id`` map for a class mode.

    :func:`class_id_of` runs once per placed object, so rebuilding the vocabulary and scanning it linearly there costs a
    fresh 48-entry list plus a lookup per object; the map is built once per mode instead. Module-private and never
    handed out — callers must not mutate it.

    """
    return {name: index for index, name in enumerate(class_names(class_mode))}


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
            animal silhouettes instead, ``tuple(AnimalShape)`` for every animal, or
            ``(*GeomShape, *AnimalShape)`` for the full mixed vocabulary. Restricting this does
            **not** renumber classes: class ids always index the full :class:`Shape` vocabulary.
        colors: Fill colors the generator may draw, sampled uniformly. Defaults to
            :data:`DEFAULT_COLORS` (all three); pass e.g. ``(Color.RED,)`` to draw only red
            objects. Restricting this does **not** renumber classes either, for the same reason
            as ``shapes``.
        task: Annotation task the generated samples target. Only :attr:`Task.KEYPOINTS`
            changes what the generator computes (it adds the landmark table); the other
            tasks all read the same polygon/box fields, so they differ at write time only.

    Raises:
        ValueError: On non-positive sizes, inverted min/max ranges, an ``overlap_iou`` or
            ``boundary_tolerance`` outside ``[0, 1]``, ``max_placement_attempts`` below 1,
            a ``shapes`` tuple that is empty or holds a non-:class:`Shape` element, a ``colors``
            tuple that is empty or holds a non-:class:`Color` element, a ``task`` that is not a
            :class:`Task` member, or a :attr:`Task.KEYPOINTS` task combined with a shape that has
            no keypoint table (anything but an
                :class:`~fuse_augmentations.data.animals.AnimalShape`).

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.animals import AnimalShape
        >>> from fuse_augmentations.data.config import Color, SyntheticConfig, Task
        >>> SyntheticConfig(img_size=128).img_size
        128
        >>> SyntheticConfig(shapes=(AnimalShape.DUCK, AnimalShape.CAMEL)).shapes
        (<AnimalShape.DUCK: 'duck'>, <AnimalShape.CAMEL: 'camel'>)
        >>> SyntheticConfig(task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK,)).task
        <Task.KEYPOINTS: 'keypoints'>
        >>> SyntheticConfig(colors=(Color.RED,)).colors
        (<Color.RED: 'red'>,)

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
    colors: tuple[Color, ...] = DEFAULT_COLORS
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
        """Reject an unusable shape/color tuple, a non-:class:`Task` task, or an unannotatable pairing.

        Split out of :meth:`__post_init__` so neither routine outgrows the project's complexity
        budget; it carries every check that reads the enum-valued fields rather than the numbers.

        Raises:
            ValueError: If ``shapes`` is empty or holds a non-:class:`Shape` element, ``colors`` is
                empty or holds a non-:class:`Color` element, ``task`` is not a :class:`Task` member,
                or a :attr:`Task.KEYPOINTS` task is paired with a shape that has no keypoint table.

        """
        if not self.shapes:
            raise ValueError("shapes must name at least one Shape, got an empty sequence")
        # ``Shape`` is a str-Enum, so a bare "duck" compares equal to AnimalShape.DUCK yet is not an
        # instance -- reject it here rather than let it surface as an opaque lookup failure.
        invalid = [value for value in self.shapes if not isinstance(value, Shape)]
        if invalid:
            raise ValueError(f"shapes must contain only Shape members, got {invalid!r}")
        if not self.colors:
            raise ValueError("colors must name at least one Color, got an empty sequence")
        # Same "equal but not an instance" trap as `shapes` above -- a bare "red" would otherwise
        # reach the generator's `color.rgb` lookup and fail there instead of at construction.
        invalid_colors = [value for value in self.colors if not isinstance(value, Color)]
        if invalid_colors:
            raise ValueError(f"colors must contain only Color members, got {invalid_colors!r}")
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
