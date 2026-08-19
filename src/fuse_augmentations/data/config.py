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
    its OBB collapses to the axis-aligned box. The twelve *animal* shapes are fixed
    side-profile silhouettes traced from public-domain reference art (see
    :mod:`fuse_augmentations.data.animal_shapes`); each is asymmetric and belongs to a
    distinct silhouette archetype, so they stay separable at a glance and every outline
    point keeps an unambiguous identity under rotation.

    Only the shapes listed in :attr:`SyntheticConfig.shapes` are drawn; that field
    defaults to :data:`DEFAULT_SHAPES` (the four geometric shapes), so appending the
    animals left every existing caller's seeded output unchanged. Use
    :func:`animal_shapes` to select the animals by count instead of naming them.

    Attributes:
        SQUARE: Axis-aligned equal-sided quadrilateral.
        RECTANGLE: Non-square quadrilateral.
        TRIANGLE: Equilateral triangle.
        CIRCLE: Polygon-approximated circle.
        DUCK: Compact duck silhouette with an S-curved neck and a beak.
        ELEPHANT: Bulky elephant silhouette with a trunk, a large ear, and thick legs.
        GIRAFFE: Tall, thin giraffe silhouette with a very long neck and thin legs.
        FISH: Streamlined fish silhouette with a forked tail fin.
        RABBIT: Compact rabbit silhouette with long upright ears.
        CAMEL: Humped camel silhouette on four long legs.
        EAGLE: Perched eagle silhouette with a hooked beak and a long tail.
        PENGUIN: Upright penguin silhouette with flippers and webbed feet.
        WHALE: Streamlined whale silhouette with a pectoral flipper and a tail fluke.
        KANGAROO: Hopping kangaroo silhouette with a heavy tail and one large hind foot.
        FLAMINGO: Long-legged flamingo silhouette with an S-curved neck.
        CROCODILE: Low, elongated crocodile silhouette with a long snout and sprawled legs.

    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    DUCK = "duck"
    ELEPHANT = "elephant"
    GIRAFFE = "giraffe"
    FISH = "fish"
    RABBIT = "rabbit"
    CAMEL = "camel"
    EAGLE = "eagle"
    PENGUIN = "penguin"
    WHALE = "whale"
    KANGAROO = "kangaroo"
    FLAMINGO = "flamingo"
    CROCODILE = "crocodile"


#: Shapes drawn when :attr:`SyntheticConfig.shapes` is not overridden — the four
#: geometric shapes, i.e. the vocabulary that predates the animal silhouettes.
DEFAULT_SHAPES: tuple[Shape, ...] = (Shape.SQUARE, Shape.RECTANGLE, Shape.TRIANGLE, Shape.CIRCLE)

#: Shapes carrying a landmark table and therefore usable with :attr:`Task.KEYPOINTS` — the twelve
#: animal silhouettes, in :class:`Shape` declaration order. The geometric shapes are excluded on
#: purpose: a square is 4-fold symmetric and a circle rotation-invariant, so a fixed landmark on them
#: has no identity a model could learn. Listed explicitly rather than imported from
#: :data:`~fuse_augmentations.data.animal_shapes.ANIMAL_KEYPOINTS` so this module stays plain Python;
#: a test pins the two against each other.
KEYPOINT_SHAPES: tuple[Shape, ...] = (
    Shape.DUCK,
    Shape.ELEPHANT,
    Shape.GIRAFFE,
    Shape.FISH,
    Shape.RABBIT,
    Shape.CAMEL,
    Shape.EAGLE,
    Shape.PENGUIN,
    Shape.WHALE,
    Shape.KANGAROO,
    Shape.FLAMINGO,
    Shape.CROCODILE,
)


def animal_shapes(count: int | None = None) -> tuple[Shape, ...]:
    """Return the animal silhouettes, optionally just the first ``count`` of them.

    A convenience selector for :attr:`SyntheticConfig.shapes`, which still takes (and stores) a
    plain ``tuple[Shape, ...]`` — naming members explicitly stays equally valid. "First ``count``"
    means :class:`Shape` declaration order, the same order :data:`KEYPOINT_SHAPES` and the class-id
    vocabulary use, so ``animal_shapes(3)`` names the same three animals on every call and across
    releases; appending a thirteenth animal can only extend the tail of that list.

    Args:
        count: How many animals to take, from the start of :data:`KEYPOINT_SHAPES`. ``None`` (the
            default) returns every animal.

    Returns:
        The selected animal :class:`Shape` members, in declaration order.

    Raises:
        ValueError: If ``count`` is negative or exceeds the number of animals.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import SyntheticConfig, Task, animal_shapes
        >>> animal_shapes(3)
        (<Shape.DUCK: 'duck'>, <Shape.ELEPHANT: 'elephant'>, <Shape.GIRAFFE: 'giraffe'>)
        >>> len(animal_shapes())
        12
        >>> SyntheticConfig(task=Task.KEYPOINTS, shapes=animal_shapes(4)).shapes[-1]
        <Shape.FISH: 'fish'>

        ```

    """
    if count is None:
        return KEYPOINT_SHAPES
    if not 0 <= count <= len(KEYPOINT_SHAPES):
        raise ValueError(f"count must be within [0, {len(KEYPOINT_SHAPES)}], got {count}")
    return KEYPOINT_SHAPES[:count]


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
            :data:`KEYPOINT_NAMES`; restricted to :data:`KEYPOINT_SHAPES`. Points an animal does
            not have (e.g. a whale's hind limbs) are absent rather than faked — see
            :mod:`fuse_augmentations.data.animal_shapes` for the NaN-row contract.

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
#: label row uses. One shared anatomical schema across all animals: Ultralytics' YOLO pose format
#: carries a single dataset-wide ``kpt_shape``, so a per-class name list is not representable.
#: The ``front_limb_*`` pair covers whatever the animal actually has at that slot — paws, wings, or
#: flippers/fins; ``left`` is the limb nearer the viewer (fully visible), ``right`` the far one — a
#: documented convention, since a side-profile silhouette cannot truly tell left from right. Every
#: limb is articulated in two points, proximal before distal: ``front_elbow_*`` (elbow, wing wrist,
#: or flipper bend) then ``front_limb_*`` (the paw/wing tip/fin tip), and ``hind_knee_*`` (the
#: knee/hock bend) then ``hind_limb_*`` (the foot) — a limb's bend is the most visible pose cue on a
#: silhouette. All four hind points are optional — an animal whose silhouette shows no hind leg at
#: all (a fish, a whale) carries NaN rows there instead of faked points, see
#: :mod:`fuse_augmentations.data.animal_shapes`.
KEYPOINT_NAMES: tuple[str, ...] = (
    "mouth",
    "eye",
    "ear",
    "head",
    "neck",
    "body_top",
    "body_bottom",
    "tail",
    "front_elbow_left",
    "front_elbow_right",
    "front_limb_left",
    "front_limb_right",
    "hind_knee_left",
    "hind_knee_right",
    "hind_limb_left",
    "hind_limb_right",
)

#: Skeleton edges as index pairs into :data:`KEYPOINT_NAMES`: ``mouth-head``, ``eye-head``,
#: ``ear-head``, the ``head-neck-body_top-body_bottom-tail`` chain, a two-segment
#: ``body_top-front_elbow-front_limb`` chain per front limb, and a two-segment
#: ``body_bottom-hind_knee-hind_limb`` chain per hind leg. 15 edges over 16 nodes is a spanning tree
#: whose optional hind points hang off the end of their own chain, so an absent hind leg drops
#: exactly its own two edges and orphans nothing. Purely a visualization aid (COCO viewers connect
#: the dots with it); nothing in generation, writing, or validation depends on it.
KEYPOINT_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (5, 8),
    (5, 9),
    (8, 10),
    (9, 11),
    (6, 12),
    (6, 13),
    (12, 14),
    (13, 15),
)

#: Fill color per landmark name, as written into every packaged zoo SVG's ``<circle>`` elements and
#: mirrored by ``examples/edit_zoo_keypoints.py``. Fixed and identical across all animals, so a
#: color identifies a landmark at a glance in any SVG viewer. Package-internal: nothing in
#: generation or writing reads it — it exists so the asset convention has exactly one definition
#: instead of one per authoring tool. A test pins it against the packaged documents.
_KEYPOINT_COLORS: dict[str, str] = {
    "mouth": "#e6194b",
    "eye": "#ffe119",
    "ear": "#f58231",
    "head": "#911eb4",
    "neck": "#4363d8",
    "body_top": "#42d4f4",
    "body_bottom": "#3cb44b",
    "tail": "#f032e6",
    "front_elbow_left": "#bfef45",
    "front_elbow_right": "#ffd8b1",
    "front_limb_left": "#9a6324",
    "front_limb_right": "#fabed4",
    "hind_knee_left": "#808000",
    "hind_knee_right": "#aaffc3",
    "hind_limb_left": "#469990",
    "hind_limb_right": "#dcbeff",
}


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
        >>> SyntheticConfig(shapes=(Shape.DUCK, Shape.CAMEL)).shapes
        (<Shape.DUCK: 'duck'>, <Shape.CAMEL: 'camel'>)
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
