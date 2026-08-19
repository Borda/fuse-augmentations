"""Config validation and class-vocabulary tests (no Pillow required)."""

from __future__ import annotations

import pytest

from fuse_augmentations.data.animals import AnimalShape, animal_shapes
from fuse_augmentations.data.config import (
    DEFAULT_SHAPES,
    ClassMode,
    Color,
    SplitRatios,
    SyntheticConfig,
    Task,
    class_id_of,
    class_names,
)
from fuse_augmentations.data.geometry import GeomShape


def test_split_ratios_default_sums_to_one() -> None:
    """Default split ratios sum to one."""
    assert SplitRatios().to_dict() == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_split_ratios_drop_zero_split() -> None:
    """Zero split is dropped from dictionary."""
    assert SplitRatios(0.8, 0.2, 0.0).to_dict() == {"train": 0.8, "val": 0.2}


def test_split_ratios_reject_bad_sum() -> None:
    """Invalid split ratios that don't sum to 1 are rejected."""
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(0.5, 0.2, 0.1)


def test_split_ratios_reject_negative() -> None:
    """Negative split ratios are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        SplitRatios(1.2, -0.2, 0.0)


def test_synthetic_config_rejects_inverted_object_range() -> None:
    """Inverted min/max object range is rejected."""
    with pytest.raises(ValueError, match="min_objects <= max_objects"):
        SyntheticConfig(min_objects=5, max_objects=2)


def test_synthetic_config_rejects_inverted_size_ratio() -> None:
    """Inverted min/max size ratio is rejected."""
    with pytest.raises(ValueError, match="min_size_ratio"):
        SyntheticConfig(min_size_ratio=0.4, max_size_ratio=0.2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"overlap_iou": -0.1}, "overlap_iou"),
        ({"overlap_iou": 1.5}, "overlap_iou"),
        ({"boundary_tolerance": -0.2}, "boundary_tolerance"),
        ({"boundary_tolerance": 2.0}, "boundary_tolerance"),
        ({"max_placement_attempts": 0}, "max_placement_attempts"),
        ({"max_placement_attempts": -5}, "max_placement_attempts"),
    ],
)
def test_synthetic_config_rejects_out_of_range_placement_knobs(kwargs: dict[str, float], match: str) -> None:
    """Out-of-range placement knobs are rejected."""
    with pytest.raises(ValueError, match=match):
        SyntheticConfig(**kwargs)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_synthetic_config_accepts_boundary_values(value: float) -> None:
    """Boundary values for overlap and tolerance are accepted."""
    config = SyntheticConfig(overlap_iou=value, boundary_tolerance=value, max_placement_attempts=1)
    assert config.overlap_iou == value
    assert config.max_placement_attempts == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ClassMode.SHAPE, [s.value for s in (*GeomShape, *AnimalShape)]),
        (ClassMode.COLOR, [c.value for c in Color]),
    ],
)
def test_class_names(mode: ClassMode, expected: list[str]) -> None:
    """Class names match expected values for mode."""
    assert class_names(mode) == expected


def test_shape_class_vocabulary_is_pinned_in_order() -> None:
    """The 16 shape class names, pinned as a literal in their exact class-id order.

    A class id is this list's index, and seeded runs are documented as byte-identical across releases, so reordering or
    renaming a member silently relabels every previously exported dataset. Derived assertions cannot catch that — only a
    literal can.

    """
    assert class_names(ClassMode.SHAPE) == [
        "square",
        "rectangle",
        "triangle",
        "circle",
        "duck",
        "elephant",
        "giraffe",
        "fish",
        "rabbit",
        "camel",
        "eagle",
        "penguin",
        "whale",
        "kangaroo",
        "flamingo",
        "crocodile",
    ]


def test_class_names_shape_color_product_size() -> None:
    """Shape-color class count is product of shape and color counts."""
    assert len(class_names(ClassMode.SHAPE_COLOR)) == (len(GeomShape) + len(AnimalShape)) * len(Color)


def test_class_names_shape_color_pins_shape_major_color_minor_order() -> None:
    """Shape-color class names follow shape-major, color-minor order.

    SHAPE_COLOR carries the same class-id-is-list-index guarantee as SHAPE (see class_names docstring): reordering
    silently relabels every exported dataset. A count-only check would still pass if shape-major/color-minor nesting
    changed, so pin the actual sequence for the first 12 (pre-animal-shape) ids.

    """
    assert class_names(ClassMode.SHAPE_COLOR)[:12] == [
        "red_square",
        "green_square",
        "blue_square",
        "red_rectangle",
        "green_rectangle",
        "blue_rectangle",
        "red_triangle",
        "green_triangle",
        "blue_triangle",
        "red_circle",
        "green_circle",
        "blue_circle",
    ]


def test_class_id_round_trips() -> None:
    """Class ID round-trips work for all modes, shapes, and colors."""
    for mode in ClassMode:
        for shape in (*GeomShape, *AnimalShape):
            for color in Color:
                idx = class_id_of(shape, color, mode)
                assert 0 <= idx < len(class_names(mode))


def test_shapes_defaults_to_the_four_geometric_shapes() -> None:
    """An untouched config still draws only square/rectangle/triangle/circle.

    Appending the animals to `Shape` would otherwise change what every existing seeded caller generates; pinning the
    default here is the guard that kept that upgrade non-breaking.

    """
    assert SyntheticConfig().shapes == (GeomShape.SQUARE, GeomShape.RECTANGLE, GeomShape.TRIANGLE, GeomShape.CIRCLE)
    assert SyntheticConfig().shapes == DEFAULT_SHAPES


def test_default_shapes_keeps_the_original_class_ids() -> None:
    """The default vocabulary still occupies class ids 0-3 of the full `Shape` order.

    Datasets generated before the animals landed carry those ids in their label files, so the enum had to grow by
    appending rather than by re-ordering.

    """
    assert [class_id_of(shape, Color.RED, ClassMode.SHAPE) for shape in DEFAULT_SHAPES] == [0, 1, 2, 3]


def test_shapes_accepts_an_animal_override() -> None:
    """A custom tuple is stored verbatim and restricts the drawable vocabulary.

    This is the opt-in path documented for animal shapes; the config must not silently widen or reorder what the caller
    asked for.

    """
    config = SyntheticConfig(shapes=(AnimalShape.GIRAFFE, AnimalShape.DUCK))
    assert config.shapes == (AnimalShape.GIRAFFE, AnimalShape.DUCK)


def test_animal_shapes_returns_every_animal_by_default() -> None:
    """`animal_shapes()` is the whole roster, which is exactly `tuple(AnimalShape)`.

    The helper exists so callers can ask for "the animals" without importing and re-listing the roster; if it ever
    returned a different set than the keypoint-capable one, `Task.KEYPOINTS` would start rejecting its own default.

    """
    assert animal_shapes() == tuple(AnimalShape)
    assert len(animal_shapes()) == 12
    assert not set(animal_shapes()) & set(DEFAULT_SHAPES)


@pytest.mark.parametrize("count", range(13))
def test_animal_shapes_by_count_is_a_declaration_order_prefix(count: int) -> None:
    """`animal_shapes(n)` takes the first `n` animals in `Shape` declaration order.

    "First n" has to mean a stable prefix, not an arbitrary subset: a dataset regenerated with the same `n` after a
    thirteenth animal is appended must still contain the same species, or its class ids stop meaning one thing.

    """
    selected = animal_shapes(count)
    assert len(selected) == count
    assert selected == tuple(AnimalShape)[:count]


def test_animal_shapes_by_count_feeds_a_keypoints_config() -> None:
    """A count-selected tuple is accepted verbatim by the field it exists to fill.

    The helper is a convenience constructor for `SyntheticConfig.shapes`, so the round trip through the dataclass —
    including its `Task.KEYPOINTS` vocabulary check — is the behaviour worth pinning, not the tuple in isolation.

    """
    config = SyntheticConfig(task=Task.KEYPOINTS, shapes=animal_shapes(5))
    assert config.shapes == (
        AnimalShape.DUCK,
        AnimalShape.ELEPHANT,
        AnimalShape.GIRAFFE,
        AnimalShape.FISH,
        AnimalShape.RABBIT,
    )


@pytest.mark.parametrize("count", [-1, 13, 99])
def test_animal_shapes_rejects_a_count_outside_the_roster(count: int) -> None:
    """An out-of-range count raises instead of silently clamping to the roster length.

    Asking for 13 of 12 animals is a caller bug — quietly returning 12 would hide it until the dataset came out one
    class short of what the caller believed it had requested.

    """
    with pytest.raises(ValueError, match="count must be within"):
        animal_shapes(count)


def test_shapes_rejects_an_empty_tuple() -> None:
    """An empty vocabulary is refused at construction rather than at draw time.

    With no shape to sample the generator would fail deep inside the placement loop with an opaque index error, long
    after the real mistake was made.

    """
    with pytest.raises(ValueError, match="at least one Shape"):
        SyntheticConfig(shapes=())


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(("duck",), id="bare-string"),
        pytest.param((AnimalShape.DUCK, "square"), id="mixed-string"),
        pytest.param((AnimalShape.DUCK, None), id="none"),
    ],
)
def test_shapes_rejects_non_shape_elements(bad: tuple[object, ...]) -> None:
    """Anything that is not a `Shape` member is refused, including an equal bare string.

    Both members subclass `str`, so `"duck" == AnimalShape.DUCK` is True and a plain string would sail through a naive
    equality check while breaking identity comparisons downstream.

    """
    with pytest.raises(ValueError, match="only Shape members"):
        SyntheticConfig(shapes=bad)


@pytest.mark.parametrize(
    "bad_shapes",
    [
        pytest.param(DEFAULT_SHAPES, id="all-geometric"),
        pytest.param((GeomShape.SQUARE,), id="single-geometric"),
        pytest.param((AnimalShape.DUCK, GeomShape.SQUARE), id="mixed-animal-and-geometric"),
    ],
)
def test_keypoints_task_rejects_shapes_without_a_keypoint_table(bad_shapes: tuple[object, ...]) -> None:
    """`Task.KEYPOINTS` refuses any shape tuple that includes a non-animal shape.

    Only `AnimalShape` members carry a landmark table; pairing `Task.KEYPOINTS` with a geometric shape (alone or mixed
    with an animal) would otherwise reach the generator and fail deep inside landmark lookup instead of at construction.

    """
    with pytest.raises(ValueError, match="needs a keypoint table"):
        SyntheticConfig(task=Task.KEYPOINTS, shapes=bad_shapes)


def test_task_rejects_a_value_that_is_not_a_task_member() -> None:
    """A bare string equal to a `Task` value is refused despite passing equality checks.

    `Task` is a str-Enum, so `"keypoints" == Task.KEYPOINTS` is True even though `"keypoints"` fails `isinstance(...,
    Task)`; without this check a plain string would silently skip every downstream identity comparison instead of
    raising here.

    """
    with pytest.raises(ValueError, match="must be a Task member"):
        SyntheticConfig(task="keypoints", shapes=(AnimalShape.DUCK,))
