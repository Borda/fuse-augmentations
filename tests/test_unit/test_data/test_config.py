"""Config validation and class-vocabulary tests (no Pillow required)."""

from __future__ import annotations

import pytest

from fuse_augmentations.data.config import (
    DEFAULT_SHAPES,
    KEYPOINT_SHAPES,
    ClassMode,
    Color,
    Shape,
    SplitRatios,
    SyntheticConfig,
    Task,
    animal_shapes,
    class_id_of,
    class_names,
)


def test_split_ratios_default_sums_to_one():
    assert SplitRatios().to_dict() == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_split_ratios_drop_zero_split():
    assert SplitRatios(0.8, 0.2, 0.0).to_dict() == {"train": 0.8, "val": 0.2}


def test_split_ratios_reject_bad_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(0.5, 0.2, 0.1)


def test_split_ratios_reject_negative():
    with pytest.raises(ValueError, match="non-negative"):
        SplitRatios(1.2, -0.2, 0.0)


def test_synthetic_config_rejects_inverted_object_range():
    with pytest.raises(ValueError, match="min_objects <= max_objects"):
        SyntheticConfig(min_objects=5, max_objects=2)


def test_synthetic_config_rejects_inverted_size_ratio():
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
def test_synthetic_config_rejects_out_of_range_placement_knobs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SyntheticConfig(**kwargs)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_synthetic_config_accepts_boundary_values(value):
    config = SyntheticConfig(overlap_iou=value, boundary_tolerance=value, max_placement_attempts=1)
    assert config.overlap_iou == value
    assert config.max_placement_attempts == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ClassMode.SHAPE, [s.value for s in Shape]),
        (ClassMode.COLOR, [c.value for c in Color]),
    ],
)
def test_class_names(mode, expected):
    assert class_names(mode) == expected


def test_class_names_shape_color_product_size():
    assert len(class_names(ClassMode.SHAPE_COLOR)) == len(Shape) * len(Color)


def test_class_id_round_trips():
    for mode in ClassMode:
        for shape in Shape:
            for color in Color:
                idx = class_id_of(shape, color, mode)
                assert 0 <= idx < len(class_names(mode))


def test_shapes_defaults_to_the_four_geometric_shapes():
    """An untouched config still draws only square/rectangle/triangle/circle.

    Appending the animals to `Shape` would otherwise change what every existing seeded caller generates; pinning the
    default here is the guard that kept that upgrade non-breaking.

    """
    assert SyntheticConfig().shapes == (Shape.SQUARE, Shape.RECTANGLE, Shape.TRIANGLE, Shape.CIRCLE)
    assert SyntheticConfig().shapes == DEFAULT_SHAPES


def test_default_shapes_keeps_the_original_class_ids():
    """The default vocabulary still occupies class ids 0-3 of the full `Shape` order.

    Datasets generated before the animals landed carry those ids in their label files, so the enum had to grow by
    appending rather than by re-ordering.

    """
    assert [class_id_of(shape, Color.RED, ClassMode.SHAPE) for shape in DEFAULT_SHAPES] == [0, 1, 2, 3]


def test_shapes_accepts_an_animal_override():
    """A custom tuple is stored verbatim and restricts the drawable vocabulary.

    This is the opt-in path documented for animal shapes; the config must not silently widen or reorder what the caller
    asked for.

    """
    config = SyntheticConfig(shapes=(Shape.GIRAFFE, Shape.DUCK))
    assert config.shapes == (Shape.GIRAFFE, Shape.DUCK)


def test_animal_shapes_returns_every_animal_by_default():
    """`animal_shapes()` is the whole roster, which is exactly `KEYPOINT_SHAPES`.

    The helper exists so callers can ask for "the animals" without importing and re-listing the roster; if it ever
    returned a different set than the keypoint-capable one, `Task.KEYPOINTS` would start rejecting its own default.

    """
    assert animal_shapes() == KEYPOINT_SHAPES
    assert len(animal_shapes()) == 12
    assert not set(animal_shapes()) & set(DEFAULT_SHAPES)


@pytest.mark.parametrize("count", range(13))
def test_animal_shapes_by_count_is_a_declaration_order_prefix(count):
    """`animal_shapes(n)` takes the first `n` animals in `Shape` declaration order.

    "First n" has to mean a stable prefix, not an arbitrary subset: a dataset regenerated with the same `n` after a
    thirteenth animal is appended must still contain the same species, or its class ids stop meaning one thing.

    """
    selected = animal_shapes(count)
    assert len(selected) == count
    assert selected == KEYPOINT_SHAPES[:count]


def test_animal_shapes_by_count_feeds_a_keypoints_config():
    """A count-selected tuple is accepted verbatim by the field it exists to fill.

    The helper is a convenience constructor for `SyntheticConfig.shapes`, so the round trip through the dataclass —
    including its `Task.KEYPOINTS` vocabulary check — is the behaviour worth pinning, not the tuple in isolation.

    """
    config = SyntheticConfig(task=Task.KEYPOINTS, shapes=animal_shapes(5))
    assert config.shapes == (Shape.DUCK, Shape.ELEPHANT, Shape.GIRAFFE, Shape.FISH, Shape.RABBIT)


@pytest.mark.parametrize("count", [-1, 13, 99])
def test_animal_shapes_rejects_a_count_outside_the_roster(count):
    """An out-of-range count raises instead of silently clamping to the roster length.

    Asking for 13 of 12 animals is a caller bug — quietly returning 12 would hide it until the dataset came out one
    class short of what the caller believed it had requested.

    """
    with pytest.raises(ValueError, match="count must be within"):
        animal_shapes(count)


def test_shapes_rejects_an_empty_tuple():
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
        pytest.param((Shape.DUCK, "square"), id="mixed-string"),
        pytest.param((Shape.DUCK, None), id="none"),
    ],
)
def test_shapes_rejects_non_shape_elements(bad):
    """Anything that is not a `Shape` member is refused, including an equal bare string.

    `Shape` subclasses `str`, so `"duck" == Shape.DUCK` is True and a plain string would sail through a naive equality
    check while breaking identity comparisons downstream.

    """
    with pytest.raises(ValueError, match="only Shape members"):
        SyntheticConfig(shapes=bad)
