"""Config validation and class-vocabulary tests (no Pillow required)."""

from __future__ import annotations

import pytest

from fuse_augmentations.data.config import (
    ClassMode,
    Color,
    Shape,
    SplitRatios,
    SyntheticConfig,
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
