"""Generator tests: reproducibility, placement constraints, and class modes."""

from __future__ import annotations

import numpy as np
import pytest

from fuse_augmentations.data.config import ClassMode, Color, Shape, SyntheticConfig, class_names
from fuse_augmentations.data.generator import SyntheticGenerator, _boundary_overlap
from fuse_augmentations.data.shapes import bbox_iou


def _generate(**kwargs):
    config = SyntheticConfig(**kwargs)
    return SyntheticGenerator(config).sample(np.random.default_rng(123)), config


def test_image_shape_and_dtype():
    sample, _config = _generate(img_size=96)
    assert sample.image.shape == (96, 96, 3)
    assert sample.image.dtype == np.uint8
    assert sample.width == 96
    assert sample.height == 96


def test_object_count_within_range():
    sample, _ = _generate(img_size=128, min_objects=2, max_objects=5)
    assert 2 <= len(sample.annotations) <= 5


def test_same_seed_is_identical():
    config = SyntheticConfig(img_size=64)
    gen = SyntheticGenerator(config)
    a = gen.sample(np.random.default_rng(7))
    b = gen.sample(np.random.default_rng(7))
    assert np.array_equal(a.image, b.image)
    assert [ann.bbox_xyxy for ann in a.annotations] == [ann.bbox_xyxy for ann in b.annotations]


def test_different_seed_differs():
    config = SyntheticConfig(img_size=64)
    gen = SyntheticGenerator(config)
    a = gen.sample(np.random.default_rng(1))
    b = gen.sample(np.random.default_rng(2))
    assert not np.array_equal(a.image, b.image)


def test_placement_respects_boundary_tolerance():
    sample, config = _generate(img_size=128, min_objects=5, max_objects=10)
    for ann in sample.annotations:
        assert _boundary_overlap(ann.bbox_xyxy, config.img_size) <= config.boundary_tolerance + 1e-9


def test_kept_boxes_respect_overlap_threshold():
    sample, config = _generate(img_size=128, min_objects=5, max_objects=10)
    boxes = [ann.bbox_xyxy for ann in sample.annotations]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert bbox_iou(boxes[i], boxes[j]) <= config.overlap_iou + 1e-9


def test_annotation_fields_are_consistent():
    sample, _ = _generate(img_size=96, min_objects=3, max_objects=3)
    for ann in sample.annotations:
        assert len(ann.obb_corners) == 8
        assert len(ann.polygon) >= 6
        x1, y1, x2, y2 = ann.bbox_xyxy
        assert x2 > x1
        assert y2 > y1


def test_generate_yields_lazily():
    import types

    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    stream = gen.generate(3, seed=0)
    assert isinstance(stream, types.GeneratorType)


def test_generate_yields_exact_count():
    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    assert len(list(gen.generate(5, seed=0))) == 5


def test_generate_same_seed_is_deterministic():
    gen = SyntheticGenerator(SyntheticConfig(img_size=32))
    first = [ann.bbox_xyxy for s in gen.generate(3, seed=7) for ann in s.annotations]
    second = [ann.bbox_xyxy for s in gen.generate(3, seed=7) for ann in s.annotations]
    assert first == second


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ClassMode.SHAPE, {s.value for s in Shape}),
        (ClassMode.COLOR, {c.value for c in Color}),
        (ClassMode.SHAPE_COLOR, {f"{c.value}_{s.value}" for s in Shape for c in Color}),
    ],
)
def test_class_names_belong_to_mode(mode, expected):
    config = SyntheticConfig(img_size=128, min_objects=8, max_objects=10, class_mode=mode)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    vocab = set(class_names(mode))
    assert vocab == expected
    for ann in sample.annotations:
        assert ann.class_name in vocab
        assert class_names(mode)[ann.class_id] == ann.class_name
