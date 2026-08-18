"""Generator tests: reproducibility, placement constraints, and class modes."""

from __future__ import annotations

import numpy as np
import pytest

from fuse_augmentations.data.config import DEFAULT_SHAPES, ClassMode, Color, Shape, SyntheticConfig, class_names
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


def test_sample_reaches_requested_count_under_pressure():
    # min==max fixes num_objects; a tight-but-feasible scene forces retries yet must still
    # place every object rather than silently dropping any.
    config = SyntheticConfig(
        img_size=96, min_objects=8, max_objects=8, min_size_ratio=0.16, max_size_ratio=0.26, overlap_iou=0.05
    )
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    assert len(sample.annotations) == 8


def test_sample_retry_is_reproducible_for_a_seed():
    config = SyntheticConfig(
        img_size=96, min_objects=8, max_objects=8, min_size_ratio=0.16, max_size_ratio=0.26, overlap_iou=0.05
    )
    gen = SyntheticGenerator(config)
    first = [ann.bbox_xyxy for ann in gen.sample(np.random.default_rng(3)).annotations]
    second = [ann.bbox_xyxy for ann in gen.sample(np.random.default_rng(3)).annotations]
    assert first == second


ANIMAL_SHAPES = tuple(s for s in Shape if s not in DEFAULT_SHAPES)


def test_default_config_draws_only_the_original_four_shapes():
    """A config that never mentions `shapes` still produces only the pre-animal vocabulary.

    This is the compatibility guarantee behind appending to `Shape`: existing seeded callers must keep getting
    square/rectangle/triangle/circle and nothing else.

    """
    config = SyntheticConfig(img_size=192, min_objects=12, max_objects=12, class_mode=ClassMode.SHAPE)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(0))
    drawn = {ann.class_name for ann in sample.annotations}
    assert drawn <= {s.value for s in DEFAULT_SHAPES}


@pytest.mark.parametrize("shape", ANIMAL_SHAPES)
def test_single_animal_shape_is_the_only_one_drawn(shape):
    """Restricting `cfg.shapes` to one animal makes every annotation carry that class.

    This is the documented opt-in for the animal family; a sampler still reading the full enum would leak geometric
    shapes into a dataset the caller asked to be animals-only.

    """
    config = SyntheticConfig(img_size=160, min_objects=4, max_objects=6, shapes=(shape,))
    sample = SyntheticGenerator(config).sample(np.random.default_rng(11))
    assert {ann.class_name for ann in sample.annotations} == {shape.value}


def test_animal_shapes_respect_boundary_tolerance():
    """Animal placements obey the same off-canvas budget as the geometric shapes.

    Animals are far less convex than a square, so this confirms the rejection test still works on a box derived from a
    concave outline rather than only on well-behaved ones.

    """
    config = SyntheticConfig(img_size=192, min_objects=5, max_objects=8, shapes=ANIMAL_SHAPES)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(4))
    for ann in sample.annotations:
        assert _boundary_overlap(ann.bbox_xyxy, config.img_size) <= config.boundary_tolerance + 1e-9


def test_animal_shapes_respect_overlap_threshold():
    """Kept animal boxes stay under `overlap_iou` pairwise.

    Elongated silhouettes (snake, fish) have large bounding boxes relative to their filled area, which is exactly the
    case where a broken IoU guard would let objects pile up.

    """
    config = SyntheticConfig(img_size=192, min_objects=5, max_objects=8, shapes=ANIMAL_SHAPES)
    sample = SyntheticGenerator(config).sample(np.random.default_rng(4))
    boxes = [ann.bbox_xyxy for ann in sample.annotations]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert bbox_iou(boxes[i], boxes[j]) <= config.overlap_iou + 1e-9


def test_animal_shapes_are_seed_deterministic():
    """Two runs of the same seed over the animal vocabulary agree pixel- and label-wise.

    Animal outlines come from lookup tables rather than trigonometry, so this pins that the table path introduces no
    ordering or floating-point nondeterminism of its own.

    """
    config = SyntheticConfig(img_size=128, min_objects=3, max_objects=5, shapes=ANIMAL_SHAPES)
    gen = SyntheticGenerator(config)
    first = gen.sample(np.random.default_rng(21))
    second = gen.sample(np.random.default_rng(21))
    assert np.array_equal(first.image, second.image)
    assert [(a.class_name, a.bbox_xyxy, a.polygon) for a in first.annotations] == [
        (b.class_name, b.bbox_xyxy, b.polygon) for b in second.annotations
    ]


def test_animal_annotations_carry_a_multi_vertex_polygon():
    """An animal annotation exports its full outline, not a four-corner approximation.

    Segmentation labels are taken straight from the drawn polygon, so an animal collapsing to its box would produce
    masks that no longer match the rendered pixels.

    """
    config = SyntheticConfig(img_size=192, min_objects=3, max_objects=3, shapes=(Shape.ELEPHANT,))
    sample = SyntheticGenerator(config).sample(np.random.default_rng(2))
    for ann in sample.annotations:
        assert len(ann.polygon) >= 2 * 15
        assert len(ann.obb_corners) == 8


def test_sample_raises_when_min_objects_unreachable():
    # Near-full shapes with zero tolerated overlap and full-containment cannot fit min_objects.
    config = SyntheticConfig(
        img_size=32,
        min_objects=8,
        max_objects=8,
        min_size_ratio=0.9,
        max_size_ratio=1.0,
        overlap_iou=0.0,
        boundary_tolerance=0.0,
        max_placement_attempts=5,
        rotate=False,
    )
    gen = SyntheticGenerator(config)
    with pytest.raises(RuntimeError, match="min_objects"):
        gen.sample(np.random.default_rng(0))
