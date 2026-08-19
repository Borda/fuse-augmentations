"""COCO writer tests: JSON validity, categories, and per-task segmentation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fuse_augmentations.data.config import KEYPOINT_NAMES, KEYPOINT_SHAPES, Shape, SyntheticConfig, Task, class_names
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.writers import CocoWriter


def _samples(n=4):
    config = SyntheticConfig(img_size=96, min_objects=2, max_objects=4)
    gen = SyntheticGenerator(config)
    rng = np.random.default_rng(5)
    return [gen.sample(rng) for _ in range(n)], config


def _write(tmp_path, task):
    samples, config = _samples()
    names = class_names(config.class_mode)
    CocoWriter(task, names).write({"train": samples}, tmp_path)
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    return doc, samples, names


def test_json_is_parseable_and_complete(tmp_path):
    doc, samples, names = _write(tmp_path, Task.DETECTION)
    assert len(doc["images"]) == len(samples)
    assert len(doc["annotations"]) == sum(len(s.annotations) for s in samples)
    assert len(doc["categories"]) == len(names)


def test_category_ids_are_one_based(tmp_path):
    doc, _, names = _write(tmp_path, Task.DETECTION)
    assert [c["id"] for c in doc["categories"]] == list(range(1, len(names) + 1))
    for ann in doc["annotations"]:
        assert 1 <= ann["category_id"] <= len(names)


def test_bbox_area_positive(tmp_path):
    doc, _, _ = _write(tmp_path, Task.DETECTION)
    for ann in doc["annotations"]:
        assert ann["area"] > 0
        assert len(ann["bbox"]) == 4


def test_detection_has_no_segmentation(tmp_path):
    doc, _, _ = _write(tmp_path, Task.DETECTION)
    assert all("segmentation" not in ann for ann in doc["annotations"])


def test_segmentation_polygon_present(tmp_path):
    doc, _, _ = _write(tmp_path, Task.SEGMENTATION)
    for ann in doc["annotations"]:
        assert "segmentation" in ann
        assert len(ann["segmentation"][0]) >= 6


def test_obb_stores_four_corner_polygon(tmp_path):
    doc, _, _ = _write(tmp_path, Task.OBB)
    for ann in doc["annotations"]:
        assert len(ann["segmentation"][0]) == 8


def test_images_written_to_disk(tmp_path):
    _write(tmp_path, Task.DETECTION)
    jpgs = sorted((tmp_path / "train").glob("*.jpg"))
    assert len(jpgs) == 4


@pytest.mark.parametrize("task", list(Task))
def test_image_ids_match_files(tmp_path, task):
    doc, _, _ = _write(tmp_path, task)
    for image in doc["images"]:
        assert (tmp_path / "train" / image["file_name"]).exists()


@pytest.mark.parametrize("task", list(Task))
def test_animal_shape_dataset_round_trips(tmp_path, task):
    """A giraffe-only dataset writes valid COCO JSON for every task without writer changes.

    Animal outlines are concave and carry many more vertices than a square; this is the regression guard that the writer
    stays purely format-agnostic and needed no special casing.

    """
    config = SyntheticConfig(img_size=192, min_objects=2, max_objects=3, shapes=(Shape.GIRAFFE,))
    gen = SyntheticGenerator(config)
    rng = np.random.default_rng(31)
    samples = [gen.sample(rng) for _ in range(3)]
    names = class_names(config.class_mode)
    CocoWriter(task, names).write({"train": samples}, tmp_path)

    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    assert len(doc["annotations"]) == sum(len(s.annotations) for s in samples)
    giraffe_id = names.index(Shape.GIRAFFE.value) + 1  # COCO category ids are 1-based
    assert {ann["category_id"] for ann in doc["annotations"]} == {giraffe_id}
    for ann in doc["annotations"]:
        assert ann["area"] > 0
        if task is Task.SEGMENTATION:
            assert len(ann["segmentation"][0]) >= 2 * 15
        elif task is Task.OBB:
            assert len(ann["segmentation"][0]) == 8


def test_keypoints_task_declares_the_sixteen_point_schema_and_fifteen_edge_skeleton(tmp_path):
    """The category schema carries all 16 landmark names and the 15-edge skeleton, 1-based.

    COCO viewers connect the dots via `skeleton`, which indexes into `keypoints` starting at 1 (not 0); getting that
    off-by-one wrong draws every edge one landmark short.

    """
    config = SyntheticConfig(img_size=192, min_objects=2, max_objects=3, task=Task.KEYPOINTS, shapes=KEYPOINT_SHAPES)
    gen = SyntheticGenerator(config)
    rng = np.random.default_rng(11)
    samples = [gen.sample(rng) for _ in range(3)]
    names = class_names(config.class_mode)
    CocoWriter(Task.KEYPOINTS, names).write({"train": samples}, tmp_path)

    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    for category in doc["categories"]:
        assert category["keypoints"] == list(KEYPOINT_NAMES)
        assert len(category["skeleton"]) == 15
        for i, j in category["skeleton"]:
            assert 1 <= i <= 16
            assert 1 <= j <= 16


def test_keypoints_num_keypoints_excludes_absent_landmarks(tmp_path):
    """`num_keypoints` counts only `v>0` triples, so a whale's absent hind-leg points are excluded.

    A whale's silhouette shows pectoral flippers but no hind legs, so it carries 12 present landmarks (16 minus the four
    `hind_knee_*`/`hind_limb_*` points); `num_keypoints` must reflect that, not the full schema length, or a consumer
    would expect a triple that never arrives.

    """
    config = SyntheticConfig(img_size=192, min_objects=2, max_objects=2, task=Task.KEYPOINTS, shapes=(Shape.WHALE,))
    gen = SyntheticGenerator(config)
    rng = np.random.default_rng(3)
    samples = [gen.sample(rng) for _ in range(2)]
    names = class_names(config.class_mode)
    CocoWriter(Task.KEYPOINTS, names).write({"train": samples}, tmp_path)

    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    for ann in doc["annotations"]:
        assert len(ann["keypoints"]) == 16 * 3
        assert ann["num_keypoints"] <= 12
