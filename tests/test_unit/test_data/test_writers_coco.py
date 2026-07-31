"""COCO writer tests: JSON validity, categories, and per-task segmentation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fuse_augmentations.data.config import SyntheticConfig, Task, class_names
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
