"""YOLO writer tests: normalized labels, row shape per task, and data.yaml."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from fuse_augmentations.data.config import SyntheticConfig, Task, class_names
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.writers import YoloWriter


def _write(tmp_path, task, splits=("train", "val")):
    config = SyntheticConfig(img_size=96, min_objects=2, max_objects=4)
    gen = SyntheticGenerator(config)
    rng = np.random.default_rng(9)
    data = {split: [gen.sample(rng) for _ in range(3)] for split in splits}
    names = class_names(config.class_mode)
    YoloWriter(task, names).write(data, tmp_path)
    return data, names


def _rows(tmp_path, split="train"):
    out = []
    for txt in sorted((tmp_path / "labels" / split).glob("*.txt")):
        out.extend(r for r in txt.read_text().splitlines() if r)
    return out


@pytest.mark.parametrize(("task", "tokens"), [(Task.DETECTION, 5), (Task.SEGMENTATION, None), (Task.OBB, 9)])
def test_row_token_counts(tmp_path, task, tokens):
    _write(tmp_path, task)
    for row in _rows(tmp_path):
        parts = row.split()
        if tokens is None:  # segmentation: cls + 2n polygon coords, even coord count
            assert len(parts) >= 7
            assert (len(parts) - 1) % 2 == 0
        else:
            assert len(parts) == tokens


@pytest.mark.parametrize("task", list(Task))
def test_all_coords_normalized(tmp_path, task):
    _write(tmp_path, task)
    for row in _rows(tmp_path):
        coords = [float(v) for v in row.split()[1:]]
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_class_id_is_zero_based_integer(tmp_path):
    _write(tmp_path, Task.DETECTION)
    for row in _rows(tmp_path):
        cls = row.split()[0]
        assert cls.isdigit()
        assert 0 <= int(cls) < 4


def test_data_yaml_contents(tmp_path):
    _, names = _write(tmp_path, Task.DETECTION)
    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["nc"] == len(names)
    assert list(doc["names"].values()) == names
    assert doc["train"] == "images/train"
    assert doc["val"] == "images/val"


def test_data_yaml_omits_absent_split(tmp_path):
    _write(tmp_path, Task.DETECTION, splits=("train",))
    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert "val" not in doc
    assert "test" not in doc


def test_image_label_parity(tmp_path):
    _write(tmp_path, Task.DETECTION)
    for split in ("train", "val"):
        imgs = {p.stem for p in (tmp_path / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (tmp_path / "labels" / split).glob("*.txt")}
        assert imgs == lbls
        assert len(imgs) == 3
