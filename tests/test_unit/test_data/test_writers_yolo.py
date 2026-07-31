"""YOLO writer tests: normalized labels, row shape per task, and data.yaml."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from fuse_augmentations.data.config import SyntheticConfig, Task, class_names
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.sample import Annotation, Sample
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


def test_write_rejects_empty_splits(tmp_path):
    names = class_names(SyntheticConfig().class_mode)
    with pytest.raises(ValueError, match="at least one split"):
        YoloWriter(Task.DETECTION, names).write({}, tmp_path)
    assert list(tmp_path.iterdir()) == []  # guarded before any partial output is created


def test_detection_label_clamps_edge_crossing_box(tmp_path):
    # bbox crosses the left/top edges; the YOLO label must describe the clipped visible box.
    ann = Annotation(
        class_id=0,
        class_name="square",
        polygon=[-20.0, 10.0, 40.0, 10.0, 40.0, 50.0, -20.0, 50.0],
        bbox_xyxy=(-20.0, 10.0, 40.0, 50.0),
        obb_corners=[-20.0, 10.0, 40.0, 10.0, 40.0, 50.0, -20.0, 50.0],
    )
    sample = Sample(image=np.zeros((100, 100, 3), dtype=np.uint8), annotations=[ann], width=100, height=100)
    YoloWriter(Task.DETECTION, class_names(SyntheticConfig().class_mode)).write({"train": [sample]}, tmp_path)
    cx, cy, w, h = (float(v) for v in (tmp_path / "labels" / "train" / "img_000000.txt").read_text().split()[1:])
    # Clipped box is (0, 10, 40, 50): cx=0.2, cy=0.3, w=0.4, h=0.4 — not the unclipped cx=0.1, w=0.6.
    assert (cx, cy, w, h) == pytest.approx((0.2, 0.3, 0.4, 0.4))
