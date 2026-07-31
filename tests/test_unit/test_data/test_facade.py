"""End-to-end facade tests: all format x task combos, splits, and determinism."""

from __future__ import annotations

import itertools

import pytest

from fuse_augmentations.data import SplitRatios, generate_dataset

_FORMATS = ["coco", "yolo"]
_TASKS = ["detection", "segmentation", "obb"]


@pytest.mark.parametrize(("fmt", "task"), list(itertools.product(_FORMATS, _TASKS)))
def test_generate_all_combos(tmp_path, fmt, task):
    out = tmp_path / f"{fmt}_{task}"
    counts = generate_dataset(out, num_images=10, fmt=fmt, task=task, img_size=64, seed=0)
    assert sum(counts.values()) == 10
    assert out.exists()


def test_split_counts_follow_ratios(tmp_path):
    counts = generate_dataset(tmp_path, num_images=10, fmt="coco", seed=0)
    assert counts == {"train": 7, "val": 2, "test": 1}


def test_custom_split_ratios(tmp_path):
    counts = generate_dataset(tmp_path, num_images=10, fmt="yolo", split_ratios=SplitRatios(0.5, 0.5, 0.0), seed=0)
    assert counts == {"train": 5, "val": 5}


def test_coco_file_counts_match(tmp_path):
    counts = generate_dataset(tmp_path, num_images=10, fmt="coco", seed=1)
    for split, count in counts.items():
        assert len(list((tmp_path / split).glob("*.jpg"))) == count


def test_yolo_file_counts_match(tmp_path):
    counts = generate_dataset(tmp_path, num_images=10, fmt="yolo", seed=1)
    for split, count in counts.items():
        assert len(list((tmp_path / "images" / split).glob("*.jpg"))) == count


def test_deterministic_labels(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(a, num_images=6, fmt="yolo", task="obb", img_size=64, seed=99)
    generate_dataset(b, num_images=6, fmt="yolo", task="obb", img_size=64, seed=99)
    label_a = (a / "labels" / "train" / "img_000000.txt").read_text()
    label_b = (b / "labels" / "train" / "img_000000.txt").read_text()
    assert label_a == label_b


def test_enum_and_string_args_equivalent(tmp_path):
    from fuse_augmentations.data import ClassMode, OutputFormat, Task

    a = tmp_path / "str"
    b = tmp_path / "enum"
    counts_str = generate_dataset(a, num_images=5, fmt="coco", task="detection", class_mode="color", seed=3)
    counts_enum = generate_dataset(
        b, num_images=5, fmt=OutputFormat.COCO, task=Task.DETECTION, class_mode=ClassMode.COLOR, seed=3
    )
    assert counts_str == counts_enum
