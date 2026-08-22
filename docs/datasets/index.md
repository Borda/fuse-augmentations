---
title: Synthetic shape datasets
description: Generate COCO and YOLO datasets of colored shapes for detection, segmentation, oriented-bounding-box (OBB), and keypoint tasks with fuse-augmentations.
---

# Synthetic shape datasets

`fuse_augmentations.data` draws colored shapes on a canvas and writes ready-to-train datasets. It is a standalone generation utility: no file-backed dataset loader, no model, no training loop. Use it for pipeline smoke tests, augmentation demos, teaching material, and quick detector sanity checks where a real dataset is overkill.

It supports two output formats — **COCO** and **YOLO** — across four tasks — **detection**, **segmentation**, **oriented bounding box (OBB)**, and **keypoints / pose**.

## Install

Rendering uses [Pillow](https://python-pillow.github.io/), which ships as a base dependency — nothing extra to install:

```bash
pip install fuse-augmentations
```

## Quickstart

```python
import tempfile

from fuse_augmentations import generate_dataset

with tempfile.TemporaryDirectory() as out_dir:
    counts = generate_dataset(
        out_dir,
        num_images=100,
        fmt="coco",
        task="detection",
        class_mode="shape",
        seed=0,
    )

print(counts)
```

<details>
<summary>Per-split image counts</summary>

```
{'train': 70, 'val': 20, 'test': 10}
```

</details>

Pass a real path instead of the temporary directory to keep the dataset. The same call with `fmt="yolo"` writes an Ultralytics-style layout; `generate_dataset` returns the number of images written per split.

## In this section

- [Shape families](shapes.md) — the four vocabularies (geometric, animals, symbols, letters), the visual shape reference, and how to select shapes and colors.
- [Tasks and keypoints](tasks.md) — the four annotation tasks and every family's keypoint schema.
- [Output formats](outputs.md) — COCO and YOLO on-disk layouts, plus the in-memory streaming training feed.
- [Customization and extension](customization.md) — reproducibility knobs, custom splits and fills, registering new families and writers, and editing the packaged assets.
