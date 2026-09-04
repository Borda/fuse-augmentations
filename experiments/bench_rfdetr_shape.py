"""Benchmark the rf-detr training shape: image plus boxes plus per-instance metadata.

The sibling scripts in this directory time an image-only pipeline. A detection training step does
more than that: it carries a box table and a per-instance label vector alongside the image, and it
has to drop the instances a warp pushed off the canvas — keeping the labels in step with the boxes.
An image-only speed comparison says nothing about whether that whole step is faster, which is the
number an rf-detr adoption decision actually needs.

This script times three implementations of the same step at two resolutions:

``albumentations``
    The native path rf-detr uses today: ``A.Compose(..., bbox_params=A.BboxParams(label_fields=))``
    on an HWC ``uint8`` array, with Albumentations doing its own clipping and instance dropping.
``fuse_cv2`` / ``fuse_torch``
    ``fuse_augmentations.Compose`` with ``data_keys=["input", "bbox_xyxy"]`` on the same NumPy
    inputs, followed by ``clip_bbox_xyxy`` and ``instance_keep_mask`` to make the survival decision
    and the label filtering the caller owns in this package. The two differ only in ``execution=``.

The post-warp survival work is deliberately inside the timed region for the fuse variants: it is
work Albumentations does internally, so leaving it out would compare a complete step against a
partial one.

Usage
-----
Run as a script::

    python experiments/bench_rfdetr_shape.py

Results are written to ``experiments/results/rfdetr_shape.json``.

"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as albu
import numpy as np
import torch

from fuse_augmentations import Compose
from fuse_augmentations.targets import clip_bbox_xyxy, instance_keep_mask

RESOLUTIONS = (640, 1024)
NUM_INSTANCES = 12
WARMUP_CALLS = 20
TIMED_CALLS = 200
RESULT_PATH = Path(__file__).parent / "results" / "rfdetr_shape.json"


@dataclass(frozen=True, slots=True)
class Measurement:
    """One timed configuration: milliseconds per call for a pipeline at one resolution."""

    pipeline: str
    resolution: int
    median_ms: float
    p10_ms: float
    p90_ms: float


def _sample(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one deterministic ``(image, boxes, labels)`` sample at the given resolution."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8)
    centres = rng.uniform(0.15, 0.85, size=(NUM_INSTANCES, 2)) * resolution
    sizes = rng.uniform(0.05, 0.20, size=(NUM_INSTANCES, 2)) * resolution
    boxes = np.concatenate([centres - sizes / 2, centres + sizes / 2], axis=1).astype(np.float32)
    labels = rng.integers(0, 80, size=NUM_INSTANCES).astype(np.int64)
    return image, boxes, labels


def _albu_transforms() -> list[albu.BasicTransform]:
    """Return the geometric chain shared by every benchmarked implementation."""
    return [
        albu.HorizontalFlip(p=0.5),
        albu.Affine(rotate=(-10.0, 10.0), scale=(0.9, 1.1), p=1.0),
    ]


def _albu_step(resolution: int) -> Callable[[], object]:
    """Build the native Albumentations detection step, boxes and labels included."""
    pipeline = albu.Compose(
        _albu_transforms(),
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"], filter_invalid_bboxes=True),
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        return pipeline(image=image, bboxes=boxes, labels=labels)

    return run


def _fuse_step(resolution: int, execution: str) -> Callable[[], object]:
    """Build the fuse-augmentations detection step for one execution engine.

    The clip and survival-mask calls are part of the step rather than an afterthought: this package
    hands back every instance and leaves the decision to the caller, so a fair comparison against
    Albumentations' internal filtering has to pay for it here.

    """
    pipeline = Compose(
        _albu_transforms(),
        data_keys=["input", "bbox_xyxy"],
        execution=execution,
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        out = pipeline(image=image, bboxes=boxes)
        warped = torch.from_numpy(out["bboxes"])
        clipped = clip_bbox_xyxy(warped, resolution, resolution)
        keep = instance_keep_mask(warped, clipped, min_size=1.0, min_visibility=0.1)
        return out["image"], clipped[keep], labels[keep.numpy()]

    return run


def _fuse_tensor_step(resolution: int, execution: str) -> Callable[[], object]:
    """Build the same step with inputs already in tensor form.

    Timed alongside the NumPy variants to attribute the difference: this one pays for the warp and
    the survival decision but not for the array-to-tensor normalisation, so the gap between it and
    ``fuse_cv2`` is the cost of the conversion rather than of the augmentation itself.

    """
    pipeline = Compose(
        _albu_transforms(),
        data_keys=["input", "bbox_xyxy"],
        execution=execution,
    )
    image, boxes, labels = _sample(resolution)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    boxes_tensor = torch.from_numpy(boxes).unsqueeze(0)
    labels_tensor = torch.from_numpy(labels)

    def run() -> object:
        warped_image, warped_boxes = pipeline(image_tensor, boxes_tensor)
        clipped = clip_bbox_xyxy(warped_boxes, resolution, resolution)
        keep = instance_keep_mask(warped_boxes, clipped, min_size=1.0, min_visibility=0.1)
        return warped_image, clipped[keep], labels_tensor[keep[0]]

    return run


def _time(run: Callable[[], object], pipeline: str, resolution: int) -> Measurement:
    """Time one step, reporting the median and the 10th/90th percentile in milliseconds."""
    for _ in range(WARMUP_CALLS):
        run()
    samples: list[float] = []
    for _ in range(TIMED_CALLS):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return Measurement(
        pipeline=pipeline,
        resolution=resolution,
        median_ms=round(statistics.median(samples), 3),
        p10_ms=round(samples[int(0.1 * len(samples))], 3),
        p90_ms=round(samples[int(0.9 * len(samples))], 3),
    )


def main() -> None:
    """Run every pipeline at every resolution and write the results as JSON."""
    torch.manual_seed(0)
    builders: dict[str, Callable[[int], Callable[[], object]]] = {
        "albumentations": _albu_step,
        "fuse_cv2": lambda resolution: _fuse_step(resolution, "cv2"),
        "fuse_torch": lambda resolution: _fuse_step(resolution, "torch"),
        "fuse_cv2_tensor_in": lambda resolution: _fuse_tensor_step(resolution, "cv2"),
    }

    measurements = [
        _time(builder(resolution), pipeline=name, resolution=resolution)
        for resolution in RESOLUTIONS
        for name, builder in builders.items()
    ]

    for entry in measurements:
        print(f"{entry.pipeline:>16}  {entry.resolution:>5}  {entry.median_ms:8.3f} ms")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "albumentations": albu.__version__,
                    "numpy": np.__version__,
                },
                "config": {
                    "num_instances": NUM_INSTANCES,
                    "warmup_calls": WARMUP_CALLS,
                    "timed_calls": TIMED_CALLS,
                },
                "measurements": [asdict(entry) for entry in measurements],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten: {RESULT_PATH}")


if __name__ == "__main__":
    main()
