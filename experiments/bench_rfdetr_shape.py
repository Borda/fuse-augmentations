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
import random
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

#: Instance-survival thresholds, applied identically on both sides. Albumentations owns the decision
#: internally through ``BboxParams``; this package hands it to the caller through ``instance_keep_mask``.
#: They have to agree or the two are not doing the same amount of work -- Albumentations' defaults keep
#: every sliver, while the fused side was dropping instances below 1 px or 10% visibility.
MIN_INSTANCE_SIZE = 1.0
MIN_INSTANCE_VISIBILITY = 0.1

#: The two chain lengths timed. See :func:`_albu_transforms` for why both are reported.
CHAINS = ("two_op", "four_op")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One timed configuration: milliseconds per call for a pipeline at one resolution."""

    pipeline: str
    chain: str
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


def _albu_transforms(chain: str) -> list[albu.BasicTransform]:
    """Return one of the two geometric chains, shared by every benchmarked implementation.

    ``"two_op"`` is a flip plus one affine. It is the honest floor rather than a showcase: Albumentations
    executes it as one array flip and one warp, so there is no second resampling pass for fusion to
    remove, and any win has to come from per-call overhead alone.

    ``"four_op"`` is what a detection recipe actually stacks. Albumentations resamples once per affine;
    the fused pipeline still resamples exactly once, which is the property the package exists for.

    """
    if chain == "two_op":
        return [
            albu.HorizontalFlip(p=0.5),
            albu.Affine(rotate=(-10.0, 10.0), scale=(0.9, 1.1), p=1.0),
        ]
    return [
        albu.HorizontalFlip(p=0.5),
        albu.Affine(rotate=(-10.0, 10.0), scale=(0.9, 1.1), p=1.0),
        albu.Affine(translate_percent=(-0.05, 0.05), p=1.0),
        albu.Affine(shear=(-5.0, 5.0), p=1.0),
    ]


def _albu_step(resolution: int, chain: str) -> Callable[[], object]:
    """Build the native Albumentations detection step, boxes and labels included.

    ``min_width``/``min_height``/``min_visibility`` are set rather than left at their defaults so
    Albumentations drops exactly the instances the fused variants drop. With the defaults it keeps every
    sliver, which would have it doing less work than the side it is being compared against.

    """
    pipeline = albu.Compose(
        _albu_transforms(chain),
        bbox_params=albu.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            filter_invalid_bboxes=True,
            min_width=MIN_INSTANCE_SIZE,
            min_height=MIN_INSTANCE_SIZE,
            min_visibility=MIN_INSTANCE_VISIBILITY,
        ),
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        return pipeline(image=image, bboxes=boxes, labels=labels)

    return run


def _fuse_step(resolution: int, execution: str, chain: str) -> Callable[[], object]:
    """Build the fuse-augmentations detection step for one execution engine.

    The clip and survival-mask calls are part of the step rather than an afterthought: this package
    hands back every instance and leaves the decision to the caller, so a fair comparison against
    Albumentations' internal filtering has to pay for it here.

    """
    pipeline = Compose(
        _albu_transforms(chain),
        data_keys=["input", "bbox_xyxy"],
        execution=execution,
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        out = pipeline(image=image, bboxes=boxes)
        warped = torch.from_numpy(out["bboxes"])
        clipped = clip_bbox_xyxy(warped, resolution, resolution)
        keep = instance_keep_mask(warped, clipped, min_size=MIN_INSTANCE_SIZE, min_visibility=MIN_INSTANCE_VISIBILITY)
        return out["image"], clipped[keep], labels[keep.numpy()]

    return run


def _fuse_tensor_step(resolution: int, execution: str, chain: str) -> Callable[[], object]:
    """Build the same step with inputs already in tensor form.

    Timed alongside the NumPy variants to attribute the difference: this one pays for the warp and
    the survival decision but not for the array-to-tensor normalisation, so the gap between it and
    ``fuse_cv2`` is the cost of the conversion rather than of the augmentation itself.

    The conversion back to a channel-last array is inside the timed region. Every other variant here
    ends holding one, so a variant that stopped at a tensor would be measuring a shorter step and
    would understate its own cost by exactly the copy the others pay.

    """
    pipeline = Compose(
        _albu_transforms(chain),
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
        keep = instance_keep_mask(
            warped_boxes, clipped, min_size=MIN_INSTANCE_SIZE, min_visibility=MIN_INSTANCE_VISIBILITY
        )
        image_hwc = warped_image[0].permute(1, 2, 0).contiguous().numpy()
        return image_hwc, clipped[keep], labels_tensor[keep[0]]

    return run


def _time(run: Callable[[], object], pipeline: str, chain: str, resolution: int) -> Measurement:
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
        chain=chain,
        resolution=resolution,
        median_ms=round(statistics.median(samples), 3),
        p10_ms=round(samples[int(0.1 * len(samples))], 3),
        p90_ms=round(samples[int(0.9 * len(samples))], 3),
    )


def _seed_everything() -> None:
    """Seed every RNG domain the benchmarked pipelines draw from.

    ``torch.manual_seed`` alone leaves the augmentation parameters unseeded: Albumentations samples
    through ``numpy.random`` and its own per-transform ``py_random``, and the fused NumPy path draws its
    activation gates from ``numpy.random`` too. An unseeded run still produces valid timings, but the two
    sides then warp by different angles, which makes the comparison unrepeatable rather than merely
    noisy.

    """
    random.seed(0)
    # The legacy global RandomState, deliberately: albumentations and the fused NumPy path both
    # draw from it rather than from a Generator.
    np.random.seed(0)
    torch.manual_seed(0)


def main() -> None:
    """Run every pipeline at every resolution and chain length, and write the results as JSON."""
    builders: dict[str, Callable[[int, str], Callable[[], object]]] = {
        "albumentations": _albu_step,
        "fuse_cv2": lambda resolution, chain: _fuse_step(resolution, "cv2", chain),
        "fuse_torch": lambda resolution, chain: _fuse_step(resolution, "torch", chain),
        "fuse_cv2_tensor_in": lambda resolution, chain: _fuse_tensor_step(resolution, "cv2", chain),
    }

    measurements: list[Measurement] = []
    for chain in CHAINS:
        for resolution in RESOLUTIONS:
            for name, builder in builders.items():
                _seed_everything()
                measurements.append(
                    _time(builder(resolution, chain), pipeline=name, chain=chain, resolution=resolution)
                )

    for entry in measurements:
        print(f"{entry.pipeline:>18}  {entry.chain:>8}  {entry.resolution:>5}  {entry.median_ms:8.3f} ms")

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
                    "min_instance_size": MIN_INSTANCE_SIZE,
                    "min_instance_visibility": MIN_INSTANCE_VISIBILITY,
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
