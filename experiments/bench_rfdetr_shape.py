"""Benchmark the rf-detr training shape: image plus boxes plus per-instance metadata.

The sibling scripts in this directory time an image-only pipeline. A detection training step does
more than that: it carries a box table and a per-instance label vector alongside the image, and it
has to drop the instances a warp pushed off the canvas — keeping the labels in step with the boxes.
An image-only speed comparison says nothing about whether that whole step is faster, which is the
number an rf-detr adoption decision actually needs.

This script times four reproducible implementations of the same model-ready step at two resolutions:

``albumentations``
    The native path rf-detr uses today: ``A.Compose(..., bbox_params=A.BboxParams(label_fields=))``
    on an HWC ``uint8`` array, with Albumentations doing its own clipping and instance dropping.
``fuse_cv2`` / ``fuse_torch``
    ``fuse_augmentations.Compose`` with ``data_keys=["input", "bbox_xyxy"]`` on the same NumPy
    inputs, followed by ``clip_bbox_xyxy`` and ``instance_keep_mask`` to make the survival decision
    and the label filtering the caller owns in this package. The two differ only in ``execution=``.
``fuse_cv2_tensor_in``
    The same fused execution with a model-ready tensor supplied at the input boundary. It retains the common
    model-ready output boundary, so its result measures augmentation-only work rather than an incomplete endpoint.

The post-warp survival work and every conversion needed to reach the common model-ready endpoint
are deliberately inside the timed region. The rows are independently reproducible; they are not a
paired raster or geometric-parity experiment because the pipelines do not replay a shared sampled
parameter sequence.

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
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as albu
import cv2
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
BENCHMARK_SEED = 0
MODEL_READY_ENDPOINT = {
    "image": {"dtype": "float32", "range": [0.0, 1.0], "layout": "BCHW", "device": "cpu"},
    "boxes": {"dtype": "float32", "layout": "N4", "device": "cpu"},
    "labels": {"dtype": "int64", "layout": "N", "device": "cpu"},
}


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
    labels = np.arange(NUM_INSTANCES, dtype=np.int64)
    return image, boxes, labels


def _albu_transforms(chain: str) -> list[albu.BasicTransform]:
    """Return one of the two geometric chains, shared by every benchmarked implementation.

    ``"two_op"`` is a flip plus one affine. It is the honest floor rather than a showcase: Albumentations executes it as
    one array flip and one warp, so there is no second resampling pass for fusion to remove, and any win has to come
    from per-call overhead alone.

    ``"four_op"`` is what a detection recipe actually stacks. Albumentations resamples once per affine; the fused
    pipeline still resamples exactly once, which is the property the package exists for.

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


def _seed_albumentations_transforms(transforms: list[albu.BasicTransform], seed: int) -> None:
    """Seed the private random stream each Albumentations transform owns."""
    for transform in transforms:
        transform.set_random_seed(seed)


def _model_ready_endpoint(
    image: np.ndarray | torch.Tensor,
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a completed augmentation into the benchmark's common training boundary."""
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    else:
        image_tensor = image.to(dtype=torch.float32)
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
    return image_tensor, boxes_tensor, labels_tensor


def _validate_model_ready_endpoint(result: tuple[torch.Tensor, torch.Tensor, torch.Tensor], resolution: int) -> None:
    """Reject a timed row that does not return the declared image and target boundary."""
    image, boxes, labels = result
    if image.shape != (1, 3, resolution, resolution) or image.dtype is not torch.float32 or image.device.type != "cpu":
        raise ValueError("Detection benchmark image endpoint must be CPU float32 BCHW at the requested resolution.")
    if not (torch.all(image >= 0) and torch.all(image <= 1)):
        raise ValueError("Detection benchmark image endpoint must be normalized to [0, 1].")
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or boxes.dtype is not torch.float32 or boxes.device != image.device:
        raise ValueError("Detection benchmark box endpoint must be CPU float32 with shape (N, 4).")
    labels_are_aligned = labels.ndim == 1 and labels.shape[0] == boxes.shape[0]
    labels_match_endpoint = labels.dtype is torch.int64 and labels.device == image.device
    if not labels_are_aligned or not labels_match_endpoint:
        raise ValueError("Detection benchmark labels must remain aligned with the surviving box rows.")
    source_labels = torch.from_numpy(_sample(resolution)[2])
    if not torch.isin(labels, source_labels).all() or torch.unique(labels).numel() != labels.numel():
        raise ValueError("Detection benchmark labels must be unique source labels for each surviving box.")


def _git_revision() -> str:
    """Return the checked-out revision that produced a benchmark result."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _albu_step(resolution: int, chain: str) -> Callable[[], object]:
    """Build the native Albumentations detection step, boxes and labels included.

    ``min_width``/``min_height``/``min_visibility`` are set rather than left at their defaults so Albumentations drops
    exactly the instances the fused variants drop. With the defaults it keeps every sliver, which would have it doing
    less work than the side it is being compared against.

    """
    transforms = _albu_transforms(chain)
    pipeline = albu.Compose(
        transforms,
        bbox_params=albu.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            filter_invalid_bboxes=True,
            min_width=MIN_INSTANCE_SIZE,
            min_height=MIN_INSTANCE_SIZE,
            min_visibility=MIN_INSTANCE_VISIBILITY,
        ),
        seed=BENCHMARK_SEED,
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        result = pipeline(image=image, bboxes=boxes, labels=labels)
        return _model_ready_endpoint(result["image"], np.asarray(result["bboxes"]), np.asarray(result["labels"]))

    return run


def _fuse_step(resolution: int, execution: str, chain: str) -> Callable[[], object]:
    """Build the fuse-augmentations detection step for one execution engine.

    The clip and survival-mask calls are part of the step rather than an afterthought: this package
    hands back every instance and leaves the decision to the caller, so a fair comparison against
    Albumentations' internal filtering has to pay for it here.

    """
    transforms = _albu_transforms(chain)
    _seed_albumentations_transforms(transforms, BENCHMARK_SEED)
    pipeline = Compose(
        transforms,
        data_keys=["input", "bbox_xyxy"],
        execution=execution,
    )
    image, boxes, labels = _sample(resolution)

    def run() -> object:
        out = pipeline(image=image, bboxes=boxes)
        warped = torch.from_numpy(out["bboxes"])
        clipped = clip_bbox_xyxy(warped, resolution, resolution)
        keep = instance_keep_mask(warped, clipped, min_size=MIN_INSTANCE_SIZE, min_visibility=MIN_INSTANCE_VISIBILITY)
        return _model_ready_endpoint(out["image"], clipped[keep], labels[keep.numpy()])

    return run


def _fuse_tensor_step(resolution: int, execution: str, chain: str) -> Callable[[], object]:
    """Build the same step with inputs already in tensor form.

    This augmentation-only attribution starts with the same model-ready tensor format that every row returns. It still
    includes the target survival work, so it isolates input preparation without comparing against a shorter output
    contract.

    """
    transforms = _albu_transforms(chain)
    _seed_albumentations_transforms(transforms, BENCHMARK_SEED)
    pipeline = Compose(
        transforms,
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
        return _model_ready_endpoint(warped_image, clipped[keep], labels_tensor[keep[0]])

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

    ``torch.manual_seed`` alone leaves the augmentation parameters unseeded: Albumentations samples through
    ``numpy.random`` and its own per-transform ``py_random``, and the fused NumPy path draws its activation gates from
    ``numpy.random`` too. An unseeded run still produces valid timings, but the two sides then warp by different angles,
    which makes the comparison unrepeatable rather than merely noisy.

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
                step = builder(resolution, chain)
                _validate_model_ready_endpoint(step(), resolution)
                measurements.append(_time(step, pipeline=name, chain=chain, resolution=resolution))

    for entry in measurements:
        print(f"{entry.pipeline:>18}  {entry.chain:>8}  {entry.resolution:>5}  {entry.median_ms:8.3f} ms")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "environment": {
                    "revision": _git_revision(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "albumentations": albu.__version__,
                    "numpy": np.__version__,
                    "torch_threads": torch.get_num_threads(),
                    "opencv_threads": cv2.getNumThreads(),
                },
                "config": {
                    "num_instances": NUM_INSTANCES,
                    "min_instance_size": MIN_INSTANCE_SIZE,
                    "min_instance_visibility": MIN_INSTANCE_VISIBILITY,
                    "warmup_calls": WARMUP_CALLS,
                    "timed_calls": TIMED_CALLS,
                    "seed_policy": {
                        "global_seed": BENCHMARK_SEED,
                        "albumentations_transform_seed": BENCHMARK_SEED,
                        "comparison": "per-variant reproducible; no shared sampled-geometry replay",
                    },
                    "endpoint": MODEL_READY_ENDPOINT,
                    "input_boundaries": {
                        "albumentations": "HWC uint8 NumPy image with NumPy boxes and labels",
                        "fuse_cv2": "HWC uint8 NumPy image with NumPy boxes and labels",
                        "fuse_torch": "HWC uint8 NumPy image with NumPy boxes and labels",
                        "fuse_cv2_tensor_in": "BCHW float32 [0, 1] CPU tensor with dense tensor boxes and labels",
                    },
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
