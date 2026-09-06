"""Measure opt-in antialias CPU cost separately from scale estimation and the warp.

Run ``NO_ALBUMENTATIONS_UPDATE=1 python experiments/bench_antialias.py`` with Kornia installed on an otherwise idle
host. The fixed RGB224 cases use B1/8/32; results are written to ignored ``experiments/results/antialias_cpu.json``. The
scalar-extraction profile counts CPU events, not accelerator sync latency.

"""

import json
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from kornia.augmentation import RandomResizedCrop

from fuse_augmentations import Compose
from fuse_augmentations.affine.segment import _antialias_axis_scales, _maybe_antialias_prefilter


def measure(call):
    """Return median and p95 milliseconds from five warmups and thirty calls."""
    for _ in range(5):
        call()
    elapsed = []
    for _ in range(30):
        start = time.perf_counter_ns()
        call()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95))}


def main():
    """Record repeated fixed-shape costs and scalar extraction in a CPU profile."""
    torch.set_num_threads(1)
    rows = []
    for batch in (1, 8, 32):
        image = torch.rand(batch, 3, 224, 224, generator=torch.Generator().manual_seed(23))
        matrix = torch.eye(3).repeat(batch, 1, 1)
        for index in range(batch):
            matrix[index, 0, 0], matrix[index, 1, 1] = ((0.25, 0.5), (0.4, 0.2), (0.9, 0.9), (1.0, 1.0))[index % 4]
        for run in range(3):
            row = {"batch": batch, "run": run}
            row["heterogeneous_scales"] = measure(partial(_antialias_axis_scales, matrix))
            row["heterogeneous_prefilter"] = measure(partial(_maybe_antialias_prefilter, image, matrix, True))
            for enabled in (False, True):
                pipe = Compose(
                    [RandomResizedCrop(size=(56, 56), scale=(1.0, 1.0), ratio=(1.0, 1.0), p=1.0)], antialias=enabled
                )
                row[f"uniform_crop_antialias_{enabled}"] = measure(partial(pipe, image))
            rows.append(row)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        _maybe_antialias_prefilter(image, matrix, True)
    scalar_events = [
        {"key": event.key, "count": event.count, "self_cpu_time_us": event.self_cpu_time_total}
        for event in profile.key_averages()
        if "scalar" in event.key or "item" in event.key
    ]
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scope": "CPU float32 RGB224, threads1, five warmups/thirty calls/three repeats; fixed seed23",
        "prefilter_scales_xy": [[0.25, 0.5], [0.4, 0.2], [0.9, 0.9], [1.0, 1.0]],
        "crop": "full-canvas Kornia RandomResizedCrop 224 to 56; full public pipeline with flag off/on",
        "rows": rows,
        "B32_prefilter_scalar_events": scalar_events,
        "limits": "CPU scalar counts exclude accelerator sync latency; no model-quality claim",
    }
    output = Path(__file__).parent / "results" / "antialias_cpu.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
