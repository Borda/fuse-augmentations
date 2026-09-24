"""Measure Albumentations CPU matrix preparation and a complete image/box endpoint.

Run with ``NO_ALBUMENTATIONS_UPDATE=1 python experiments/bench_albu_preparation.py OUTPUT.json --revision LABEL``.
Select another source tree with ``PYTHONPATH``. The private preparation call intentionally attributes this
implementation's cost; the separate public pipeline measurement includes the image warp and bbox routing. Run on an
otherwise idle host. Results exclude decode, transfer and model execution.

"""

import cProfile
import io
import json
import platform
import pstats
import time
from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from fused_transforms import Compose
from fused_transforms.__about__ import __version__


def pipeline(seed):
    """Construct identical independent transform streams for each measured case."""
    transforms = [A.Rotate(limit=15, p=0.7), A.Affine(scale=(0.9, 1.1), p=0.8), A.HorizontalFlip(p=0.5)]
    for transform in transforms:
        transform.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Compose(transforms, execution="torch", data_keys=["input", "bbox_xyxy"])


def measure(call):
    """Time five warmups and thirty calls; return median and p95 in milliseconds."""
    for _ in range(5):
        call()
    elapsed = []
    for _ in range(30):
        start = time.perf_counter_ns()
        call()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95))}


def main(output: str, revision: str) -> None:
    """Persist timings, revision label, and one attributed preparation profile."""
    output_path = Path(output)
    torch.set_num_threads(1)
    rows = []
    for batch in (1, 8, 32):
        image = torch.rand(batch, 3, 224, 224, generator=torch.Generator().manual_seed(23))
        boxes = torch.tensor([[[30.0, 40.0, 150.0, 180.0]]]).repeat(batch, 1, 1)
        for run in range(3):
            pipe = pipeline(17)
            prepare = measure(partial(pipe._segments[0]._compose_matrices, image))
            pipe = pipeline(17)
            whole = measure(partial(pipe, image, boxes))
            rows.append({"batch": batch, "run": run, "prepare": prepare, "image_boxes_pipeline": whole})
    pipe = pipeline(17)
    profiler = cProfile.Profile()
    profiler.runcall(lambda: [pipe._segments[0]._compose_matrices(image) for _ in range(20)])
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "revision": revision,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "fused_transforms": __version__,
                "seed_policy": {"pipeline_streams": 17, "input_generator": 23},
                "profile_batch": 32,
                "torch": torch.__version__,
                "albumentations": A.__version__,
                "numpy": np.__version__,
                "threads": 1,
                "warmups": 5,
                "timed_calls": 30,
                "repeats": 3,
                "canvas": [224, 224],
                "endpoint": (
                    "CPU float32 BCHW image and dense pixel-edge bbox tensor; no decode, host/device transfer or model"
                ),
                "measurements": rows,
                "preparation_profile": stream.getvalue(),
            },
            indent=2,
        )
        + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
