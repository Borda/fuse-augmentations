---
title: Reproducible augmentation pipelines
description: Control PyTorch, NumPy, Albumentations, DataLoader, reorder, and execution settings for reproducible fuse-augmentations experiments.
---

# Reproducible augmentation pipelines

Reproducibility requires more than one seed. Record the pipeline definition, backend versions, tensor shapes, execution strategy, reorder policy, and every random-number domain that the selected transforms consume.

## Short answer

- Backend-free, Kornia, and TorchVision fused paths primarily consume PyTorch randomness.
- Albumentations fused geometry consumes both global NumPy randomness for package activation gates and each transform's internal random generator for parameters.
- `randomness="backend"` preserves batch-sampling style; it does not promise an identical native random stream or pixel-identical output.
- Batch size, fast-path selection, skipped transforms, and backend version can change random draw consumption.

## Randomness matrix

| Pipeline path                                  | Activation/probability source | Parameter source            | Required control                                                        |
| ---------------------------------------------- | ----------------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `Compose.from_params` without a backend        | PyTorch                       | PyTorch                     | `torch.manual_seed` and fixed batch/shape/order                         |
| Kornia transforms                              | PyTorch                       | Kornia through PyTorch      | `torch.manual_seed`; record Kornia version and `same_on_batch` settings |
| TorchVision transforms                         | PyTorch                       | TorchVision through PyTorch | `torch.manual_seed`; record v1/v2, batch shape, and randomness policy   |
| Albumentations transforms on fused tensor path | Global NumPy                  | Transform-internal RNG      | Seed `numpy.random` and call the transform's supported seed method      |
| Albumentations image-only native NumPy path    | Albumentations/native path    | Transform-internal RNG      | Use Albumentations' supported seeding API and record its version        |

## A deterministic backend-free run

Use an explicit order-preserving policy and reset the seed before building or calling each comparison pipeline:

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy


def run_once(seed: int, images: torch.Tensor) -> torch.Tensor:
    """Build and execute one reproducible backend-free augmentation call."""
    torch.manual_seed(seed)
    pipe = Compose.from_params(
        rotation=(-20.0, 20.0),
        scale=(0.9, 1.1),
        hflip_p=0.5,
        reorder=ReorderPolicy.NONE,
    )
    return pipe(images.clone())


images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(7))
first = run_once(1234, images)
second = run_once(1234, images)
torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Keep the batch size and input shape fixed when asserting byte-for-byte reproduction. Some optimized paths consume draws differently when batch size or the active subset changes.

## Seed Albumentations' two RNG domains

For the fused Albumentations tensor path, seed global NumPy for package probability gates and seed each transform's internal generator for parameters:

```python
import numpy as np
import torch
import albumentations as A

from fuse_augmentations import Compose, ReorderPolicy


def build_albumentations_pipe(seed: int) -> Compose:
    """Build an Albumentations-backed pipeline with both RNG domains seeded."""
    transforms = [
        A.Rotate(limit=(-15, 15), p=0.8),
        A.HorizontalFlip(p=0.5),
    ]
    for index, transform in enumerate(transforms):
        transform.set_random_seed(seed + index)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Compose(transforms, reorder=ReorderPolicy.NONE)


images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(9))
first = build_albumentations_pipe(123)(images.clone())
second = build_albumentations_pipe(123)(images.clone())
torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Calling `np.random.seed` alone does not reset an existing Albumentations 2.x transform's internal parameter stream. Conversely, seeding only the transform does not control the package's NumPy activation gates.

## DataLoader workers

PyTorch assigns a distinct base seed to each worker, but custom NumPy and backend-owned generators must be seeded explicitly. The exact dataset structure is application-specific; this pattern assumes the dataset stores the pipeline as `dataset.augment`:

```python
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info


def seed_augmentation_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and stored Albumentations transforms in one worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is None:
        return

    pipeline = worker_info.dataset.augment
    for index, transform in enumerate(pipeline.original_transforms):
        set_seed = getattr(transform, "set_random_seed", None)
        if callable(set_seed):
            set_seed((worker_seed + index) % (2**32))


generator = torch.Generator().manual_seed(2026)
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_augmentation_worker,
    generator=generator,
)
```

Adapt the `dataset.augment` lookup to your dataset. Constructing a fresh pipeline inside every `__getitem__` call has a different state and cost model; document that choice if you use it.

## Reorder and execution settings are part of the experiment

`ReorderPolicy.POINTWISE` can change results because color operations do not always commute with padding, interpolation, and clipping. `from_params` and `from_config` default to `POINTWISE`; set `NONE` explicitly when comparing runs or native references.

Albumentations `execution="cv2"` and `execution="torch"` can use the same sampled matrices but differ in border and subpixel numerics. Record the chosen strategy. Also record `mask_interpolation`, `padding_mode`, `clip_policy`, `antialias`, and `compile`.

## Record the effective pipeline

Save enough information to explain a future mismatch:

```python
import json

output, last_matrix = pipe(images, return_matrix=True)
plan = [descriptor.to_dict() for descriptor in pipe.fusion_plan_descriptors]

run_metadata = {
    "fusion_plan": plan,
    "last_matrix_shape": None if last_matrix is None else list(last_matrix.shape),
    "torch_version": torch.__version__,
}
print(json.dumps(run_metadata, indent=2))
```

The returned matrix is only the last matrix-producing segment, not a whole-pipeline matrix. Save it when it is useful for audit, but do not use it to reconstruct pipelines that cross barriers or backend boundaries.

For research artifacts, also record:

- `fuse-augmentations`, Python, NumPy, and backend versions;
- device type and model, torch build, and accelerator runtime;
- transform configuration and construction route;
- input batch size, shape, dtype, and value range;
- every seed and worker count;
- randomness, reorder, execution, interpolation, padding, mask, color-clipping, antialias, and compile settings;
- `fusion_plan_descriptors` and warnings;
- whether timing includes compilation/warmup and device synchronization.

## What reproducibility does not imply

A repeatable fused run is not necessarily equal to a native backend run. The fused engine can use different coordinate, interpolation, clipping, and execution rules. Reproducibility answers “can I repeat this pipeline?”; numerical parity answers “does it match another implementation?” Treat them as separate validation gates.

See [Known limitations](../known-limitations.md) for backend parity and accelerator constraints.
