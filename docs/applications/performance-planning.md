---
title: Planning fused augmentation performance
description: Read the fusion plan to predict whether fusion pays for your pipeline before you benchmark it, and recognize the orderings that save nothing.
---

# Performance planning

Every pipeline reports what it will do before it runs. Reading that plan costs one line and answers the only question that matters up front: how many resampling passes did fusion actually remove?

<!--phmdoctest-share-names-->

```python
import torch
import torchvision.transforms.v2 as T

from fuse_augmentations import Compose, ReorderPolicy

chain = Compose(
    [
        T.RandomRotation(degrees=15.0),
        T.RandomAffine(degrees=0.0, scale=(0.9, 1.1)),
        T.RandomAffine(degrees=0.0, translate=(0.05, 0.05)),
        T.RandomHorizontalFlip(p=0.5),
    ],
    reorder=ReorderPolicy.NONE,
)

print(chain.fusion_plan)
print(chain.n_warps_saved)
```

<details>
<summary>Four consecutive geometric transforms collapse into one segment</summary>

```
fused(RandomRotation, RandomAffine, RandomAffine, RandomHorizontalFlip)
3
```

</details>

Four transforms, one segment, three resampling passes removed. This is the shape of pipeline the package was built for, and the shape the [benchmarks](../research/benchmarks.md) measure most favorably.

## A colour operation in the middle costs everything

```python
split = Compose(
    [
        T.RandomRotation(degrees=15.0),
        T.ColorJitter(brightness=0.2),
        T.RandomAffine(degrees=0.0, scale=(0.9, 1.1)),
    ],
    reorder=ReorderPolicy.NONE,
)

print(split.fusion_plan)
print(split.n_warps_saved)
```

<details>
<summary>Interleaved colour splits the geometry into two separate warps</summary>

```
fused(RandomRotation) → color(ColorJitter) → fused(RandomAffine)
0
```

</details>

Nothing fused. The two geometric transforms are separated by a pointwise operation, so each still resamples, and the package has added a plan and a wrapper for no benefit.

Grouping the geometry restores the win:

```python
grouped = Compose(
    [
        T.RandomRotation(degrees=15.0),
        T.RandomAffine(degrees=0.0, scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2),
    ],
    reorder=ReorderPolicy.NONE,
)

print(grouped.fusion_plan)
print(grouped.n_warps_saved)
```

<details>
<summary>Colour moved to the end lets the geometry fuse</summary>

```
fused(RandomRotation, RandomAffine) → color(ColorJitter)
1
```

</details>

Brightness applied before or after a geometric warp is not the same operation in general, so this reordering is a decision about your augmentation policy, not a free optimization. Make it deliberately, with `ReorderPolicy.NONE` set, rather than letting a policy reorder your pipeline behind you.

## What the plan does and does not tell you

`n_warps_saved` counts resampling passes the planner removed. It is a structural quantity, not a measured speedup — a saved warp on a small tensor may cost less than the fused path's overhead.

Use it as a gate, not as a result:

- `n_warps_saved == 0` — there is no fusion win available; stop here and keep the native pipeline.
- `n_warps_saved >= 2` — worth benchmarking on your device, dtype, and batch shape.
- Any value — the wall-clock outcome still has to be measured. The benchmarks report a 1.7861x fixed-bank score over 168 CPU variants and also publish a case where the fused path runs slower, at TorchVision batch 32.

All published latency evidence is CPU-only. Nothing in this repository measures GPU or MPS, so treat a device change as an unmeasured configuration.

For the meaning of each segment field, see [Introspection](../guides/introspection.md). For the numbers themselves and how they were produced, see [Benchmarks](../research/benchmarks.md) and [Methodology](../research/methodology.md).
