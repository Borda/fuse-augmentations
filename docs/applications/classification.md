---
title: Fused augmentation for image classification
description: Drop a fused geometric pipeline into a classification training step, inspect the plan, and benchmark the whole step rather than augmentation alone.
---

# Classification

Classification is the lowest-risk application. The label is a class index, so no target carries spatial coordinates and there is nothing to desynchronize when geometry changes. Everything the package refuses to guarantee about masks, boxes, and keypoints is irrelevant here.

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(7)

augment = Compose.from_params(
    rotation=(-12.0, 12.0),
    scale=(0.9, 1.1),
    hflip_p=0.5,
    reorder=ReorderPolicy.NONE,
)

loader = [(torch.rand(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))]
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
criterion = torch.nn.CrossEntropyLoss()

for images, labels in loader:
    predictions = model(augment(images))
    loss = criterion(predictions, labels)
    assert torch.isfinite(loss)
```

The pipeline is a callable that takes a BCHW float tensor and returns one. It goes wherever your current `transforms` object goes, provided that object was already operating on batched tensors rather than PIL images.

## Confirm the geometry actually fused

Reading the plan takes one line and tells you whether the pipeline you built is the pipeline you meant to build.

```python
print(augment.fusion_plan)
print(augment.n_warps_saved)
```

<details>
<summary>Classification pipeline plan and saved warp count</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
1
```

</details>

The plan has one fused segment; `n_warps_saved` is a legacy heuristic and is not a literal count of native interpolation calls. If a color operation had been declared between the rotation and the flip, the plan would show three segments and report no collapsed operations — [Performance planning](performance-planning.md) walks through that case.

## Benchmark the training step, not the transform

Augmentation latency in isolation is the wrong measurement. Host-to-device transfers, the model forward and backward passes, and the optimizer step usually dominate a classification loop, and a 6x faster transform inside a step that spends 5% of its time augmenting is a 4% win.

Measure the full step, with and without fusion, on your device and batch shape. The [benchmarks](../research/benchmarks.md) report both the wins and a case where the fused path is slower — TorchVision at batch 32 — so a local measurement is not a formality.

## Randomness

`Compose.from_params` samples parameters independently per image on the direct path. If you are comparing a fused run against a native run, pair the randomness or store the sampled matrices, otherwise the two runs differ by sampling noise before they differ by anything you meant to test. [Reproducibility](../guides/reproducibility.md) covers the seeding surface.
