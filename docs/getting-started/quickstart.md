---
title: Quickstart with the native augmentation builder
description: Run a deterministic, backend-free BCHW tensor example that fuses rotation, scale, translation, and flips.
---

# Quickstart

Start with the native builder. It needs no Kornia, TorchVision, or Albumentations installation and makes the package's actual input contract explicit: a floating BCHW tensor.

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(7)

augment = Compose.from_params(
    rotation=(-15.0, 15.0),
    scale=(0.9, 1.1),
    translate_x=(-8.0, 8.0),
    hflip_p=0.5,
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(4, 3, 128, 128, dtype=torch.float32)
augmented, matrix = augment(images, return_matrix=True)

assert augmented.shape == images.shape
assert augmented.dtype == images.dtype
assert matrix is not None and matrix.shape == (4, 3, 3)

print(augment.fusion_plan)
print(augment.n_warps_saved)
```

<details>
<summary>Quickstart fusion plan and saved warp count</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
1
```

</details>

`return_matrix=True` is unambiguous here because this pipeline has one matrix-producing segment. In a pipeline with backend changes, projective boundaries, or passthrough operations, the returned matrix represents only the last matrix-producing segment.

## What this example guarantees

- Rotation, scale, translation, and the supported flip are sampled independently per image on the direct-parameter path.
- Compatible geometry is represented in homogeneous pixel-space matrices.
- The declared order is preserved because `ReorderPolicy.NONE` is explicit.
- Output remains a BCHW tensor on the same torch device.

It does not guarantee pixel identity with any native backend. Native libraries may use different centers, fill rules, interpolation kernels, clipping, or random-number streams.

## Inspect before you benchmark

Use the plan to confirm that your intended operations formed a useful segment:

```python
for segment in augment.fusion_plan_descriptors:
    print(segment.kind, segment.transforms, segment.backend, segment.split_reason)
```

<details>
<summary>Quickstart segment kind, transforms, backend, and split reason</summary>

```
fused ('_DirectParamTransform', '_DirectFlipTransform') None None
```

</details>

Fewer planned resampling passes are structural. Faster execution is not: benchmark the exact device, image shape, batch size, dtype, and transform mix. See [Benchmarks](../research/benchmarks.md).

## Next steps

- Bring an existing library pipeline with [Backend pipelines](../guides/backend-pipelines.md).
- Build portable specs with [Declarative configuration](../guides/configuration.md).
- Read the [Known limitations](../known-limitations.md) before routing masks, boxes, or keypoints.
