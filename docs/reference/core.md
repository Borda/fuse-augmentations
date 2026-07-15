---
title: Core API
description: Construct, run, inspect, and convert fuse-augmentations pipelines.
---

# Core API

`Compose` is the main public entry point. It accepts a list of recognized backend transform objects and builds a segmented `torch.nn.Module` around the supported `(B, C, H, W)` tensor contract.

`Compose` and `FusedCompose` refer to the same class. `AugmentationSequential` is another naming alias; these aliases do not imply complete behavioral compatibility with every native container option.

## Choose a construction style

<!--phmdoctest-share-names-->

```python
import torch
import torchvision.transforms.v2 as transforms

from fuse_augmentations import Compose

pipe = Compose(
    [
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    ]
)

image = torch.rand(8, 3, 224, 224)
output = pipe(image)
print(pipe.fusion_plan)
```

<details>
<summary>Fusion plan for the native TorchVision pipeline</summary>

```
fused(RandomRotation, RandomHorizontalFlip, RandomAffine)
```

</details>

Use `Compose.from_params` for an augmentation-backend-free pipeline and `Compose.from_config` for declarative backend-specific construction. Both are documented in [Configuration API](configuration.md).

## Input and output forms

The standard call accepts a BCHW tensor. Set `data_keys` when positional auxiliary targets are present:

!!! danger "Use only explicitly supported spatial transforms with auxiliary targets"

```text
The runtime refusal policy is a finite class-name list. An unknown crop,
resize, or custom spatial transform can modify the image while leaving
masks, boxes, or keypoints stale. Treat every
`Unknown ... SPATIAL_KERNEL barrier` warning as unsafe with `data_keys`.
Read [Auxiliary targets](../guides/auxiliary-targets.md) before using this
path for training data.
```

```python
import torch
import torchvision.transforms.v2 as transforms

from fuse_augmentations import Compose

safe_transforms = [
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
]

pipe = Compose(
    safe_transforms,
    data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
)

image = torch.rand(2, 3, 64, 64)
mask = torch.zeros(2, 1, 64, 64, dtype=torch.long)
boxes = torch.tensor([[[8.0, 8.0, 32.0, 32.0]]] * 2)
keypoints = torch.tensor([[[16.0, 16.0]]] * 2)

image_out, mask_out, boxes_out, keypoints_out = pipe(
    image,
    mask,
    boxes,
    keypoints,
)
```

`output_backend="numpy"` converts images and masks to channel-last NumPy arrays. Coordinate outputs remain tensors.

An image-only Albumentations pipeline additionally accepts `pipe(image=hwc_array)` and returns an Albumentations-style dictionary. Auxiliary keyword keys are not supported on that native NumPy path; use tensor inputs and `data_keys` instead.

## Inspect a pipeline

```python
for descriptor in pipe.fusion_plan_descriptors:
    print(descriptor.kind, descriptor.transforms, descriptor.n_warps_saved)

output, last_matrix = pipe(image, return_matrix=True)
```

<details>
<summary>Segment descriptor and saved warp count for the pipeline</summary>

```
fused ('RandomRotation', 'RandomHorizontalFlip', 'RandomAffine') 2
```

</details>

`fusion_plan`, `fusion_plan_descriptors`, and `n_warps_saved` describe the constructed plan. `transform_matrix` and `return_matrix=True` report only the most recent call's **last matrix-producing segment**, not a whole-pipeline composition.

## Constructor options that change semantics

| Option                          | Effect                                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| `reorder`                       | Preserves order by default; `POINTWISE` may move color operations to extend geometric runs     |
| `randomness`                    | Preserves backend sampling by default; `per_sample` requests independent draws where supported |
| `execution`                     | Selects `cv2` or `torch` execution for fused Albumentations geometry only                      |
| `interpolation`, `padding_mode` | Segment-level sampling choices for fused geometry                                              |
| `clip_policy`                   | Controls boundaries/clamping in fused color runs                                               |
| `mask_interpolation`            | `nearest` for hard labels or `bilinear` for floating soft masks                                |
| `compile`                       | Opts eligible non-CPU torch warp cores into `torch.compile`                                    |
| `antialias`                     | Enables supported downscale prefiltering                                                       |
| `substitute_passthrough`        | Opts into behavior-changing registered passthrough substitutions                               |

Extra `backend_kwargs` are currently reserved and unused; do not rely on them as an extension mechanism.

## `Compose` / `FusedCompose`

::: fuse_augmentations.FusedCompose
    options:
        show_root_heading: true
        show_source: false
        members:
            - __init__
            - forward
            - from_params
            - from_config
            - supported_ops
            - capability_matrix
            - fusion_plan
            - fusion_plan_descriptors
            - n_warps_saved
            - transform_matrix

## NumPy converters

The converters are useful when conversion should be explicit rather than attached to a pipeline.

::: fuse_augmentations.NumpyToTorchConverter
    options:
        show_root_heading: true
        show_source: false
        members:
            - convert

::: fuse_augmentations.TorchToNumpyConverter
    options:
        show_root_heading: true
        show_source: false
        members:
            - convert
