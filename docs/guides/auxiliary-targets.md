---
title: Auxiliary targets — masks, boxes, and keypoints
description: Safely transform segmentation masks, bounding boxes, and keypoints with fuse-augmentations without silent target misalignment.
---

# Auxiliary targets: masks, boxes, and keypoints

Use `data_keys` to route dense tensor targets through a supported fused geometric segment. The image must be first and use the key `"input"`.

!!! danger "Stop on unknown spatial transforms"

```
Before using any example on this page, inspect construction warnings and
`pipe.fusion_plan`.

If you see `Unknown ... transform ... treating as SPATIAL_KERNEL barrier`,
do not use that pipeline with masks, boxes, or keypoints until you have
proved the transform preserves coordinates. `RandomCrop`, `CenterCrop`, and
`Resize` can transform only the image and leave the mask at its old shape.
The runtime refusal list catches several named distortions, but it is not a
complete spatial-transform detector.

Replace an unsupported spatial transform with a registered operation, run
it through a native target-aware pipeline, or transform every target
yourself. A warning is not a safety guarantee.
```

## Supported target contract

| Key           | Required tensor shape                        | Output behavior                                               |
| ------------- | -------------------------------------------- | ------------------------------------------------------------- |
| `"input"`     | `(B, C, H, W)` floating image                | Warped by the selected image interpolation and padding policy |
| `"mask"`      | `(B, C_mask, H, W)` integer or floating mask | Nearest sampling by default; zero-filled out of bounds        |
| `"bbox_xyxy"` | `(B, N, 4)` floating `[x1, y1, x2, y2]`      | Four corners transformed, then wrapped in an axis-aligned box |
| `"bbox_xywh"` | `(B, N, 4)` floating `[x, y, width, height]` | Converted through xyxy, transformed, then converted back      |
| `"keypoints"` | `(B, N, 2)` floating `[x, y]`                | Transformed by the forward homogeneous matrix                 |

The batch dimension is mandatory. Box and keypoint counts are dense and fixed within a batch. This API does not carry class labels, visibility flags, or per-image variable-length lists.

## A contract-safe example

This example uses the augmentation-backend-free builder and only registered operations. It avoids unknown passthrough transforms.

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

batch_size, height, width = 2, 128, 128
images = torch.rand(batch_size, 3, height, width)
masks = torch.randint(0, 4, (batch_size, 1, height, width), dtype=torch.int64)
boxes = torch.tensor(
    [
        [[12.0, 18.0, 70.0, 92.0]],
        [[24.0, 10.0, 100.0, 80.0]],
    ]
)
keypoints = torch.tensor(
    [
        [[20.0, 30.0], [60.0, 75.0]],
        [[40.0, 20.0], [90.0, 70.0]],
    ]
)

pipe = Compose.from_params(
    rotation=(-15.0, 15.0),
    hflip_p=0.5,
    data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
    reorder=ReorderPolicy.NONE,
)

images_out, masks_out, boxes_out, keypoints_out = pipe(
    images,
    masks,
    boxes,
    keypoints,
)

assert images_out.shape == images.shape
assert masks_out.shape == masks.shape
assert boxes_out.shape == boxes.shape
assert keypoints_out.shape == keypoints.shape
```

`ReorderPolicy.NONE` is explicit because moving pointwise operations across a warp can change border and clipping behavior. It is the safer default for experiments that care about declared operation order.

## Choose mask interpolation deliberately

### Hard-label masks

The default `mask_interpolation="nearest"` preserves label values, accepts integer masks, and deliberately detaches the mask output from autograd.

```python
pipe = Compose.from_params(
    rotation=(-10.0, 10.0),
    data_keys=["input", "mask"],
    mask_interpolation="nearest",
    reorder=ReorderPolicy.NONE,
)
```

Nearest sampling itself can propagate gradients to floating input values in PyTorch, but this package intentionally executes the nearest mask path without autograd. Do not use it when learning through mask values or geometry is part of the model.

### Soft masks

Use bilinear sampling only for floating probability or soft-label masks:

```python
soft_masks = torch.rand(batch_size, 1, height, width, requires_grad=True)

pipe = Compose.from_params(
    rotation=(-10.0, 10.0),
    data_keys=["input", "mask"],
    mask_interpolation="bilinear",
    reorder=ReorderPolicy.NONE,
)

images_out, soft_masks_out = pipe(images, soft_masks)
assert soft_masks_out.grad_fn is not None
```

Bilinear sampling mixes neighboring values. It is not appropriate for integer class IDs, and the package rejects integer masks in bilinear mode.

## Mask padding is always zero

Mask sampling uses zero padding independently of image `padding_mode`.

If the image uses `padding_mode="border"` or `"reflection"`, the image and mask share the same geometric grid but not the same out-of-bounds fill rule. Ensure label `0` means background or an acceptable ignore/background value. If it does not, remap labels before augmentation and restore them afterward, or avoid warps that sample outside the image.

## Postprocess boxes and keypoints

The package applies coordinate math; it is not an Albumentations-style detection processor. It does not clip, filter, or update related labels.

At minimum, clip xyxy boxes and remove invalid boxes after augmentation:

```python
def clip_and_filter_xyxy(
    boxes_xyxy: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip dense xyxy boxes and return a validity mask for positive-area boxes."""
    clipped = boxes_xyxy.clone()
    clipped[..., 0::2].clamp_(0, image_width - 1)
    clipped[..., 1::2].clamp_(0, image_height - 1)
    valid = (clipped[..., 2] > clipped[..., 0]) & (clipped[..., 3] > clipped[..., 1])
    return clipped, valid


boxes_out, valid_boxes = clip_and_filter_xyxy(boxes_out, height, width)
```

Apply `valid_boxes` to the corresponding class labels, scores, instance masks, and metadata in your own data model. For minimum visibility or area thresholds, compute those policies after clipping. Keypoints likewise need your own in-bounds and visibility rules.

## Safe and unsafe pipeline matrix

| Pattern                                                         | Decision                           | Reason                                                 |
| --------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------ |
| Registered rotation/affine/flip with declared targets           | Use                                | Targets share the supported grid or matrix path        |
| Registered `RandomResizedCrop`                                  | Use with output-size awareness     | Image and targets change to the configured output size |
| Blur, noise, or color passthrough known to preserve coordinates | Use with backend-domain validation | Coordinate targets may remain unchanged legitimately   |
| Named elastic/grid/optical distortion refused at runtime        | Do not expect support              | Raising prevents known target desynchronization        |
| Unknown crop, resize, spatial transform, or custom callable     | Do not use with targets            | It can transform only the image                        |
| Albumentations HWC NumPy call with `image` and `mask` keywords  | Do not use                         | The native NumPy compatibility path is image-only      |
| BCHW tensor call with supported `data_keys`                     | Use                                | This is the intended multi-target API                  |

## Validate every production pipeline

For each pipeline used in training or evaluation:

1. Fail CI on unexpected construction warnings.
2. Assert image and mask spatial shapes after every pipeline call.
3. Use asymmetric synthetic masks and landmark points so a missed flip or crop cannot pass accidentally.
4. Check boxes for finite coordinates, positive area, and bounds.
5. Record `fusion_plan_descriptors` and compare them after dependency upgrades.
6. Compare a small fixed-seed batch with an independently trusted geometric reference before accepting a new transform or backend version.

See [Known limitations](../known-limitations.md) for parity, randomness, and device constraints.
