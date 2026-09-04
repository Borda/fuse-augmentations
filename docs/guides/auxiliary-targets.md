---
title: Auxiliary targets — masks, boxes, and keypoints
description: Safely transform segmentation masks, bounding boxes, and keypoints with fuse-augmentations without silent target misalignment.
---

# Auxiliary targets: masks, boxes, and keypoints

Use `data_keys` to route dense tensor targets through a supported fused geometric segment. The image must be first and use the key `"input"`.

!!! danger "Stop on unknown spatial transforms"

    Before using any example on this page, inspect construction warnings and `pipe.fusion_plan`.

    If you see `Unknown ... transform ... treating as SPATIAL_KERNEL barrier`, do not use that pipeline with masks, boxes, or keypoints until you have proved the transform preserves coordinates. `RandomCrop`, `CenterCrop`, and `Resize` can transform only the image and leave the mask at its old shape. The runtime refusal list catches several named distortions, but it is not a complete spatial-transform detector.

    Replace an unsupported spatial transform with a registered operation, run it through a native target-aware pipeline, or transform every target yourself. A warning is not a safety guarantee.

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

<!--phmdoctest-share-names-->

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


clipped_boxes, valid_boxes = clip_and_filter_xyxy(boxes_out, height, width)
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
| Albumentations HWC NumPy call with no `data_keys`               | Do not use with targets            | Without `data_keys` the NumPy path is image-only       |
| HWC NumPy call with supported `data_keys`                       | Use                                | Targets route through the same matrix as tensors       |
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

## Keypoint pairs under a mirror (`keypoint_flip_index`)

A mirrored image has its left and right anatomy swapped. The warp already puts the coordinates in the right places; what has to follow is the *identity* of each keypoint slot — "left elbow" now sits where the right elbow is. `keypoint_flip_index` is that permutation: slot `i` takes its value from slot `flip_index[i]`.

```python
import torch

from fuse_augmentations import Compose

points = torch.tensor([[[4.0, 4.0], [8.0, 6.0], [12.0, 10.0]]])
augment = Compose.from_params(
    hflip_p=1.0,
    data_keys=["input", "keypoints"],
    keypoint_flip_index=(0, 2, 1),
)

_, swapped = augment(torch.zeros(1, 3, 16, 16), points)
print(swapped[0].tolist())
```

```
[[11.0, 4.0], [3.0, 10.0], [7.0, 6.0]]
```

- **When it fires is decided by the composed matrix's determinant**, never by whether a flip transform appears in the pipeline. After fusion the mirror is part of a larger matrix and is no longer a discrete op, and two mirrors compose back to a rotation — which must *not* swap. A rotation past 90 degrees looks like a flip and is not one; the determinant is what tells them apart.
- **Every path applies it**, including the ones that never build a warp grid: the exact-flip route and the D4 fast path (a pure flip is applied by tensor reversal, skipping `grid_sample` entirely). A permutation wired only into the interpolating route would leave those images mirrored with unswapped labels, and nothing about the shapes would say so.
- **The pair table is dataset schema and stays with you.** This package validates only that it is a permutation of `range(len(flip_index))` — a table that repeated or dropped a slot would duplicate one landmark and silently delete another.
- `None` (the default) leaves the keypoint axis in input order, unchanged.
- The same decision is available directly: `orientation_reversed(matrix)` and `permute_keypoint_pairs(points, flip_index, reversed_mask)`.

## Rotated boxes (`rboxes`)

`"rboxes"` is a data key like any other: `(B, N, 5)` as `(cx, cy, w, h, theta)`, `theta` in radians, routed by every path that routes plain boxes — fused affine, exact flip, crop-resize, letterbox, and the Albumentations tensor path.

```python
import torch

from fuse_augmentations import Compose, rbox_envelopes

image = torch.rand(1, 3, 16, 32)
rboxes = torch.tensor([[[16.0, 8.0, 8.0, 4.0, 0.0]]])
augment = Compose.from_params(rotation=(90.0, 90.0), data_keys=["input", "rboxes"])

_, warped = augment(image, rboxes)
print([round(value, 3) for value in warped[0, 0].tolist()])
print([round(value, 3) for value in rbox_envelopes(warped)[0, 0].tolist()])
```

```
[15.0, 8.0, 8.0, 4.0, 1.571]
[13.0, 4.0, 17.0, 12.0]
```

What the transport does and does not promise:

- **A general affine does not map a rectangle to a rectangle** — only a similarity does. Shear sends a rectangle to a parallelogram, which no `(cx, cy, w, h, theta)` describes. So the box is expanded to four corners, those are mapped, and a box is re-fitted to them. Under rotation, uniform scale, translation and mirroring the fit is exact; under a shear of angle `s` the fitted corners sit `(h / 2) * sqrt(tan(s)**2 + (1 - 1/cos(s))**2)` from the warped ones, where `h` is the extent across the shear. The first-order form of that is the familiar `h * sin(s / 2)`.
- **No canonical form is imposed.** The package returns whatever the fit produced: no `w >= h` swap, no angle range. The long-edge convention is one reading of the literature among several and belongs with the assigner, the loss and the evaluation kernel that share it. `transform_rboxes`, `mirror_rboxes` and `shift_rboxes` take an optional `canonicalize=` callable if you want yours applied in one place.
- **Mirroring uses this package's flip axis**, `(width - 1) / 2`, matching the image flip under `align_corners=True` — not the `width / 2` an extent-convention implementation uses.
- **Clipping is not provided**, because a rotated box clipped to the canvas is generally a polygon rather than a rotated box. Use `rbox_envelopes` with `clip_bbox_xyxy` and `instance_keep_mask` to decide survival on the axis-aligned envelope.
- Helpers: `rboxes_to_corners`, `corners_to_rboxes`, `transform_rboxes`, `mirror_rboxes`, `shift_rboxes`, `rbox_envelopes`.

The Albumentations *native NumPy* dict path without `data_keys` (`pipe(image=<ndarray>)`) transforms the image only and refuses auxiliary targets outright. Declare `data_keys` and rotated boxes route through the same composed matrix the tensor path uses, in NumPy or in tensors alike.

## Deciding which instances survive a warp

A warp pushes some instances off the canvas and clips others to slivers. This package supplies the geometry for that decision and leaves the decision itself to the caller:

```python
import torch

from fuse_augmentations import Compose, clip_bbox_xyxy, instance_keep_mask

image = torch.rand(1, 3, 32, 32)
boxes = torch.tensor([[[2.0, 2.0, 10.0, 10.0], [26.0, 26.0, 31.0, 31.0]]])
augment = Compose.from_params(
    translate_x=(12.0, 12.0), data_keys=["input", "bbox_xyxy"]
)

warped_image, warped_boxes = augment(image, boxes)
clipped = clip_bbox_xyxy(warped_boxes, height=32, width=32)
keep = instance_keep_mask(warped_boxes, clipped, min_size=2.0, min_visibility=0.25)
print(keep.tolist())
```

```
[[True, False]]
```

- `clip_bbox_xyxy` clamps boxes to the `[0, width] x [0, height]` canvas extent — geometry, not policy.
- `instance_keep_mask` keeps an instance whose clipped box is at least `min_size` on both axes **and** retains at least `min_visibility` of its unclipped area. Both thresholds are yours; the defaults are `0.0`, which drops nothing.
- **It returns the mask, not filtered boxes.** Labels, keypoints, rotated boxes, polygon rings and any per-instance flag live on the same instance axis, and only you hold all of them. Filter every one of them with this single mask: a pipeline that filtered boxes but not the keypoints on the same instances is corrupt while every shape still lines up and nothing raises.

Do not confuse this with `clip_policy`, which decides when a fused *colour* chain clamps to `[0, 1]`. The names are close; the concepts share nothing.
