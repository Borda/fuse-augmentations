---
title: Types and targets
description: Enums, fusion descriptors, masks, boxes, keypoints, and target transformation helpers.
---

# Types and targets

The high-level `data_keys` interface keeps supported auxiliary targets on the same sampled geometric path as the image. Low-level target functions are also public for applying a known matrix or sampling grid directly.

## Data keys

`data_keys` describes positional arguments. The first entry must be `"input"`.

| Key         | Expected shape | Output behavior                                               |
| ----------- | -------------- | ------------------------------------------------------------- |
| `input`     | `(B, C, H, W)` | Image transformed by every segment                            |
| `mask`      | `(B, C, H, W)` | Nearest-neighbor by default; bilinear for floating soft masks |
| `bbox_xyxy` | `(B, N, 4)`    | Transforms corners and returns an enclosing axis-aligned box  |
| `bbox_xywh` | `(B, N, 4)`    | Converts to/from xyxy internally                              |
| `keypoints` | `(B, N, 2)`    | Applies the forward homogeneous pixel matrix                  |

Unknown keys are passed through with a warning. Duplicate auxiliary keys are rejected.

## Mask interpolation

Nearest-neighbor sampling preserves discrete class values, but the package deliberately detaches the entire nearest-mask path. Its output has no autograd path with respect to either mask values or the sampling grid. This is package policy; direct PyTorch nearest `grid_sample` can propagate gradients to floating input values even though nearest coordinates are nondifferentiable almost everywhere. `mask_interpolation="bilinear"` supports differentiable soft masks and requires floating-point mask input; class values mix at boundaries.

## Coordinate-changing passthrough

A passthrough transform normally changes only the image. That is safe only for an operation known to preserve coordinates, such as a verified blur or pointwise color transform. When auxiliary targets are present, the pipeline raises for passthroughs whose exact class name appears on its finite coordinate-changing refusal list.

Unknown spatial transforms are not structurally detected. Unsupported crops, resizes, and custom callables can still modify the image while targets remain unchanged. Treat every `Unknown ... SPATIAL_KERNEL barrier` warning as unsafe with `data_keys`; use a registered transform or route all targets through a native target-aware pipeline.

## Output conversion

With `output_backend="numpy"`, image and mask targets become channel-last NumPy arrays. Boxes and keypoints stay as tensors because image channel layout does not apply to coordinate arrays.

## Segment descriptors

`fusion_plan_descriptors` is available before the first forward call. Each immutable descriptor exposes the segment kind, transform names, saved-application count, backend name where available, and structured boundary/refusal reasons.

::: fuse_augmentations.SegmentDescriptor
    options:
        show_root_heading: true
        show_source: false
        members:
            - to_dict

## Transform categories

::: fuse_augmentations.TransformCategory
    options:
        show_root_heading: true
        show_source: false

## Sampling enums

::: fuse_augmentations.InterpolationMode
    options:
        show_root_heading: true
        show_source: false

::: fuse_augmentations.PaddingMode
    options:
        show_root_heading: true
        show_source: false

## Low-level target functions

These functions expect matrices and grids that already follow the package's pixel-coordinate and sampling conventions. Most users should prefer `data_keys`.

::: fuse_augmentations.transform_keypoints
    options:
        show_root_heading: true
        show_source: false

::: fuse_augmentations.transform_bbox_xyxy
    options:
        show_root_heading: true
        show_source: false

::: fuse_augmentations.transform_bbox_xywh
    options:
        show_root_heading: true
        show_source: false

::: fuse_augmentations.transform_mask
    options:
        show_root_heading: true
        show_source: false
