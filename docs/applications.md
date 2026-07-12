---
title: Image augmentation applications and use cases
description: Apply fused augmentation to classification, restoration, segmentation, detection, and research while respecting target and parity contracts.
---

# Applications and use cases

The package is best when the cost or fidelity problem is repeated geometric resampling. It is a poor fit when the pipeline is mostly nonlinear color, blur, erasing, elastic deformation, or unregistered spatial preprocessing.

## Image classification

Classification is the lowest-risk application because the label does not carry spatial coordinates. A native builder pipeline is a clear starting point:

```python
augment = Compose.from_params(
    rotation=(-12.0, 12.0),
    scale=(0.9, 1.1),
    hflip_p=0.5,
    reorder=ReorderPolicy.NONE,
)

for images, labels in loader:
    predictions = model(augment(images))
    loss = criterion(predictions, labels)
```

Benchmark the whole training step, not augmentation latency in isolation. Host/device transfers and model compute can dominate.

## Restoration and image-to-image models

When input and target images must receive identical geometry, both can be represented as image-like tensors in a multi-target pipeline only if every spatial operation is registered and target-safe. Decide the interpolation and boundary policy for each target explicitly; a hard label mask has different needs from a continuous restoration target.

## Semantic segmentation

Use `data_keys=["input", "mask"]` only with an allowlisted transform sequence. Nearest mask interpolation preserves discrete labels but is intentionally detached, and masks always use zero padding even if images use border or reflection padding.

If zero is not the background class, pad/remap deliberately outside the package or avoid transforms that sample outside the image.

## Object detection

Bounding boxes and keypoints are raw dense tensors, not a detection framework's processor. The package applies matrix math but does not clip boxes, remove invalid or low-visibility instances, propagate labels, support ragged `N`, or update keypoint visibility.

A production detection pipeline must postprocess coordinates after augmentation:

1. clip coordinates to the output image extent;
2. remove boxes with invalid or too-small area;
3. apply the same keep mask to labels and instance metadata;
4. update keypoint visibility or validity flags;
5. test empty and fully out-of-frame cases.

## Research and ablations

Fused and native pipelines are different interventions. Reducing resampling can change both image quality and backend numerics, while reordering can change operation semantics.

For an interpretable study:

- set `ReorderPolicy.NONE` unless reordering is the independent variable;
- record plan descriptors and dependency versions;
- pair random parameters or store sampled matrices;
- compare task metrics as well as pixel errors;
- publish regressions and skipped cases;
- separate warmup/compilation from steady-state timing.

Use the [research methodology](research/methodology.md) as the experiment checklist.

## When not to use this package

Choose the native backend container when you require PIL/CHW input, complete Albumentations dictionary processors, exact native pixels, per-transform fill/interpolation semantics, segment hooks, unregistered spatial transforms with targets, or a backend-specific random-number stream. A smaller native pipeline is better than an unsafe fused one.
