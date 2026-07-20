---
title: How fusion works
description: How fuse-augmentations groups transforms, composes matrices, executes segments, and reports the result.
---

# How fusion works

`fuse-augmentations` saves resampling passes by replacing a compatible run of transforms with one composed operation. It does not merge an entire pipeline indiscriminately: backend changes, transform categories, and unsupported operations define explicit segment boundaries.

## The short version

For a compatible affine run such as rotation → translation → horizontal flip, the package:

1. asks the backend adapter to sample each transform's parameters;
2. converts those parameters to forward pixel-space matrices;
3. composes the matrices in declared execution order; and
4. applies the result in one resampling step.

If the forward matrices are (M_1, M_2, \\ldots, M_n), the composed forward transform is:

$$
M_{\text{composed}} = M_n \cdots M_2 M_1
$$

The implementation inverts that forward mapping when it builds the sampling grid. Matrices exposed to users remain forward, pixel-coordinate, homogeneous (3 \\times 3) matrices.

## Segmentation comes before execution

Construction classifies every recognized transform and partitions the pipeline into segments.

| Segment       | What joins the segment                                                                                    | Execution result                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Affine        | Consecutive registered affine and compatible exact transforms from one backend                            | One affine resampling step for the run                                                                |
| Exact         | A run containing only supported discrete transforms such as flips                                         | Lossless tensor operations; no interpolation                                                          |
| Projective    | Consecutive registered perspective transforms from one backend                                            | One projective resampling step for the run                                                            |
| Color         | Consecutive supported per-channel affine color transforms                                                 | One composed color-matrix application                                                                 |
| Color-LUT     | Consecutive supported per-channel nonlinear scalar maps, such as gamma, solarize, posterize, and equalize | One composed lookup-table application                                                                 |
| Crop-resize   | A registered `RandomResizedCrop` without a preceding compatible affine run                                | One resampling step at the target size                                                                |
| Gaussian blur | Consecutive Gaussian blurs, optionally commuted across a following non-downscaling affine                 | One folded blur application; the affine run collapses to a single warp when the blur commutes past it |
| Passthrough   | A recognized backend transform without a fusion representation                                            | Native transform execution on the package's supported data contract                                   |

An affine run and a projective run are separate today, even though both use homogeneous matrices. For example, rotation → perspective uses two segments. Consecutive perspective transforms can still fuse with each other.

## What forms a boundary

A new segment starts when any of these conditions applies:

- the backend changes;
- execution changes between affine/exact and projective geometry;
- a nonlinear pointwise operation appears;
- a spatial-kernel operation appears, unless it is a Gaussian blur eligible to fold with an adjacent blur or commute across a following non-downscaling affine (see the segmentation table above);
- a coordinate-changing passthrough such as elastic distortion appears;
- a crop-resize cannot be absorbed by the preceding affine run.

Transforms from Kornia, TorchVision, and Albumentations may share one `Compose`, but a backend boundary always prevents them from sharing one fused matrix segment.

## Exact operations

Supported flips and discrete D4-style operations do not need interpolation. An exact-only segment uses lossless tensor operations. If an exact operation is part of an interpolating affine run, its matrix is composed with the other affine matrices and the whole run still uses one resampling step.

Auxiliary coordinate targets can require an exact run to use the matrix/grid route so the same sampled transform reaches images, boxes, and keypoints. The package keeps image and target sampling coupled; it does not independently resample the transform for each target.

## Projective operations

Perspective transforms use full (3 \\times 3) homographies. Consecutive projective transforms from the same backend compose into one projective segment. They do not currently merge with adjacent affine/exact segments.

## Color operations

Brightness, contrast, and supported normalization modes can be written as a homogeneous color transform:

[ c' = A c + b ]

The package represents this as a (4 \\times 4) matrix and composes a compatible run into one application. The exact supported operations differ by backend; see [Capabilities](capabilities.md#live-transform-registry-and-execution-coverage).

Nonlinear color operations do not silently disappear. A `ColorJitter` with nontrivial saturation or hue falls back as a native passthrough transform because those components are not represented by the fused color matrix.

Color output clipping is controlled by `clip_policy`:

- `"final"` applies one final clamp for a fused run;
- `"per_op_parity"` introduces boundaries when an intermediate affine color result can escape the image gamut.

A run containing normalization does not apply the final image-gamut clamp because normalized values are intentionally allowed outside `[0, 1]`.

## Crop-resize fusion

Crop behavior depends on the execution path:

- On the torch paths used by Kornia and TorchVision, an immediately preceding affine run can be composed with `RandomResizedCrop` as (M\_{crop} M\_{affine}). The result is one resampling step at the crop's output size.
- Without a preceding affine run, crop-resize is a standalone segment.
- On the Albumentations NumPy/cv2 path, crop-resize is never merged with the preceding affine run. It remains native passthrough for image-only pipelines; when the pipeline declares `data_keys` auxiliary targets it becomes a standalone crop-resize segment so masks, boxes, and keypoints are routed to the crop's output size alongside the image (both the `cv2` and `torch` execution strategies).
- Projective → crop is not a combined segment.

This asymmetry matters when comparing fusion plans or expected warp counts across backends.

## Execution is path-dependent

Adapters provide backend-specific classification, random-parameter sampling, matrix construction, and passthrough calls. The source augmentation backend does not always perform the final fused warp.

Depending on the segment, device, batch size, and options, execution may use:

- the package's batched torch sampling engine;
- an OpenCV CPU fast path;
- the Albumentations cv2 or torch strategy;
- lossless tensor operations for exact segments; or
- a native backend call for passthrough and selected single-operation paths.

`execution="cv2"|"torch"` controls fused Albumentations geometric execution. It does not switch the Kornia or TorchVision path.

## Reordering is opt-in

`Compose` preserves declared order by default. `ReorderPolicy.POINTWISE` moves reorderable pointwise operations after compatible geometric stretches, which can create a longer affine run. `ReorderPolicy.AGGRESSIVE` currently behaves the same as `POINTWISE`.

Reordering is a semantic choice, not merely a performance switch. Use `ReorderPolicy.NONE` when exact declared ordering is part of the intended augmentation.

`Compose.from_params` and `Compose.from_config` default to `POINTWISE`; direct `Compose(...)` defaults to `NONE`.

## Reading the fusion plan

The plan is available immediately after construction:

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose

image = torch.rand(2, 3, 32, 32)
pipe = Compose.from_params(rotation=(-15.0, 15.0), hflip_p=0.5)

print(pipe.fusion_plan)
for segment in pipe.fusion_plan_descriptors:
    print(segment.kind, segment.transforms, segment.n_warps_saved)
print(pipe.n_warps_saved)
```

<details>
<summary>Fusion plan, descriptor summary, and saved warp count</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
fused ('_DirectParamTransform', '_DirectFlipTransform') 1
1
```

</details>

The three introspection properties answer different questions:

| Property                  | Meaning                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------ |
| `fusion_plan`             | Human-readable segment order                                                         |
| `fusion_plan_descriptors` | Structured, JSON-serializable plan metadata                                          |
| `n_warps_saved`           | Aggregate eliminated geometric interpolation passes and collapsed color applications |

`transform_matrix` is different: it is mutable execution state populated by the most recent call. It contains only the matrix produced by the **last affine or projective segment that ran**. It is not a whole-pipeline matrix across backend boundaries, passthrough operations, or multiple geometric segments. It can be `None` for exact-only, color-only, passthrough-only, or empty pipelines.

Use `return_matrix=True` when the output and last-segment matrix should be retrieved from the same call:

```python
output, last_segment_matrix = pipe(image, return_matrix=True)
```

The same last-segment limitation applies. Because the property is per-instance mutable state, do not share one pipeline across concurrent threads when matrix inspection must be race-free.

## Next steps

- Use [Capabilities](capabilities.md) to check exact transform and configuration coverage.
- Use the [Core API](../reference/core.md) to construct and inspect a pipeline.
- Use [Configuration API](../reference/configuration.md) for backend-free or declarative construction.
