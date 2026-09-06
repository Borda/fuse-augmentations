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

`return_matrix=True` is unambiguous here because this pipeline has one matrix-producing segment. In a pipeline with backend changes, projective boundaries, or passthrough operations, the returned matrix is the actual pixel-centre matrix from only the last supported matrix-producing segment.

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

## The full parameter palette

`Compose.from_params` accepts every direct-path knob in one call: independent axis scaling, shear, both translations, both flips, photometric brightness and contrast, and the resampling controls.

```python
torch.manual_seed(11)

full = Compose.from_params(
    rotation=(-15.0, 15.0),
    scale_x=(0.9, 1.1),
    scale_y=(0.9, 1.1),
    shear_x=(-5.0, 5.0),
    shear_y=(-5.0, 5.0),
    translate_x=(-8.0, 8.0),
    translate_y=(-8.0, 8.0),
    hflip_p=0.5,
    vflip_p=0.2,
    brightness=0.2,
    contrast=0.2,
    interpolation="bicubic",
    padding_mode="reflection",
    clip_policy="final",
    reorder=ReorderPolicy.NONE,
)

full_out = full(images)

print(full.fusion_plan)
print(full.n_warps_saved)
print(tuple(full_out.shape), full_out.dtype)
```

<details>
<summary>Full-palette fusion plan, saved warps, and output shape</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform, _DirectFlipTransform) → color(_DirectParamTransform, _DirectParamTransform)
3
(4, 3, 128, 128) torch.float32
```

</details>

The geometry collapses into one fused segment; brightness and contrast form a separate pointwise colour segment because they are not matrix operations. `scale` and `scale_x`/`scale_y` are alternatives — pass the isotropic form or the two axis forms, not both.

## Route a mask with the image

Pass `data_keys` to carry auxiliary targets through the same fused geometry. The first key must be `"input"`, naming the image tensor.

```python
torch.manual_seed(13)

targeted = Compose.from_params(
    rotation=(-10.0, 10.0),
    translate_x=(-4.0, 4.0),
    hflip_p=0.5,
    data_keys=["input", "mask"],
    mask_interpolation="nearest",
    reorder=ReorderPolicy.NONE,
)

masks = (torch.rand(4, 1, 128, 128) > 0.5).float()
image_out, mask_out = targeted(images, masks)

print(tuple(image_out.shape), tuple(mask_out.shape))
print(sorted(set(mask_out.unique().tolist())))
```

<details>
<summary>Routed image and mask shapes with preserved mask labels</summary>

```
(4, 3, 128, 128) (4, 1, 128, 128)
[0.0, 1.0]
```

</details>

`mask_interpolation="nearest"` keeps label values discrete. With `data_keys`, an unknown or unclassified spatial transform is refused before execution; image-only passthroughs have a separate native contract. Read [Known limitations](../known-limitations.md) before routing targets through a backend pipeline.

## Hand back NumPy

`output_backend` converts the result on the way out, so a torch-internal pipeline can serve a NumPy consumer.

```python
torch.manual_seed(17)

as_numpy = Compose.from_params(
    rotation=(-5.0, 5.0),
    output_backend="numpy",
    reorder=ReorderPolicy.NONE,
)

array = as_numpy(images)

print(type(array).__name__, array.shape, array.dtype)
```

<details>
<summary>NumPy output type, shape, and dtype</summary>

```
ndarray (4, 128, 128, 3) float32
```

</details>

The `"numpy"` backend returns channel-last `NHWC`; use `"numpy_hwc"` or `"torch"` to select a different output contract.

## Next steps

- Bring an existing library pipeline with [Backend pipelines](../guides/backend-pipelines.md).
- Build portable specs with [Declarative configuration](../guides/configuration.md).
- Read the [Known limitations](../known-limitations.md) before routing masks, boxes, or keypoints.
