---
title: Fused augmentation for segmentation and dense targets
description: Route masks and continuous image targets through fused geometry, keep label values discrete, and handle the zero-padding rule deliberately.
---

# Segmentation and dense targets

A segmentation pipeline transforms two tensors that must agree pixel for pixel after augmentation. Declare both through `data_keys`, and the same fused geometry is applied to each.

!!! danger "Only an allowlisted transform sequence is safe here"

    With `data_keys`, an unknown or unclassified spatial transform is refused before any segment executes, so the image and mask cannot silently diverge. Image-only passthroughs remain a separate native contract. Read [Known limitations](../known-limitations.md) before routing any target; the direct-parameter path used on this page is safe.

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(3)

segment = Compose.from_params(
    rotation=(-8.0, 8.0),
    scale=(0.95, 1.05),
    hflip_p=0.5,
    data_keys=["input", "mask"],
    mask_interpolation="nearest",
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(2, 3, 96, 96)
masks = torch.zeros(2, 1, 96, 96)
masks[:, :, :48, :48] = 1.0
masks[:, :, 48:, 48:] = 2.0

image_out, mask_out = segment(images, masks)

print(tuple(image_out.shape), tuple(mask_out.shape))
print(sorted(set(mask_out.unique().tolist())))
```

<details>
<summary>Segmentation output shapes and surviving label values</summary>

```
(2, 3, 96, 96) (2, 1, 96, 96)
[0.0, 1.0, 2.0]
```

</details>

The first key must be `"input"`, naming the image. Remaining keys name the positional targets in order. `mask_interpolation="nearest"` is what keeps class indices discrete — bilinear sampling would invent label `1.5` on every boundary pixel.

## Mask fill is independent of image padding

Images can pad with zeros, border replication, or reflection. Masks use their own scalar `mask_fill`, which defaults to `0`, regardless of the image padding mode. Set `mask_fill` to a finite dtype-compatible background or ignore value when zero is not valid.

```python
torch.manual_seed(5)

rotate = Compose.from_params(
    rotation=(25.0, 25.0),
    padding_mode="reflection",
    data_keys=["input", "mask"],
    mask_interpolation="nearest",
    reorder=ReorderPolicy.NONE,
)

filled = torch.ones(1, 1, 64, 64)
_, rotated_mask = rotate(torch.rand(1, 3, 64, 64), filled)

print(sorted(set(rotated_mask.unique().tolist())))
```

<details>
<summary>Label values after rotating a mask that was entirely class 1</summary>

```
[0.0, 1.0]
```

</details>

The input mask was class `1` everywhere. With the default `mask_fill=0`, corners sample outside the original extent and come back as `0`, even though the image used reflection padding.

That is correct when `0` is your background class. If another value represents background or ignore, pass it as `mask_fill` and keep nearest interpolation for hard labels.

## Continuous image targets

Restoration and image-to-image models have the same structural requirement — input and target must receive identical geometry — but different interpolation needs. A continuous target should be sampled with the same smooth interpolation as the image, not with nearest, which would quantize it to the source grid.

Decide interpolation per target rather than inheriting the mask default, and confirm the choice against the [auxiliary targets](../guides/auxiliary-targets.md) contract, which owns the full routing rules.
