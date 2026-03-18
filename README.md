# fuse-augmentations

Fuse consecutive geometric augmentation ops (rotation, scale, shear, flip) into a **single `grid_sample` pass**, eliminating redundant interpolation and improving image quality.

Drop-in replacement for Kornia's `AugmentationSequential` / `Sequential` — same call interface, fewer warps.

## Installation

```bash
pip install fuse-augmentations
```

Requires: Python 3.10+, PyTorch 2.0+. Kornia is optional (needed only when passing Kornia transform objects).

## Quick start

```python
import torch
import kornia.augmentation as K
from fuse_aug import Compose

pipe = Compose([
    K.RandomRotation(degrees=30, p=0.8),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=0, scale=(0.8, 1.2), p=0.7),
])

image = torch.rand(4, 3, 256, 256)   # (B, C, H, W)
out = pipe(image)                     # one grid_sample instead of three
```

## Auxiliary targets — masks, boxes, keypoints

Pass `data_keys` to route auxiliary tensors through the same fused transform:

```python
from fuse_aug import Compose
import kornia.augmentation as K

pipe = Compose([
    K.RandomRotation(degrees=30, p=0.8),
    K.RandomHorizontalFlip(p=0.5),
], data_keys=["input", "mask"])

img_out, mask_out = pipe(image, mask)   # mask warped with mode='nearest'
```

Supported keys:

| Key | Tensor shape | Notes |
|---|---|---|
| `"input"` | `(B, C, H, W)` | Image; always the first argument |
| `"mask"` | `(B, C, H, W)` | Nearest-neighbour sampling; integer class labels preserved |
| `"bbox_xyxy"` | `(B, N, 4)` | Pixel-space `[x1, y1, x2, y2]`; AABB wrapping after rotation |
| `"bbox_xywh"` | `(B, N, 4)` | Pixel-space `[x, y, w, h]`; converted internally to xyxy and back |
| `"keypoints"` | `(B, N, 2)` | Pixel-space `[x, y]`; exact homogeneous transform, no AABB |

Bounding boxes and keypoints use the composed forward matrix; masks share the same sampling grid as the image.

## Backend-free pipelines with `from_params`

Construct a fused pipeline from numeric parameter ranges — no Kornia import required:

```python
from fuse_aug import Compose

pipe = Compose.from_params(rotation=(-30, 30), scale=(0.8, 1.2), hflip_p=0.5)
out = pipe(image)
```

`from_params` accepts: `rotation`, `scale`, `scale_x`, `scale_y`, `shear_x`, `shear_y`, `translate_x`, `translate_y`, `hflip_p`, `vflip_p`, `interpolation`, `padding_mode`, `reorder`, and `data_keys`.

## How fusion works

Given a pipeline `[Rotate, Scale, HFlip, GaussianBlur, Rotate]`:

1. `[Rotate, Scale, HFlip]` are grouped into a `FusedAffineSegment` — their matrices are composed and one `grid_sample` call is made.
2. `GaussianBlur` is a spatial-kernel barrier — it passes through unchanged.
3. The trailing `Rotate` forms its own `FusedAffineSegment`.

```python
print(pipe.fusion_plan)
# fused(RandomRotation, RandomResizedCrop, RandomHorizontalFlip) -> passthrough(RandomGaussianBlur) -> fused(RandomRotation)

print(pipe.n_warps_saved)
# 2  (saved 2 interpolation passes)
```

## Reorder policy

```python
from fuse_aug import Compose, ReorderPolicy
import kornia.augmentation as K

pipe = Compose([
    K.RandomRotation(degrees=15, p=0.8),
    K.ColorJitter(brightness=0.3, p=0.5),   # POINTWISE
    K.RandomHorizontalFlip(p=0.5),
], reorder=ReorderPolicy.POINTWISE)

# ColorJitter is moved after HFlip, letting both geometric ops fuse:
# fused(RandomRotation, RandomHorizontalFlip) -> passthrough(ColorJitter)
```

## Transform matrix access

```python
out = pipe(image)
M = pipe.transform_matrix   # (B, 3, 3) composed forward matrix
```

Use `M` to transform stored coordinates (keypoints, boxes) that were not passed as `data_keys`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
