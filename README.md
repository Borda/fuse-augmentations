# fuse-augmentations

**Fuse consecutive geometric augmentation transforms into a single interpolation pass -- fewer warps, better image quality.**

> **Summary**: `fuse-augmentations` is a framework-agnostic library that automatically **groups** consecutive fusible geometric transforms in your augmentation pipeline, then **fuses** their matrices into a single composed transform applied via one interpolation pass. Linear color transforms (brightness, contrast, and standard Normalize) are additionally fused into a single matrix multiply. Non-fusible operations such as blur and nonlinear color adjustments pass through unchanged. Drop-in replacement for Kornia's `AugmentationSequential`, TorchVision, and Albumentations compose classes.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/) [![PyPI version](https://img.shields.io/pypi/v/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![CI complete testing](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml/badge.svg?event=push)](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml) [![codecov](https://codecov.io/github/Borda/fuse-augmentations/graph/badge.svg?token=hw3VOHuzKk)](https://codecov.io/github/Borda/fuse-augmentations)

<details>
<summary><b>Contents</b></summary>

- [💡 Motivation](#-motivation)
- [🔍 Overview](#-overview)
- [✨ Features](#-features)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙️ How Fusion Works](#%EF%B8%8F-how-fusion-works)
- [📖 API Reference](#-api-reference)
- [🎯 Auxiliary Targets](#-auxiliary-targets)
- [🔌 Backend-Free Pipelines](#-backend-free-pipelines)
- [🔄 NumPy I/O](#-numpy-io)
- [🎨 Color Fusion (POINTWISE_LINEAR)](#-color-fusion-pointwise_linear)
- [✂️ Crop+Resize (CROP_RESIZE_FIXED)](#%EF%B8%8F-cropresize-crop_resize_fixed)
- [🔧 Backend-Agnostic Meta-Config](#-backend-agnostic-meta-config)
- [🔗 Multi-Backend Pipelines](#-multi-backend-pipelines)
- [🔀 Reorder Policy](#-reorder-policy)
- [🔬 Fusion Introspection](#-fusion-introspection)
- [🏋️ Training Loop](#%EF%B8%8F-training-loop)
- [⚠️ Limitations](#%EF%B8%8F-limitations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

</details>

______________________________________________________________________

## 💡 Motivation

People pick specific transforms -- `RandomRotation`, `RandomHorizontalFlip`, `RandomScale` -- because they want intuitive, independent control over each one. That is why nobody just uses a single monolithic `RandomAffine` for everything: it does not let you set different probabilities per parameter (e.g. flip with `p=0.5`, rotation with `p=0.8`, scale with `p=0.7`, each drawn independently).

The problem is that chaining these individual transforms applies a separate interpolation for each one, compounding quality loss across your pipeline.

`fuse-augmentations` gives you the best of both worlds. You keep writing your pipeline with individual, independently-controlled transforms, and `Compose` is a drop-in replacement for your existing backend's compose class (`AugmentationSequential`, `transforms.Compose`, etc.) -- no pipeline rewrite needed. Under the hood, the library **groups** consecutive fusible geometric transforms and **fuses** their matrices, applying a single interpolation pass. The fusion is an implementation detail that gives you quality improvement for free.

## 🔍 Overview

Given a pipeline of transforms, `fuse-augmentations` performs two steps:

1. **Grouping**: consecutive fusible transforms are identified and collected into segments -- geometric transforms (rotation, flip, scale, perspective) into one type of segment, and linear color transforms (brightness, contrast, Normalize) into another. Non-fusible operations (Gaussian blur, statistical normalization modes, saturation) act as natural segment boundaries and pass through via their native backend unchanged.
2. **Fusing**: within each segment, the individual affine (or projective) matrices are composed mathematically -- `M_composed = M_n @ ... @ M_2 @ M_1` -- and a single interpolation pass applies the entire group.

A pipeline of three affine transforms saves two interpolation passes. At training time, with thousands of images and many augmentation steps, this translates to measurably better effective resolution in your augmented dataset.

## ✨ Features

- Automatic fusion of consecutive geometric transforms -- no manual configuration needed.
- Use `ReorderPolicy.POINTWISE` to bubble color ops past geometric chains, enabling fusion across non-consecutive geometric runs.
- All affine transforms from each supported backend (Kornia, TorchVision, Albumentations) are mapped and fusible -- not just a subset.
- Backend-preserving randomness by default, with opt-in `randomness="per_sample"` for independent draws where the adapter exposes canonical sampling.
- Auxiliary target support: masks, bounding boxes (`xyxy` and `xywh`), and keypoints warped by the same composed matrix for fused/interpolating geometric segments and flip-only exact chains.
- Multi-backend: Kornia, TorchVision, and Albumentations transforms in the same pipeline.
- Backend-free mode: construct a pipeline from numeric parameter ranges with no framework imports.
- Meta-config mode: describe a pipeline as a list of `TransformSpec` objects and resolve it to any supported backend at construction time -- swap backends without rewriting the pipeline.
- NumPy I/O: `NumpyToTorchConverter` and `TorchToNumpyConverter` bridge OpenCV/PIL/Albumentations workflows; `output_backend="numpy"` returns NumPy arrays directly from the pipeline.
- Reorder policy: `NONE` (default), `POINTWISE` (bubble color ops after geometric runs), or `AGGRESSIVE` (currently an alias of `POINTWISE`).
- Fusion introspection: inspect `fusion_plan`, `n_warps_saved`, and `transform_matrix` after each forward pass.
- Projective (perspective) transform fusion via full 3x3 homography matrices.
- Pickle-safe: pipelines survive `pickle.dumps`/`pickle.loads` for use with `DataParallel` and multiprocess `DataLoader` workers.

## 📦 Installation

```bash
pip install fuse-augmentations
```

Backend extras are optional -- install only what your pipeline uses:

```bash
pip install "fuse-augmentations[kornia]"       # Kornia transforms
pip install "fuse-augmentations[torchvision]"  # TorchVision transforms
pip install "fuse-augmentations[albumentations]"  # Albumentations transforms
pip install "fuse-augmentations[all]"          # All backends
```

**Requirements**: Python 3.10+, PyTorch >= 2.2.

## 🚀 Quick Start

```python
import torch
import albumentations as aug_a
from fuse_aug import Compose  # or: from fuse_augmentations import Compose

pipe = Compose(
    [
        aug_a.Rotate(limit=30, p=0.8),
        aug_a.HorizontalFlip(p=0.5),
        aug_a.Affine(scale=(0.8, 1.2), p=0.7),
    ]
)

image = torch.rand(4, 3, 256, 256)  # (batch_size, channels, height, width)
out = pipe(image)  # one interpolation pass instead of three

print(pipe.fusion_plan)
# fused(Rotate, HorizontalFlip, Affine)

print(pipe.n_warps_saved)
# 2
```

The short import `fuse_aug` is a canonical alias for `fuse_augmentations` -- both expose the same public API. All affine transforms from each backend are supported, not just the ones shown in this example.

## ⚙️ How Fusion Works

Given a pipeline `[Rotate, Scale, HFlip, GaussianBlur, Rotate]`:

1. **Grouping**: `[Rotate, Scale, HFlip]` are consecutive geometric transforms and are collected into one segment. `GaussianBlur` is a spatial-kernel operation that is not yet fusible, so it acts as a segment boundary. The trailing `Rotate` forms its own segment.
2. **Fusing**: the first segment's affine matrices are composed: `M = M_hflip @ M_scale @ M_rot`. One interpolation pass applies all three. The trailing `Rotate` segment applies its own single pass.

All matrices are `(batch_size, 3, 3)` homogeneous in pixel coordinates with `align_corners=True`. To apply the interpolation, the composed forward matrix is inverted once to yield backward (sampling) grid coordinates.

For flip-only chains, `fuse-augmentations` uses an `ExactAffineSegment` that applies `tensor.flip` directly -- zero interpolation error.

### Architecture in three scenarios

The backend (Albumentations, Kornia, or TorchVision) is always used for the final execution step — `fuse-augmentations` acts as a meta-proxy that reduces the number of backend calls by composing transform matrices upfront.

**Scenario 1 — consecutive geometric + color ops (`ReorderPolicy.NONE`)**

```text
WITHOUT fuse-augmentations
  pipeline:   Rotate     Translate    HFlip      Brightness
  backend:   [warp 1]   [warp 2]    [warp 3]   [pixel op]    3 warps

WITH fuse-augmentations
  pipeline:   Rotate     Translate    HFlip      Brightness
               └─────────────┴──────────┘             │
                   FusedAffineSegment         FusedColorSegment
              M = M_hflip @ M_trans @ M_rot     M_brightness
  backend:        [warp 1]                      [pixel op]    1 warp  ✓
```

**Scenario 2 — color op interleaved, solved by `ReorderPolicy.POINTWISE`**

```text
Pipeline:  Rotate, Brightness, Translate, HFlip   (color op splits the geometric chain)

ReorderPolicy.NONE (default):
  segments:  [Rotate] │ [Brightness] │ [Translate, HFlip]
  backend:   [warp 1]   [pixel op]    [warp 2]              2 warps

ReorderPolicy.POINTWISE (bubble color past geometric):
  reordered: Rotate, Translate, HFlip, Brightness
  segments:  [Rotate, Translate, HFlip] │ [Brightness]
              FusedAffineSegment            FusedColorSegment
  backend:         [warp 1]                [pixel op]        1 warp  ✓
```

**Scenario 3 — consecutive color ops fused by `FusedColorSegment`**

```text
Pipeline:  Rotate, Translate, HFlip, Brightness, Contrast

  pipeline:   Rotate    Translate    HFlip         Brightness    Contrast
               └─────────────┴────────┘              └────────────────┘
                   FusedAffineSegment                FusedColorSegment
             M = M_hflip @ M_trans @ M_rot    M = M_contrast @ M_brightness
                                                (4×4 RGBA color matrix)
  backend:        [warp 1]                          [pixel op]    1 warp + 1 pixel op  ✓
```

### Execution strategies for Albumentations pipelines

Fused Albumentations segments support an `execution` flag on `Compose`:

- `execution="cv2"` (default) — each sample in the batch is warped with one `cv2.warpAffine`/`cv2.warpPerspective` call. Outputs are bit-identical to earlier releases and to what the composed cv2 matrices produce natively; this is the fastest choice on CPU at small batch sizes.
- `execution="torch"` (opt-in) — the same per-sample matrices (identical random sampling stream) are applied as **one batched** `grid_sample` for the whole batch, giving batch-size-independent throughput and native GPU/MPS execution. Border handling and bilinear sub-pixel weights differ slightly from cv2 (interior pixels agree to ~1e-3 on smooth content), so this strategy is opt-in rather than a silent default change.

```python
pipe = Compose(albu_transforms, execution="torch")  # batched grid_sample path
```

On CUDA/MPS devices the torch strategy is the natural choice — the cv2 path would require a CPU round-trip.

### Compiled warp core (`compile=True`)

`Compose(..., compile=True)` opts the geometric warp core (matrix inversion → grid build → `grid_sample`) into `torch.compile`. It is **off by default**, and:

- On CPU, or on torch older than 2.2, it is a **no-op** — the eager path runs and the output is unchanged, so the flag is always safe to set.
- On GPU/MPS the warp core runs as one compiled graph. Probability masking and per-sample selection stay outside the compiled region, so there are no graph breaks. `dynamic=True` keeps a single graph across varying batch/height/width instead of recompiling per shape.
- The **first** GPU/MPS call after construction pays a one-time compilation cost; steady-state calls are faster. On darwin/arm64 (Apple MPS) the inductor backend is still maturing — a small local speedup is typical; the larger win is on CUDA hosts.

```python
pipe = Compose(transforms, compile=True)  # compiled warp on GPU/MPS, eager elsewhere
```

### Antialiased downscale (`antialias=True`)

`Compose(..., antialias=True)` reduces aliasing when a crop-resize step shrinks the image aggressively. It is **off by default**, and only acts when the worst-axis scale drops below `0.5` (a downscale of more than 2×). In that case the input is Gaussian pre-filtered (mipmap-rule sigma) before the single warp, so high-frequency detail is band-limited instead of aliased. When the scale is gentler than `0.5`, or the flag is off, the output is **bit-identical** to the default single-warp path. The pre-filter uses the installed Kornia Gaussian blur; if Kornia is not available it falls back to the un-filtered warp.

```python
pipe = Compose(
    transforms, antialias=True
)  # prefilter only when a downscale would alias
```

## 📖 API Reference

### Core

| Class / Function                      | Description                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Compose`                             | Main entry point. Wraps a list of transforms, groups them into fusible runs, and fuses each group on `forward()`. Accepts `output_backend="numpy"` to return NumPy arrays, `execution="cv2"`, and `mask_interpolation="nearest"` (default) or `"bilinear"`; bilinear gives differentiable soft masks, mixes labels at boundaries, and requires float mask input. |
| `Compose.from_params(...)`            | Classmethod. Build a backend-free pipeline from numeric parameter ranges, or from a `specs` list of `TransformSpec` objects. Defaults to `ReorderPolicy.POINTWISE`.                                                                                                                                                                                              |
| `Compose.from_config(specs, backend)` | Classmethod. Resolve a list of `TransformSpec` objects to a specific backend and build the pipeline -- no backend imports needed at spec time. Defaults to `ReorderPolicy.POINTWISE`.                                                                                                                                                                            |
| `TransformSpec`                       | Frozen dataclass for declarative, backend-agnostic pipeline configuration: `operation`, `params`, `prob`. JSON-serialisable via `to_dict()` / `from_dict()`.                                                                                                                                                                                                     |
| `NumpyToTorchConverter`               | Converts NumPy `(H, W, C)` / `(B, H, W, C)` arrays (uint8 or float32) to `(B, C, H, W)` torch tensors. uint8 is normalised to float32 `[0, 1]`.                                                                                                                                                                                                                  |
| `TorchToNumpyConverter`               | Converts `(B, C, H, W)` torch tensors to NumPy arrays. Single-image batches are squeezed to `(H, W, C)`; multi-image batches produce `(B, H, W, C)`.                                                                                                                                                                                                             |
| `FusedAffineSegment`                  | Handles one fusible run: samples random params, composes matrices, applies a single interpolation pass.                                                                                                                                                                                                                                                          |
| `ExactAffineSegment`                  | Lossless segment for flip-only chains. Uses `tensor.flip` -- no interpolation.                                                                                                                                                                                                                                                                                   |
| `ProjectiveSegment`                   | Fuses projective transforms using 3x3 homography matrices.                                                                                                                                                                                                                                                                                                       |
| `FusedColorSegment`                   | Fuses consecutive `POINTWISE_LINEAR` transforms (brightness, contrast, and standard Normalize) into one `(B, 4, 4)` matrix multiply. Constructor accepts `clip_output: bool = True` and `clip_policy` to control gamut clipping.                                                                                                                                 |
| `CropResizeSegment`                   | Handles a single crop+resize operation (`RandomResizedCrop`). Samples crop coordinates, builds the crop-to-output affine matrix, applies one interpolation pass at the target output size. Output spatial size differs from input.                                                                                                                               |
| `build_segments()`                    | Internal. Partitions a transform list into fusible segments and passthrough barriers.                                                                                                                                                                                                                                                                            |
| `SegmentDescriptor`                   | Frozen dataclass describing one pipeline segment: `kind`, `transforms`, `n_warps_saved`, `backend`, plus machine-readable `barrier`, `split_reason`, `refused`. Returned by `FusedCompose.fusion_plan_descriptors`.                                                                                                                                              |

### Enums

<details>
<summary><code>ReorderPolicy</code></summary>

- `NONE` — default for `Compose()`; preserves declared order, merges consecutive fusible transforms.
- `POINTWISE` — default for `from_params`/`from_config`; bubbles `POINTWISE` and `POINTWISE_LINEAR` ops out of geometric chains before segmentation.
- `AGGRESSIVE` — currently same as `POINTWISE`; accepted for forward compatibility.

</details>

<details>
<summary><code>InterpolationMode</code></summary>

Ordered by quality (`BICUBIC > BILINEAR > NEAREST`); useful for programmatic comparison:

- `NEAREST`
- `BILINEAR`
- `BICUBIC`

</details>

<details>
<summary><code>PaddingMode</code></summary>

Ordered by quality:

- `ZEROS`
- `BORDER`
- `REFLECTION`

</details>

<details>
<summary><code>TransformCategory</code></summary>

- `GEOMETRIC_INTERP` — interpolation-based affine (rotation, scale, shear, translate).
- `GEOMETRIC_EXACT` — lossless discrete ops (flips, 90° rotations); fused via `ExactAffineSegment`.
- `POINTWISE` — pixel-wise ops (normalize, gamma) that are not yet fusible; act as passthrough.
- `SPATIAL_KERNEL` — kernel-based ops (GaussianBlur, Sharpen); act as fusion barriers.
- `PROJECTIVE` — perspective transforms; fused via `ProjectiveSegment` using 3×3 homographies.
- `POINTWISE_LINEAR` — brightness/contrast ops fused by `FusedColorSegment`; see the Color Fusion section for supported ops per backend.
- `CROP_RESIZE_FIXED` — handled by `CropResizeSegment`; changes output spatial size.

</details>

### Auxiliary Target Functions

| Function                                      | Shape                                   | Description                                                                                                 |
| --------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `transform_keypoints(keypoints, mtx_forward)` | `(batch_size, num_points, 2)`           | Apply forward affine matrix to keypoint coordinates. Differentiable.                                        |
| `transform_bbox_xyxy(boxes, mtx_forward)`     | `(batch_size, num_boxes, 4)`            | Transform `[x1, y1, x2, y2]` boxes by forward homography; AABB-wrap after rotation.                         |
| `transform_bbox_xywh(boxes, mtx_forward)`     | `(batch_size, num_boxes, 4)`            | Transform `[x, y, w, h]` boxes; converts to/from xyxy internally.                                           |
| `transform_mask(mask, grid, mode="nearest")`  | `(batch_size, channels, height, width)` | Apply a sampling grid with nearest labels or differentiable bilinear mixing; bilinear requires float input. |

## 🎯 Auxiliary Targets

Pass `data_keys` to route masks, boxes, or keypoints through the same fused transform:

```python
import torchvision.transforms.v2 as aug_tv
from fuse_aug import Compose

image = ...  # (batch_size, channels, height, width) float32 tensor
mask = ...  # (batch_size, channels, height, width) integer label tensor
bboxes = ...  # (batch_size, num_boxes, 4) pixel-space boxes
keypoints = ...  # (batch_size, num_points, 2) pixel-space keypoints

pipe = Compose(
    [aug_tv.RandomRotation(degrees=30), aug_tv.RandomHorizontalFlip(p=0.5)],
    data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
)

img_out, mask_out, bboxes_out, kpts_out = pipe(image, mask, bboxes, keypoints)
# mask warped with nearest-neighbour -- integer class labels preserved
# bboxes AABB-wrapped after rotation
# keypoints transformed exactly via homogeneous matrix
```

Supported `data_keys` values:

| Key           | Tensor shape   | Notes                                                                                                                                |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `"input"`     | `(B, C, H, W)` | Image; always the first argument                                                                                                     |
| `"mask"`      | `(B, C, H, W)` | Nearest-neighbour by default; pass `mask_interpolation="bilinear"` for differentiable float soft masks that mix labels at boundaries |
| `"bbox_xyxy"` | `(B, N, 4)`    | Pixel-space `[x1, y1, x2, y2]`; AABB wrapping after rotation                                                                         |
| `"bbox_xywh"` | `(B, N, 4)`    | Pixel-space `[x, y, w, h]`; converted internally to xyxy                                                                             |
| `"keypoints"` | `(B, N, 2)`    | Pixel-space `[x, y]`; exact homogeneous transform                                                                                    |

Exact discrete chains route all auxiliary targets. Masks are transformed by the same lossless flip/rotation applied to the image, for every exact op including non-flip ops (`RandomRotate90`, `D4`, `Transpose`). Boxes and keypoints are routed through the composed pixel matrix: flip-only exact chains stay lossless, and a chain containing a non-flip exact op is executed on the interpolating grid path when box/keypoint targets are present, so they are always routed and never raise.

## 🔌 Backend-Free Pipelines

No Kornia or TorchVision import needed:

```python
from fuse_aug import Compose

image = ...  # your (B, C, H, W) tensor

pipe = Compose.from_params(
    rotation=(-30, 30),
    scale=(0.8, 1.2),
    hflip_p=0.5,
    vflip_p=0.3,
    interpolation="bicubic",
    padding_mode="reflection",
)

out = pipe(image)
```

`from_params` accepts: `rotation`, `scale`, `scale_x`, `scale_y`, `shear_x`, `shear_y`, `translate_x`, `translate_y`, `hflip_p`, `vflip_p`, `interpolation` (`"bilinear"`, `"nearest"`, `"bicubic"`), `padding_mode` (`"zeros"`, `"border"`, `"reflection"`), `mask_interpolation` (`"nearest"` or `"bilinear"`), `reorder`, `data_keys`, `output_backend`, `randomness`, `clip_policy`, `specs`. (`brightness` and `contrast` are reserved for a future version.)

> **Note**: `from_params` and `from_config` default to `ReorderPolicy.POINTWISE`, while `Compose()` defaults to `ReorderPolicy.NONE`. Pass `reorder=ReorderPolicy.NONE` explicitly if you need to preserve the declared order in a `from_params` pipeline.

## 🔄 NumPy I/O

`fuse-augmentations` pipelines operate on `(B, C, H, W)` torch tensors internally. Two converters bridge the gap for OpenCV, PIL, and Albumentations workflows that use NumPy arrays:

```python
import numpy as np
from fuse_aug import Compose, NumpyToTorchConverter, TorchToNumpyConverter

# NumPy (H, W, C) uint8 -> torch (B, C, H, W) float32 [0, 1]
to_torch = NumpyToTorchConverter()
image_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
image_tensor = to_torch.convert(image_np)  # (1, 3, 256, 256)

pipe = Compose.from_params(rotation=(-15, 15), hflip_p=0.5)
out_tensor = pipe(image_tensor)

# torch (B, C, H, W) -> NumPy (H, W, C) for B=1, or (B, H, W, C) for B>1
to_numpy = TorchToNumpyConverter()
out_np = to_numpy.convert(out_tensor)  # (256, 256, 3)
```

For pipelines where NumPy output is always wanted, pass `output_backend="numpy"` directly to `Compose`, `from_params`, or `from_config`:

```python
from fuse_aug import Compose

image_tensor = ...  # your (B, C, H, W) float32 tensor

pipe = Compose.from_params(
    rotation=(-15, 15),
    hflip_p=0.5,
    output_backend="numpy",
)

out = pipe(image_tensor)  # returns NumPy (H, W, C) array directly
```

`output_backend` values: `"numpy"` / `"numpy_hwc"` (channel-last NumPy array), `"torch"` or `None` (native tensor, default). Conversion is applied per target: the image and mask outputs are converted to the requested backend, while coordinate targets (bounding boxes, keypoints) stay as tensors because the channel-last image layout does not apply to them.

`NumpyToTorchConverter` accepts arrays of shape `(H, W)`, `(H, W, C)`, or `(B, H, W, C)`. `uint8` inputs are normalised to `float32 [0, 1]`; `float32` inputs are passed through unchanged.

### Randomness Policy

`randomness="backend"` is the default and preserves the native backend's batch sampling semantics. For example, TorchVision v2 transforms sample one parameter set for the whole batched tensor, and `Compose` preserves that by default.

Use `randomness="per_sample"` to ask fused segments to draw independent probability and parameter samples per image where the adapter exposes canonical sampling:

```python
pipe = Compose(
    [aug_tv.RandomRotation(degrees=30)],
    randomness="per_sample",
)
```

## 🎨 Color Fusion (POINTWISE_LINEAR)

Consecutive color transforms registered as `POINTWISE_LINEAR` are fused into a single `FusedColorSegment` that applies one matrix multiply instead of N sequential operations:

```python
import kornia.augmentation as K
from fuse_augmentations import Compose

pipe = Compose(
    [
        K.RandomRotation(degrees=30),
        K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
        K.RandomContrast(contrast=(0.9, 1.1), p=1.0),
    ]
)
out = pipe(image)
print(pipe.fusion_plan)
# fused(RandomRotation) → color(RandomBrightness, RandomContrast)
```

The `"color"` kind appears in `fusion_plan_descriptors` for `FusedColorSegment` runs; its `n_warps_saved` reflects eliminated sequential color applies.

Supported color operations per backend:

| Backend        | Supported                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------- |
| Kornia         | `RandomBrightness`, `RandomContrast`, `ColorJitter` (brightness+contrast only), `Normalize` |
| TorchVision    | `ColorJitter` (brightness+contrast only), v1/v2 `Normalize`                                 |
| Albumentations | `RandomBrightnessContrast`, standard `Normalize` (including `max_pixel_value`)              |

By default, `FusedColorSegment` clamps the fused output to `[0, 1]` after the matrix multiply (`clip_output=True`). `FusedCompose(..., clip_policy="final")` keeps that one-pass behavior. Use `clip_policy="per_op_parity"` when native per-operation clamping is required; it inserts a clamp only when the composed affine range can escape `[0, 1]`. A fused run containing Normalize disables the final gamut clamp because normalized values are intentionally outside image gamut. Pass `clip_output=False` when constructing a `FusedColorSegment` directly if your pipeline intentionally produces values outside this range.

> **Contrast midpoint**: ColorJitter contrast now uses the per-image mean luminance, matching the native backend. TorchVision uses RGB weights `(0.2989, 0.587, 0.114)` and Kornia uses `(0.299, 0.587, 0.114)`. If brightness precedes contrast in the same fused run, the mean is propagated through the preceding affine operations. This is a deliberate behavior change from the old fixed `0.5` midpoint.

Normalize fusion folds `alpha = 1 / std` and `beta = -mean / std` into the color matrix for Kornia and TorchVision. Albumentations standard Normalize also folds its `max_pixel_value` scaling; image-statistics modes are left as native passthrough operations because their coefficients depend on the image.

Color fusion relies on each supported op being a per-channel affine map `c' = alpha * c + beta`, expressible as a 4×4 homogeneous matrix -- composing N such ops reduces to a single matrix product, so one fused multiply replaces N sequential applies.

## ✂️ Crop+Resize (`CROP_RESIZE_FIXED`)

`RandomResizedCrop` from any supported backend is registered as `CROP_RESIZE_FIXED` and handled by `CropResizeSegment`. Unlike `FusedAffineSegment`, it is not fused with adjacent geometric transforms -- it acts as a segment boundary and applies exactly one interpolation pass at the configured output size:

```python
import torchvision.transforms.v2 as aug_tv
from fuse_aug import Compose

pipe = Compose(
    [
        aug_tv.RandomRotation(degrees=15),
        aug_tv.RandomResizedCrop(
            size=(224, 224)
        ),  # CROP_RESIZE_FIXED — segment boundary
        aug_tv.RandomHorizontalFlip(p=0.5),
    ]
)

out = pipe(image)  # (B, C, 224, 224)
print(pipe.fusion_plan)
# fused(RandomRotation) → crop_resize(RandomResizedCrop) → exact(RandomHorizontalFlip)
```

The output tensor has the target spatial size specified in the `RandomResizedCrop` constructor. Supported in all three backends: Kornia, TorchVision (v1 and v2), and Albumentations.

## 🔧 Backend-Agnostic Meta-Config

`TransformSpec` is a frozen, JSON-serialisable dataclass that describes one augmentation operation without importing any backend. Use it to define pipelines in configuration files or experiment configs, then materialise them at runtime with either `from_config` (backend-specific) or `from_params(specs=...)` (backend-free):

For the fastest zero-dependency, fully batched path, opt in to `backend="native"`. It uses the same direct Torch matrix engine as `from_params` and supports rotation, scale, shear, translation, flips, brightness, and contrast. Optional-backend-only operations remain visible in the capability matrix and are rejected by the native builder rather than silently approximated.

```python
from fuse_aug import Compose, TransformSpec

specs = [
    TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8),
    TransformSpec(operation="hflip", params={}, prob=0.5),
]

image = ...
# Resolve to a specific backend -- backend imports happen here, not at spec time
pipe = Compose.from_config(specs, backend="kornia")
out = pipe(image)

# Or stay fully backend-free using from_params(specs=...)
pipe2 = Compose.from_params(specs=specs)
out2 = pipe2(image)
```

`TransformSpec` fields:

| Field       | Type                | Description                                                                   |
| ----------- | ------------------- | ----------------------------------------------------------------------------- |
| `operation` | `str`               | Canonical operation name: `"rotation"`, `"hflip"`, `"vflip"`, `"scale"`, etc. |
| `params`    | `dict[str, object]` | Operation-specific parameters associated with the canonical operation.        |
| `prob`      | `float`             | Per-sample application probability. Default `1.0`.                            |

For `from_config`, `operation` names are canonical and `params` are first passed through `translate_params()` before being forwarded to the backend constructor. A small set of canonical parameter names (for example, `degrees` for rotation-like operations or `factor` for scale) are translated into the appropriate backend-specific kwargs for each supported backend. Any keys that are not recognized by `translate_params()` remain backend-specific constructor kwargs and are passed through unchanged. This means a `TransformSpec` list that uses only the canonical subset of parameters is generally portable across backends, while specs that rely on backend-only parameters may still need adjustment when switching backends.

Specs are JSON round-trip safe via `to_dict()` / `from_dict()`:

```python
import json
from fuse_aug import TransformSpec

spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
payload = json.dumps(spec.to_dict())
restored = TransformSpec.from_dict(json.loads(payload))
assert restored == spec
```

Supported ops for `from_config`: all ops in `SUPPORTED_OPS` (`"rotation"`, `"affine"`, `"shear"`, `"translate"`, `"hflip"`, `"vflip"`, `"scale"`, `"perspective"`, `"rotation90"`, `"brightness"`, `"contrast"`), subject to each backend's coverage:

| Op            | Kornia | TorchVision | Albumentations | Native |
| ------------- | :----: | :---------: | :------------: | :----: |
| `rotation`    |   ✓    |      ✓      |       ✓        |   ✓    |
| `affine`      |   ✓    |      ✓      |       ✓        |   –    |
| `shear`       |   ✓    |      –      |       –        |   ✓    |
| `translate`   |   ✓    |      –      |       –        |   ✓    |
| `hflip`       |   ✓    |      ✓      |       ✓        |   ✓    |
| `vflip`       |   ✓    |      ✓      |       ✓        |   ✓    |
| `scale`       |   ✓    |      ✓      |       ✓        |   ✓    |
| `perspective` |   ✓    |      ✓      |       ✓        |   –    |
| `rotation90`  |   ✓    |      –      |       ✓        |   –    |
| `brightness`  |   ✓    |      –      |       –        |   ✓    |
| `contrast`    |   ✓    |      –      |       –        |   ✓    |

Supported ops for `from_params(specs=...)`: `"rotation"`, `"scale"`, `"scale_x"`, `"scale_y"`, `"shear_x"`, `"shear_y"`, `"translate_x"`, `"translate_y"`, `"hflip"`, `"vflip"`, `"brightness"`, `"contrast"`.

> **Note**: `from_config` defaults to `ReorderPolicy.POINTWISE`. Pass `reorder=ReorderPolicy.NONE` to preserve the declared order.

### Hydra / OmegaConf integration

`TransformSpec` is designed to round-trip through YAML. A typical Hydra config:

```yaml
# config/augmentation.yaml
augmentation:
  backend: kornia
  specs:
    - operation: rotation
      params:
        degrees: [-30.0, 30.0]
      prob: 0.8
    - operation: hflip
      params: {}
      prob: 0.5
    - operation: scale
      params:
        factor: [0.8, 1.2]
      prob: 0.7
```

```python
from omegaconf import OmegaConf
from fuse_aug import Compose, TransformSpec


def build_pipeline(cfg):
    specs = [
        TransformSpec.from_dict(s)
        for s in OmegaConf.to_container(cfg.augmentation.specs)
    ]
    return Compose.from_config(specs, backend=cfg.augmentation.backend)
```

`TransformSpec.from_dict` restores tuple semantics from any sequence type (JSON/YAML `list`, OmegaConf `ListConfig`, etc.) automatically for canonical range-parameter keys (`degrees`, `factor`, `scale`, etc.).

**Plain PyYAML** (no Hydra):

```python
import yaml
from fuse_augmentations import Compose, TransformSpec

with open("augmentation.yaml") as f:
    data = yaml.safe_load(f)

specs = [TransformSpec.from_dict(s) for s in data["augmentation"]["specs"]]
pipe = Compose.from_config(specs, backend=data["augmentation"]["backend"])
```

> **Note**: `DictConfig` objects can be passed directly to `from_dict` — `OmegaConf.to_container()` is not required. Range keys (`degrees`, `factor`, etc.) are restored to tuples automatically regardless of input type.

## 🔗 Multi-Backend Pipelines

Kornia, TorchVision, and Albumentations transforms can be mixed in the same `Compose`:

```python
import albumentations as aug_a
import torchvision.transforms.v2 as aug_tv
from kornia import augmentation as aug_k
from fuse_aug import Compose

image = ...  # your (B, C, H, W) tensor

pipe = Compose(
    [
        aug_a.Rotate(limit=15),  # Albumentations
        aug_tv.RandomHorizontalFlip(),  # TorchVision
        aug_k.ColorJitter(brightness=0.3),  # Kornia (POINTWISE_LINEAR — color-fused)
    ]
)

out = pipe(image)
# fused(Rotate) → exact(RandomHorizontalFlip) → color(ColorJitter)
```

Each transform is resolved to the correct adapter at construction time. Backend boundaries split fusion groups, so transforms from different frameworks can share one `Compose` pipeline but are not fused into the same matrix segment. Framework-specific behavior (parameter sampling, matrix building, passthrough for operations not yet fusible) is handled by `KorniaAdapter`, `TorchVisionAdapter`, or `AlbumentationsAdapter`.

## 🔀 Reorder Policy

When a color operation sits between two geometric transforms, fusion is broken by default. `ReorderPolicy.POINTWISE` bubbles color ops to the end of each geometric stretch, extending the fusion window:

```python
import torchvision.transforms.v2 as aug_tv
from fuse_aug import Compose, ReorderPolicy

pipe = Compose(
    [
        aug_tv.RandomRotation(degrees=15),
        aug_tv.ColorJitter(brightness=0.3),  # POINTWISE_LINEAR — would break fusion
        aug_tv.RandomHorizontalFlip(p=0.5),
    ],
    reorder=ReorderPolicy.POINTWISE,
)

print(pipe.fusion_plan)
# fused(RandomRotation, RandomHorizontalFlip) → color(ColorJitter)
```

**`ReorderPolicy.NONE`** (default for `Compose()`): preserves declared order, merges consecutive fusible transforms.

**`ReorderPolicy.POINTWISE`** (default for `from_params` and `from_config`): moves `POINTWISE` and `POINTWISE_LINEAR` ops out of geometric chains before segmentation.

**`ReorderPolicy.AGGRESSIVE`**: currently behaves the same as `POINTWISE`. It is accepted for forward compatibility, but today it preserves the same pointwise ordering and yields the same fusion plan as `POINTWISE`.

## 🔬 Fusion Introspection

After any forward pass:

```python
from fuse_aug import Compose

image = ...  # your (B, C, H, W) tensor
pipe = Compose(...)  # built in a previous step

out = pipe(image)

print(pipe.fusion_plan)
# fused(RandomRotation, RandomAffine) → passthrough(RandomGaussianBlur) → exact(RandomHorizontalFlip)

print(pipe.n_warps_saved)
# 1  -- one interpolation pass saved

M = pipe.transform_matrix  # (B, 3, 3) composed forward matrix
```

`transform_matrix` gives the composed forward affine matrix for each sample in the batch. Use it to transform stored coordinates that were not passed as `data_keys`.

For machine-readable inspection, use `fusion_plan_descriptors`:

```python
from fuse_aug import Compose
import json

pipe = Compose(...)  # built in a previous step

for desc in pipe.fusion_plan_descriptors:
    print(desc.kind, desc.transforms, desc.n_warps_saved)
# fused ('RandomRotation', 'RandomAffine') 1
# passthrough ('RandomGaussianBlur',) 0

# Each descriptor is also JSON-serialisable:
plan_json = [d.to_dict() for d in pipe.fusion_plan_descriptors]
print(json.dumps(plan_json, indent=2))
```

`SegmentDescriptor` fields:

| Field           | Type              | Description                                                                                                                                                                                  |
| --------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`          | `str`             | Segment type: `"fused"`, `"exact"`, `"projective"`, `"color"`, `"crop_resize"`, or `"passthrough"`                                                                                           |
| `transforms`    | `tuple[str, ...]` | Class names of transforms in this segment (`list` in `to_dict()` output)                                                                                                                     |
| `n_warps_saved` | `int`             | Interpolation passes eliminated by this segment                                                                                                                                              |
| `backend`       | `str \| None`     | Adapter class name (`"KorniaAdapter"`, `"AlbumentationsAdapter"`, `"TorchVisionAdapter"`) for fused/exact/projective segments; `None` for passthrough segments and backend-free pipelines    |
| `barrier`       | `str \| None`     | Machine-readable reason this segment ends a fusion run: `"spatial_kernel"` (blur/noise), `"coordinate_change"` (elastic/grid/optical distortion), `"crop_resize"`; `None` for fused segments |
| `split_reason`  | `str \| None`     | Why an otherwise-fusible run was split here: `"backend_boundary"` in a mixed-backend pipeline, else `None`                                                                                   |
| `refused`       | `str \| None`     | Why an op stayed on the passthrough path: `"not_fusible"` for passthrough segments, `None` for fused segments                                                                                |

## 🏋️ Training Loop

`fuse-augmentations` pipelines are `nn.Module` instances -- construct them once, then call per batch:

```python
import torch
import torchvision.transforms.v2 as aug_tv
from torch.utils.data import DataLoader, Dataset
from fuse_aug import Compose


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # list of (C, H, W) float32 tensors
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Build pipeline once; it is pickle-safe for multiprocess DataLoader workers
augment = Compose(
    [
        aug_tv.RandomRotation(degrees=15),
        aug_tv.RandomHorizontalFlip(p=0.5),
        aug_tv.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        aug_tv.ColorJitter(brightness=0.2),  # POINTWISE_LINEAR — color-fused
    ]
)

images, labels = ...  # your dataset tensors
model = ...  # your nn.Module
optimizer = ...  # your optimizer

loader = DataLoader(
    ImageDataset(images, labels), batch_size=32, shuffle=True, num_workers=4
)

for batch_images, batch_labels in loader:
    augmented = augment(
        batch_images
    )  # fused: 1 geometric warp instead of 3; color ops in one matrix multiply
    loss = model(augmented, batch_labels)
    loss.backward()
    optimizer.step()
```

For segmentation and detection tasks, pass `data_keys` to keep auxiliary targets in sync:

```python
import albumentations as aug_a
from fuse_aug import Compose

loader = ...  # your DataLoader yielding (imgs, masks, boxes, labels)

augment = Compose(
    [aug_a.Rotate(limit=15, p=0.8), aug_a.HorizontalFlip(p=0.5)],
    data_keys=["input", "mask", "bbox_xyxy"],
)

for imgs, masks, boxes, labels in loader:
    imgs_out, masks_out, boxes_out = augment(imgs, masks, boxes)
```

Pipelines survive `pickle` round-trips, so they work transparently with `torch.nn.DataParallel` and multiprocess `DataLoader` workers (the index-keyed adapter map is preserved across deserialisation).

Note for multi-worker loaders: PyTorch reseeds only the *torch* RNG per worker. Albumentations transforms sample activation (`p<1`) and parameters from numpy's global RNG, which is **not** reseeded per worker — pass a `worker_init_fn` that seeds `numpy.random` (e.g. from `torch.initial_seed()`) if you need independent augmentation streams across workers.

## ⚠️ Limitations

- **Pixel-wise ops** (gamma, equalize, saturation, hue) are not yet fusible -- they are nonlinear operations and currently act as passthrough. Standard Normalize and linear color ops (brightness, contrast) are fusible via `FusedColorSegment`; image-statistics Normalize modes remain passthrough.
- **Spatial-kernel ops** (GaussianBlur, Sharpen) act as fusion barriers; transforms on either side of a barrier form separate segments. These are not yet fusible. On a GPU/MPS pipeline a passthrough op forces a device-to-host round-trip; `fusion_plan` marks these entries with `[CPU passthrough]` so the cost is visible. Set `substitute_passthrough=True` to opt in to replacing a passthrough op with an already-installed backend's torch-native equivalent (currently Albumentations `GaussianBlur` -> Kornia `RandomGaussianBlur`) so the pipeline stays on-device. This is off by default and **behaviour-changing**: the substitute uses a different kernel, border handling, and random stream, so outputs and RNG differ -- each substitution emits a `UserWarning`. Substitution happens only when the target backend is importable; otherwise the original op is kept.
- **Padding mode** is segment-level: all transforms in a fused run share the padding mode passed to `Compose` (or the adapter default). Individual transforms cannot override it per-segment.
- **Crop+resize ops** (`RandomResizedCrop`): `CropResizeSegment` applies one interpolation pass at the target output size, but the output spatial dimensions differ from the input. `data_keys` auxiliary targets (masks, bounding boxes, keypoints) are warped through the crop affine matrix at the target size.
- **Geometric passthrough ops** (`ElasticTransform`, `GridDistortion`, `OpticalDistortion`, thin-plate-spline, piecewise-affine) act as fusion barriers and apply to the **image only**. Because they move image content, masks, bounding boxes, and keypoints that skip them would silently misalign with the image. In a multi-target `data_keys` pipeline this is a correctness bug, so the pipeline **raises `ValueError`** at runtime when such a coordinate-changing passthrough executes with auxiliary targets present (all backends: Kornia, TorchVision, Albumentations). Place geometric ops before the barrier, or transform the auxiliary targets manually. Kernel/pointwise passthrough ops (blur, noise, gamma) leave geometry unchanged, so auxiliary targets legitimately pass through them untouched -- no error and no warning.
- **Gradients**: image transforms are differentiable; mask sampling defaults to `mode='nearest'`, which is not. Opt in to differentiable soft masks with `mask_interpolation="bilinear"` (requires a floating-point mask; labels mix at boundaries).
- **Hooks**: the pipeline directly dispatches segment `forward` methods for speed, so `nn.Module` forward hooks registered on individual segment modules are bypassed. Register hooks on the pipeline or use the backend transform path when segment-level hook observation is required.

## 🤝 Contributing

Bug fixes are always welcome -- just open a pull request on [GitHub](https://github.com/Borda/fuse-augmentations). For new features or bigger ideas, open an issue first so we can discuss the direction -- all suggestions are genuinely appreciated.

## 📄 License

Apache-2.0. Copyright (c) 2025-2026 Jiri Borovec.
