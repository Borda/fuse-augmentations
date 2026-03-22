# fuse-augmentations

**Fuse consecutive geometric augmentation transforms into a single interpolation pass -- fewer warps, better image quality.**

> **Summary**: `fuse-augmentations` is a framework-agnostic library that automatically **groups** consecutive fusible geometric transforms in your augmentation pipeline, then **fuses** their matrices into a single composed transform applied via one interpolation pass. Operations that are not yet fusible (blur, color jitter) pass through unchanged. Drop-in replacement for Kornia's `AugmentationSequential`, TorchVision, and Albumentations compose classes.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/)
[![PyPI version](https://img.shields.io/pypi/v/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![CI complete testing](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml/badge.svg?event=push)](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml)
[![codecov](https://codecov.io/github/Borda/fuse-augmentations/graph/badge.svg?token=hw3VOHuzKk)](https://codecov.io/github/Borda/fuse-augmentations)

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
- [🔧 Backend-Agnostic Meta-Config](#-backend-agnostic-meta-config)
- [🔗 Multi-Backend Pipelines](#-multi-backend-pipelines)
- [🔀 Reorder Policy](#-reorder-policy)
- [🔬 Fusion Introspection](#-fusion-introspection)
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

1. **Grouping**: consecutive fusible geometric transforms are identified and collected into segments. Operations that are not yet fusible (Gaussian blur, color jitter, normalization) act as natural segment boundaries and pass through via their native backend unchanged.
2. **Fusing**: within each segment, the individual affine (or projective) matrices are composed mathematically -- `M_composed = M_n @ ... @ M_2 @ M_1` -- and a single interpolation pass applies the entire group.

A pipeline of three affine transforms saves two interpolation passes. At training time, with thousands of images and many augmentation steps, this translates to measurably better effective resolution in your augmented dataset.

## ✨ Features

- Automatic fusion of (consecutive) geometric transforms -- no manual configuration needed. Reordering with quality preservation is coming soon, so non-consecutive runs will also be fusible.
- All affine transforms from each supported backend (Kornia, TorchVision, Albumentations) are mapped and fusible -- not just a subset.
- Per-sample randomness: independent probability draws per image in the batch.
- Auxiliary target support: masks, bounding boxes (`xyxy` and `xywh`), and keypoints warped by the same composed matrix.
- Multi-backend: Kornia, TorchVision, and Albumentations transforms in the same pipeline.
- Backend-free mode: construct a pipeline from numeric parameter ranges with no framework imports.
- Meta-config mode: describe a pipeline as a list of `TransformSpec` objects and resolve it to any supported backend at construction time — swap backends without rewriting the pipeline.
- Reorder policy: `NONE` (default), `POINTWISE` (bubble color ops after geometric runs), or `AGGRESSIVE` (currently an alias of `POINTWISE`).
- Fusion introspection: inspect `fusion_plan`, `n_warps_saved`, and `transform_matrix` after each forward pass.
- Projective (perspective) transform fusion via full 3x3 homography matrices.
- **Roadmap**: future versions will allow passing transforms as meta-configuration and receiving output in any supported backend -- e.g. build a pipeline with Albumentations ops and switch to Kornia output when changing accelerator, without rewriting the pipeline.

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
import kornia.augmentation as K
from fuse_aug import Compose  # or: from fuse_augmentations import Compose

pipe = Compose(
    [
        K.RandomRotation(degrees=30, p=0.8),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(degrees=0, scale=(0.8, 1.2), p=0.7),
    ]
)

image = torch.rand(4, 3, 256, 256)  # (B, C, H, W)
out = pipe(image)  # one interpolation pass instead of three

print(pipe.fusion_plan)
# fused(RandomRotation, RandomHorizontalFlip, RandomAffine)

print(pipe.n_warps_saved)
# 2
```

The short import `fuse_aug` is a canonical alias for `fuse_augmentations` -- both expose the same public API. All affine transforms from each backend are supported, not just the ones shown in this example.

## ⚙️ How Fusion Works

Given a pipeline `[Rotate, Scale, HFlip, GaussianBlur, Rotate]`:

1. **Grouping**: `[Rotate, Scale, HFlip]` are consecutive geometric transforms and are collected into one segment. `GaussianBlur` is a spatial-kernel operation that is not yet fusible, so it acts as a segment boundary. The trailing `Rotate` forms its own segment.
2. **Fusing**: the first segment's affine matrices are composed: `M = M_hflip @ M_scale @ M_rot`. One interpolation pass applies all three. The trailing `Rotate` segment applies its own single pass.

All matrices are `(B, 3, 3)` homogeneous in pixel coordinates with `align_corners=True`. To apply the interpolation, the composed forward matrix is inverted once to yield backward (sampling) grid coordinates.

For flip-only chains, `fuse-augmentations` uses an `ExactAffineSegment` that applies `tensor.flip` directly -- zero interpolation error.

## 📖 API Reference

### Core

| Class / Function                      | Description                                                                                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Compose`                             | Main entry point. Wraps a list of transforms, groups them into fusible runs, and fuses each group on `forward()`. Aliases: `FusedCompose`, `AugmentationSequential`. |
| `Compose.from_params(...)`            | Classmethod. Build a backend-free pipeline from numeric parameter ranges, or from a `specs` list of `TransformSpec` objects in backend-free mode.                    |
| `Compose.from_config(specs, backend)` | Classmethod. Resolve a list of `TransformSpec` objects to a specific backend and build the pipeline -- no backend imports needed at spec time.                       |
| `TransformSpec`                       | Frozen dataclass for declarative, backend-agnostic pipeline configuration: `op`, `params`, `p`. JSON-serialisable via `to_dict()` / `from_dict()`.                   |
| `FusedAffineSegment`                  | Handles one fusible run: samples random params, composes matrices, applies a single interpolation pass.                                                              |
| `ExactAffineSegment`                  | Lossless segment for flip-only chains. Uses `tensor.flip` -- no interpolation.                                                                                       |
| `ProjectiveSegment`                   | Fuses projective transforms using 3x3 homography matrices.                                                                                                           |
| `build_segments()`                    | Internal. Partitions a transform list into fusible segments and passthrough barriers.                                                                                |
| `SegmentDescriptor`                   | Frozen dataclass describing one pipeline segment: `kind`, `transforms`, `n_warps_saved`, `backend`. Returned by `FusedCompose.fusion_plan_descriptors`.              |

### Enums

| Enum                | Values                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `ReorderPolicy`     | `NONE` (default), `POINTWISE` (bubble color ops after geometric runs), `AGGRESSIVE` (currently same as `POINTWISE`) |
| `InterpolationMode` | `NEAREST`, `BILINEAR`, `BICUBIC`                                                                                    |
| `PaddingMode`       | `ZEROS`, `BORDER`, `REFLECTION`                                                                                     |
| `TransformCategory` | `GEOMETRIC_INTERP`, `GEOMETRIC_EXACT`, `POINTWISE`, `SPATIAL_KERNEL`, `PROJECTIVE`                                  |

### Auxiliary Target Functions

| Function                               | Shape          | Description                                                                 |
| -------------------------------------- | -------------- | --------------------------------------------------------------------------- |
| `transform_keypoints(kps, M)`          | `(B, N, 2)`    | Apply affine matrix to keypoint coordinates. Differentiable.                |
| `transform_bbox_xyxy(bboxes, M, H, W)` | `(B, N, 4)`    | Transform `[x1, y1, x2, y2]` bboxes; AABB-wrap after rotation.              |
| `transform_bbox_xywh(bboxes, M, H, W)` | `(B, N, 4)`    | Transform `[x, y, w, h]` bboxes; converts to/from xyxy internally.          |
| `transform_mask(mask, grid)`           | `(B, C, H, W)` | Apply sampling grid with `mode='nearest'` to preserve integer class labels. |

## 🎯 Auxiliary Targets

Pass `data_keys` to route masks, boxes, or keypoints through the same fused transform:

```python
from fuse_aug import Compose
import kornia.augmentation as K

pipe = Compose(
    [K.RandomRotation(degrees=30, p=0.8), K.RandomHorizontalFlip(p=0.5)],
    data_keys=["input", "mask", "bbox_xyxy", "keypoints"],
)

img_out, mask_out, bboxes_out, kpts_out = pipe(image, mask, bboxes, keypoints)
# mask warped with nearest-neighbour -- integer class labels preserved
# bboxes AABB-wrapped after rotation
# keypoints transformed exactly via homogeneous matrix
```

Supported `data_keys` values:

| Key           | Tensor shape   | Notes                                                        |
| ------------- | -------------- | ------------------------------------------------------------ |
| `"input"`     | `(B, C, H, W)` | Image; always the first argument                             |
| `"mask"`      | `(B, C, H, W)` | Nearest-neighbour sampling; integer labels preserved         |
| `"bbox_xyxy"` | `(B, N, 4)`    | Pixel-space `[x1, y1, x2, y2]`; AABB wrapping after rotation |
| `"bbox_xywh"` | `(B, N, 4)`    | Pixel-space `[x, y, w, h]`; converted internally to xyxy     |
| `"keypoints"` | `(B, N, 2)`    | Pixel-space `[x, y]`; exact homogeneous transform            |

## 🔌 Backend-Free Pipelines

No Kornia or TorchVision import needed:

```python
from fuse_aug import Compose, InterpolationMode, PaddingMode

pipe = Compose.from_params(
    rotation=(-30, 30),
    scale=(0.8, 1.2),
    hflip_p=0.5,
    vflip_p=0.3,
    interpolation=InterpolationMode.BICUBIC,
    padding_mode=PaddingMode.REFLECTION,
)

out = pipe(image)
```

`from_params` accepts: `rotation`, `scale`, `scale_x`, `scale_y`, `shear_x`, `shear_y`, `translate_x`, `translate_y`, `hflip_p`, `vflip_p`, `interpolation`, `padding_mode`, `reorder`, `data_keys`, `specs`.

## 🔧 Backend-Agnostic Meta-Config

`TransformSpec` is a frozen, JSON-serialisable dataclass that describes one augmentation operation without importing any backend. Use it to define pipelines in configuration files or experiment configs, then materialise them at runtime with either `from_config` (backend-specific) or `from_params(specs=...)` (backend-free):

```python
from fuse_aug import Compose, TransformSpec

specs = [
    TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8),
    TransformSpec(op="hflip", params={}, p=0.5),
]

# Resolve to a specific backend -- backend imports happen here, not at spec time
pipe = Compose.from_config(specs, backend="kornia")
out = pipe(image)

# Or stay fully backend-free using from_params(specs=...)
pipe2 = Compose.from_params(specs=specs)
out2 = pipe2(image)
```

`TransformSpec` fields:

| Field    | Type                | Description                                                            |
| -------- | ------------------- | ---------------------------------------------------------------------- |
| `op`     | `str`               | Canonical op name: `"rotation"`, `"hflip"`, `"vflip"`, `"scale"`, etc. |
| `params` | `dict[str, object]` | Op-specific parameters associated with the canonical op.               |
| `p`      | `float`             | Per-sample application probability. Default `1.0`.                     |

For `from_config`, `op` names are canonical and `params` are first passed through `translate_params()` before being forwarded to the backend constructor. A small set of canonical parameter names (for example, `degrees` for rotation-like ops or `factor` for scale) are translated into the appropriate backend-specific kwargs for each supported backend. Any keys that are not recognized by `translate_params()` remain backend-specific constructor kwargs and are passed through unchanged. This means a `TransformSpec` list that uses only the canonical subset of parameters is generally portable across backends, while specs that rely on backend-only parameters may still need adjustment when switching backends.

Specs are JSON round-trip safe via `to_dict()` / `from_dict()`:

```python
import json
from fuse_aug import TransformSpec

spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
payload = json.dumps(spec.to_dict())
restored = TransformSpec.from_dict(json.loads(payload))
assert restored == spec
```

Supported ops for `from_config`: all ops in `SUPPORTED_OPS` (`"rotation"`, `"affine"`, `"shear"`, `"translate"`, `"hflip"`, `"vflip"`, `"scale"`, `"perspective"`, `"rotation90"`), subject to each backend's coverage (see `_resolver.py`).

Supported ops for `from_params(specs=...)`: `"rotation"`, `"scale"`, `"scale_x"`, `"scale_y"`, `"shear_x"`, `"shear_y"`, `"translate_x"`, `"translate_y"`, `"hflip"`, `"vflip"`.

## 🔗 Multi-Backend Pipelines

Kornia, TorchVision, and Albumentations transforms can be mixed in the same `Compose`:

```python
import torchvision.transforms.v2 as T
import kornia.augmentation as K
from fuse_aug import Compose

pipe = Compose(
    [
        K.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        K.ColorJitter(brightness=0.3),
    ]
)

out = pipe(image)
# fused(RandomRotation, RandomHorizontalFlip) -> passthrough(ColorJitter)
```

Each transform is resolved to the correct adapter at construction time. Framework-specific behavior (parameter sampling, matrix building, passthrough for operations not yet fusible) is handled by `KorniaAdapter`, `TorchVisionAdapter`, or `AlbumentationsAdapter`.

## 🔀 Reorder Policy

When a color operation sits between two geometric transforms, fusion is broken by default. `ReorderPolicy.POINTWISE` bubbles color ops to the end of each geometric stretch, extending the fusion window:

```python
from fuse_aug import Compose, ReorderPolicy
import kornia.augmentation as K

pipe = Compose(
    [
        K.RandomRotation(degrees=15, p=0.8),
        K.ColorJitter(brightness=0.3, p=0.5),  # POINTWISE -- would break fusion
        K.RandomHorizontalFlip(p=0.5),
    ],
    reorder=ReorderPolicy.POINTWISE,
)

print(pipe.fusion_plan)
# fused(RandomRotation, RandomHorizontalFlip) -> passthrough(ColorJitter)
```

**`ReorderPolicy.NONE`** (default): preserves declared order, merges consecutive fusible transforms.

**`ReorderPolicy.AGGRESSIVE`**: currently behaves the same as `POINTWISE`. It is accepted for forward compatibility, but today it preserves the same pointwise ordering and yields the same fusion plan as `POINTWISE`.

## 🔬 Fusion Introspection

After any forward pass:

```python
out = pipe(image)

print(pipe.fusion_plan)
# fused(RandomRotation, RandomAffine) -> passthrough(RandomGaussianBlur) -> fused(RandomHorizontalFlip)

print(pipe.n_warps_saved)
# 1  -- one interpolation pass saved

M = pipe.transform_matrix  # (B, 3, 3) composed forward matrix
```

`transform_matrix` gives the composed forward affine matrix for each sample in the batch. Use it to transform stored coordinates that were not passed as `data_keys`.

For machine-readable inspection, use `fusion_plan_descriptors`:

```python
for desc in pipe.fusion_plan_descriptors:
    print(desc.kind, desc.transforms, desc.n_warps_saved)
# fused ('RandomRotation', 'RandomAffine') 1
# passthrough ('RandomGaussianBlur',) 0

# Each descriptor is also JSON-serialisable:
import json

plan_json = [d.to_dict() for d in pipe.fusion_plan_descriptors]
print(json.dumps(plan_json, indent=2))
```

`SegmentDescriptor` fields:

| Field           | Type              | Description                                                                                                                                                                               |
| --------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`          | `str`             | Segment type: `"fused"`, `"exact"`, `"projective"`, or `"passthrough"`                                                                                                                    |
| `transforms`    | `tuple[str, ...]` | Class names of transforms in this segment (`list` in `to_dict()` output)                                                                                                                  |
| `n_warps_saved` | `int`             | Interpolation passes eliminated by this segment                                                                                                                                           |
| `backend`       | `str \| None`     | Adapter class name (`"KorniaAdapter"`, `"AlbumentationsAdapter"`, `"TorchVisionAdapter"`) for fused/exact/projective segments; `None` for passthrough segments and backend-free pipelines |

## ⚠️ Limitations

- **Pixel-wise ops** (ColorJitter, Normalize) are not yet fusible -- they are single-pixel operations and currently act as passthrough.
- **Spatial-kernel ops** (GaussianBlur, Sharpen) act as fusion barriers; transforms on either side of a barrier form separate segments. These are not yet fusible.
- **Padding mode** is segment-level: all transforms in a fused run share the same padding mode (the highest-quality setting among them).
- **Gradients**: image transforms are differentiable; mask sampling (`mode='nearest'`) is not.

## 🤝 Contributing

Bug fixes are always welcome -- just open a pull request on [GitHub](https://github.com/Borda/fuse-augmentations). For new features or bigger ideas, open an issue first so we can discuss the direction -- all suggestions are genuinely appreciated.

## 📄 License

Apache-2.0. Copyright (c) 2025-2026 Jiri Borovec.
