# 🔗 fuse-augmentations

**Write image augmentation as independent transforms. Execute compatible geometry with fewer resampling passes.**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/) [![PyPI version](https://img.shields.io/pypi/v/fuse-augmentations)](https://pypi.org/project/fuse-augmentations/) [![Documentation](https://img.shields.io/badge/docs-MkDocs%20Material-4051b5)](https://borda.github.io/fuse-augmentations/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Borda/fuse-augmentations/blob/main/LICENSE) [![CI](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml/badge.svg)](https://github.com/Borda/fuse-augmentations/actions/workflows/ci_testing.yml)

`fuse-augmentations` is a PyTorch tensor-first engine for reducing repeated interpolation in image augmentation pipelines. It recognizes a finite set of Kornia, TorchVision, and Albumentations transforms—or builds a pipeline directly from numeric ranges—then composes compatible transform matrices before pixels are sampled.

You keep the readable pipeline: rotate, scale, shear, translate, flip. The engine finds compatible runs and can replace several geometric warps with one.

> [!IMPORTANT]
>
> This package is Alpha and is **not** a general drop-in replacement for native Compose containers. It does not guarantee native pixels, input types, target processors, random streams, hooks, or universal speedups.

> [!WARNING]
>
> Use auxiliary targets only with explicitly supported spatial transforms. An unknown crop, resize, or other spatial passthrough can modify the image while leaving a mask, box, or keypoint tensor stale. Treat every `Unknown ... SPATIAL_KERNEL barrier` warning as unsafe when `data_keys` is present.

## 🔄 The problem: every warp resamples the image

A conventional chain may interpolate the same pixels after every geometric operation:

```text
Conventional chain — 4 resampling passes

  Input ──▶ Rotate ─warp 1─▶ Scale ─warp 2─▶ Shear ─warp 3─▶ Translate ─warp 4─▶ Output

Fused compatible chain — 1 resampling pass

  Input ──▶ sample parameters ──▶ M = M_translate · M_shear · M_scale · M_rotate
        ──▶ one sampling grid ─warp 1─▶ Output
```

Repeated interpolation adds work, creates intermediate tensors, and can progressively discard high-frequency detail. Matrix composition is cheap by comparison: for a compatible run, the package samples the individual parameters, multiplies their homogeneous matrices in declared order, and evaluates the result through one sampling grid.

This does **not** mean the fused output is pixel-identical to the native chain. It is a different resampling strategy, and backend centers, fill rules, clipping, and interpolation conventions still matter.

### Matched-parameter visual example

The same fixed Kornia rotation → scale → shear recipe is evaluated below. The native route resamples three times; Fuse Compose composes the geometry and samples once. The red/green/yellow overlay makes local disagreement visible without claiming native-pixel parity.

![Fixed Kornia parameters: native sequential versus Fuse Compose resampling](https://raw.githubusercontent.com/Borda/fuse-augmentations/main/docs/assets/images/sequential-vs-fused-kornia-framing.webp)

See [three fixed recipes for each of Kornia, TorchVision, and Albumentations](https://borda.github.io/fuse-augmentations/research/quality-and-fidelity/), including their exact limits.

## ✨ What the package can do

| Capability             | What is implemented                                                                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Affine fusion          | Registered rotation, affine, shear, translation, scale, and exact discrete operations are grouped within compatible same-backend runs.                                            |
| Exact geometry         | Supported flips and discrete operations use lossless tensor paths where the segment contract permits it.                                                                          |
| Projective fusion      | Consecutive registered perspective transforms compose as 3×3 homographies; affine↔projective transitions remain boundaries.                                                       |
| Linear color fusion    | Supported brightness, contrast, brightness/contrast-only `ColorJitter`, and standard RGB Normalize paths can collapse into color-matrix segments.                                 |
| Crop-resize            | Registered `RandomResizedCrop` has a dedicated segment; a preceding affine run can be absorbed on Kornia/TorchVision torch paths.                                                 |
| Flexible construction  | Native numeric ranges, Kornia transforms, TorchVision transforms, Albumentations transforms, or a mixed-backend list.                                                             |
| Portable configuration | Frozen `TransformSpec` values resolve a declarative pipeline against a chosen backend with strict unsupported-operation handling.                                                 |
| Auxiliary coordinates  | Masks, dense xyxy/xywh boxes, and dense keypoints can follow supported fused matrices, subject to the safety limits below.                                                        |
| NumPy bridges          | HWC/BHWC NumPy ↔ BCHW torch converters and NumPy output are available; conversion to NumPy detaches and moves data to CPU.                                                        |
| Execution controls     | Albumentations cv2 or torch execution, optional torch compilation on non-CPU paths, Kornia-dependent downscale antialiasing, interpolation, padding, and color clipping policies. |
| Plan inspection        | Human-readable plans, structured descriptors, a warp-saving estimate, and the last matrix-producing segment are exposed.                                                          |
| Training integration   | Pipelines are `nn.Module` objects and have tested pickle/serialization paths for common worker use.                                                                               |

### Built-in live-transform coverage

This is an allowlist, not a claim about every upstream transform:

| Backend           | Registered geometry                                                                                                   | Registered linear color                                                                             | Crop-resize                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Kornia            | `RandomRotation`, `RandomAffine`, `RandomShear`, `RandomTranslate`, H/V flip, `RandomRotation90`, `RandomPerspective` | `RandomBrightness`, `RandomContrast`, brightness/contrast-only `ColorJitter`, 3-channel `Normalize` | `RandomResizedCrop`                                        |
| TorchVision v1/v2 | `RandomRotation` (`expand=False`), `RandomAffine`, H/V flip, `RandomPerspective`                                      | Brightness/contrast-only `ColorJitter`, 3-channel `Normalize`                                       | `RandomResizedCrop`                                        |
| Albumentations    | `Affine`, `Rotate`, `SafeRotate`, `ShiftScaleRotate`, H/V flip, `RandomRotate90`, `D4`, `Transpose`, `Perspective`    | `RandomBrightnessContrast`, standard 3-channel `Normalize`                                          | `RandomResizedCrop` with backend-specific execution limits |
| Native/direct     | Rotation, scale, x/y shear, x/y translation, H/V flip                                                                 | Brightness, contrast                                                                                | Not available                                              |

Unknown and nonlinear operations generally become passthrough barriers. That preserves pipeline construction in many image-only cases, but passthrough is not automatically numerically transparent, device-efficient, or auxiliary-target safe.

## 📦 Install

```bash
pip install fuse-augmentations
```

The base package requires Python 3.10+ and PyTorch 2.2+. It includes the native/direct builder.

Install optional adapter ecosystems only when needed:

```bash
pip install "fuse-augmentations[kornia]"
pip install "fuse-augmentations[torchvision]"
pip install "fuse-augmentations[albumentations]"
pip install "fuse-augmentations[all]"
```

## 🚀 Quick start: no optional augmentation backend

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(7)

augment = Compose.from_params(
    rotation=(-15.0, 15.0),
    scale=(0.9, 1.1),
    shear_x=(-4.0, 4.0),
    translate_x=(-8.0, 8.0),
    hflip_p=0.5,
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(4, 3, 128, 128, dtype=torch.float32)
augmented, matrix = augment(images, return_matrix=True)

assert augmented.shape == images.shape
assert augmented.dtype == images.dtype
assert matrix is not None
assert matrix.shape == (4, 3, 3)

print(augment.fusion_plan)
print(
    [
        (descriptor.kind, descriptor.n_warps_saved)
        for descriptor in augment.fusion_plan_descriptors
    ]
)
```

<details>
<summary>Fusion plan and saved warp count for the quick-start pipeline</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
[('fused', 1)]
```

</details>

The common fused input contract is a floating BCHW torch tensor. Both `fuse_augmentations` and the shorter `fuse_aug` import expose the same public objects.

## 🔌 Bring an existing backend pipeline

```python
import torch
import torchvision.transforms.v2 as T

from fuse_augmentations import Compose, ReorderPolicy

augment = Compose(
    [
        T.RandomRotation(15.0),
        T.RandomAffine(degrees=0.0, scale=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
    ],
    reorder=ReorderPolicy.NONE,
)

output = augment(torch.rand(8, 3, 224, 224))
```

Kornia and Albumentations transform objects follow the same BCHW entry point when used in tensor pipelines. Albumentations also has an image-only HWC NumPy compatibility call, but it is not a replacement for native multi-target dictionaries and processors.

Mixed-backend pipelines are supported, with every backend change acting as a hard fusion boundary.

## 📊 What the measurements say

These numbers are from the 2026-07-12 local audit on macOS arm64, `fuse-augmentations 0.9.0.dev0`, Python 3.12, PyTorch 2.10, 256×256 inputs. They demonstrate the shape of the opportunity—not a release-wide promise.

| Measurement                          |                                             Observed result | Interpretation                                                                             |
| ------------------------------------ | ----------------------------------------------------------: | ------------------------------------------------------------------------------------------ |
| Fixed 45-case CPU, batch-1 score     |            **1.79×** geometric mean of native/fused latency | Real aggregate gain for the synthetic bank; mixed cases use opt-in reordering.             |
| Five-op geometric chain, CPU batch 1 |   **6.52× Kornia**, **14.48× TorchVision** in one quick run | Long geometric chains are the strongest fit; quick-run effect sizes are noisy.             |
| Sampled CPU tensor peak memory       |                           Lower in **12/12** compared pairs | Fewer intermediate warped tensors can reduce peak; allocation count improved in only 8/12. |
| TorchVision 3-op, CPU batch 8 peak   |                        **117.5 MB → 38.0 MB** (~3.1× lower) | One profiler result, not a universal memory ratio.                                         |
| Apple MPS quick sweep                | Faster in only **9/28** comparable Kornia/TorchVision pairs | GPU-capable does not mean faster; TorchVision MPS often regressed here.                    |
| CUDA                                 |                                              **Not tested** | No CUDA performance claim is made by this audit.                                           |

Single operations can be slower because there is no resampling to eliminate. Color-heavy pipelines retain color cost. Small accelerator workloads can be dominated by launch, sampling, compilation, or conversion overhead. Always benchmark the exact production pipeline and keep a correctness/parity gate beside the timing.

## 🔬 Quality and semantics

Fusing geometry changes *when* interpolation happens. That can preserve more detail than repeatedly warping an already warped image, but it also means the result is not a byte-for-byte substitute for the native chain.

For research or parity-sensitive use:

- set `ReorderPolicy.NONE` explicitly;
- pair or record sampled parameters and matrices;
- compare output coordinates and task metrics, not only shapes;
- report backend, device, versions, dtype, image size, and batch size;
- separate compile warmup from steady-state timing;
- publish losses and skips alongside wins.

`POINTWISE` and `AGGRESSIVE` reordering are behavior-changing optimizations: moving color across geometry can alter border and clipped pixels. `AGGRESSIVE` currently behaves like `POINTWISE`.

## 🎯 Auxiliary targets: supported, but narrowly

Registered fused geometry can route:

- masks as BCHW tensors with nearest or bilinear sampling;
- boxes as dense `(B, N, 4)` xyxy or xywh tensors;
- keypoints as dense `(B, N, 2)` tensors.

Important boundaries:

- unknown spatial transforms may desynchronize image and targets;
- mask padding is always zero even when image padding is border/reflection;
- nearest masks are intentionally detached; bilinear requires floating soft masks;
- boxes are AABB-wrapped after rotation but are not clipped or filtered;
- labels, visibility, ragged instances, invalid-box removal, and keypoint validity are application responsibilities;
- the Albumentations HWC NumPy path is image-only.

For detection and segmentation, validate every transform class and warning before training.

## 🔍 Introspection without overreading it

- `fusion_plan` describes the current segment structure.
- `fusion_plan_descriptors` provides structured, serializable segment metadata.
- `n_warps_saved` is a plan estimate, not a literal native interpolation counter for exact operations.
- `return_matrix=True` and `transform_matrix` expose the **last matrix-producing segment**, not an automatic whole-pipeline matrix across backend, projective, crop, or passthrough boundaries.

Use per-call matrix return when output and transform provenance must stay paired.

## 🧭 Where it fits

Use `fuse-augmentations` when:

- data is already a BCHW torch tensor;
- the pipeline contains several registered geometric transforms;
- repeated resampling is a fidelity, memory, or throughput concern;
- you can validate output semantics for the exact backend and task.

Prefer the native backend container when you require:

- PIL or unbatched CHW input;
- full Albumentations dictionary processors;
- exact native centers, fills, pixels, hooks, or RNG behavior;
- unsupported spatial transforms with masks, boxes, or keypoints;
- backend-specific per-transform interpolation and padding semantics.

## 📚 Documentation

The repository includes a complete MkDocs Material site:

- [Overview](https://borda.github.io/fuse-augmentations/)
- [Installation and backend-free quickstart](https://borda.github.io/fuse-augmentations/getting-started/quickstart/)
- [How fusion works](https://borda.github.io/fuse-augmentations/concepts/how-fusion-works/)
- [Exact capabilities](https://borda.github.io/fuse-augmentations/concepts/capabilities/)
- [Backend and configuration guides](https://borda.github.io/fuse-augmentations/guides/backend-pipelines/)
- [Auxiliary-target safety](https://borda.github.io/fuse-augmentations/guides/auxiliary-targets/)
- [Reproducibility](https://borda.github.io/fuse-augmentations/guides/reproducibility/)
- [Quality and benchmark evidence](https://borda.github.io/fuse-augmentations/research/benchmarks/)
- [Research methodology](https://borda.github.io/fuse-augmentations/research/methodology/)
- [Known limitations](https://borda.github.io/fuse-augmentations/known-limitations/)
- [FAQ](https://borda.github.io/fuse-augmentations/faq/)
- [Application walkthroughs](https://borda.github.io/fuse-augmentations/applications/)
- generated references for the notable public API

The site configuration provides local search, per-page descriptions, canonical/Open Graph metadata, sitemap and crawler files, an `llms.txt` agent index, and GitHub Pages publication automation.

## 🧪 Reproduce the evidence

Benchmark and memory scripts live in [`experiments/`](https://github.com/Borda/fuse-augmentations/tree/main/experiments). They expose cases where fusion loses as well as wins.

```bash
uv run --all-extras --group benchmark python experiments/optimize_score.py
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py --quick
uv run --all-extras --group benchmark python experiments/bench_memory.py --quick
```

Treat quick runs as smoke evidence. Release-grade comparisons need independent processes, uncertainty intervals, paired RNG state, output-parity assertions, and full environment provenance.

## 🤝 Contributing

Bug reports and focused pull requests are welcome. Open an issue before a public API or architecture change.

Documentation example authoring and generated-test instructions are in [`CONTRIBUTING.md`](.github/CONTRIBUTING.md). Generated documentation tests are recreated in CI and should not be committed.

Build the docs locally with:

```bash
uv sync --group docs
uv run --group docs mkdocs build --strict
```

## 📄 License

[Apache-2.0](https://github.com/Borda/fuse-augmentations/blob/main/LICENSE) © 2025–2026 Jiri Borovec.
