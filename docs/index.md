---
title: Image augmentation and synthetic computer vision datasets
description: Fuse compatible image augmentations and generate synthetic COCO or YOLO datasets for prototyping, convergence checks, and controlled difficulty experiments.
---

# Synthetic vision datasets and fused image augmentation

`vision-synth` generates labelled synthetic computer vision datasets and fuses compatible PyTorch image augmentations. Use generated shapes to prototype a model, check its training pipeline, and explore controlled changes in scene difficulty.

## Generate a synthetic dataset

Start here when you need **COCO or YOLO training data** for **object detection**, **instance segmentation**, **oriented bounding boxes (OBB)**, or **keypoint / pose estimation**. Generate primitives, animal silhouettes, symbols, or letters with reproducible seeds and configurable backgrounds, object sizes, clutter, occlusion, and camera effects. No source images or optional augmentation backend is required.

```bash
pip install vision-synth
```

- [Generate your first dataset](datasets/index.md#quickstart) — one Python call, images and labels, train/validation/test splits.
- [Choose a task and output format](datasets/index.md#choose-your-computer-vision-task) — copy a recipe for your trainer.
- [Check convergence and scale difficulty](datasets/prototyping.md) — fixed tiny sets, held-out evaluation, and controlled experiments.
- [Stream into a DataLoader](datasets/outputs.md#in-memory-streaming-and-training-feed) — generate samples without exporting files.

The generator draws synthetic shapes; your application supplies the model and training loop. A passing synthetic experiment checks that setup, while real-world accuracy requires real held-out data.

## Fuse compatible augmentations

`vision-synth` is a PyTorch-based matrix-fusion engine for image augmentation pipelines. It recognizes a finite set of Kornia, TorchVision, and Albumentations transforms—or builds a pipeline directly from numeric ranges—then composes compatible transforms so a geometric chain can use fewer interpolation passes. This part of the package needs the `torch` extra: `pip install "vision-synth[torch]"` — see [Install](getting-started/installation.md).

The strongest use case is a BCHW tensor pipeline with several consecutive, registered geometric transforms. Reducing repeated resampling can preserve more image detail, lower peak tensor memory, and accelerate long CPU chains.

!!! tip "This is not a general drop-in Compose replacement"

    Native PIL, CHW, Albumentations dictionary, target-processing, fill, center, random-number, and hook contracts are not preserved in general. Use the package as a tensor-first fusion engine and validate the exact pipeline you intend to ship.

!!! warning "Auxiliary targets require an allowlisted pipeline"

    With `data_keys` present, an unknown or unclassified spatial transform is rejected before any segment executes, preventing silent image/target divergence. Image-only calls may still use native passthrough; review every `Unknown ... SPATIAL_KERNEL barrier` warning. See [Known limitations](known-limitations.md).

## What is verified

### Synthetic generation

- Four shape vocabularies — geometric primitives, animal silhouettes, symbols, and letters — export COCO or YOLO labels for detection, instance segmentation, oriented boxes, and keypoints.
- Both writers convert point fields back to pixel-edge space at the file boundary, so an exported COCO or YOLO file carries one coordinate convention throughout.
- Background, clutter, and degradation knobs each draw from a side stream of their own, so switching one on at a fixed seed cannot move an object.
- Direct generation and YOLO export keep sample storage bounded and need only Pillow and NumPy — no torch, no source images.
- Output formats, shape families, and keypoint schemas are registered surfaces: `register_writer` adds a format, and new families can be registered.

### Fused augmentation

- Registered consecutive affine transforms from one backend can be represented by a composed matrix and applied in one resampling pass.
- Exact discrete operations such as supported flips have lossless execution paths.
- Projective-to-projective chains, supported linear color chains, and some affine-to-crop paths have dedicated fusion strategies.
- Kornia, TorchVision, and Albumentations transforms can share one pipeline, with backend changes acting as hard fusion boundaries.
- A native builder supports rotation, scale, shear, translation, flips, brightness, and contrast without an optional augmentation backend.
- The pipeline exposes a structured fusion plan and per-call matrix access for inspection.

## What is conditional

### Synthetic generation

- **Realism:** objects are drawn shapes. A synthetic result measures your pipeline, not accuracy on photographs or production data.
- **Difficulty:** there is no `difficulty=` argument and no curriculum scheduler. Difficulty is a combination of ordinary `SyntheticConfig` fields, and the published bands are ranked by training-free image statistics rather than by model performance.
- **Keypoints:** only animals, symbols, and letters carry a landmark schema; geometric primitives have none, and one dataset uses one family.
- **Reproducibility:** a seed reproduces generation within one environment. Record the configuration, package versions, and any source background files as well; changing `num_images` or the split ratios can move samples between splits.

### Fused augmentation

- **Speed:** long CPU geometric chains are the clearest win; single operations, mixed pipelines, and sampled TorchVision batch-8/32 workloads can be slower.
- **Native parity:** fewer resampling passes deliberately change numerics, and TorchVision center/fill/interpolation behavior is not generally pixel-equivalent.
- **Targets:** only registered and explicitly handled spatial operations are safe for multi-target routing.
- **Reordering:** `POINTWISE` and `AGGRESSIVE` can change pixels because border handling and clipping make operation order observable.
- **Accelerators:** device execution is supported on torch paths. The published evidence includes a July 12, 2026 CPU run and a separate September 5, 2026 historical CUDA sweep; current-head CUDA/MPS measurements and runner availability remain unverified. See the [benchmark record](research/benchmarks.md) before making a device claim.

## Choose your route

| Goal                                              | Start here                                           |
| ------------------------------------------------- | ---------------------------------------------------- |
| Try the package with only its base dependencies   | [Quickstart](getting-started/quickstart.md)          |
| Bring an existing backend pipeline                | [Backend pipelines](guides/backend-pipelines.md)     |
| Build one configuration for several backends      | [Declarative configuration](guides/configuration.md) |
| Route masks, boxes, or keypoints                  | [Auxiliary targets](guides/auxiliary-targets.md)     |
| Evaluate image quality or performance             | [Research guide](research/quality-and-fidelity.md)   |
| Check an exact supported transform or restriction | [Capabilities](concepts/capabilities.md)             |
| Understand known unsafe or approximate behavior   | [Known limitations](known-limitations.md)            |
| Inspect signatures and docstrings                 | [API reference](reference/core.md)                   |
| Prototype a labelled vision pipeline              | [Synthetic datasets](datasets/index.md)              |

## Evidence standard

This documentation separates structural guarantees from measurements. Warp reduction follows from the fusion plan; speed and memory are measured properties of a particular backend, device, batch, shape, dtype, and transform mix. Benchmark pages publish losses and measurement blind spots alongside wins.

The package is currently classified **Beta**. Treat advanced segment classes and the third-party adapter extension point as provisional.
