---
title: Fewer resampling passes for image augmentation
description: Learn where fuse-augmentations safely composes image transforms, what it accelerates, and where its compatibility and parity limits matter.
---

# Fuse compatible augmentations, once

`fuse-augmentations` is a PyTorch-based matrix-fusion engine for image augmentation pipelines. It recognizes a finite set of Kornia, TorchVision, and Albumentations transforms—or builds a pipeline directly from numeric ranges—then composes compatible transforms so a geometric chain can use fewer interpolation passes.

The strongest use case is a BCHW tensor pipeline with several consecutive, registered geometric transforms. Reducing repeated resampling can preserve more image detail, lower peak tensor memory, and accelerate long CPU chains.

!!! warning "This is not a general drop-in Compose replacement" Native PIL, CHW, Albumentations dictionary, target-processing, fill, center, random-number, and hook contracts are not preserved in general. Use the package as a tensor-first fusion engine and validate the exact pipeline you intend to ship.

!!! critical "Auxiliary targets require an allowlisted pipeline" An unknown spatial transform can be passed through on the image while a mask, box, or keypoint tensor remains unchanged. Ordinary unsupported TorchVision crops and resize operations can therefore desynchronize targets. Treat every `Unknown ... SPATIAL_KERNEL barrier` warning as unsafe when `data_keys` is present. See [Known limitations](known-limitations.md).

## What is verified

- Registered consecutive affine transforms from one backend can be represented by a composed matrix and applied in one resampling pass.
- Exact discrete operations such as supported flips have lossless execution paths.
- Projective-to-projective chains, supported linear color chains, and some affine-to-crop paths have dedicated fusion strategies.
- Kornia, TorchVision, and Albumentations transforms can share one pipeline, with backend changes acting as hard fusion boundaries.
- A native builder supports rotation, scale, shear, translation, flips, brightness, and contrast without an optional augmentation backend.
- The pipeline exposes a structured fusion plan and per-call matrix access for inspection.

## What is conditional

- **Speed:** long CPU geometric chains are the clearest win; single operations, mixed pipelines, and sampled TorchVision batch-8/32 workloads can be slower.
- **Native parity:** fewer resampling passes deliberately change numerics, and TorchVision center/fill/interpolation behavior is not generally pixel-equivalent.
- **Targets:** only registered and explicitly handled spatial operations are safe for multi-target routing.
- **Reordering:** `POINTWISE` and `AGGRESSIVE` can change pixels because border handling and clipping make operation order observable.
- **Accelerators:** device execution is supported on torch paths, but the current full latency run provides CPU evidence only; CUDA and stable MPS latency still need measurement on the deployment host.

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

## Evidence standard

This documentation separates structural guarantees from measurements. Warp reduction follows from the fusion plan; speed and memory are measured properties of a particular backend, device, batch, shape, dtype, and transform mix. Benchmark pages publish losses and measurement blind spots alongside wins.

The package is currently classified **Alpha**. Treat advanced segment classes and the third-party adapter extension point as provisional.
