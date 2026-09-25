---
title: Applications and use cases
description: Choose between synthetic dataset generation and fused augmentation, then follow the per-task recipe for synthetic experiments, classification, segmentation, detection, test-time augmentation, and performance planning.
---

# Applications and use cases

`vision-synth` ships two independent capabilities, and most questions belong clearly to one of them.

| You want to                                                      | Start here                                                  |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| Produce labelled images without collecting or annotating data    | [Synthetic data experiments](synthetic-data-experiments.md) |
| Cut repeated resampling out of an existing augmentation pipeline | [Fused augmentation](#fused-augmentation)                   |
| Feed generated samples into a fused pipeline                     | [Generate and augment](generate-and-augment.md)             |

## Synthetic data generation

`synth_datasets` draws labelled shapes and exports COCO or YOLO for detection, segmentation, oriented boxes, or keypoints. It needs no source images and no `torch` install. It generates data; your application owns training and evaluation, so a synthetic run answers questions about your pipeline rather than about real-image accuracy.

- [Synthetic data experiments](synthetic-data-experiments.md) — pick the experiment that answers your question, and decode the exported labels correctly.
- [Synthetic data generation](../datasets/index.md) — the full guide, with runnable recipes for every task and format.

## Fused augmentation

The augmentation engine solves one problem: a chain of registered geometric transforms resamples the image once per transform, and each resampling costs time and image detail. Fusing the chain into a single warp removes the repeats.

That framing decides whether it fits your pipeline.

| Your pipeline                                                                   | Verdict                                                                                  |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Three or more consecutive registered geometric transforms on BCHW float tensors | Strong fit — this is the measured case                                                   |
| One geometric transform, or geometry separated by color operations              | Weak fit — there is nothing to fuse; see [Performance planning](performance-planning.md) |
| Mostly nonlinear color, blur, erasing, or elastic deformation                   | Poor fit — those are pointwise or non-affine and never merge                             |
| Unregistered spatial transforms alongside masks, boxes, or keypoints            | Unsafe — see [Known limitations](../known-limitations.md)                                |
| PIL input, Albumentations dictionaries, or exact native pixel parity required   | Wrong tool — use the native backend container                                            |

### Per-task recipes

- [Classification](classification.md) — the lowest-risk application, because the label carries no spatial coordinates.
- [Segmentation and dense targets](segmentation.md) — routing masks and continuous image targets through the same geometry without desynchronizing them.
- [Detection and keypoints](detection-and-keypoints.md) — what the package does to coordinate tensors, and the postprocessing you still own.
- [Test-time augmentation](test-time-augmentation.md) — inverting a fused segment to map predictions back to the original frame.
- [Performance planning](performance-planning.md) — read the fusion plan and predict the win before you benchmark.

## What the evidence supports

The [benchmarks](../research/benchmarks.md) retain a historical 1.7861x fixed-bank score across 168 timed CPU variants, with geometric chains reaching far higher ratios and a published regression at TorchVision batch 32. These are dated smoke measurements rather than current-head guarantees; the memory section and unpaired/model-endpoint comparisons remain withdrawn pending corrected tools. Fewer resampling passes is structural — it follows from the plan, not from timing — but faster wall-clock is not: it depends on device, shape, batch size, dtype, and transform mix.

Image quality is argued from the resampling count and visual overlays. Whether fusion changes downstream task metrics is not measured in this repository; treat that as an open question in your own ablation rather than a property of the package. The [research methodology](../research/methodology.md) is the checklist for running that comparison honestly.

### When not to fuse

Choose the native backend container when you require PIL/CHW input, complete Albumentations dictionary processors, exact native pixels, per-transform fill and interpolation semantics, segment hooks, unregistered spatial transforms with targets, or a backend-specific random-number stream.

A smaller native pipeline is better than an unsafe fused one.
