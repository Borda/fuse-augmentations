---
title: Image augmentation applications and use cases
description: Decide whether fused augmentation fits your task, then follow the per-task recipe for classification, segmentation, detection, test-time augmentation, and performance planning.
---

# Applications and use cases

This package solves one problem: a chain of registered geometric transforms resamples the image once per transform, and each resampling costs time and image detail. Fusing the chain into a single warp removes the repeats.

That framing decides whether it fits your pipeline.

| Your pipeline                                                                   | Verdict                                                                                  |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Three or more consecutive registered geometric transforms on BCHW float tensors | Strong fit — this is the measured case                                                   |
| One geometric transform, or geometry separated by color operations              | Weak fit — there is nothing to fuse; see [Performance planning](performance-planning.md) |
| Mostly nonlinear color, blur, erasing, or elastic deformation                   | Poor fit — those are pointwise or non-affine and never merge                             |
| Unregistered spatial transforms alongside masks, boxes, or keypoints            | Unsafe — see [Known limitations](../known-limitations.md)                                |
| PIL input, Albumentations dictionaries, or exact native pixel parity required   | Wrong tool — use the native backend container                                            |

## Per-task recipes

- [Classification](classification.md) — the lowest-risk application, because the label carries no spatial coordinates.
- [Segmentation and dense targets](segmentation.md) — routing masks and continuous image targets through the same geometry without desynchronizing them.
- [Detection and keypoints](detection-and-keypoints.md) — what the package does to coordinate tensors, and the postprocessing you still own.
- [Test-time augmentation](test-time-augmentation.md) — inverting a fused segment to map predictions back to the original frame.
- [Performance planning](performance-planning.md) — read the fusion plan and predict the win before you benchmark.

## What the evidence supports

The [benchmarks](../research/benchmarks.md) report a 1.7861x fixed-bank score across 168 timed CPU variants, with geometric chains reaching far higher ratios and a published regression at TorchVision batch 32. Fewer resampling passes is structural — it follows from the plan, not from timing — but faster wall-clock is not: it depends on device, shape, batch size, dtype, and transform mix.

Image quality is argued from the resampling count and visual overlays. Whether fusion changes downstream task metrics is not measured in this repository; treat that as an open question in your own ablation rather than a property of the package. The [research methodology](../research/methodology.md) is the checklist for running that comparison honestly.

## When not to use this package

Choose the native backend container when you require PIL/CHW input, complete Albumentations dictionary processors, exact native pixels, per-transform fill and interpolation semantics, segment hooks, unregistered spatial transforms with targets, or a backend-specific random-number stream.

A smaller native pipeline is better than an unsafe fused one.
