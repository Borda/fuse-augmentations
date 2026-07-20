---
title: Frequently asked questions
description: Direct answers about fuse-augmentations compatibility, speed, GPU support, numerical parity, target safety, randomness, NumPy I/O, and introspection.
---

# Frequently asked questions

## Is `fuse-augmentations` a drop-in replacement for native Compose classes?

No. It accepts supported Kornia, TorchVision, and Albumentations transform objects through its own tensor-first API. The normal fused path expects floating BCHW PyTorch tensors. It does not reproduce every native input type, target processor, hook, metadata, or output contract.

Use “Compose-like tensor pipeline” as the mental model, not “behaviorally identical container.”

## What does the package actually optimize?

It composes compatible geometric matrices inside a same-backend segment and applies the segment with one interpolation pass. It can also combine supported linear RGB color operations into fewer matrix multiplications. Exact operations such as flips use lossless paths.

Fusion stops at backend changes, unsupported operations, many passthroughs, and affine/projective boundaries.

## Will it make my training pipeline faster?

Possibly, but not universally. Speed depends on batch size, resolution, device, transform mix, passthrough operations, warmup, and backend. Fewer resampling passes can improve image quality without improving wall-clock time.

Benchmark the complete production pipeline on the target hardware. Separate first-call compilation time from steady-state timing, synchronize accelerators, and compare against the exact native baseline you would otherwise deploy.

## Does it run on CUDA or Apple MPS?

Torch-backed fused segments can run on CUDA and MPS. That does not mean every pipeline stays on device: a CPU/native passthrough can introduce transfers. Inspect `fusion_plan` and benchmark end to end.

`compile=True` is experimental and environment-dependent on non-CPU devices; it is a no-op on CPU. There is no universal CUDA/MPS speedup guarantee.

## Is fused output pixel-identical to the source backend?

No general parity guarantee exists. A fused pipeline intentionally replaces multiple warps with one, and execution can differ in coordinate centers, interpolation, borders, clipping, and random draw order.

TorchVision geometry has a known center/`align_corners` difference. The Albumentations torch execution strategy differs numerically from OpenCV. Color `clip_policy="final"` differs from native per-operation clipping when intermediate values leave the image gamut.

Validate task metrics and representative images, not only output shapes.

## Are all Kornia, TorchVision, and Albumentations transforms supported?

No. Each adapter has a finite registry. Registered operations may fuse; unregistered operations generally become passthrough barriers, warn, or are refused. Some configurations of registered classes are unsupported, including TorchVision `RandomRotation(expand=True)`.

Use the capability documentation and inspect `fusion_plan_descriptors` after construction. Do not infer support from an upstream class merely existing.

## Is an `Unknown ... SPATIAL_KERNEL barrier` warning safe to ignore?

Not in a pipeline with masks, boxes, or keypoints.

!!! danger "Unknown spatial transforms can corrupt training targets"

```
`RandomCrop`, `CenterCrop`, and `Resize` can change the image while an
auxiliary mask remains unchanged. The runtime refusal list is not exhaustive.
Treat every unknown transform as unsafe with `data_keys` until you have
independently proved it preserves coordinates.
```

For image-only pipelines, the warning still means fusion stopped and native passthrough behavior applies.

## Which auxiliary targets are supported?

The tensor `data_keys` path supports masks, dense xyxy or xywh boxes, and dense keypoints for registered geometric operations. See [Auxiliary targets](guides/auxiliary-targets.md) before using it for detection or segmentation.

It does not provide a full detection processor: no box clipping, visibility or area filtering, class-label propagation, variable-length batching, or keypoint visibility policy is included.

## Why do masks contain zeros near a transformed boundary?

Mask sampling always uses zero padding. This is independent of the image's `padding_mode`. If label `0` is not a valid background or ignore label, remap labels around augmentation or avoid out-of-bounds sampling.

## Can I backpropagate through a transformed mask?

The default nearest mask path is deliberately detached and has no `grad_fn`. Use `mask_interpolation="bilinear"` with a floating soft mask when gradients through mask values are required. Bilinear interpolation mixes neighboring values and must not be used for integer class IDs.

## Are boxes clipped to the output image?

No. Boxes are transformed and axis-aligned, but coordinates can remain negative or exceed image bounds. Clip and filter them after augmentation, and apply the same validity mask to labels and instance metadata.

## Can I pass Albumentations `image=...`, `mask=...` NumPy dictionaries?

No. The native HWC NumPy compatibility path is image-only. For auxiliary targets, use BCHW tensors with `data_keys`, and accept the package's target contract. Use native Albumentations when its dictionary target processors are a requirement.

## What does `output_backend="numpy"` return?

Images and masks become channel-last NumPy arrays on CPU. A batch of one is squeezed to HWC; larger batches are BHWC. Conversion detaches autograd and moves device data to CPU. Boxes and keypoints remain PyTorch tensors because image layout conversion does not apply to coordinates.

## How should I seed a pipeline?

For backend-free, Kornia, and TorchVision paths, control PyTorch randomness and keep batch shape/order fixed. Albumentations fused geometry also needs global NumPy activation gates and transform-internal parameter generators seeded.

See [Reproducibility](guides/reproducibility.md) for single-process and DataLoader examples.

## What does `randomness="backend"` guarantee?

It preserves the backend's intended batch sampling style where the adapter can represent it. It does not guarantee the same random stream, parameters, or pixels as calling the native container.

`randomness="per_sample"` requests independent samples where the adapter has a canonical per-sample sampler. The backend-free direct path is already per-sample, so this option does not change its parameter sampling.

## Does `ReorderPolicy.POINTWISE` preserve my declared operation order?

No. It moves pointwise/color operations across geometric operations to enlarge fusion windows. Padding, interpolation, and clipping can make the reordered result materially different.

Use `ReorderPolicy.NONE` for order-sensitive or parity-oriented experiments. Remember that `from_params` and `from_config` currently default to `POINTWISE`.

## What does `transform_matrix` represent?

It is the last matrix-producing affine or projective segment from the most recent call. It is not a whole-pipeline matrix across multiple segments, backend boundaries, passthroughs, or affine/projective boundaries. It can be `None` for exact-only or passthrough-only pipelines.

Use `return_matrix=True` to retrieve the same last-segment matrix with the call. The property is mutable per-call state, so do not read it concurrently from a shared pipeline instance.

## Is `n_warps_saved` a measured speedup?

No. It is a planning heuristic. It does not measure runtime, transfers, or native interpolation calls exactly. Exact flips can contribute even though native flips are already non-interpolating.

## When should I use `antialias=True`?

Use it experimentally for aggressive crop-resize downscales and validate image quality and speed. It depends on Kornia; without Kornia it falls back to the unfiltered warp. The antialias decision is batch-global, so one aggressively downscaled sample can filter the entire batch.

## Can I undo augmentation on predictions?

Yes, with `pipe.inverse(prediction, matrix=matrix)`, where `matrix` is the matrix returned by the exact paired `forward(..., return_matrix=True)` call. Each `inverse` call must be paired with its own forward call's matrix; do not read the mutable `transform_matrix` property instead.

This supports one fused affine or projective segment only. It raises for crop-resize, color/LUT/blur or passthrough segments, exact-only segments, and multi-segment pipelines, since `return_matrix` records only the last segment's matrix. Boxes are axis-aligned, so a forward-then-inverse box is exact only for axis-aligned transforms and inflates under rotation, shear, or a projective warp.

See [Introspection](guides/introspection.md) and the README's [Test-time de-augmentation](https://github.com/Borda/fuse-augmentations#test-time-de-augmentation) example.

## When should I keep the native backend instead?

Keep the native pipeline when you require PIL/TVTensor/native dictionary input, native target processors, exact native pixels or RNG streams, unsupported spatial transforms, or an upstream option the adapter does not preserve.

Use `fuse-augmentations` when BCHW tensor input, registered transforms, fused numerics, and measured task-level results meet your requirements. Review [Known limitations](known-limitations.md) before production use.
