---
title: Known limitations and safety boundaries
description: Verified compatibility, target-safety, numerical-parity, randomness, GPU, and performance limits of fuse-augmentations.
---

# Known limitations and safety boundaries

`fuse-augmentations` is a tensor-first matrix-fusion engine. It has a real, tested advantage: compatible geometric transforms in one segment can share a single interpolation pass. It is not a behaviorally identical replacement for every Kornia, TorchVision, or Albumentations pipeline.

!!! danger "Auxiliary targets can silently desynchronize"

    When `data_keys` contains a mask, boxes, or keypoints, treat any warning of the form `Unknown ... transform ... treating as SPATIAL_KERNEL barrier` as unsafe until you have proved that the transform leaves pixel coordinates unchanged.

    Common TorchVision transforms such as `RandomCrop`, `CenterCrop`, and `Resize` are not registered target-aware operations. They can change the image while the mask remains at its original shape. The runtime refusal list catches several named distortions, but it is not a complete detector for every spatial transform or custom callable.

    Use only explicitly supported geometric transforms in a multi-target pipeline. Otherwise, apply the operation through a native target-aware pipeline or transform every target yourself. See [Auxiliary targets](guides/auxiliary-targets.md).

## Compatibility at a glance

| Question                                          | Honest answer                                                                                                                                                                                      |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is `Compose` a drop-in native compose class?      | No. It accepts transform objects from supported backends through its own tensor-first contract.                                                                                                    |
| What is the normal image format?                  | A floating PyTorch tensor shaped `(B, C, H, W)`. Fused geometry does not generally accept PIL images or unbatched `(C, H, W)` tensors.                                                             |
| Is Albumentations NumPy input supported?          | Yes. `image=HWC_array` works on its own, and with `data_keys` declared the same call carries masks, boxes, keypoints and rotated boxes. Albumentations' label processors are still not replicated. |
| Are all upstream transforms fused?                | No. Built-in adapters use finite registries. Unknown transforms become passthrough barriers or are refused.                                                                                        |
| Does fused output equal native output?            | Not universally. Sampling, coordinate conventions, interpolation, padding, clipping, and operation order can differ.                                                                               |
| Does `transform_matrix` cover the whole pipeline? | No. It is the last matrix-producing affine or projective segment from the most recent call.                                                                                                        |
| Is every GPU or MPS configuration faster?         | No. Speed depends on device, batch, image size, operation mix, warmup, and passthrough transfers. Benchmark your exact pipeline.                                                                   |

## Input and backend limits

The main fused path expects BCHW tensors. Native compose classes accept broader families of inputs and metadata that this package does not reproduce:

- TorchVision pipelines may accept PIL images, unbatched tensors, TVTensors, and nested sample structures. Fused geometric segments require BCHW tensors.
- Albumentations' HWC NumPy keyword path accepts auxiliary targets only when the pipeline declares `data_keys`, and only for chains whose every segment publishes a matrix; a crop, an exact affine or an opaque passthrough sends the call back through tensors. Either way the targets are this package's contract, not Albumentations' label-processor semantics.
- Kornia's `AugmentationSequential` has container behavior and metadata contracts beyond the transform-object compatibility provided here.
- Multi-backend pipelines are supported for BCHW tensors, but a backend change creates a segment boundary. Transforms from different backends do not share one fused matrix.

TorchVision `RandomRotation(expand=True)` is explicitly unsupported. The fused TorchVision geometry uses an `align_corners=True` pixel convention with center `((W - 1) / 2, (H - 1) / 2)`, which differs from native TorchVision geometry. Fill, interpolation, and center behavior must be validated rather than assumed from the original transform object.

## Auxiliary-target limits

### Safe only within the declared contract

| Pipeline element                                          | Mask/box/keypoint behavior                               | Safety decision                                                               |
| --------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Registered affine/projective transform                    | Routes supported targets with the segment grid or matrix | Supported, subject to the numerical limits below                              |
| Registered exact flip or discrete operation               | Routes supported targets through exact or matrix logic   | Supported for the registered operation                                        |
| Registered `RandomResizedCrop`                            | Routes targets to the target spatial size                | Supported; output size changes                                                |
| Known pointwise or kernel passthrough                     | Leaves coordinate targets unchanged                      | Safe only when the operation truly preserves coordinates                      |
| Named coordinate-changing passthrough on the refusal list | Raises rather than silently misaligning targets          | Safe refusal, not target support                                              |
| Unknown transform or custom callable                      | May run on the image only                                | Unsafe with auxiliary targets unless independently proved geometry-preserving |

Masks and coordinates have additional contracts:

- Nearest-neighbor mask sampling uses zero padding even when the image uses `padding_mode="border"` or `"reflection"`. Label `0` must therefore be an acceptable out-of-bounds fill value.
- `padding_mode="reflection"` reflects about the outer pixel *centres* (OpenCV `BORDER_REFLECT_101`), following this package's `align_corners=True` sampling. An implementation sampling with `align_corners=False` reflects about the outer pixel *edges* (`BORDER_REFLECT`) and its mirrored band differs by a pixel of phase. `"zeros"` and `"border"` are convention-free; see "Half-pixel convention" in `guides/configuration.md`.
- A canvas thinner than two pixels on either axis is refused: the `align_corners=True` normalization divides by `L - 1`. A `(H, 1)` or `(1, W)` image raises naming the axis rather than warping through an infinite scale.
- The nearest mask path is deliberately executed without autograd. This removes gradients with respect to both the mask values and sampling grid. Bilinear sampling is available for floating soft masks, mixes values at boundaries, and keeps an autograd path.
- Boxes are dense `(B, N, 4)` tensors and keypoints are dense `(B, N, 2)` tensors. The package does not clip them to image bounds, filter invisible or zero-area boxes, calculate visibility, carry labels, or manage variable `N`.
- Rotated boxes are returned as axis-aligned bounding boxes, which can be larger than the rotated object.
- Coordinate targets remain PyTorch tensors when image and mask outputs are converted with `output_backend="numpy"`.

## Passthrough is not transparent

An unregistered transform can execute through its native backend, but "passthrough" does not mean identical data or behavior:

- It splits fusion segments and can add transfers or native calls.
- A coordinate-changing passthrough may be unsafe for auxiliary targets as described above.
- Albumentations passthrough on the tensor path receives HWC float data derived from the BCHW tensor. Operations designed around uint8 magnitude, including some noise, fog, and compression transforms, may behave differently or become ineffective.
- `substitute_passthrough=True` is explicitly behavior-changing. The current substitution can replace Albumentations Gaussian blur with a Kornia blur that has different kernels, borders, and randomness.

Two common Albumentations transforms produce a passthrough for reasons worth knowing before you plan a chain around them:

- `HueSaturationValue` always does. It is registered `POINTWISE`: reorderable, but non-linear in RGB, so it composes into neither a color matrix nor a per-channel lookup table. There is no fused segment for it and, in the current design, cannot be one. A chain of geometry plus `HueSaturationValue` therefore fuses the geometry and then leaves the tensor for the color step.
- `RandomResizedCrop` and `RandomSizedCrop` do so **only on image-only calls**. When the call carries a mask, box, or keypoint target, the crop routes through a `CropResizeSegment` and no passthrough appears. Without an auxiliary target the crop runs natively instead, which keeps it bit-exact against Albumentations at the cost of a segment break. This is a parity choice, not a missing registration — but it means the same chain plans differently depending on whether you declared `data_keys`, and the image-only plan is the slower one on an accelerator.

Kornia and TorchVision fuse a crop into the preceding affine as a single segment regardless of auxiliary targets, so a plan comparison across backends is not a like-for-like comparison of the same chain.

## Reordering changes semantics

`ReorderPolicy.POINTWISE` moves color and pointwise operations across geometric operations to create larger fusion groups. Pointwise math does not generally commute with finite-image warps because padding, interpolation, and clipping make order observable.

Use `ReorderPolicy.NONE` when declared order, native comparison, or experiment reproduction matters. `from_params` and `from_config` currently default to `POINTWISE`, so pass `reorder=ReorderPolicy.NONE` explicitly for parity-oriented work. `AGGRESSIVE` currently behaves like `POINTWISE`; it is not a stronger parity guarantee.

## Numerical parity and color limits

Fusing intentionally replaces multiple resampling steps with one. The result can be higher quality than sequential interpolation while still differing from the original backend output.

- TorchVision geometry has a known center/`align_corners` difference. Native pixel parity is not promised.
- Albumentations `execution="torch"` uses `grid_sample`; its border and subpixel weights differ from OpenCV. Use `execution="cv2"` when the OpenCV execution convention matters.
- Affine and projective transforms form separate segment types. Crossing between them can require another resampling pass.
- Default color `clip_policy="final"` clamps after the fused color run, whereas native chains may clamp after every operation.
- `clip_policy="per_op_parity"` improves parity for gamut-escaping chains but is still approximate for some contrast sequences. It can differ at roughly the `1e-2` scale in the known mean-relative contrast case.
- Fused color matrices are defined for three-channel RGB. Other channel counts fall back to sequential native execution.

## Randomness and reproducibility limits

`randomness="backend"` preserves the intended batch sampling style, not a guarantee of an identical random stream or identical native output.

Albumentations-backed fused geometry uses two random-number domains: package activation gates use global NumPy randomness, while transform parameters use the Albumentations transform's internal RNG. Seeding only `torch`, or only `numpy.random`, is insufficient. See [Reproducibility](guides/reproducibility.md).

Fast paths and different batch sizes can consume random draws differently. For strict experiments, record package/backend versions, batch size, execution strategy, reorder policy, seeds, and the machine-readable fusion plan.

## GPU, compilation, and antialiasing limits

`compile=True` is off by default and is a no-op on CPU. Non-CPU compilation is environment-dependent; the test suite does not establish a universal CUDA or MPS speedup, dynamic-shape guarantee, or compiler compatibility matrix. Measure warmup and steady state separately on the deployment host.

`antialias=True` is limited to aggressive crop-resize downscaling. It depends on Kornia's Gaussian blur and silently falls back to the unfiltered warp when Kornia is absent. The decision and blur strength are batch-global: one strongly downscaled sample can cause the whole batch to be filtered. Scale estimation also reads device values into Python, which can synchronize accelerator work.

Passthrough operations are particularly important on accelerators. A native CPU-only passthrough can erase the advantage of a fused GPU segment through device transfers. Inspect the plan and benchmark the complete pipeline, not only the fused warp.

That cost is now measured rather than asserted. On an NVIDIA L4, chains carrying exactly one passthrough ran 2.0x to 3.0x slower on CUDA than the same chains on the CPU engine, while chains with no passthrough tied or won — and the penalty grew with batch size, because `call_nonfused` copies the whole batch to the host and back once per passthrough per call. One passthrough is enough to erase a fused warp's advantage.

Fusion's value on an accelerator is also backend-dependent, and not in the direction the CPU results suggest. In the same run, fused Kornia beat native Kornia on multi-operation chains, while fused TorchVision lost to native TorchVision almost everywhere on CUDA despite planning to a single segment with no passthrough at all. The cause is fixed per-call host-side overhead rather than warp work: fused timings barely move with batch size (a single rotate cost 1.0 ms at batch 1 and 2.6 ms at batch 64), so on a device fast enough to make the warps nearly free, that fixed cost decides the comparison. Do not assume a CPU speedup transfers to a GPU; measure on the deployment device. See [Benchmarks](research/benchmarks.md#cuda-batch-sweep-september-5-2026).

## Introspection limits

`fusion_plan` and `fusion_plan_descriptors` describe segmentation. They are the right tools for detecting barriers and backend boundaries.

`transform_matrix` and `return_matrix=True` expose only the last matrix-producing segment. They do not compose across backend boundaries, passthrough barriers, separate affine/projective segments, exact-only segments, or multiple fused segments. The property is mutable per-call state and should not be read concurrently from a shared pipeline instance.

`n_warps_saved` is a planning heuristic, not a literal count of native interpolations or an observed speedup. In particular, exact flips can contribute to the metric even though native flips are already non-interpolating.

## Test-time inverse limits

`pipe.inverse(prediction, matrix=matrix)` maps a prediction back to the original geometric frame, but only for a pipeline that reduces to one fused affine or projective segment. It raises for crop-resize (cropped pixels are lost and cannot be recovered), color/LUT/blur or passthrough segments, exact-only segments, and multi-segment pipelines, because `return_matrix` records only the last segment's matrix.

The inverse is geometric-only. It cannot recover values discarded by interpolation or padding, and it does not undo color, LUT, or blur operations. Recovered boxes are axis-aligned, so a forward-then-inverse box is exact only for axis-aligned transforms (flip, scale, translation) and inflates under rotation, shear, or a projective warp. Always pass the matrix returned by the same `forward(..., return_matrix=True)` call rather than the mutable `transform_matrix` property.

## How to decide whether the package fits

Use `fuse-augmentations` when all of the following are true:

1. Your main path uses BCHW PyTorch tensors.
2. Your geometric transforms appear in the documented capability surface.
3. You accept fused-engine numerics instead of requiring native pixel identity.
4. Any auxiliary targets stay inside the explicitly supported routing contract.
5. A representative benchmark and task-quality check pass on your hardware.

Use the native backend, or keep a native reference pipeline, when input compatibility, native target processors, exact random streams, native pixel parity, or unsupported spatial transforms are requirements.
