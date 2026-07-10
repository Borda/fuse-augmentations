# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pluggable adapter registry: public `register_adapter()` plus the `fuse_augmentations.adapters` entry-point group (experimental); `Compose.supported_ops(backend)` and `Compose.capability_matrix()` report config-time op coverage, and `from_config` aggregates all invalid specs in one error.
- Exact execution for composed flip / quarter-turn (90°/180°/270°) chains: dispatched via `tensor.flip`/`rot90` with zero interpolation error; auxiliary targets (masks, boxes, keypoints) fall back to the grid path automatically instead of raising.
- Crop+resize fusion: a geometric chain followed by `RandomResizedCrop` now fuses into a single warp at the target output size.
- `execution="cv2" | "torch"` flag on `Compose` for fused Albumentations segments: `"cv2"` (default) keeps per-sample cv2 warps bit-identical to earlier releases; `"torch"` opts into one batched `grid_sample` per segment (batch-size-independent throughput, native GPU/MPS execution).
- Multi-target `data_keys` with Albumentations fused segments: masks, bounding boxes, and keypoints are routed through the composed pixel matrix (previously a construction-time `ValueError`).
- Albumentations-style keyword calls on multi-target pipelines (`pipe(image=..., mask=..., bboxes=...)`) return a dict keyed by the caller's keyword names; the positional tuple API is unchanged. Colliding keyword aliases raise `ValueError`.
- `output_backend="numpy"` now converts each convertible target of a multi-target output (image, mask); coordinate targets remain tensors.
- A `UserWarning` is emitted when a passthrough (non-fusible) transform runs in a multi-target forward, since auxiliary targets skip passthrough segments and geometric passthrough ops (e.g. elastic/grid distortion) would silently misalign them.

### Fixed

- Corrupt rotation matrix in the TorchVision batch-size-1 CPU cv2 fast path (`sin` computed as `cos`).
- `from_params(scale=...)` now draws a single isotropic factor shared by both axes, as documented; explicit `scale_x`/`scale_y` keep independent draws.
- cv2 `"reflection"` padding now maps to `BORDER_REFLECT_101`, matching torch `grid_sample(padding_mode="reflection", align_corners=True)`.
- Bounding-box zero-`w` guard uses `finfo.eps` (the previous `finfo.tiny` clamp overflowed float32 to `inf`).
- Near-singular affine matrices raise consistently across the torch and cv2 inversion paths; eager and `torch.compile` branches of `inv3x3` share one threshold.
- cv2 fast-path activation gates respond to `torch.manual_seed`; Albumentations segment `forward` no longer consumes RNG draws for inactive transforms.
- `uint16` NumPy inputs are normalised to `[0, 1]` (previously cast without rescaling).
- Albumentations native dict path raises instead of silently dropping non-image keys; unrecognised transforms are rejected in Albumentations-backed pipelines.
- `transform_matrix` resets to `None` at every forward, so exact/passthrough-only calls no longer report a stale matrix.
- `fuse_aug.__version__` is exported, matching `fuse_augmentations.__version__`.

### Changed

- `same_on_batch=True` on Albumentations-backed fused segments now shares the sampled parameters across the batch, not just the activation decision.
- Documented color-fusion accuracy caveats (final-only clamping; fixed 0.5 contrast midpoint) and the seeding contract limits between warp backends.

## [0.7.0] and earlier

No changelog was kept prior to 0.8.0 development; see the git history for details.
