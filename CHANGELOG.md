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
- `Normalize` (Kornia, TorchVision v2, standard Albumentations) now fuses into the color matrix as a per-channel affine, deleting one full-tensor pass from pipelines that end in normalization; the final gamut clamp is suppressed for the normalized output (image-statistics Normalize modes remain passthrough).
- `clip_policy="final" | "per_op_parity"` on `Compose`: `"final"` (default, unchanged) clamps once after the fused color matmul; `"per_op_parity"` splits the fused color run wherever an intermediate would leave `[0, 1]`, matching a native per-op clamped chain.
- Opt-in `compile=True` on `Compose`: wraps the warp core (matrix normalize → `affine_grid` → `grid_sample`) in `torch.compile` on torch ≥ 2.2 (no-op otherwise and on CPU; default off, outputs unchanged).
- Opt-in `antialias=True` on `Compose`: crop-resize segments prefilter aggressive downscales (worst-axis scale < 0.5) before the single warp, removing aliasing; default off, outputs bit-identical.
- Opt-in `substitute_passthrough=True` on `Compose`: replaces registered non-fusible ops with an installed backend's torch-native equivalent (initially Albumentations `GaussianBlur` → Kornia `RandomGaussianBlur`) so GPU pipelines stay on-device; behaviour-changing and warns per substitution.
- Passthrough segments now cross the CPU boundary once per batch (one device-to-host and one host-to-device transfer per segment instead of per sample), with identical numerics.
- `fusion_plan` marks passthrough entries with `[CPU passthrough]` on non-CPU pipelines, and `fusion_plan_descriptors` carries machine-readable `split_reason` / `barrier` / `refused` fields.
- Opt-in `mask_interpolation="bilinear"` on `Compose` and `from_params`: differentiable soft-mask sampling for auxiliary masks (float masks required; labels mix at boundaries). Default `"nearest"` is unchanged and bit-identical.
- Memory benchmark (`experiments/bench_memory.py`): peak memory + allocation counts, fused vs native, per pipeline and batch size.
- `backend="native"` is now a first-class option for `from_config` (and `from_params` gains a `native` flag): the zero-dependency, fully batched pure-torch engine, including native `brightness`/`contrast` builders. Opt-in — backend auto-detection remains the default.
- `return_matrix=True` per-call flag: returns `(output, matrix)` without reading shared instance state, making matrix retrieval thread-safe; the `transform_matrix` property remains for compatibility.
- One `finfo(dtype).eps`-scaled near-singular threshold shared by all three matrix-inversion paths (torch, compile-friendly, numpy); `fusion_plan` / `fusion_plan_descriptors` results are cached (device-aware, pickle-safe).

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

- **Fused contrast midpoint is now the per-image mean luminance** (matching native TorchVision/Kornia `ColorJitter` semantics) instead of a fixed `0.5`. Fused pipelines containing contrast produce different (more native-faithful) values than previous releases; pin the previous behavior only by comparing against your own stored baselines. Parity holds under `reorder=NONE`; with pointwise reordering the mean is taken over the warped image and diverges from native by construction.
- Coordinate-changing passthrough ops (elastic/grid/optical distortion and similar) now **raise `ValueError`** when they execute in a multi-target pipeline (previously a `UserWarning`): auxiliary targets skip passthrough segments, so continuing would silently misalign masks/boxes/keypoints. Kernel/pointwise passthrough (blur, noise) with auxiliary targets no longer warns — skipping them is the correct semantics.
- `same_on_batch=True` on Albumentations-backed fused segments now shares the sampled parameters across the batch, not just the activation decision.
- Documented color-fusion accuracy caveats (final-only clamping; fixed 0.5 contrast midpoint) and the seeding contract limits between warp backends.

## [0.7.0] and earlier

No changelog was kept prior to 0.8.0 development; see the git history for details.
