# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All `0.x` versions below were published to PyPI as `dev0` snapshots on 2026-07-11 (batch-uploaded from their respective `bump vX` commits); no stable release has shipped yet.

## [Unreleased]

### Added

- `inverse()` test-time de-augmentation: pass the image and matrix from a `return_matrix=True` forward call to map predictions (image, masks, xyxy/xywh boxes, keypoints) back to the original frame through the inverse of the fused pixel matrix in one `grid_sample`. Supported only for a pipeline that fuses to a single affine or projective segment; raises a named error for crop-resize, color/lookup/blur, exact-only or passthrough segments, multi-segment pipelines, and a missing paired matrix. Keypoints and masks recover to sampling precision; bounding boxes are axis-aligned (AABB) and recover exactly only under axis-aligned transforms (flip, scale, translation), inflating under rotation, shear, or a projective warp. The paired matrix is not validated against the image — passing a matrix from a different call yields silently wrong geometry.
- `pipeline_dtype="bfloat16"|"float16"` opt-in on `Compose`: runs the image warp and fused color/lookup applies in half precision on a non-CPU device for roughly 2x memory bandwidth. Matrix composition and inversion stay float32/float64 and are cast to the low precision only at the sampling-grid boundary, so the public `transform_matrix` always keeps full precision. Default (`None`) is bit-identical to before; CPU ignores the option.

### Fixed

- `inverse()` normalizes and inverts the paired matrix in full precision even when the augmented image is low precision, casting only the sampling grid to the image dtype at the `grid_sample` boundary (float32/float64 images unaffected). The Albumentations-backed affine/projective segments' public matrix now keeps the image's own precision (float32 or float64) instead of always promoting to float32; only a float16/bfloat16 image promotes it to float32, so float64 Albumentations pipelines no longer lose matrix precision.

### Changed

- `compile=True` on `Compose` now also wraps the fused color-matrix application and the lookup-table application in their own dynamic-shape `torch.compile` regions (previously only the warp core), cutting kernel launches for color- and lookup-heavy pipelines on GPU; each tensor core compiles separately so varying height, width, and batch do not trigger a recompile storm. Per-sample probability masks and the equalize runtime histogram table stay outside compiled regions. Default `compile=False` path and outputs are unchanged.

## [0.8.0.dev0] - 2026-07-11

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

## [0.7.0.dev0] - 2026-05-14

### Added

- `FusedCompose.__call__` gains a native Albumentations-dict input/output fast path (`_forward_albu_native`).

### Fixed

- `RandomSaturation` and `HueSaturationValue` are now registered as `POINTWISE` (previously misclassified, bypassing color fusion).

### Changed

- Performance: single-op fast paths that skip the matrix pipeline and `grid_sample` for one-transform chains, numpy-direct matrix builders with cached identity/inverse buffers for the cv2 and Albumentations warp paths, and fused sample+build for the TorchVision cv2 path; cumulative gains tracked via an expanded 45-case benchmark suite.
- CI gains a matrix strategy exercising the optional Kornia/TorchVision/Albumentations extras independently.

### Performance

- Single-op fast paths for `FusedAffineSegment` (Kornia/TorchVision) and the Albumentations numpy path skip matrix reconstruction and `grid_sample`/`cv2.warpAffine` entirely for one-transform chains, and bypass `nn.Module.__call__` in favor of direct `.forward()` dispatch in the compose loop.
- `matmul3x3` moved to `torch.bmm` and the eager `inv3x3` path to `torch.linalg.inv` (~150x and ~6x faster per call respectively, measured); the Albumentations numpy path gained a closed-form Cramer's-rule 3x3 inverse and an `np.flip` bypass for pure horizontal/vertical-flip chains, replacing `scipy.ndimage` with `cv2.warpAffine` for its warp step.
- Fused sample+build helpers (`sample_and_build_matrix_numpy_b1_kornia`/`_tv`) combine parameter sampling and matrix construction into one call on the Kornia and TorchVision cv2 fast paths, cutting several intermediate tensor allocations per active transform.
- Pre-allocated matrix buffers, cached identity matrices, and pre-classified segment-dispatch tags remove per-call allocations and `isinstance` checks from the hot cv2/Albumentations forward path.
- Individual optimization commits are pinned to measured per-change deltas against the running 45-case composite benchmark score (e.g. `A.Rotate` numpy fast path +1.45%, fused sample+build for the TorchVision cv2 path +1.68%, Albumentations direct-dispatch bypass +0.94%, redundant-copy removal in the cv2 batch-size-1 path +0.61%).
- `experiments/optimize_score.py` grew from a 15-case to a 45-case benchmark with a computed theoretical-target ceiling per case; `examples/bench_augmentation_pipelines.py` and `examples/bench_primitive_vs_affine.py` were added to compare fused vs. native throughput across all three backends.

## [0.6.0.dev0] - 2026-03-28

### Added

- `Compose.from_config()` classmethod, backed by a backend resolver, an op-name registry, and a frozen `TransformSpec` dataclass, for declarative pipeline construction.
- `output_backend` parameter on `Compose.__init__` for cross-backend output conversion, backed by new `NumpyToTorchConverter` / `TorchToNumpyConverter` and a `BackendConverter` protocol.
- `CROP_RESIZE_FIXED` op category and `CropResizeSegment`, with adapter registrations across all three backends.
- `POINTWISE_LINEAR` color fusion: `build_color_matrix` per adapter, `FusedColorSegment`, and `reorder_pointwise`/`build_segments` integration.
- `ReorderPolicy.AGGRESSIVE`, and extended `GEOMETRIC_EXACT` dispatch with an `exact_apply` protocol method.
- `fusion_plan_descriptors` property (backed by a new `SegmentDescriptor` dataclass) and a `backend=` kwarg on `from_params` for full-parity delegation.

### Changed

- `ExactSegment` renamed to `ExactAffineSegment` (deprecation alias kept); expanded Kornia and Albumentations adapter coverage surveys (`SafeRotate`, `RandomShear`, `RandomTranslate` registrations).

### Fixed

- Aux-target corruption, batch-randomness, and backend-attribution bugs found across the review cycle; `_d4_matrix` now guards shape-changing D4 elements on non-square images.

## [0.5.0.dev0] - 2026-03-20

### Added

- `ProjectiveSegment` and `AlbuProjectiveSegment` for fused perspective-warp chains, with perspective division applied to auxiliary targets (masks, boxes, keypoints).
- `RandomPerspective` / `Perspective` registered across all three adapters (Kornia, TorchVision, Albumentations), wired into `Compose` via `ProjectiveSegment`.
- `PROJECTIVE` op-category enum and perspective matrix utilities.

## [0.4.0.dev0] - 2026-03-20

### Added

- `TorchVisionAdapter` for TorchVision v1 and v2 transforms, wired into `Compose` dispatch.
- Mixed-backend restriction lifted: pipelines can now mix adapters per transform, dispatched individually.

### Fixed

- `RandomAffine` matrix composition corrected to match TorchVision semantics; TorchVision v2 batch semantics fixed.
- `id()`-keyed adapter map replaced with a stable lookup (fixes pickle-stability of passthrough adapter dispatch); `Backend.UNKNOWN` handling clarified.

## [0.3.0.dev0] - 2026-03-19

### Added

- `AlbumentationsAdapter` implementing the `TransformAdapter` protocol, wired into `Compose` and segment dispatch.
- `NumpyFusedAffineSegment` for the Albumentations (cv2) backend, plus `_np_matrix.py` matrix builders for `hflip`/`vflip`.

### Changed

- Affine engine restructured into an `affine/` subpackage; `cv2` replaced with `scipy` in the shared matrix path.

### Fixed

- `torch.from_numpy` incompatibility with NumPy 2.x.

## [0.2.0.dev0] - 2026-03-18

### Added

- `data_keys` routing and auxiliary-target (mask/bbox/keypoint) transform helpers (`_targets.py`), wired through segments.
- `Compose.from_params()` classmethod.

### Fixed

- `transform_mask` now supports integer masks with dtype preservation (previously float32-only).
- Duplicate `data_keys` handling in the forward loop.

## [0.1.0.dev0] - 2026-03-18

- `ExactSegment` for lossless flip-only chains, dispatched via `build_segments` detection of EXACT-only op chains.
- `ReorderPolicy.POINTWISE` reordering support.
- `same_on_batch` support verified and extended in `KorniaAdapter`.
- `FusedCompose` renamed and reworked for `Protocol` conformance.
