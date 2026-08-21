# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All `0.x` versions below were published to PyPI as `dev0` snapshots on 2026-07-11 (batch-uploaded from their respective `bump vX` commits); no stable release has shipped yet.

## [Unreleased]

### Added

- `fuse_augmentations.data`: standalone synthetic dataset generator that draws colored shapes (square, rectangle, triangle, circle) on a canvas and exports **COCO** or **YOLO** datasets for **detection**, **segmentation**, and **oriented-bounding-box (OBB)** tasks. One `generate_dataset(output_dir, num_images, fmt=, task=, class_mode=, split_ratios=, seed=)` facade (also re-exported as `fuse_augmentations.generate_dataset`) over a format-agnostic `SyntheticGenerator` and `CocoWriter`/`YoloWriter`; `class_mode` selects shape / color / shape_color classes; rectangle plus per-shape rotation give OBB real orientation, and generation is fully seeded for byte-identical output. Generation and writing are streaming (one sample materialized at a time): `SyntheticGenerator.generate(n, seed=)` yields a lazy `Sample` iterator, the writers persist single-pass, and `generate_dataset` streams to disk so arbitrarily large datasets stay memory-bounded. `fuse_augmentations.data.datasets.SyntheticIterableDataset` (a worker-shard-aware `torch.utils.data.IterableDataset`) feeds samples straight into a `DataLoader` with no disk round-trip. Rendering uses Pillow, now a base dependency. See `docs/guides/synthetic-datasets.md` and `examples/generate_synthetic_dataset.py`.
- `fuse_augmentations.data.animals.AnimalShape` adds 12 animal side-profile silhouettes — `duck`, `elephant`, `giraffe`, `fish`, `rabbit`, `camel`, `eagle`, `penguin`, `whale`, `kangaroo`, `flamingo`, `crocodile` — fixed outline tables loaded in that same module. Unlike the geometric shapes, none is rotationally symmetric, so every silhouette carries real orientation under any rotation angle; they were added as a basis for future keypoint annotation work, where a symmetric outline makes a landmark's identity ambiguous under rotation. `SyntheticConfig` gains a `shapes: tuple[Shape, ...]` field (default `DEFAULT_SHAPES`, the original 4 geometric shapes) that restricts which shapes the generator draws, so existing callers' seeded output is unchanged; pass e.g. `shapes=(AnimalShape.DUCK, AnimalShape.GIRAFFE)` to draw animals instead, or `shapes=animal_shapes(4)` to take the first four animals in declaration order (`animal_shapes()` with no argument is the whole roster). `class_names()` and class ids still always span the full vocabulary for the selected `class_mode` regardless of `cfg.shapes` (`ClassMode.SHAPE` spans all 16 shapes, `ClassMode.SHAPE_COLOR` spans all 16 shapes times 3 colors -- 48 combined classes -- and `ClassMode.COLOR` spans only the 3 colors), so a class id means the same thing across every configuration. Works with the existing detection/segmentation/OBB tasks and both writers with no writer code changes. See the "Animal shapes" section of `docs/guides/synthetic-datasets.md` and `examples/animate_synthetic_dataset.py --shapes animals`.
- `Task.KEYPOINTS` adds a fixed sixteen-keypoint anatomical pose schema — `mouth`, `eye`, `ear`, `head`, `neck`, `body_top`, `body_bottom`, `tail`, plus `front_elbow_left/right`, `front_limb_left/right`, `hind_knee_left/right` and `hind_limb_left/right` pairs — shared across quadrupeds, birds, and swimmers (a front limb is the paw, wing, or fin/flipper; every limb is articulated in two points because a limb's bend is the clearest pose cue; `left` = viewer-near by convention; placement rules documented in `fuse_augmentations/data/zoo/README.md`), with COCO keypoint annotations and a 15-edge category skeleton plus Ultralytics YOLO pose rows and `kpt_shape: [16, 3]`. `ear` and the four hind-leg points are the only optional landmarks (a fish's or a whale's silhouette shows pectoral fins/flippers but no hind legs, and neither taxon has an external ear); an animal without them carries NaN rows in its packaged table rather than faked points, and every writer already treats a NaN coordinate the same as a canvas-clipped one (`0.0 <= nan` is `False`), so it needs no special-casing. Every keypoint renders in a fixed per-name color, identical across all animals, and `examples/edit_zoo_keypoints.py` opens a minimal drag-and-save editor for fixing placements by hand. The per-animal outline, keypoints, skeleton edges, and CC0/Public Domain Mark provenance now live in one editable SVG per animal (`fuse_augmentations/data/zoo/<animal>.svg`, superseding the earlier per-animal JSON) — keypoints and their skeleton render visible by default, so opening the file shows shape and pose together; keypoint placement is hand-edited per the documented rule, not script-derived.
- `inverse()` test-time de-augmentation: pass the image and matrix from a `return_matrix=True` forward call to map predictions (image, masks, xyxy/xywh boxes, keypoints) back to the original frame through the inverse of the fused pixel matrix in one `grid_sample`. Supported only for a pipeline that fuses to a single affine or projective segment; raises a named error for crop-resize, color/lookup/blur, exact-only or passthrough segments, multi-segment pipelines, and a missing paired matrix. Keypoints and masks recover to sampling precision; bounding boxes are axis-aligned (AABB) and recover exactly only under axis-aligned transforms (flip, scale, translation), inflating under rotation, shear, or a projective warp. The paired matrix is not validated against the image — passing a matrix from a different call yields silently wrong geometry.
- `pipeline_dtype="bfloat16"|"float16"` opt-in on `Compose`: runs the image warp and fused color/lookup applies in half precision on a non-CPU device for roughly 2x memory bandwidth. Matrix composition and inversion stay float32/float64 and are cast to the low precision only at the sampling-grid boundary, so the public `transform_matrix` always keeps full precision. Default (`None`) is bit-identical to before; CPU ignores the option.
- `fuse_augmentations.data.symbols.SymbolShape` adds a third, analytically-computed shape family — 7 straight-edge 2D symbols (`kite`, `trapezoid`, `house`, `arrow`, `cross`, `teardrop`, `anchor`) needing no traced artwork, unlike `AnimalShape`. There is deliberately no plain-triangle member: an isosceles triangle both collides in name with `GeomShape.TRIANGLE` and, being acute, has a minimum-area OBB with no unique answer (all three candidate edges tie), the same problem that motivated redesigning that geometric shape into an obtuse-scalene one — not worth solving twice for a shape this family does not need to keep. `Shape` widens to `GeomShape | AnimalShape | SymbolShape`, and `class_names()`/class ids append the 7 symbol classes after the existing 16 (23 total; `ClassMode.SHAPE_COLOR` 69 combined classes), so every prior class id is unchanged. Three of the seven (`arrow`, `cross`, `anchor`) are concave, so their segmentation polygon and OBB carry real orientation information an axis-aligned box does not. `SyntheticConfig(shapes=symbol_shapes())` selects the family (`symbol_shapes(n)` mirrors `animal_shapes(n)`); works with the existing detection/segmentation/OBB tasks with no writer code changes. See the "Symbol shapes" section of `docs/guides/synthetic-datasets.md` and `examples/animate_synthetic_dataset.py --shapes symbols`.
- `Task.KEYPOINTS` now also works with `SymbolShape`, via a second, smaller 7-point structural schema — `center` (mandatory, each symbol's own area centroid/center of mass), `apex`, `tail`, `flank_left/right`, `base_left/right` (all optional) — distinct from the animals' 16-point anatomical one, with its own star skeleton (six edges from `center`) and, unlike the animals' identity mapping, a genuine `flip_idx` swap (`flank_left`↔`flank_right`, `base_left`↔`base_right`) since every symbol is drawn mirror-symmetric about its own vertical axis. A dataset carries exactly one keypoint-bearing family: pairing `Task.KEYPOINTS` with a geometric shape, or mixing `AnimalShape` and `SymbolShape` in the same `shapes` tuple, now raises `ValueError` at `SyntheticConfig` construction (previously only the geometric-shape case was checked). `fuse_augmentations.data.config.keypoint_schema_for(shapes)` returns the active family's schema (or `None`); `fuse_augmentations.data.landmarks.KeypointSchema` is the shared type both families' schemas (and the writers' `keypoint_schema` parameter) now use. See the "Symbol keypoint schema" section of `docs/guides/synthetic-datasets.md`.
- `SyntheticConfig.asymmetry_jitter` (default `0.0`, range `[0, 0.5)`) optionally narrows a randomly chosen half — left or right of a shape's own local vertical axis, before rotation — of each placed object by up to that fraction, breaking the identical left/right OBB margins that every mirror-symmetric shape in this package otherwise always shows. Applied via a new shared `fuse_augmentations.data.geometry._skewed`/`_placed(..., skew=)` step (skew, then rotate, then translate), so the polygon and, under `Task.KEYPOINTS`, its landmark table are always skewed together and never drift apart; `shape_polygon`, `animal_keypoints`, and `symbol_keypoints` all gain a matching `skew` parameter, defaulting to `0.0`. `circle` is always excluded (it never rotates either, so an unrotated skew would bias every instance toward the same absolute image direction). `0.0` (the default) draws no extra randomness, so every existing seeded configuration's output is unchanged. See "Breaking left/right symmetry" in `docs/guides/synthetic-datasets.md`.
- `examples/render_shape_reference.py` renders one static, monochrome (black-on-white) reference image per shape into `docs/assets/shape-references/`, upright at each shape's own authored orientation — every symbol and animal is drawn mirror-symmetric about its own vertical axis, so this keeps a reference recognizable (an `arrow` pointing up, a `house` with its roof up) — with its plain axis-aligned detection box in blue and, for the two keypoint-bearing families (animals, symbols), keypoint dots and skeleton in the orange `animate_synthetic_dataset.py` uses for an occluded keypoint. The blue box is deliberately the detection box at this fixed reference pose, not `polygon_to_obb`'s minimum-area OBB, which the generator's actually-rotated samples carry and which is a rotated quadrilateral in general (not always axis-aligned even at this same unrotated pose, e.g. the `arrow`'s) — drawing that box at one fixed angle would show an arbitrary rotation (or, for a shape whose OBB candidates tie, an arbitrary tied choice) rather than the shape itself. Each shape is scaled independently to the largest size that fits its own outline inside the frame, rather than sharing one scale sized for the whole family, and carries no in-image filename caption — one shape per file already names it. See the new "Shape reference" tabs in `docs/guides/synthetic-datasets.md`.

### Fixed

- `inverse()` normalizes and inverts the paired matrix in full precision even when the augmented image is low precision, casting only the sampling grid to the image dtype at the `grid_sample` boundary (float32/float64 images unaffected). The Albumentations-backed affine/projective segments' public matrix now keeps the image's own precision (float32 or float64) instead of always promoting to float32; only a float16/bfloat16 image promotes it to float32, so float64 Albumentations pipelines no longer lose matrix precision.
- **`polygon_to_obb`'s chosen box orientation no longer wobbles between equal-area candidates as a shape rotates.** A shape with reflective symmetry (every animal and symbol this package draws, plus the geometric square and rectangle) or a right/acute triangle's altitude-inside-the-opposite-side tie (see `GeomShape.TRIANGLE`'s docstring) can have more than one candidate rectangle achieve the true minimum area. `polygon_to_obb` picked among tied candidates via a plain `area < best_area` comparison over hull edges ordered by `_convex_hull`'s lexicographic sort of the *already-rotated* coordinates — so which tied candidate won depended on the shape's absolute orientation in the image, not its own geometry, and the OBB's edge alignment visibly flipped between renders of the identical shape at different rotation angles (e.g. `kite`'s and `teardrop`'s OBBs showed only one flush edge in roughly half of random rotations, two in the other half, when it should be consistently two). `_convex_hull` now also returns each hull point's original vertex index (stable across rotation, since `_placed` rotates the whole vertex array rigidly); `polygon_to_obb` uses that to break near-ties (relative tolerance `1e-9`) by a rotation-invariant edge-index key instead of iteration order. No public signature changed; `polygon_to_obb`'s *area* was always correct — only the tied candidate's *orientation* was unstable.

### Changed

- **`fuse_augmentations.data.shapes` moved to `fuse_augmentations.data.geometry`** — the old path keeps working as a deprecated re-export shim that emits a `DeprecationWarning` on import, so existing `from fuse_augmentations.data.shapes import ...` code is unaffected for now; update the import to `fuse_augmentations.data.geometry`, as the shim will be removed in a future release. It forwards the geometry surface the module published (`CIRCLE_POINTS`, `RECT_ASPECT`, `GEOMETRIC_SHAPES`, `rotate_polygon`, `shape_polygon`, `polygon_to_bbox_xyxy`, `polygon_to_obb`, `bbox_iou`).
- **`Shape` is now `GeomShape | AnimalShape`, not a single `Enum`** — the drawable vocabulary is split by family, `GeomShape` (square, rectangle, triangle, circle) in `fuse_augmentations.data.geometry` and `AnimalShape` (the 12 animal silhouettes) in `fuse_augmentations.data.animals`. `isinstance(value, Shape)` still works and still rejects a bare string, but `Shape.SQUARE`, `Shape("square")`, `tuple(Shape)`, and `for s in Shape` no longer work — a plain union has no members and is not iterable. Use `GeomShape` directly for the old enum-only behavior, or build the full vocabulary as `[*GeomShape, *AnimalShape]`.
- `compile=True` on `Compose` now also wraps the fused color-matrix application and the lookup-table application in their own dynamic-shape `torch.compile` regions (previously only the warp core), cutting kernel launches for color- and lookup-heavy pipelines on GPU; each tensor core compiles separately so varying height, width, and batch do not trigger a recompile storm. Per-sample probability masks and the equalize runtime histogram table stay outside compiled regions. Default `compile=False` path and outputs are unchanged.
- **`GeomShape.TRIANGLE` is now an obtuse-scalene triangle, not an equilateral triangle** — an equilateral triangle's 3-fold rotational symmetry left the *outline's* orientation ambiguous mod 120°; the new outline has neither rotational nor reflective symmetry, so its rotation is always visually recoverable from the silhouette. Its minimum-area OBB is also now a genuine unique minimum rather than a tie: a right or acute triangle's altitude from every vertex lands inside the opposite side, tying that side's flush candidate with another (see the `Fixed` entry below), but an obtuse triangle's altitude from its obtuse vertex falls outside the opposite side, so no other candidate can match the longest side's area.
- **Animal and symbol outline normalization now centers on the polygon's area centroid (center of mass), not the arithmetic vertex mean** — `fuse_augmentations.data.landmarks._frame` (shared by `AnimalShape` and `SymbolShape` loading) previously subtracted `points.mean(axis=0)`, which drifts off a shape's true visual middle whenever vertices are spread unevenly along its edges (an arrow's barbs, a giraffe's long neck); it now subtracts the polygon's shoelace-formula area centroid instead. `GeomShape`'s own analytically-constructed shapes (square, rectangle, circle, and the new obtuse-scalene triangle) are unaffected, since their vertex mean already equals their area centroid by construction. This shifts every animal's and symbol's rendered position by the (typically small) offset between its old vertex-mean center and its true center of mass; no public API changed.

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
