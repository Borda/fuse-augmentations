# Campaign: fuse-augmentations — score from 1.13 → 1.40

## Goal

Reach a composite boost score ≥ 1.40, where score = geometric mean of native/fused latency ratios across 15 cases (Kornia×5 + TorchVision×5 + Albumentations×5: b02_geom_3, b04_geom_5, b05_geom_5_warp, a01_rotate, a04_scale).

**Target reached** (score ≈ 1.59 as of 2026-03-31 with 15 cases).

## Sequence naming conventions

Keys follow `<group>_<number>_<name>` so `sorted(SEQUENCE_BANK)` gives the intended display order.

- **a** — single-op baselines (`a01_rotate`, `a04_scale`, …): no fusion possible; measure wrapper overhead
- **b** — geometric chains (`b01_geom_2` … `b05_geom_5_warp`): `_geom_` prefix signals pure geometric ops
- **c** — colour chains (`c01_color_2` … `c03_color_4`): pointwise-only sequences
- **d** — mixed geo+colour (`d01_mixed_g3c2` … `d05_mixed_g5c4`): `gN` = geo count, `cN` = colour count; sequences are strict-prefix slices of a 9-op interleaved pool

`b05_geom_5_warp` has 5 pure affine warps (Rotate + Scale + Shear + Translate + Rotate, all `p=1.0`, no flips) and is the canonical test for the fixed-cost architecture: fused pays 1 warp regardless of N ops, so 5 full warps yield ≥ 3× speedup.

## Reorder policy variants

d-group sequences are benchmarked under all three policies. Variants appear as separate rows in the results table, named with a `__<suffix>` appended to the base key:

| Suffix   | Policy       | Description                                                                     |
| -------- | ------------ | ------------------------------------------------------------------------------- |
| *(none)* | `NONE`       | No reordering — interleaved geo/colour worst case                               |
| `__pw`   | `POINTWISE`  | Moves colour ops across POINTWISE barriers; groups all geo ops into one segment |
| `__agr`  | `AGGRESSIVE` | Reorders across all non-geometric barriers                                      |

`K.RandomSaturation` and `A.HueSaturationValue` are registered as `TransformCategory.POINTWISE` so `POINTWISE` reordering can group geo ops across them without claiming they are linearly composable colour transforms.

## Architecture notes

`FusedAffineSegment` has **fixed cost**: N affine ops fuse into a single composed matrix and one `grid_sample` call. The fused pipeline pays identical cost for 1 op or 5 ops — only the matrix multiply count changes (cheap). This is why `b05_geom_5_warp` yields ~4× speedup and d-group sequences benefit strongly from reordering (more geo ops per segment).

Single-op passthrough is implemented: when `len(segment.transforms) == 1`, the segment skips matrix compose + `grid_sample` and calls the native adapter transform directly. Do NOT apply tensor-path passthrough to `AlbuFusedAffineSegment` — Albumentations single ops are already ≈ 1.0 via the native I/O path.

## Constraints

Every kept commit must not break:

01. `pipe.fusion_plan`, `pipe.n_warps_saved`, `pipe.transform_matrix` remain accessible and correct after every forward pass.
02. Per-sample randomness — each image in a batch draws independent probabilities (no `same_on_batch` regression).
03. Auxiliary targets — masks, `bbox_xyxy`, `bbox_xywh`, keypoints routed through the composed matrix.
04. `ReorderPolicy.NONE`, `POINTWISE`, and `AGGRESSIVE` all produce valid output.
05. Multi-backend pipelines (Kornia + TorchVision + Albumentations in the same pipe) work correctly.
06. `FusedCompose.from_params()` and `.from_config()` still construct valid runnable pipelines.
07. `output_backend="numpy"` returns HWC NumPy; `NumpyToTorchConverter` and `TorchToNumpyConverter` work.
08. `pickle.dumps(pipe)` / `pickle.loads(...)` round-trips without error (DataLoader workers).
09. `FusedCompose(transforms)(image)` call signature unchanged — no new required arguments.
10. `FusedCompose(albu_transforms)(image=ndarray)` returns `{"image": ndarray}` matching `A.Compose` convention.

## Metric

```
command: uv run python experiments/optimize_score.py
direction: higher
target: 1.40
baseline: 1.1298  (2026-03-30, 12 cases — pre b05)
achieved: ~1.59   (2026-03-31, 15 cases — post b05 + single-op passthrough + POINTWISE reorder)
```

## Guard

```
command: uv run pytest tests/ -x -q --tb=short
```

## Config

```
max_iterations: 20
agent_strategy: perf
scope_files:
  - src/fuse_augmentations/_compose.py
  - src/fuse_augmentations/affine/_segment.py
  - src/fuse_augmentations/affine/_matrix.py
  - src/fuse_augmentations/_interpolation.py
  - src/fuse_augmentations/adapters/_albumentations.py
  - src/fuse_augmentations/adapters/_kornia.py
  - experiments/optimize_score.py
  - experiments/bench_augmentation_pipelines.py
compute: local
```

## Notes

`optimize_score.py` runs in ~20 s (10 warmup + 50 reps × 15 cases) — within the 120 s `VERIFY_TIMEOUT_SEC` limit.

Kornia colour ops (`K.RandomSaturation`, `K.ColorJitter`) are CPU-heavy (2–3 ms each). For d-group sequences with Kornia, the colour ops dominate total pipeline time so the geo-fusion saving is a small fraction — d-group Kornia boost with `__pw` is ≈ 1.06–1.12× even with perfect reordering. This is a Kornia CPU characteristic, not addressable by the adapter.

## References

- AutoResearch (Karpathy) — autonomous experiment-loop pattern this campaign follows: https://github.com/karpathy/autoresearch
