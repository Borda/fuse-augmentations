# Campaign: fuse-augmentations — eliminate passthrough overhead & reach theoretical ceiling

## Goal

Reach `real_score ≥ theoretical_target` across the expanded 45-case benchmark (all five a-group single-ops, all five b-group pure-geo chains, all five d-group sequences under `AGGRESSIVE` reordering; 15 sequences × 3 backends). `theoretical_target` is printed by `optimize_score.py` on every run as `theoretical_target=X.XXXX`; it is the visual success criterion — no hard `target:` is set. The campaign runs to `max_iterations`.

Primary objectives by priority:

1. **Eliminate single-op passthrough overhead** — every a-group case must reach boost ≥ 0.95 across all backends. Current worst: `a03_vflip/tv = 0.58`, `a05_shear/alb = 0.69`, `a01_rotate/alb = 0.63`. Root cause: `AlbuFusedAffineSegment` and TV single-element segments likely fall through to `grid_sample` instead of calling the native adapter directly — check the `len(segment.transforms) == 1` passthrough path.

2. **Fix Albumentations short-chain regression** — `b02_geom_3/alb = 0.85` (3-op chain is slower fused than native). Investigate whether the compose path in `AlbuFusedAffineSegment` re-allocates tensors unnecessarily for short chains.

3. **Close d-group aggressive-reordering gap** — `d02_mixed_g3c3__agr/alb = 0.91` and `d04_mixed_g4c4__agr` are the weakest mixed-pipeline cases after reordering; the gap between actual and `nb_geom`-derived ceiling is the remaining opportunity.

`real_score > theoretical_target` is a valid and expected outcome for b-group TV/Kornia cases (flips are nearly free in native, giving super-theoretical ratios). Do not modify `optimize_score.py` to clamp or renormalise scores.

## Sequence naming conventions

Keys follow `<group>_<number>_<name>` so `sorted(SEQUENCE_BANK)` gives the intended display order.

- **a** — single-op baselines (`a01_rotate` … `a05_shear`): no fusion possible; measure wrapper overhead only. All have `nb_geom = 1` → theoretical ceiling is 1.0×.
- **b** — geometric chains (`b01_geom_2` … `b05_geom_5_warp`): `_geom_` signals pure affine ops; `_warp` means all ops have `p=1.0` (no flip short-circuits).
- **d** — mixed geo+colour (`d01_mixed_g3c2` … `d05_mixed_g5c4`): benchmarked here under `AGGRESSIVE` reordering only. `gN` = geo count, `cN` = colour count. NONE-policy variants are expected to be ≤ 1× (interleaved ops, no fusion benefit) and are not in the metric.

`b05_geom_5_warp` has 5 pure affine warps (all `p=1.0`, no flips) and is the canonical fixed-cost case: fused pays 1 warp regardless of N ops.

## Architecture notes

`FusedAffineSegment` has **fixed cost**: N affine ops fuse into one composed matrix and one `grid_sample` call. Single-op passthrough is implemented: when `len(segment.transforms) == 1`, the segment skips matrix compose + `grid_sample` and calls the native adapter transform directly. This must hold for all three backends including Albumentations — if `AlbuFusedAffineSegment` has a different passthrough path or is missing it entirely, that explains the a-group albu overhead.

For d-group sequences, `AGGRESSIVE` reordering moves all geo ops into one contiguous segment before fusion. The theoretical ceiling for a d-group case is still `nb_geom` (same formula as b-group), but the actual speedup is bounded below the ceiling by the colour-op time fraction. Kornia colour ops (`K.RandomSaturation`, `K.ColorJitter`) are CPU-heavy (2–3 ms each); this is a Kornia characteristic, not addressable via the adapter.

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
11. Any change to `_types.py` is **additive only** — new enum members are allowed; no renames, value changes, or removal of `GEOMETRIC`, `POINTWISE`, `COLOR`, `SPATIAL_KERNEL`, or `CROP_RESIZE`.

## Metric

```
command: uv run python experiments/optimize_score.py
direction: higher
baseline: 1.6550  # confirmed on 45-case metric (2026-04-02)
```

`theoretical_target` is printed on every run. Campaign success = `real_score ≥ theoretical_target` (~2.375 for the 45-case set). No `target:` is set — the campaign runs to `max_iterations` and the operator uses the printed `theoretical_target` as the visual stopping criterion.

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
  - src/fuse_augmentations/adapters/_torchvision.py
  - src/fuse_augmentations/_types.py
  - experiments/bench_augmentation_pipelines.py
compute: local
```

## Notes

`optimize_score.py` runs in ~60 s on the expanded 45-case metric (WARMUP=10 + REPS=50 × 45 cases × 2 backends per case) — within the 120 s `VERIFY_TIMEOUT_SEC` limit.

The 45-case theoretical target ≈ 2.375 = geomean([1⁵ × 2×3×4×5×5 × 3×3×4×4×5]^3)^(1/45). This is almost identical to the 15-case theoretical target (2.371) because adding a-group ones (pulls down) and d-group 3–5s (pulls up) balance out.

Kornia d-group results with `AGGRESSIVE` reordering are bounded by colour-op time; even perfect geo-fusion leaves the colour ops unsaved. The d-group Kornia ceiling with `__agr` is ≈ 1.1–2.1× — this is expected, not a bug.
