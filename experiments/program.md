# Campaign: fuse-augmentations — push fused geomean toward the theoretical ceiling, then open new fusion frontiers

## Goal

Drive `real_score` as high as it will go against the printed `theoretical_target` on the 45-case benchmark (five a-group single-ops, five b-group pure-geo chains, five d-group mixed sequences under `AGGRESSIVE` reordering; 15 sequences × 3 backends). `theoretical_target` is printed by `optimize_score.py` on every run as `theoretical_target=X.XXXX` and IS the ceiling — no hard `target:` is set. The campaign runs to the diminishing-returns window, not a fixed finish line.

This is an **open exploration**, not a checklist. The opportunity areas below are where the current gap lives, but you are explicitly invited to find and exploit fusions that are *not* listed here — any change that lifts the geomean without breaking the Guard or the Constraints is fair game. Do not treat the objectives as an exhaustive boundary; treat them as the known-weak spots and keep going past them.

### Opportunity areas (known gaps — not a boundary)

1. **Residual single-op passthrough overhead (a-group).** Every a-group case should reach boost ≥ 0.95 across all three backends. The geomean floor has been cleared repeatedly (~1.76–1.87 local), but the a-group **per-case** ≥ 0.95 check has never been individually confirmed — read the per-case rows from a fresh run and close any case still under 0.95. Historic worst offenders were `a03_vflip/tv`, `a05_shear/alb`, `a01_rotate/alb`; re-measure before assuming they are still weak.

2. **Mixed geo+colour headroom (d-group).** `d02`/`d04` were the weakest mixed cases; `d02/alb` has since crossed parity (0.91 → 1.09). Kornia d-group speedups are bounded by colour-op wall time (2–3 ms per `K.RandomSaturation`/`K.ColorJitter`) — that bound is a Kornia characteristic, not an addressable defect; do not chase it. Report the gap-to-ceiling per group and mark colour-time-bound cases as non-addressable rather than optimizing into them.

3. **New fusion frontiers (see "Open frontiers" below).** Filter algebra, per-transform border modes, and LUT composition open speedups the current 45-case metric cannot see. These are premise-gated and correctness-critical — they are delivered as staged, source-verified feature work with dedicated parity/PSNR gates, NOT by opportunistic score-chasing. When they land, their gains surface in the separately-reported frontier bank, never in the gated 45-case score.

`real_score > theoretical_target` is a valid, expected outcome for b-group TV/Kornia cases (flips are nearly free natively, giving super-theoretical ratios). Do **not** modify `optimize_score.py` to clamp or renormalise scores, and do **not** add, remove, or reweight the 45 gated cases (see Constraints).

**Ceiling note (verified 2026-07-09).** An earlier revision claimed Albumentations 2.0 composes consecutive affine matrices internally and capped the realistic geomean at ~1.85–1.95. Source-verified FALSE — Albu 2.0.8 dispatches one `cv2.warpAffine` per transform, no cross-transform matrix composition (checked against installed 2.0.8 source). The printed `theoretical_target` (~2.375) IS the ceiling. Near-identical native b02-vs-a01 timings reflect cv2's low per-warp cost at bench image sizes, not matrix composition.

## Sequence naming conventions

Keys follow `<group>_<number>_<name>` so `sorted(...)` gives the intended display order.

- **a** — single-op baselines (`a01_rotate` … `a05_shear`): no fusion possible; measure wrapper overhead only. All have `nb_geom = 1` → theoretical ceiling 1.0×.
- **b** — geometric chains (`b01_geom_2` … `b05_geom_5_warp`): `_geom_` = pure affine ops; `_warp` = all ops `p=1.0` (no flip short-circuits).
- **d** — mixed geo+colour (`d01_mixed_g3c2` … `d05_mixed_g5c4`): benchmarked under `AGGRESSIVE` reordering only. `gN` = geo count, `cN` = colour count. NONE-policy variants are ≤ 1× (interleaved, no fusion benefit) and are not in the metric.

`b05_geom_5_warp` is the canonical fixed-cost case: 5 pure affine warps (all `p=1.0`, no flips) — fused pays 1 warp regardless of N ops.

## Architecture notes

`FusedAffineSegment` has **fixed cost**: N affine ops fuse into one composed matrix and one `grid_sample` call. Single-op passthrough is implemented: when a segment holds exactly one interpolating transform, it skips the matrix pipeline and delegates to the native adapter — this holds for all three backends (Albumentations included; the cv2 path is the default CPU strategy, torch is opt-in via `execution="torch"`). If an a-group case still regresses, profile the tensor dispatch path (`nn.Module.__call__` chain, identity-buffer allocation, `grid_sample` mode-string lookup) rather than assuming the passthrough is missing — it is present as of the single-op dispatch refactor.

For d-group, `AGGRESSIVE` reordering groups all geo ops into one contiguous segment before fusion. The theoretical ceiling is still `nb_geom`, but actual speedup is bounded below it by the colour-op time fraction. Kornia colour ops are CPU-heavy — this bound is expected.

## Open frontiers (premise-gated; delivered as staged feature work, measured separately here)

These extend the fusion moat beyond affine+colour. Each is **verify-first**: its enabling premise must be source-confirmed against the installed backend before any implementation, and each ships with its own correctness gate (bit-exact / PSNR-vs-float64-reference parity), not the geomean. They are listed here so the campaign can *measure* their payoff once they land — via the frontier bank (below), reported separately and **excluded from the gated 45-case score** so an unlanded feature never drags the number.

- **Filter algebra (blur stops being a barrier).** `blur∘blur → one blur` (σ² adds); blur commutes through an affine warp via covariance transform `Σ' = A₂₂·Σ·A₂₂ᵀ` (moves to chain end as an anisotropic Gaussian); LTI kernel chains → one `conv2d`. Hard guard: refuse commutation when the min singular value of `A₂₂` < 1 (downscale aliasing) — a blur commuted *after* a downscale cannot prefilter aliasing the downscale already baked in. Non-linear kernels (median) stay barriers.
- **Per-transform border modes.** Honor each transform's own border mode (kornia `padding_mode`, albu `border_mode`, TV `fill`) instead of the single Compose-level override. Split a fused run where the mode changes (`split_reason="border_mode_change"`); elide the split only when the composed prefix provably never samples out of bounds (half-pixel-shrunk corner-preimage containment). Opt-in `padding_mode="per_transform"`; current default unchanged.
- **LUT composition (non-linear pointwise ops stop being passthrough).** Per-channel scalar maps (gamma, solarize, posterize, per-image equalize — NOT hue/saturation, which are cross-channel) compose into one lookup at plan time. uint8: 256-entry LUT, exact by enumeration. float: interpolated LUT (gather+lerp, pure torch). Adjacent diagonal-linear ops (brightness/contrast) fold into the LUT exactly.

### Frontier bench bank (opt-in, non-gated)

When the frontiers land, add representative sequences to `bench_augmentation_pipelines.py` under a clearly-labelled frontier group and report their fused/native ratios **separately** from the 45-case gate (same precedent as the crop-fusion probe). Candidate sequences:

- `f_blur_geom`: `[Rotate, GaussianBlur, Affine(scale)]` — filter-commutation payoff.
- `f_lut_pointwise`: `[Gamma, Solarize, Posterize]` — LUT-composition payoff (uint8 and float paths).
- `f_border_mixed`: an affine chain mixing two border modes — border-split cost + in-bounds elision.

Do not add these to `optimize_score.py`; the 45-case metric and its `theoretical_target` stay frozen for gate continuity.

## Constraints

Every kept commit must not break:

1. `pipe.fusion_plan`, `pipe.n_warps_saved`, `pipe.transform_matrix` remain accessible and correct after every forward pass.
2. Per-sample randomness — each image in a batch draws independent probabilities (no `same_on_batch` regression).
3. Auxiliary targets — masks, `bbox_xyxy`, `bbox_xywh`, keypoints routed through the composed matrix.
4. `ReorderPolicy.NONE`, `POINTWISE`, and `AGGRESSIVE` all produce valid output.
5. Multi-backend pipelines (Kornia + TorchVision + Albumentations in one pipe) work correctly.
6. `FusedCompose.from_params()` and `.from_config()` still construct valid runnable pipelines.
7. `output_backend="numpy"` returns HWC NumPy; `NumpyToTorchConverter` / `TorchToNumpyConverter` work.
8. `pickle.dumps(pipe)` / `pickle.loads(...)` round-trips without error (DataLoader workers).
9. `FusedCompose(transforms)(image)` call signature unchanged — no new required arguments.
10. `FusedCompose(albu_transforms)(image=ndarray)` returns `{"image": ndarray}` matching `A.Compose`.
11. Any change to `types.py` is **additive only** — new enum members allowed; no renames, value changes, or removal of `GEOMETRIC`, `POINTWISE`, `COLOR`, `SPATIAL_KERNEL`, or `CROP_RESIZE`.
12. Do not touch the settled opt-in defaults: `execution="cv2"` on CPU, `padding_mode` current default, `clip_policy="final"`, `substitute_passthrough=False`, `antialias=False`, `compile=False`. Speed knobs are opt-in; the default path stays bit-preserved.
13. **Verify-first for premise-gated fusions.** Any filter-commutation, border-mode, or LUT change must respect its source-confirmed guard (e.g. the min-singular-value refusal for blur commutation). A geomean gain that passes the Guard but violates a documented safety guard is a regression, not a win — the Guard suite is necessary, not sufficient, for these paths.

## Metric

```
command: uv run python experiments/optimize_score.py
direction: higher
baseline: 1.6671  # CI runner-measured (ubuntu, py3.12); local laptop swings ±5% — trust the runner
```

**Baseline validation (2026-07-19, single local pass).** One `optimize_score.py` run on the current tree — after the gap-closure capability work (inverse/TTA, whole-pipeline compile, bf16/fp16), the MPS numeric-parity coverage, and the structural split of the compose god-module into pipeline / factories / introspection / planner / config_validation modules — measured `real_score=1.7625`, `theoretical_target=2.3752`. This sits inside the historic ~1.76–1.87 local band and above the CI-runner baseline, confirming the refactors and capability additions caused no geomean regression. The CI-measured `baseline: 1.6671` above is left unchanged (trust-the-runner). A full re-launch campaign was intentionally deferred (budget); this pass is validation only.

`theoretical_target` (~2.375) is printed on every run and IS the ceiling. No `target:` is set; the campaign runs to the diminishing-returns window. At ~1.80 vs 2.375 with the remaining gap dominated by documented colour-op time fractions and by-design opt-ins, expect this to be a prove-and-refine run with occasional micro-wins, not an unbounded optimization goldmine — but keep exploring until improvements genuinely dry up.

## Guard

```
command: uv run pytest tests/ -x -q --tb=short
```

The Guard automatically includes every parity/precision test in `tests/` — including the frontier correctness gates once they land. A change that reddens the Guard is rejected regardless of its geomean effect.

## Config

```
max_iterations: 24
agent_strategy: perf
scope_files:
  - src/fuse_augmentations/compose.py
  - src/fuse_augmentations/affine/segment.py
  - src/fuse_augmentations/affine/matrix.py
  - src/fuse_augmentations/_interpolation.py
  - src/fuse_augmentations/adapters/albumentations.py
  - src/fuse_augmentations/adapters/kornia.py
  - src/fuse_augmentations/adapters/torchvision.py
  - src/fuse_augmentations/types.py
  - experiments/bench_augmentation_pipelines.py
compute: local
```

When the frontier fusions add new modules (e.g. a LUT color segment), extend `scope_files` to include them so the campaign can micro-optimize their hot paths. Keep `optimize_score.py` out of `scope_files` — the metric definition must not be editable by the optimizing agent.

## Notes

`optimize_score.py` runs in ~60 s on the 45-case metric (WARMUP=10 + REPS=50 × 45 cases) — within the 120 s `VERIFY_TIMEOUT_SEC` limit.

The 45-case `theoretical_target` ≈ 2.375 = geomean of `nb_geom` across all cases×backends. Albu 2.0.8 dispatches N separate warps (installed-source check) — the formula's assumption holds; actual progress is bounded by colour-op time fractions in the d-group. Kornia d-group with `AGGRESSIVE` is bounded ≈ 1.1–2.1× by colour time even with perfect geo-fusion — expected, not a bug.

## Next Experiments (live — priority order)

Stale entries removed 2026-07-18: the `_last_matrix` pre-allocated buffer shipped (matrix buffers are now reused in `AlbuFusedAffineSegment`); the a-group `nn.Module.__call__` profiling item was conditional on a Phase-1 target miss that never occurred. Replaced with the currently-live threads:

### 1. Confirm a-group per-case ≥ 0.95 — measure, then close stragglers

Run `optimize_score.py` once with per-case printing (or read the per-case ratios from `bench_augmentation_pipelines.py`) and list every a-group case still below 0.95 across the 3 backends. The geomean is healthy but the per-case objective was never individually verified. For any straggler, profile the single-op dispatch path (identity-buffer allocation, mode-string lookup, wrapper overhead) before changing code. Low risk, high confidence-value.

### 2. per_op_parity clamp-aware mean rederivation — small correctness+parity win

When `clip_policy="per_op_parity"` runs after a clipped intermediate, the mean-relative contrast diverges from native by ~1e-2 (currently documented as a residual). Correct fix: rederive the per-image mean from the *clamped* intermediate rather than the pre-clamp values. Isolated to `FusedColorSegment`; gate with a new parity case. Small.

### 3. Micro-opts within scope_files — opportunistic

Identity-buffer reuse on the tensor path, duplicate CV2-flag lookup removal, `fusion_plan` string caching where not already cached, `channels_last` through `grid_sample`. Each is small; take them when profiling points at them, skip them when it doesn't.

### 4. Frontier payoff measurement (after the frontiers land)

Once filter-algebra / border-mode / LUT fusion ship, wire the frontier bench bank (above) and report their fused/native ratios separately. This is where the next real step-change in the moat shows up — the 45-case gate will not move, but the frontier bank will.
