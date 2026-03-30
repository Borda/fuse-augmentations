# Campaign: fuse-augmentations — score from 1.13 → 1.40

## Goal

Reach a composite boost score ≥ 1.40, where score = geometric mean of native/fused latency ratios across 12 cases (Kornia×4 + TorchVision×4 + Albumentations×4: b02_geo_3, b04_geo_5, a01_rotate, a04_scale).

The main bottleneck is **Kornia and TorchVision single-op overhead**: for a 1-element `FusedAffineSegment`, the fused pipeline builds a matrix, inverts it, and calls `grid_sample` — identical work to native, plus wrapper overhead. Boost for Kornia single ops is currently 0.57–0.67. The highest-leverage fix is a passthrough: when `len(segment.transforms) == 1`, skip matrix compose + grid_sample and call the native adapter transform directly.

If single-op passthrough does not reach target, profile `FusedAffineSegment.forward` on a 3-op Kornia sequence (`torch.autograd.profiler` or `cProfile`) and fix the hottest non-grid_sample line.

Do NOT apply the tensor-path passthrough to `AlbuFusedAffineSegment` — Albumentations single ops are already ≈ 1.0 via the native I/O path.

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
baseline: 1.1298  (2026-03-30, 12 cases)
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
  - experiments/optimize_score.py
compute: local
```

## Notes

`optimize_score.py` runs in ~20 s (10 warmup + 50 reps × 12 cases) — within the 120 s `VERIFY_TIMEOUT_SEC` limit. Albumentations single-op overhead is already ≈ 1.0 via the native I/O path — do not apply tensor-path passthrough to Albumentations segments.

## References

- AutoResearch (Karpathy) — autonomous experiment-loop pattern this campaign follows: https://github.com/karpathy/autoresearch
