# Experiments

Benchmark and optimization scripts used during development to measure `fuse-augmentations`' speedup over native Albumentations / Kornia / TorchVision pipelines, and to drive the automated performance-optimization campaigns recorded in `.plans/`. These are developer tools, not part of the installed package — nothing here ships to users.

All scripts are plain Python but written in the [jupytext "percent" format](https://jupytext.readthedocs.io/) (`# %%` cell markers), so each can be run as a script or converted to a notebook:

```bash
uv run python experiments/<script>.py
# or, as a notebook:
jupytext --to notebook experiments/<script>.py
jupyter lab experiments/<script>.ipynb
```

Install the extras these scripts need once:

```bash
uv pip install -e ".[all,benchmark]"
```

Every script writes its numeric results to `experiments/results/` (JSON +, for the pipeline benchmark, PNG figures). That directory is gitignored — it's scratch output, regenerated on every run, not checked into version control. The sample output below was captured from a real (short) run of each script on a Darwin/arm64 host with Torch 2.10; exact numbers will differ on your machine but the qualitative shape (which cases fuse well, which don't) should hold.

## Files

| File                              | Purpose                                                                                                                                                  | Typical runtime                   |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `optimize_score.py`               | Single composite score (geometric mean of 45 native/fused boost ratios) — the metric an optimization campaign maximizes.                                 | ~30–60 s                          |
| `bench_augmentation_pipelines.py` | Full per-sequence latency comparison (28 sequences × 3 backends × native/fused) + visual sanity figures.                                                 | ~40–60 s                          |
| `bench_primitive_vs_affine.py`    | Is routing a single op through the backend's generic `Affine` as cheap as its dedicated primitive? Answers the "can we always fuse via Affine" question. | ~5–10 s                           |
| `bench_gpu_batch.py`              | Device × batch-size sweep (CPU/CUDA/MPS, batch 1 & 8): latency + throughput.                                                                             | ~1–2 min (`--quick`), longer full |
| `bench_memory.py`                 | Peak memory and allocation-count comparison, same sequence/device/batch sweep as above.                                                                  | ~1 min (`--quick`), longer full   |

______________________________________________________________________

## `optimize_score.py` — composite optimization metric

```bash
uv run python experiments/optimize_score.py
```

Times 45 cases (single-op baselines, pure-geometric chains, mixed geo+colour chains under aggressive reordering) across Kornia, TorchVision, and Albumentations, each native vs. fused, and prints the geometric mean of all 45 boost ratios plus the theoretical ceiling (geomean of each case's `nb_geom`). This is the single number an optimization campaign tries to push toward the ceiling. No JSON/figures are written — it only prints two lines.

**Sample output (short run):**

```
real_score=1.7601
theoretical_target=2.3752
```

Interpretation: fused pipelines are currently ~1.76× faster than native on average across the 45-case bank, against a theoretical best-case of ~2.38× (the gap is mostly CPU-bound colour-op time in the mixed-pipeline cases, which fusing the geometric ops alone can't remove).

## `bench_augmentation_pipelines.py` — full pipeline comparison

```bash
uv run python experiments/bench_augmentation_pipelines.py
```

Runs 28 sequences (single-op `a*`, geometric `b*`, colour `c*`, mixed `d*` with `__pw`/`__agr` reorder variants) × 3 backends × native/fused = 168 timed benchmarks, then renders one visual-sanity PNG per sequence (native vs. fused output side by side, with a `max|native-fused|` diff annotation) and writes `experiments/results/benchmark_results.json`.

**Sample summary table (short run, 100 reps):**

```
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Sequence                          alb             |           kornia            |             tv
                      native ;   fused ;  boost   | native ;   fused ;  boost   | native ;   fused ;  boost
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
a01_rotate             0.224 ;   0.199 ;  x1.13 ✔ |  0.450 ;   0.480 ;  x0.94 ≈ |  0.726 ;   0.766 ;  x0.95 ≈
b02_geom_3             0.238 ;   0.192 ;  x1.24 ✔ |  1.126 ;   0.211 ;  x5.34 ✔ |  1.938 ;   0.214 ;  x9.05 ✔
b05_geom_5_warp        1.097 ;   0.260 ;  x4.23 ✔ |  4.295 ;   0.329 ; x13.05 ✔ |  3.626 ;   0.230 ; x15.78 ✔
c02_color_3            0.234 ;   0.239 ;  x0.98 ≈ |  3.406 ;   3.687 ;  x0.92 ≈ |  0.718 ;   0.726 ;  x0.99 ≈
d03_mixed_g4c3__agr    0.728 ;   0.465 ;  x1.57 ✔ |  4.381 ;   3.597 ;  x1.22 ✔ |  2.206 ;   0.924 ;  x2.39 ✔
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

(full table has 28 rows; `a*`/`c*` single-op and colour-only cases hover near 1×, `b*` pure-geometric chains show the largest wins since every fused chain collapses to one `grid_sample` regardless of length, `d*` mixed chains land in between depending on colour-op cost and reorder policy.)

Before the table, the script also prints one line per fused sequence/backend showing the chosen fusion plan, e.g.:

```
d03_mixed_g4c3__agr  albumentations  fused  →  fused(Rotate, HorizontalFlip, VerticalFlip, Rotate) →
  color(RandomBrightnessContrast, RandomBrightnessContrast) → passthrough(HueSaturationValue)  [4 warps saved]
```

**Illustration** — `experiments/results/visual_b02_geom_3.png` (Rotate + HFlip

- Scale, all three backends, native on top / fused below; each panel's `max|native-fused|` confirms the two paths draw identical random parameters and agree numerically):

![b02_geom_3 native vs fused](results/visual_b02_geom_3.png)

## `bench_primitive_vs_affine.py` — primitive vs. generic Affine

```bash
uv run python experiments/bench_primitive_vs_affine.py
```

For each backend, times a dedicated primitive (`A.Rotate`, `K.RandomRotation`, …) against the backend's generic `Affine`/`RandomAffine` configured to the same effect. A ratio ≈ 1.0 means fuse-aug can freely route that op through `Affine` (and therefore fuse it into a chain) at no per-op cost; a ratio ≫ 1.0 means the dedicated primitive is meaningfully cheaper and the fused path pays a tax for single ops of that kind. Also times 2–6 op chains as dedicated primitives vs. one combined `Affine` call, which is the actual saving fusion delivers.

**Sample output (short run, 100 reps, `—` = backend has no dedicated op or no Affine equivalent):**

```
                            alb                     kornia                        tv
                          prims   affine    ratio    prims   affine   ratio    prims   affine    ratio
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  Geometric primitives
    HFlip                 0.023    0.213    9.27×    0.097        —       —    0.050        —        —
    Rotate 30°            0.221    0.225    1.02×    0.406    0.494   1.22×    0.733    0.720    0.98×
    Scale                 0.084    0.216    2.58×        —    0.610       —        —    0.877        —

  Multi-op sequence
    2-op chain            0.329    0.221    0.67×    1.286    0.510   0.40×    0.804        —        —
    5-op chain            0.577    0.223    0.39×    3.126        —       —    1.675        —        —
    6-op chain            0.659    0.228    0.35×    3.106        —       —    1.644        —        —
```

Reading this: Albumentations' `Affine` costs a near-fixed ~0.22 ms regardless of op count, so an N-op Albumentations chain routed through one `Affine` call gets cheaper (relative to N dedicated-primitive calls) as N grows — exactly the fusion win `FusedAffineSegment` exploits. Cheap ops with no dedicated primitive at all (`HFlip`/`VFlip` are near-free in Albumentations, ~0.01–0.02 ms) show a high single-op ratio, which is why fuse-aug keeps exact-primitive shortcuts for those rather than always routing through `Affine`.

Results: `experiments/results/bench_primitive_vs_affine.json`.

## `bench_gpu_batch.py` — device × batch-size sweep

```bash
uv run python experiments/bench_gpu_batch.py            # full sweep
uv run python experiments/bench_gpu_batch.py --quick     # fast smoke run
```

Sweeps CPU (always) plus CUDA/MPS (auto-detected) at batch size 1 and 8 for a representative subset of sequences, reporting median/p10/p90 latency and throughput (img/s) for native vs. fused. Correct per-device synchronization (`torch.cuda.synchronize`/`torch.mps.synchronize`) is applied before/after timing so the numbers reflect real device execution, not async dispatch. Native Albumentations is CPU/NumPy-only, so it's skipped (recorded, not silently dropped) on `cuda`/`mps` device rows.

**Sample output (`--quick`, batch 8, CPU):**

```
| d03_mixed_g4c3__agr | kornia         | native | cpu | 8 | 10.430 | 9.900 | 11.305 | 767  | — |       |
| d03_mixed_g4c3__agr | kornia         | fused  | cpu | 8 |  8.387 | 8.125 |  8.883 | 954  | — | 1.24x |
| e01_geo_crop_fuse   | kornia         | native | cpu | 8 |  3.831 | 3.707 |  4.137 | 2088 | — |       |
| e01_geo_crop_fuse   | kornia         | fused  | cpu | 8 |  1.888 | 1.771 |  1.963 | 4237 | — | 2.03x |
```

**Sample output (`--quick`, batch 1, MPS):**

```
| b05_geom_5_warp | kornia      | native | mps | 1 | 20.080 | 16.281 | 20.331 |  50 | — |       |
| b05_geom_5_warp | kornia      | fused  | mps | 1 | 10.719 |  9.849 | 11.854 |  93 | — | 1.87x |
| b05_geom_5_warp | albumentations | native | mps | 1 | — | — | — | — | — | — | skip: native Albumentations is NumPy/CPU-bound; no GPU path |
```

`boost` (last numeric column) is throughput fused/native; `>1x` = fused faster. A quick smoke run reported `154 ok, 14 skipped` and wrote `experiments/results/bench_gpu_batch_darwin_arm64.json`.

## `bench_memory.py` — peak memory & allocation count

```bash
uv run python experiments/bench_memory.py            # full sweep
uv run python experiments/bench_memory.py --quick     # fast smoke subset
uv run python experiments/bench_memory.py --json      # also write JSON
```

Same sequence/device/batch matrix as `bench_gpu_batch.py`, but measures peak memory and allocation count instead of latency, testing the hypothesis that fusing an N-op chain into one `grid_sample` both lowers peak memory (no chain of intermediate warped tensors) and cuts allocation count. Uses `torch.profiler` (CPU), `torch.mps.current_allocated_memory()` (MPS), or `max_memory_allocated` (CUDA) depending on which counter is reliable per device.

**Sample output (`--quick`, CPU, batch 1 & 8; `peak x`/`alloc x` = fused/native ratio, `<1x` = fused uses less):**

```
sequence             backend        mode    device  batch peak MB   allocs   peak x  alloc x
b02_geom_3           kornia         native  cpu     1     4.0       560
b02_geom_3           kornia         fused   cpu     1     3.0       104      0.75x   0.19x
b02_geom_3           torchvision    native  cpu     1     24.5      92
b02_geom_3           torchvision    fused   cpu     1     3.0       50       0.12x   0.54x
b02_geom_3           kornia         native  cpu     8     110.5     885
b02_geom_3           kornia         fused   cpu     8     38.0      503      0.34x   0.57x
```

Reading this: the fused TorchVision path uses ~8× less peak memory than native at batch 8 for a 3-op geometric chain (no intermediate per-op tensors to hold), even though its allocation *count* is sometimes higher (small scratch buffers inside the single fused `grid_sample` call vs. TorchVision's few large per-op allocations). A quick smoke run reported `66 ok, 6 skipped`.

## Notes

- All scripts seed `torch`/`numpy` identically for native and fused runs, so any reported difference is real cost, not different random draws — the pipeline benchmark's visual figures make this explicit via the `max|native-fused|` annotation on each panel.
- `optimize_score.py`'s 45-case bank and `bench_augmentation_pipelines.py`'s 28-case bank overlap but aren't identical; see `program.md` for the optimization-campaign context these scripts were built for.
- `bench_gpu_batch.py` and `bench_memory.py` import their sequence bank from `optimize_score.py` when available, falling back to an inline copy — console output notes which provenance was used.
