---
title: Benchmarks
description: "Historical benchmark results for fuse-augmentations, with CPU latency, batch scaling, primitive routing cost, and measurement limitations."
---

# Benchmarks

Fusion is most useful when a pipeline contains several compatible geometric transforms. It is not a universal speedup: a single transform, color-heavy workload, or particular backend and batch size can be neutral or slower. The historical suite below was collected on **July 12, 2026**; the separately labeled CPU preparation comparison was collected on **September 6, 2026**.

The figures are historical measurements for the environments below, not performance promises for every workload or the current revision. The memory table is retained as an audit trail but is withdrawn as evidence because its original timeline accounting interpreted profiler events incorrectly. The current memory tool is action-aware, and the current RF-DETR probe has reproducible variants at a common model-ready endpoint, but it does not replay shared geometry. Fresh, separately scoped CPU sweeps appear below; historical device ratios remain unvalidated for this revision. See [Methodology](methodology.md) for the publication gate.

## CPU preparation comparison — September 6, 2026

The CV remediation compares committed baseline `1b454ed` with the Wave 3 implementation. On this host, reusing native Albumentations matrix preparation reduced repeated adapter conversions while retaining the tensor path's float32 rounding and random-draw order.

| Batch | Preparation before / after (ms) | Preparation ratio | Image + boxes before / after (ms) | Endpoint ratio |
| ----- | ------------------------------- | ----------------- | --------------------------------- | -------------- |
| 1     | 0.0513 / 0.0323                 | 1.59x             | 0.675 / 0.592                     | 1.14x          |
| 8     | 0.462 / 0.194                   | 2.38x             | 4.175 / 3.782                     | 1.10x          |
| 32    | 1.890 / 0.792                   | 2.39x             | 16.381 / 14.663                   | 1.12x          |

Each cell is the median of three run medians. All three batch-32 preparation runs improved. Raw per-run medians, p95 values, profiles and source identity are in the [before](../assets/benchmarks/cv-wave3-preparation-before.json) and [after](../assets/benchmarks/cv-wave3-preparation-after.json) records.

Configuration: CPU, one Torch thread, RGB `224 x 224`, float32 BCHW images, one dense pixel-edge box per image, `execution="torch"`; `Rotate(limit=15, p=0.7)`, `Affine(scale=(0.9, 1.1), p=0.8)`, `HorizontalFlip(p=0.5)`. Each independent pipeline seeds NumPy, Torch and each Albumentations transform with `17`; five warmups precede 30 timed calls per run. Host: macOS 26.6.2 arm64, Torch 2.10.0, Albumentations 2.0.8, NumPy 2.2.6.

The complete endpoint includes augmentation and box routing. It excludes decode, host/device transfers, model execution and training. These ratios are local measurements for this chain; antialiasing is disabled. No accelerator or model-quality improvement is inferred.

```bash
NO_ALBUMENTATIONS_UPDATE=1 python experiments/bench_albu_preparation.py \
  experiments/results/preparation.json --revision YOUR_SOURCE_REVISION
```

Use the same environment, script and otherwise idle host for both source trees; `PYTHONPATH` can select an exported baseline. The private preparation probe attributes internal cost and may need updating when that implementation changes.

## Corrected detector-shaped CPU endpoint — September 6, 2026

`experiments/bench_rfdetr_shape.py` ran its complete 16-row sweep: two chains, 640/1024-pixel canvases, four variants, 20 warmups and 200 timed calls. Each endpoint holds a CPU float32 BCHW image in `[0, 1]`, clipped/filtered float32 boxes and aligned int64 labels. The timed NumPy-input rows include conversion to that model-ready endpoint. A detector forward/backward pass is not timed.

| Variant                  | Two ops, 640 (ms) | Two ops, 1024 (ms) | Four ops, 640 (ms) | Four ops, 1024 (ms) |
| ------------------------ | ----------------- | ------------------ | ------------------ | ------------------- |
| Native Albumentations    | 0.751             | 1.300              | 1.999              | 2.588               |
| Fused cv2, NumPy input   | 0.659             | 1.005              | 0.712              | 0.991               |
| Fused Torch, NumPy input | 5.368             | 12.727             | 5.412              | 12.887              |
| Fused cv2, tensor input  | 0.836             | 1.546              | 0.778              | 1.311               |

The cv2 NumPy-input ratio is about 1.14x/1.29x for two ops, and 2.81x/2.61x for four ops at 640/1024. This sweep used 12 Torch and 16 OpenCV threads. Independent sweeps on this shared host varied; these are single-sweep observations, not guaranteed gains. The tensor-input row starts after image conversion and has a different input boundary. Variants are independently reproducible; they do not replay shared sampled geometry, so these timings cannot establish image fidelity or model quality. [Raw p95, endpoint, seed and source records](../assets/benchmarks/cv-wave3-rfdetr.json).

Before augmentation, a 640/1024 RGB uint8 image occupies 1,228,800/3,145,728 bytes; the corresponding float32 model image occupies 4,915,200/12,582,912 bytes. Twelve float32 boxes and int64 labels add 192 and 96 bytes before filtering. These are tensor/array payload sizes, not measured allocations or host/device transfer counts. Accelerator transfers require a device trace; none was collected here.

## Antialias CPU cost — September 6, 2026

Per-sample filtering fixes neighbor-dependent blur and has an explicit cost. A fixed full-canvas Kornia crop from RGB224 to 56 pixels measured:

| Batch | Flag off (ms) | Flag on (ms) |
| ----- | ------------- | ------------ |
| 1     | 0.288         | 0.739        |
| 8     | 0.515         | 4.247        |
| 32    | 1.394         | 14.992       |

One Torch thread, five warmups, 30 calls, three repeats; each table cell is the median of run medians. A separate heterogeneous batch-32 probe measured 0.018 ms for scale estimation and 6.355 ms for the full prefilter. Its CPU profile counted 50 scalar extractions; this identifies potential synchronization sites but does not measure accelerator synchronization latency. The simple implementation filters active rows separately so each Gaussian support matches standalone execution. Exact-support grouping is a possible future optimization if this opt-in cost blocks a measured workload.

Run `experiments/bench_antialias.py` with Kornia installed. [Raw timings and CPU profile counts](../assets/benchmarks/cv-wave3-antialias.json). These measurements demonstrate cost; they do not establish model accuracy or accelerator performance.

## Corrected CPU tensor-memory sweep — September 6, 2026

The action-aware tool completed all 72 native/fused rows: six sequences, three backends, batches 1 and 8, RGB256, three warmups, 12 Torch and 16 OpenCV threads. All rows returned counters. Representative three-geometric-operation rows:

| Backend / batch | Native / fused live peak (MiB) | Native / fused CREATE count |
| --------------- | ------------------------------ | --------------------------- |
| kornia / 1      | 2.751 / 1.500                  | 271 / 28                    |
| kornia / 8      | 21.002 / 16.002                | 429 / 260                   |
| torchvision / 1 | 4.250 / 2.250                  | 35 / 23                     |
| torchvision / 8 | 30.500 / 16.002                | 35 / 231                    |

[All live/incremental peaks, preexisting bytes, CREATE counts, tracemalloc and RSS deltas](../assets/benchmarks/cv-wave3-memory.json). These are Torch tensor-timeline metrics, not total process memory. NumPy/OpenCV allocations are outside that timeline; near-zero Albumentations tensor peaks do not mean zero memory use. RSS deltas and tracemalloc describe different scopes and are not substitutes for transient total-process peaks. The profiler warned about an allocation made before profiling whose size was unknown, so baseline accounting is limited to visible events.

Run `NO_ALBUMENTATIONS_UPDATE=1 python experiments/bench_memory.py --devices cpu --batch-sizes 1 8 --warmup 3 --json`. Old memory ratios below remain withdrawn. No CUDA/MPS memory result was collected.

??? abstract "Historical test environment"

    | Component             | Value                                                            |
    | --------------------- | ---------------------------------------------------------------- |
    | Operating system      | macOS 26.5.2, arm64                                              |
    | Python                | 3.12.13                                                          |
    | `fuse-augmentations`  | 0.9.0.dev0                                                       |
    | PyTorch               | 2.10.0                                                           |
    | TorchVision           | 0.25.0                                                           |
    | Kornia                | 0.8.2                                                            |
    | Albumentations        | 2.0.8                                                            |
    | Input                 | 256 x 256 images; tensor inputs are BCHW `float32`               |
    | CPU batch semantics   | Albumentations applies CPU images sequentially within each batch |
    | CUDA in this July run | Unavailable                                                      |

    The CPU model string was not available inside the execution environment. The full latency-and-batch run detected CPU only, so this page makes no MPS latency claim. The separate memory script did execute MPS paths, but its MPS counter is not a reliable transient-peak measurement; see [Memory-counter boundaries](#memory-counter-boundaries). CUDA was measured separately on **September 5, 2026** on different hardware; see [Historical CUDA batch sweep](#historical-cuda-batch-sweep-september-5-2026).

## Fixed-bank score: 1.7861x

`experiments/optimize_score.py` measures 45 native/fused pairs on CPU, batch one, with 256 x 256 inputs. It includes single-operation baselines, pure geometric chains, and mixed geometric/color chains across all three backends.

```text
real_score=1.7861
theoretical_target=2.3752
```

`real_score` is the geometric mean of `native latency / fused latency` for this fixed synthetic bank. In plain language: on that bank, the fused path had a 1.7861x geometric-mean latency advantage. It is not an estimate of a typical user workload.

`theoretical_target` is the geometric mean number of geometric operations in the bank. It is a warp-count reference, not a speed ceiling: backend overhead, exact-operation fast paths, color work, caching, and wrapper costs all matter.

## Exhaustive CPU pipeline latency

`experiments/bench_augmentation_pipelines.py` ran every one of its 28 sequences for native and fused implementations across Albumentations, Kornia, and TorchVision: **168 timed variants**. Each variant had 20 warmups and 100 timed repetitions. The table groups the per-sequence native/fused ratios with a geometric mean; `>1x` means the fused path was faster.

| Backend        | All 28 sequences | Single geometric (`a`) | Geometric chains (`b`) | Color-only (`c`) | Mixed (`d`) |    Wins |
| -------------- | ---------------: | ---------------------: | ---------------------: | ---------------: | ----------: | ------: |
| Albumentations |            1.26x |                  1.03x |                  1.70x |            1.02x |       1.27x | 27 / 28 |
| Kornia         |            1.56x |                  1.12x |                  6.60x |            1.01x |       1.17x | 19 / 28 |
| TorchVision    |            1.69x |                  0.94x |                  8.03x |            1.06x |       1.35x | 16 / 28 |

The category averages are descriptive only: they give equal log-space weight to every sequence and are not weighted by user traffic or production mix.

| Representative sequence                         | Backend     | Native / fused mean latency |  Ratio | Reading                       |
| ----------------------------------------------- | ----------- | --------------------------: | -----: | ----------------------------- |
| Single rotate                                   | Kornia      |            0.502 / 0.512 ms |  0.98x | Slight fused loss.            |
| Single rotate                                   | TorchVision |            0.785 / 0.816 ms |  0.96x | Slight fused loss.            |
| Three geometric transforms                      | Kornia      |            1.214 / 0.209 ms |  5.80x | Clear chain benefit.          |
| Three geometric transforms                      | TorchVision |            1.593 / 0.216 ms |  7.36x | Clear chain benefit.          |
| Five geometric transforms with warps            | Kornia      |            4.750 / 0.339 ms | 14.03x | Large backend-specific gain.  |
| Five geometric transforms with warps            | TorchVision |            3.829 / 0.237 ms | 16.16x | Large backend-specific gain.  |
| Mixed 4 geometric + 3 color, aggressive reorder | Kornia      |            4.431 / 3.574 ms |  1.24x | Moderate gain.                |
| Mixed 4 geometric + 3 color, aggressive reorder | TorchVision |            2.351 / 0.972 ms |  2.42x | Gain depends on the sequence. |

The benchmark also regenerated native/fused visual comparisons for every sequence. Those images are a sanity aid, not a proof of equivalence; see [Quality and fidelity](quality-and-fidelity.md) for the distinction.

## CPU batch scaling

`experiments/bench_gpu_batch.py` completed its normal CPU sweep: seven representative sequences, three backends, batch sizes 1, 8, and 32, ten warmups, and 30 timed samples per variant. The figures below are geometric means of the seven **median** native/fused latency ratios.

| Backend        | Batch 1 | Batch 8 | Batch 32 | Fused wins at batch 32 |
| -------------- | ------: | ------: | -------: | ---------------------: |
| Kornia         |   2.99x |   1.73x |    1.62x |                  5 / 7 |
| TorchVision    |   3.06x |   0.85x |    0.63x |                  1 / 7 |
| Albumentations |   1.52x |   1.51x |    1.51x |                  7 / 7 |

This sweep exposes an important limitation: TorchVision was favorable for the sampled batch-one workloads, but was slower in most sampled batch-8 and batch-32 workloads. Do not assume an improvement merely because a pipeline is fused.

| Representative sequence                         | Backend / batch | Native / fused median latency |  Ratio |
| ----------------------------------------------- | --------------- | ----------------------------: | -----: |
| Three geometric transforms                      | Kornia / 1      |              1.163 / 0.319 ms |  3.65x |
| Five geometric transforms with warps            | TorchVision / 1 |              3.506 / 0.308 ms | 11.38x |
| Mixed 3 geometric + 3 color, aggressive reorder | Kornia / 8      |              8.494 / 8.118 ms |  1.05x |
| Geometric plus crop/resize                      | TorchVision / 8 |              1.310 / 1.805 ms |  0.73x |
| Single rotate                                   | Kornia / 32     |              5.284 / 5.347 ms |  0.99x |

## Primitive versus generic Affine

`experiments/bench_primitive_vs_affine.py` used 20 warmups and 100 timed repetitions. It explains why exact-operation shortcuts matter: a generic Affine call can be near parity for one operation, but it can also cost far more than a dedicated flip or rotation. Conversely, one combined Affine can replace multiple expensive resampling passes in a chain.

| Case                                                   | Measured ratio | Meaning                                   |
| ------------------------------------------------------ | -------------: | ----------------------------------------- |
| Albumentations Rotate 30 degrees: Affine / primitive   |          1.02x | Near parity.                              |
| Albumentations HFlip: Affine / primitive               |         17.69x | The generic route is much more expensive. |
| Albumentations VFlip: Affine / primitive               |         17.84x | The generic route is much more expensive. |
| Kornia Rotate 30 degrees: Affine / primitive           |          1.23x | Small generic-route cost.                 |
| TorchVision Rotate 30 degrees: Affine / primitive      |          3.87x | Meaningful generic-route cost.            |
| Albumentations two-operation chain: combined / native  |          0.70x | One combined call was faster.             |
| Albumentations five-operation chain: combined / native |          0.37x | Benefit grew with chain length.           |
| Albumentations six-operation chain: combined / native  |          0.33x | Combined cost remained relatively flat.   |

For the first five rows, a ratio near 1 is parity and a ratio above 1 means the generic Affine route costs more. For the chain rows, a lower combined/native ratio is better.

## Historical CPU tensor-memory output (withdrawn)

`experiments/bench_memory.py --json` used its normal six-sequence CPU and MPS sweep, batches 1 and 8, and three warmups. The values below are the historical output retained for traceability; they must not be interpreted as current peak-memory or allocation evidence. The current tool now interprets profiler actions explicitly, separating live peak, incremental peak above preexisting memory, baseline bytes, and physical `CREATE` events; unavailable counters are recorded as null with an error. The corrected September 6 CPU sweep above supplies new tensor-counter evidence; it does not rehabilitate these historical ratios.

| Backend / batch | Fused / native peak | Fused / native allocations | Lower peak samples | Lower allocation samples |
| --------------- | ------------------: | -------------------------: | -----------------: | -----------------------: |
| Kornia / 1      |               0.44x |                      0.24x |              6 / 6 |                    6 / 6 |
| Kornia / 8      |               0.38x |                      0.57x |              6 / 6 |                    6 / 6 |
| TorchVision / 1 |               0.18x |                      0.86x |              6 / 6 |                    5 / 6 |
| TorchVision / 8 |               0.33x |                      3.50x |              6 / 6 |                    0 / 6 |

The historical output appeared to show lower profiler-reported peak tensor memory in every sampled CPU Kornia and TorchVision comparison, but that conclusion is not currently supported. It also appeared not to consistently reduce allocation count. Neither observation should guide capacity or performance decisions until a fresh sweep is run with the corrected tool.

| Sequence                             | Backend / batch | Native / fused reported peak | Fused / native peak | Fused / native allocations |
| ------------------------------------ | --------------- | ---------------------------: | ------------------: | -------------------------: |
| Three geometric transforms           | Kornia / 1      |                 5.5 / 2.3 MB |               0.41x |                      0.11x |
| Three geometric transforms           | TorchVision / 8 |              117.5 / 38.0 MB |               0.32x |                      5.49x |
| Five geometric transforms with warps | TorchVision / 8 |              284.8 / 38.0 MB |               0.13x |                      4.15x |
| Mixed 4 geometric + 3 color          | Kornia / 8      |             582.5 / 433.1 MB |               0.74x |                      0.60x |
| Geometric plus crop/resize           | TorchVision / 8 |               75.9 / 24.5 MB |               0.32x |                      4.09x |

### Memory-counter boundaries

- CPU peak and allocation events came from a Torch profiler memory timeline; they do not capture every allocation made by native code. The historical run used the old sign-based accounting; current action-aware output is not represented by the tables above.
- Native Albumentations uses NumPy/OpenCV, so its `0.0 MB` Torch-profiler rows are not evidence of zero memory use and are excluded from the comparison.
- MPS reports an allocation delta, not a reliable transient peak. Its full-run records are preserved as diagnostics, not summarized as a memory claim.
- CUDA was unavailable in the July run and has a different allocator counter; a separate historical CUDA latency sweep appears below.
- Allocation counts depend on the profiler and dependency versions. They can increase even when peak tensor memory falls.

## Historical CUDA batch sweep (September 5, 2026)

The July run had no GPU, so every figure above is CPU. This section is a separate, later run on different hardware and records historical CUDA numbers from `experiments/bench_gpu_batch.py --batch-sizes 1 8 32 64 --warmup 10 --measure 30`, with 308 measured cases and 28 recorded skips. It is evidence from that dated environment, not a current-head or current-runner claim.

??? abstract "CUDA test environment"

    | Component            | Value                                              |
    | -------------------- | -------------------------------------------------- |
    | Accelerator          | NVIDIA L4                                          |
    | Operating system     | Linux x86_64 (Google Colab)                        |
    | Python               | 3.13.15                                            |
    | `fuse-augmentations` | 0.12.0.dev0                                        |
    | PyTorch              | 2.11.0+cu128                                       |
    | Input                | 256 x 256 images; tensor inputs are BCHW `float32` |
    | Device residency     | Device tensors are allocated on the device         |

    Native Albumentations has no GPU path and is recorded as a skip on CUDA rather than a slow row. The 28 skips are those cases.

**Read the device-residency row before the numbers.** `torch.rand(..., device=cuda)` allocates the input on the GPU, so no host-to-device copy falls inside the timed region. These are the device path's best case; any workload that starts on the host pays a transfer on top.

### Engine choice on device

This package's CPU engine (`execution="cv2"`, NumPy) against its own device engine (`execution="torch"`, one batched `grid_sample`), median ms in that historical run:

| sequence            | b32 CPU | b32 CUDA | b64 CPU | b64 CUDA | b64 result       |
| ------------------- | ------- | -------- | ------- | -------- | ---------------- |
| `a01_rotate`        | 7.59    | 7.80     | 15.18   | 14.64    | tie              |
| `b02_geom_3`        | 12.88   | 9.18     | 25.88   | 17.82    | CUDA 1.45x       |
| `b04_geom_5`        | 13.20   | 12.86    | 25.82   | 25.23    | tie              |
| `b05_geom_5_warp`   | 26.26   | 26.62    | 59.63   | 52.64    | CUDA 1.13x       |
| `d02_mixed_g3c3`    | 49.33   | 107.07   | 97.90   | 197.88   | CUDA 2.02x worse |
| `d03_mixed_g4c3`    | 49.24   | 112.51   | 99.46   | 206.75   | CUDA 2.08x worse |
| `e01_geo_crop_fuse` | 18.61   | 38.85    | 36.33   | 107.24   | CUDA 2.95x worse |

No batch size separates the wins from the losses; the split is by what the chain contains. This historical measurement informed ADR-005 keeping host data on cv2 rather than routing it to the device. Revalidate the conclusion on the current revision and deployment hardware before treating it as an active performance rule.

### Fusion value on device, by backend

Fused against that backend's own native implementation, on CUDA:

| backend     | result                                                                               |
| ----------- | ------------------------------------------------------------------------------------ |
| Kornia      | 1.09x-2.79x faster on multi-op chains; 0.83x-0.89x on the single-op `a01_rotate`     |
| TorchVision | 0.26x-0.93x — slower nearly everywhere, one exception at `b05_geom_5_warp` b64 1.23x |

Fusion needs more than one operation to fuse, which is why the single-op rotate loses. The TorchVision device result is not explained by that and is an open item.

### Why three sequences lose 2x-3x

One cause. Every losing sequence contains exactly one **passthrough segment**; every tying sequence contains none.

`AlbumentationsAdapter.call_nonfused` performs one device-to-host copy, `batch` sequential host-side transforms, and one host-to-device copy — per passthrough segment, per call. On CUDA that is a full round trip of the batch plus a device sync, which is why the penalty grows with batch while the fused warp gets cheaper per image.

| sequence            | segments                                   | the passthrough is   |
| ------------------- | ------------------------------------------ | -------------------- |
| `b04_geom_5`        | 1: fused affine                            | none                 |
| `d02` / `d03`       | 3: fused affine, fused colour, passthrough | `HueSaturationValue` |
| `e01_geo_crop_fuse` | 2: fused affine, passthrough               | `RandomResizedCrop`  |

Geometry and colour fuse correctly. One passthrough is enough to erase the warp's advantage — but the two passthroughs arrive for different reasons, and only one of them is a limitation of what can be fused.

`HueSaturationValue` is registered `POINTWISE`: reorderable, but non-linear in RGB, so it composes into neither a `FusedColorSegment` colour matrix nor a per-channel LUT. It has no fused segment because it cannot have one in the current design. That is a documented limitation, not a gap.

The historical `RandomResizedCrop` row used an image-only passthrough. Current `execution="torch"` routes that registered crop through `CropResizeSegment` even for image-only calls; auxiliary targets also select the routed crop. The historical CUDA loss therefore does not measure the current route. `execution="cv2"` and `"auto"` retain the native image-only crop policy.

Kornia and TorchVision can combine a compatible crop with preceding geometry into one `_FusedGeoCropSegment`. Albumentations can retain separate fused-affine and crop segments. Inspect the current plan and profile the complete chain: a remaining CPU-only operation such as `HueSaturationValue` still causes transfers on accelerator input, even with `execution="torch"`. The [transfer-aware recipe](../guides/reproducibility.md) makes that boundary explicit; no current-device speedup is inferred from the old table.

## Reproduce this run

Synchronize the optional benchmark dependencies, then run every experiment:

```bash
uv run --all-extras --group benchmark python experiments/optimize_score.py
uv run --all-extras --group benchmark python experiments/bench_augmentation_pipelines.py
uv run --all-extras --group benchmark python experiments/bench_primitive_vs_affine.py
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py
uv run --all-extras --group benchmark python experiments/bench_memory.py --json
```

The scripts write JSON and visual sanity outputs under `experiments/results/`. That directory is gitignored because results are host-specific scratch output. For a paper, release, or public performance claim, retain the raw JSON, command, commit SHA, dependency versions, and hardware metadata alongside the claim. For a stronger protocol, follow [Methodology](methodology.md).
