---
title: Benchmarks
description: "Full benchmark results for fuse-augmentations: CPU latency, batch scaling, primitive routing cost, and reported tensor memory."
---

# Benchmarks

Fusion is most useful when a pipeline contains several compatible geometric transforms. It is not a universal speedup: a single transform, color-heavy workload, or particular backend and batch size can be neutral or slower. This page reports the complete benchmark suite collected on **July 12, 2026**.

The figures are measurements for the environment below, not performance promises for every workload.

??? abstract "Test environment"

    | Component            | Value                                                            |
    | -------------------- | ---------------------------------------------------------------- |
    | Operating system     | macOS 26.5.2, arm64                                              |
    | Python               | 3.12.13                                                          |
    | `fuse-augmentations` | 0.9.0.dev0                                                       |
    | PyTorch              | 2.10.0                                                           |
    | TorchVision          | 0.25.0                                                           |
    | Kornia               | 0.8.2                                                            |
    | Albumentations       | 2.0.8                                                            |
    | Input                | 256 x 256 images; tensor inputs are BCHW `float32`               |
    | CPU batch semantics  | Albumentations applies CPU images sequentially within each batch |
    | CUDA                 | Unavailable                                                      |

    The CPU model string was not available inside the execution environment. The full latency-and-batch run detected CPU only, so this page makes no current MPS or CUDA latency claim. The separate memory script did execute MPS paths, but its MPS counter is not a reliable transient-peak measurement; see [Memory-counter boundaries](#memory-counter-boundaries).

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

## Reported CPU tensor memory

`experiments/bench_memory.py --json` used its normal six-sequence CPU and MPS sweep, batches 1 and 8, and three warmups. CPU results below compare Kornia and TorchVision tensor paths. The figures are geometric means of six `fused / native` ratios; a smaller peak ratio means less reported peak tensor memory.

| Backend / batch | Fused / native peak | Fused / native allocations | Lower peak samples | Lower allocation samples |
| --------------- | ------------------: | -------------------------: | -----------------: | -----------------------: |
| Kornia / 1      |               0.44x |                      0.24x |              6 / 6 |                    6 / 6 |
| Kornia / 8      |               0.38x |                      0.57x |              6 / 6 |                    6 / 6 |
| TorchVision / 1 |               0.18x |                      0.86x |              6 / 6 |                    5 / 6 |
| TorchVision / 8 |               0.33x |                      3.50x |              6 / 6 |                    0 / 6 |

Fusion reduced the profiler-reported peak tensor memory in every sampled CPU Kornia and TorchVision comparison. It did **not** consistently reduce the allocation count: TorchVision batch-eight fused cases allocated more, even while their reported peak was lower.

| Sequence                             | Backend / batch | Native / fused reported peak | Fused / native peak | Fused / native allocations |
| ------------------------------------ | --------------- | ---------------------------: | ------------------: | -------------------------: |
| Three geometric transforms           | Kornia / 1      |                 5.5 / 2.3 MB |               0.41x |                      0.11x |
| Three geometric transforms           | TorchVision / 8 |              117.5 / 38.0 MB |               0.32x |                      5.49x |
| Five geometric transforms with warps | TorchVision / 8 |              284.8 / 38.0 MB |               0.13x |                      4.15x |
| Mixed 4 geometric + 3 color          | Kornia / 8      |             582.5 / 433.1 MB |               0.74x |                      0.60x |
| Geometric plus crop/resize           | TorchVision / 8 |               75.9 / 24.5 MB |               0.32x |                      4.09x |

### Memory-counter boundaries

- CPU peak and allocation events come from a Torch profiler memory timeline; they do not capture every allocation made by native code.
- Native Albumentations uses NumPy/OpenCV, so its `0.0 MB` Torch-profiler rows are not evidence of zero memory use and are excluded from the comparison.
- MPS reports an allocation delta, not a reliable transient peak. Its full-run records are preserved as diagnostics, not summarized as a memory claim.
- CUDA was unavailable and has a different allocator counter.
- Allocation counts depend on the profiler and dependency versions. They can increase even when peak tensor memory falls.

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
