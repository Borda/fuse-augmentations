---
title: Benchmarks
description: Measured fuse-augmentations latency, throughput, and memory results with backend, device, workload, and methodology limits.
---

# Benchmarks

The short answer is: fusion can be much faster for longer geometric chains on CPU, but it is not always faster. Results depend on the backend, device, batch size, image shape, transform mix, and whether behavior-changing reordering is enabled.

This page reports the bounded measurements collected on **July 12, 2026**. They are evidence for the stated environment, not universal performance promises.

## Test environment

| Component            | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| Operating system     | macOS 26.5.2, arm64                                    |
| Logical CPUs         | 16                                                     |
| Python               | 3.12.13                                                |
| `fuse-augmentations` | 0.9.0.dev0                                             |
| PyTorch              | 2.10.0                                                 |
| TorchVision          | 0.25.0                                                 |
| Kornia               | 0.8.2                                                  |
| Albumentations       | 2.0.8                                                  |
| Torch threads        | 12 intra-op, 16 inter-op                               |
| Accelerators         | Apple MPS available; CUDA unavailable                  |
| Image size           | 256 x 256                                              |
| Tensor input         | BCHW float32                                           |
| Albumentations input | HWC uint8; CPU batches executed sequentially per image |

The CPU model string was unavailable inside the execution sandbox. Optional backend packages were installed through the project's benchmark extras; they are not part of the minimal base environment.

## Fixed-bank score: 1.7707x

The current `experiments/optimize_score.py` run reported:

```text
real_score=1.7707
theoretical_target=2.3752
```

`real_score` is the geometric mean of 45 native/fused latency ratios. The bank contains single-operation baselines, pure geometric chains, and mixed geometric/color chains across Kornia, TorchVision, and Albumentations. It uses CPU, batch 1, 256 x 256 inputs, fixed equal log-space weighting, and opt-in `AGGRESSIVE` reordering for the mixed cases.

Interpret it as:

> On this fixed synthetic CPU benchmark bank, the geometric mean of native latency divided by fused latency was 1.7707.

Do not interpret it as an average speedup for user workloads. The case bank is not sampled from production usage, and its weighting does not represent a user population.

The `2.3752` value is the geometric mean of the number of geometric operations in each case. It is a **warp-count reference**, not a theoretical performance ceiling. Backend overhead, exact-operation fast paths, color cost, caching, and wrapper cost make operation count an imperfect latency model. Individual measured speedups can exceed that reference.

## CPU latency and throughput

The representative CPU quick sweep covered seven sequences, three backends, batch sizes 1 and 8, five warmups, and ten timed samples. A second independent invocation was used to expose quick-run variability.

In the second invocation, 35 of 42 comparable pairs were faster and 7 were slower. The geometric mean by backend and batch was:

| Backend        | Batch 1 | Batch 8 |
| -------------- | ------: | ------: |
| Kornia         |   2.52x |   1.73x |
| TorchVision    |   3.20x |   1.26x |
| Albumentations |   1.53x |   1.53x |

These aggregate values include wins and losses. Representative second-run cases show the actual range:

| Sequence                                        | Backend     | Batch | Native/fused ratio | Interpretation                  |
| ----------------------------------------------- | ----------- | ----: | -----------------: | ------------------------------- |
| Single rotate                                   | Kornia      |     8 |              0.88x | Fused was slower.               |
| Single rotate                                   | TorchVision |     8 |              0.81x | Fused was slower.               |
| Three geometric ops                             | Kornia      |     1 |              4.31x | Clear geometric-chain win.      |
| Three geometric ops                             | TorchVision |     1 |              5.23x | Clear geometric-chain win.      |
| Five geometric ops                              | Kornia      |     1 |              6.52x | Benefit grew with chain length. |
| Five geometric ops with warp                    | TorchVision |     1 |             14.48x | Large backend-specific win.     |
| Mixed 3 geometric + 3 color, aggressive reorder | Kornia      |     8 |              0.49x | Fused was about 2x slower.      |
| Mixed 3 geometric + 3 color, aggressive reorder | TorchVision |     8 |              0.77x | Fused was slower.               |
| Mixed 4 geometric + 3 color, aggressive reorder | TorchVision |     8 |              0.86x | Fused was slower.               |
| Geometric + crop/resize                         | Kornia      |     8 |              2.18x | Fused was faster.               |

A ratio above 1 means fused was faster; below 1 means fused was slower.

### Quick-run variability

Ten measurements are sufficient for a smoke benchmark, not for a stable effect estimate. Between two CPU invocations:

- TorchVision batch-8 geometric-plus-crop moved from `0.75x` to `1.27x`, changing from a loss to a win.
- TorchVision batch-8 three-op geometry moved from `0.72x` to `0.92x`.
- TorchVision batch-8 mixed pipelines remained slower, although the magnitude changed.

This variation is why the tables expose individual losses and why release-neutral documentation should not promise a specific speedup.

## Apple MPS results

The MPS quick sweep used the same seven sequences and batch sizes. Native Albumentations has no MPS implementation, so 14 native rows were skipped with an explicit reason. The fused Albumentations tensor path ran, but without a native MPS comparator it cannot produce a meaningful speedup ratio.

Only 9 of 28 comparable Kornia/TorchVision pairs were above 1.0 in this run.

| Sequence                     | Kornia b1 | Kornia b8 | TorchVision b1 | TorchVision b8 |
| ---------------------------- | --------: | --------: | -------------: | -------------: |
| Single rotate                |     0.66x |     1.40x |          1.15x |          1.45x |
| Three geometric ops          |     0.59x |     0.95x |          0.69x |          0.29x |
| Five geometric ops           |     0.65x |     2.05x |          0.45x |          0.35x |
| Five geometric ops with warp |     1.11x |     2.14x |          0.60x |          0.62x |
| Mixed 3 geometric + 3 color  |     0.58x |     0.78x |          0.47x |          0.28x |
| Mixed 4 geometric + 3 color  |     0.83x |     2.05x |          0.54x |          0.36x |
| Geometric + crop/resize      |     1.34x |     1.10x |          0.28x |          0.21x |

The MPS result distribution also had wide tails. For example, fused Kornia batch-8 mixed 3+3 had a 25 ms median and 133 ms p90. These measurements prove that the paths execute and reveal backend-specific limitations; they do not establish a stable MPS speedup.

CUDA was unavailable. CUDA latency, memory, compilation, and graph-reuse claims remain unverified by this benchmark session.

## Primitive versus generic Affine

The primitive benchmark explains why chain length matters and why exact-operation shortcuts are important.

| Case                                          | Measured result | Meaning                                                           |
| --------------------------------------------- | --------------: | ----------------------------------------------------------------- |
| Albumentations Rotate: Affine/primitive       |           1.01x | Generic Affine was near parity.                                   |
| Albumentations HFlip: Affine/primitive        |           9.40x | Generic Affine was much more expensive than the exact primitive.  |
| Albumentations VFlip: Affine/primitive        |          21.41x | Generic Affine was much more expensive than the exact primitive.  |
| TorchVision Rotate90: Affine/primitive        |          15.45x | Routing an exact rotation through generic Affine would be costly. |
| Albumentations two-op chain: combined/native  |           0.71x | One combined Affine was faster.                                   |
| Albumentations five-op chain: combined/native |           0.38x | The chain benefit increased.                                      |
| Albumentations six-op chain: combined/native  |           0.32x | The combined cost stayed near constant.                           |

For the last three rows, a lower ratio is better because the benchmark reports combined-Affine cost divided by primitive-chain cost. Many combined Kornia/TorchVision chain entries are absent from this script, so package-level pipeline benchmarks remain the evidence for those backends.

## Memory results

The CPU memory quick sweep measured three sequences, Kornia and TorchVision tensor paths, and batches 1 and 8. The Torch profiler reported lower peak tensor memory for all 12 comparable pairs. Allocation count was lower in 8 pairs and higher in 4.

| Sequence                    | Backend / batch | Fused/native peak | Fused/native allocations |
| --------------------------- | --------------- | ----------------: | -----------------------: |
| Three geometric ops         | Kornia / 1      |             0.75x |                    0.19x |
| Three geometric ops         | TorchVision / 1 |             0.12x |                    0.54x |
| Three geometric ops         | Kornia / 8      |             0.34x |                    0.56x |
| Three geometric ops         | TorchVision / 8 |             0.32x |                    5.49x |
| Mixed 3 geometric + 3 color | Kornia / 8      |             0.86x |                    0.72x |
| Mixed 3 geometric + 3 color | TorchVision / 8 |             0.84x |                    2.09x |
| Geometric + crop/resize     | Kornia / 8      |             0.32x |                    0.62x |
| Geometric + crop/resize     | TorchVision / 8 |             0.32x |                    4.73x |

The three-op TorchVision batch-8 case used 117.5 MB native versus 38.0 MB fused, about **3.1x less peak tensor memory**. An older README statement described this as about 8x at batch 8; the displayed 8x figures were actually from batch 1.

### Memory-counter boundaries

- CPU peak and allocation events come from a private Torch profiler memory timeline and one measured invocation per mode.
- Native NumPy/OpenCV allocations are not visible to the Torch allocator. Albumentations rows reporting 0.0 MB do not mean zero memory.
- Python `tracemalloc` does not cover every C-extension allocation.
- MPS exposes current resident allocation, not a reliable transient peak.
- CUDA uses a different allocator counter and was not exercised.
- Allocation count depends on profiler and dependency versions and can rise even when peak memory falls.

The defensible claim is therefore: fusion reduced reported peak tensor memory in the sampled CPU Kornia/TorchVision cases. It did not consistently reduce allocation count, and no native Albumentations memory conclusion is available.

## Run the benchmarks

Install or synchronize the optional benchmark environment, then select the script matching the question:

```bash
uv run --all-extras --group benchmark python experiments/optimize_score.py
uv run --all-extras --group benchmark python experiments/bench_gpu_batch.py --quick
uv run --all-extras --group benchmark python experiments/bench_memory.py --quick --json
uv run --all-extras --group benchmark python experiments/bench_primitive_vs_affine.py
```

The scripts write scratch results under `experiments/results/`; that directory is gitignored. Preserve the command, commit, metadata, raw samples, and environment separately if a result will support a paper, release, or public claim.

For a stronger protocol than the quick scripts provide, follow [Methodology](methodology.md). For the distinction between speed and output equivalence, see [Quality and fidelity](quality-and-fidelity.md).
