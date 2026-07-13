---
title: Benchmark and validation methodology
description: Reproduce augmentation benchmarks with paired randomness, synchronized timing, correctness gates, uncertainty, and complete environment reporting.
---

# Benchmark and validation methodology

Performance claims about augmentation pipelines are easy to overgeneralize. A credible comparison must control transform parameters, input contracts, device synchronization, execution order, output correctness, and statistical uncertainty.

This page describes what the repository's experiment scripts currently establish, where their evidence stops, and a stronger protocol for application and research use.

## Define the claim before measuring

A benchmark should answer one narrow question. Examples include:

- Does fusion reduce batch latency for this exact transform list on CPU?
- Does a longer geometric chain amortize planner and wrapper overhead?
- Does the fused path increase throughput at the production batch size?
- Does it reduce peak tensor memory on a named accelerator?
- Are output differences within a task-specific tolerance?

Specify these dimensions before running:

| Dimension  | Required detail                                                                   |
| ---------- | --------------------------------------------------------------------------------- |
| Workload   | Ordered transforms, probabilities, parameters, and reorder policy                 |
| Inputs     | Shape, batch, dtype, value range, memory layout, and representative content       |
| Backend    | Kornia, TorchVision, Albumentations, or backend-free construction                 |
| Execution  | CPU, CUDA, MPS, OpenCV, eager, or compiled                                        |
| Metric     | Latency, throughput, peak memory, allocation count, numeric error, or task metric |
| Comparator | Exact native pipeline and its input/call convention                               |
| Acceptance | Minimum speedup or equivalence margin and correctness tolerance                   |

Do not combine cross-backend latency into a hardware-independent ranking. HWC uint8 sequential OpenCV calls and BCHW float32 batched tensor calls are different workloads.

## What the current scripts do

| Script                            | Useful evidence                                                                      | Main limitation                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `optimize_score.py`               | One fixed 45-case CPU score with warmup and a deterministic run order                | Batch 1, zero synthetic inputs, no raw samples, paired RNG reset, or uncertainty             |
| `bench_augmentation_pipelines.py` | 168 CPU cases, raw per-call samples, summaries, plans, metadata, and visual examples | Native/fused order is fixed; timing RNG states are not paired; visual parity is not asserted |
| `bench_gpu_batch.py`              | Device synchronization, batch sweep, median, p10/p90, throughput, explicit skips     | Quick mode uses only 10 samples; one process; native always precedes fused                   |
| `bench_memory.py`                 | Counter-specific tensor peak and allocation evidence                                 | One measured invocation; counters are not comparable across devices; NumPy/OpenCV blind spot |
| `bench_primitive_vs_affine.py`    | Explains primitive fast paths and chain amortization                                 | Mean-only timing; incomplete combined-Affine coverage for some backends                      |

The scripts are valuable developer probes. Their quick output should not be promoted to a universal package claim without the controls below.

## Control randomness correctly

### Why one global seed is insufficient

The current timing scripts call `torch.manual_seed(0)` and `numpy.random.seed(0)` once, then execute native and fused pipelines sequentially. That makes the overall run order repeatable, but it does not guarantee that each native/fused pair draws identical parameters.

Albumentations 2 adds another RNG domain:

- package activation gates can use NumPy's global RNG;
- Albumentations transform parameters use transform-local RNG state;
- resetting only NumPy and Torch does not reset an existing Albumentations transform;
- `transform.set_random_seed(...)` reproduced transform sampling in the audited environment.

### Paired-randomness protocol

For correctness comparisons:

1. Construct native and fused transforms from the same immutable specification.
2. Record the Torch RNG state, NumPy RNG state, Python RNG state if relevant, and backend-local RNG state.
3. Restore the complete state before each side of a pair.
4. Prefer injecting fixed parameters or matrices when a backend exposes them.
5. Verify that the sampled matrices or parameters match before comparing pixels.

For performance comparisons, parameter sampling is part of runtime and should normally remain inside the timed call. Use a precomputed schedule of paired seeds or parameters so both paths process the same sequence without timing seed setup itself.

For multiworker data loading, seed each worker and every backend-local RNG. Record the worker count, persistence mode, process start method, and epoch reseeding strategy.

## Separate correctness from speed

Every performance result needs a correctness gate. A fast path that changes the intended transformation is a different workload.

Before timing:

- confirm output shape, dtype, device, and range;
- compare sampled parameters or transform matrices;
- measure pixel error under documented tolerances;
- verify masks, boxes, and keypoints with the image;
- test borders, padding, clipping, and degenerate inputs;
- inspect the actual fusion plan;
- run the same validation with reordering disabled.

Reordering policies other than `NONE` can change results and must be reported as part of the workload. Benchmarking an aggressively reordered fused pipeline against an original-order native pipeline measures an opt-in semantic variant, not a transparent implementation substitution.

See [Quality and fidelity](quality-and-fidelity.md) for backend-specific parity boundaries.

## Measure latency and throughput

### Warmup

Warm up every independently constructed path. Warmup should cover:

- allocator and thread-pool initialization;
- kernel and backend initialization;
- cache population;
- compilation, if enabled.

Report first-call latency separately when it matters. Do not mix compilation cost into a steady-state median unless first-call behavior is the claim.

### Synchronization

CUDA and MPS operations can be asynchronous. Synchronize once after warmup and after each timed operation. The current GPU/batch script follows this pattern.

CPU timing should use a monotonic high-resolution clock. Control or report Torch thread counts, process affinity when available, power mode, and concurrent load.

### Ordering and independent trials

Do not always time native first and fused second. Use one of these designs:

- alternate native and fused blocks with randomized initial order;
- randomize pair order from a recorded schedule;
- run each mode in separate fresh processes and counterbalance process order.

Repeat the benchmark in multiple independent processes. A practical baseline is at least five process-level trials with enough within-process samples for a stable median. Increase this when effects are small or tails are wide.

### Statistics

Report:

- median latency and a robust interval;
- p10/p90 or p5/p95 to show tails;
- throughput derived from the same latency definition;
- the paired native/fused ratio for every case;
- an aggregate only when its weighting is justified;
- the count and identity of losses and skips.

For paired trials, compute the ratio within each matched trial before summarizing. Bootstrap the paired trial ratios or report a distribution across process medians. Predefine an equivalence band, such as `0.95x-1.05x`, when the goal is to identify parity rather than declare tiny wins.

Avoid using a ratio of two noisy single-run means. A ten-sample smoke median is useful for a failure probe, not a stable performance conclusion; use independent process-level trials when the effect is small.

## Interpret the fixed-bank score

The repository score is:

```text
geometric_mean(native_latency / fused_latency for 45 fixed cases)
```

The geometric mean is appropriate for multiplicative ratios, but only after accepting the case bank and its weights. It does not answer how frequently users run each case.

When publishing the score:

- name the case-bank revision;
- list every included ratio;
- state that every case has equal log-space weight;
- report backend-specific and workload-group summaries;
- show the worst losses, not only the aggregate;
- call `2.3752` an operation-count reference, not a ceiling;
- do not compare scores across changed banks as though they were the same metric.

The July 12, 2026 environment produced `1.7861x`. That result is scoped to CPU, batch 1, 256 x 256, the fixed synthetic bank, and the recorded dependency versions.

## Measure memory responsibly

Memory requires a metric and counter that match the device.

| Device/path           | Appropriate evidence                                                              | Limitation                                                        |
| --------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| CPU Torch tensor path | Torch allocator timeline plus process-level cross-check                           | Private profiler internals and profiler/version dependence        |
| CUDA                  | Reset peak allocator stats, synchronize, then read peak allocated/reserved values | Does not represent host memory or every driver allocation         |
| MPS                   | Current/driver allocation snapshots                                               | Current allocation is not transient peak                          |
| NumPy/OpenCV          | Fresh-process RSS/high-water or native allocator instrumentation                  | Torch profiler and Python `tracemalloc` miss native C allocations |

Measure each mode in a fresh process when possible. Establish a baseline after warmup, retain the output until the counter is read, repeat across processes, and report allocated and reserved/resident memory separately.

Allocation count and peak bytes answer different questions. A fused kernel can allocate more small buffers while retaining fewer large intermediate tensors. Report both without assuming they move together.

## Reproducibility checklist

Include this checklist with every public benchmark or research artifact.

### Software

- [ ] Package version and exact commit SHA
- [ ] Python, Torch, backend, OpenCV, compiler, and driver versions
- [ ] Lockfile or exported environment
- [ ] Full command and all environment variables
- [ ] Case-bank revision and raw result artifact

### Hardware and runtime

- [ ] CPU/GPU model and memory capacity
- [ ] Operating system and architecture
- [ ] Torch intra-op and inter-op thread counts
- [ ] Device power/performance mode
- [ ] Worker/process count and affinity policy
- [ ] Eager/compiled state and compile warmup policy

### Workload

- [ ] Transform order, parameters, probabilities, and reorder policy
- [ ] Input shape, batch, layout, dtype, range, and content source
- [ ] Interpolation, padding, clipping, antialias, and execution settings
- [ ] Native and fused input/call conventions
- [ ] RNG domains, seeds, and paired parameter schedule

### Measurement

- [ ] Warmup count and measured iterations
- [ ] Device synchronization points
- [ ] Randomized or counterbalanced execution order
- [ ] Number of independent processes
- [ ] Median, tails, uncertainty interval, raw samples, losses, and skips
- [ ] Correctness tolerance and acceptance result

## Recommended artifact shape

A durable JSON or table should record at least:

```text
metadata:
  timestamp, commit, package versions, command
  hardware, device, threads, power mode
  input shape, dtype, range, batch
  warmup, samples, independent trials
  RNG and reorder policy
case:
  sequence, backend, mode, device, batch
  trial id, raw latency samples
  median, p10, p90, throughput
  correctness metrics and acceptance
  skip reason or failure
```

Keep raw samples. Summary-only artifacts cannot be reanalyzed when an outlier, regression, or changed acceptance threshold is discovered.

## Review questions for researchers

Before citing a result, ask:

1. Are native and fused paths performing the same intended transformation?
2. Are they receiving the same parameter schedule and equivalent inputs?
3. Does the device timer include completed work rather than dispatch?
4. Were execution order and thermal drift controlled?
5. Is the sample large enough to distinguish a win from noise?
6. Are losses and unsupported cases visible?
7. Does the memory counter observe the allocator being discussed?
8. Can another researcher reconstruct the environment and raw case bank?
9. Is the aggregate weighting relevant to the intended population?
10. Does the conclusion stay within the tested backend, device, batch, and version scope?

If any answer is no, narrow the claim or collect the missing evidence. Current measured results and known limits are summarized in [Benchmarks](benchmarks.md).
