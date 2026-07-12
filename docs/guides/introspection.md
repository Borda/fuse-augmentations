---
title: Inspect fusion plans and transform matrices
description: Interpret fusion plans, segment descriptors, warp-saving estimates, and the last-segment transform matrix without overreading them.
---

# Inspect a pipeline

Introspection is how you verify that a pipeline formed the segments you intended. It is also where several names need careful interpretation.

## Human-readable plan

```python
print(augment.fusion_plan)
```

The plan distinguishes fused, exact, projective, color, crop-resize, and passthrough segments. A backend change, projective/affine transition, unsupported transform, or spatial-kernel operation can split the chain.

## Structured descriptors

```python
for segment in augment.fusion_plan_descriptors:
    print(
        segment.kind,
        segment.transforms,
        segment.backend,
        segment.barrier,
        segment.split_reason,
        segment.refused,
    )
```

Descriptors are frozen and dictionary-serializable, which makes them suitable for experiment metadata. Store the dependency versions and pipeline configuration beside them.

## `n_warps_saved` is an estimate

`n_warps_saved` summarizes collapsed operations across segments. It is useful for comparing plans, but it is not a literal count of native interpolation calls in every case: an exact flip may be counted even though the native operation was already a lossless tensor reversal.

Use it as a planning metric, then profile the real workload.

## Matrix lifetime and scope

```python
output, matrix = augment(images, return_matrix=True)
```

The matrix is the forward pixel-space matrix for the **last matrix-producing segment in that call**. It is not automatically the transform of an entire pipeline containing multiple backend segments, a projective boundary, a crop boundary, or passthrough operations.

The `transform_matrix` property exposes the same mutable last-call state and can be `None` for a pipeline with no matrix-producing segment. It is not safe as shared cross-thread request state. Prefer `return_matrix=True` when the matrix must stay paired with its output.

For a pipeline deliberately constrained to one matrix segment, the returned `(B, 3, 3)` matrix can route coordinates or be stored as augmentation provenance.
