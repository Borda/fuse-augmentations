---
title: Inspect fusion plans and transform matrices
description: Interpret fusion plans, segment descriptors, warp-saving estimates, and the last-segment transform matrix without overreading them.
---

# Inspect a pipeline

Introspection is how you verify that a pipeline formed the segments you intended. It is also where several names need careful interpretation.

## Human-readable plan

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose

torch.manual_seed(7)
augment = Compose.from_params(rotation=(-15.0, 15.0), hflip_p=0.5)
images = torch.rand(2, 3, 32, 32)

print(augment.fusion_plan)
```

<details>
<summary>Human-readable plan for the configured pipeline</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
```

</details>

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

<details>
<summary>Structured descriptor fields for the configured segment</summary>

```
fused ('_DirectParamTransform', '_DirectFlipTransform') None None None None
```

</details>

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

## Test-time de-augmentation with `inverse`

Pair a `return_matrix=True` output with `inverse(prediction, matrix=...)` to map a prediction back into the original geometric frame:

```python
translate_pipe = Compose.from_params(translate_x=(2.0, 2.0))
translate_images = torch.rand(1, 3, 8, 8)
augmented, matrix = translate_pipe(translate_images, return_matrix=True)
recovered = translate_pipe.inverse(augmented, matrix=matrix)
```

Pass the `matrix` returned by the same forward call rather than reading `transform_matrix`. `inverse` does not read that mutable property, so pairing a call's own matrix this way is safe under concurrent calls.

`inverse` supports one fused affine or projective segment, including a chain already fused into that segment. It raises `ValueError` instead of guessing for:

- crop-resize segments (`CropResizeSegment`, `_FusedGeoCropSegment`) — crop-resize discards pixels outside the crop;
- non-geometric segments (`FusedColorSegment`, `FusedLUTSegment`, `FusedGaussianBlurSegment`) — color, LUT, and blur segments carry no geometric matrix;
- passthrough segments — no recorded matrix;
- exact-only segments (`ExactAffineSegment`) — flips and D4/90° ops have no recorded matrix;
- multi-segment pipelines — `return_matrix` records only the last segment's matrix;
- a missing paired `matrix` argument.

Auxiliary targets recover at different fidelities. Keypoints and masks recover to sampling precision. Bounding boxes are axis-aligned (AABB), so a forward-then-inverse box is exact only for axis-aligned transforms (flip, scale, translation) and inflates under a rotation, shear, or projective warp.
