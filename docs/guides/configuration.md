---
title: Configure portable augmentation pipelines
description: Define TransformSpec objects, query backend capabilities, choose strict unsupported-operation behavior, and preserve operation order.
---

# Declarative configuration

`TransformSpec` separates an augmentation idea from one concrete backend object. Resolve the specs at construction time and reject unsupported operations early.

<!--phmdoctest-share-names-->

```python
from fuse_augmentations import Compose, ReorderPolicy, TransformSpec

specs = [
    TransformSpec(
        operation="rotation",
        params={"degrees": (-15.0, 15.0)},
        prob=0.8,
    ),
    TransformSpec(operation="hflip", params={}, prob=0.5),
]

augment = Compose.from_config(
    specs,
    backend="kornia",
    on_unsupported="raise",
    reorder=ReorderPolicy.NONE,
)
```

## Query before resolving

The global operation vocabulary is larger than any one backend's constructible set. Query the current environment instead of copying a static assumption into application code:

```python
from fuse_augmentations import Compose

print(sorted(Compose.supported_ops("native")))
print(
    {
        backend: len(operations)
        for backend, operations in sorted(Compose.capability_matrix().items())
    }
)
```

<details>
<summary>Native operations and capability counts by backend</summary>

```
['brightness', 'contrast', 'hflip', 'rotation', 'scale', 'shear', 'translate', 'vflip']
{'albumentations': 8, 'kornia': 12, 'native': 8, 'torchvision': 7}
```

</details>

An optional backend that is not installed reports an empty capability set. The matrix describes what the declarative resolver can construct; the live-transform adapter tables include some additional concrete classes and parameter restrictions.

## Reject or skip unsupported specs

The default and recommended behavior is `on_unsupported="raise"`. It aggregates invalid specifications into one `ValueError`.

`on_unsupported="warn_skip"` drops unsupported operations with a warning. Use it only when an intentionally reduced pipeline is acceptable; skipping an operation changes the experiment or training distribution.

## Preserve semantics by default

`Compose(...)` defaults to `ReorderPolicy.NONE`, but `from_config` and `from_params` default to `POINTWISE`. Reordering can move color operations across geometry and change border pixels or clamped values.

For reproducible or parity-sensitive work, pass this explicitly:

```python
reorder = ReorderPolicy.NONE
```

Only enable `POINTWISE` after measuring the output and performance trade-off. `AGGRESSIVE` currently follows the same implementation as `POINTWISE`; it is not a stronger optimizer today.

## Constant border colour (`fill`)

`padding_mode` chooses *how* the region outside the source canvas is produced — `"zeros"` writes black, `"border"` replicates the edge pixel, `"reflection"` mirrors the image. `fill` replaces the constant that `"zeros"` writes, in the image's own value range:

```python
import torch

from fuse_augmentations import Compose

image = torch.full((1, 3, 32, 32), 0.8)
augment = Compose.from_params(translate_x=(8.0, 8.0), fill=114.0 / 255.0)
out = augment(image)
print(round(float(out[0, 0, 16, 0]), 3), round(float(out[0, 0, 16, 31]), 3))
```

```
0.447 0.8
```

- **Units are the image's own.** A float image in `[0, 1]` takes `114 / 255`; a uint8 image on the Albumentations NumPy path takes `114`. Nothing rescales the value.
- **Scalar or per-channel.** `fill=0.447` fills every channel; `fill=(0.1, 0.2, 0.3)` fills one channel each and must match the image's channel count.
- **Image only.** A routed mask keeps its zero padding whatever the image fill is — outside the canvas a mask means "no instance", not a colour. The same holds for boxes and keypoints, which are mapped rather than sampled.
- **Requires `padding_mode="zeros"`** (the default). `"border"` and `"reflection"` have no constant to replace and `"per_transform"` picks a mode per transform, so combining any of them with `fill` raises rather than ignoring the argument.
- **Executor-independent.** The torch warp has no constant padding mode in `grid_sample`, so it subtracts the fill, samples against the zero border and adds it back; the cv2 warps pass it as `borderValue`. Both produce the same border, including on the batch-size-dependent cv2 fast path.
- `fill=None` (the default) keeps the plain zero border, unchanged.

## Low-precision execution (`pipeline_dtype`)

`pipeline_dtype="bfloat16"` or `pipeline_dtype="float16"` runs the fused affine/projective/crop warp and the fused color/LUT applies in that dtype. Matrix composition and inversion stay in float32 or float64, and the returned image is cast back to its input dtype, so the low-precision path is confined to the sampling and lookup cores.

```python
augment = Compose.from_params(rotation=(-15.0, 15.0), pipeline_dtype="bfloat16")
```

CPU ignores this option and keeps the existing float32/float64 path; it only affects non-CPU execution. Reach for it when non-CPU memory pressure or throughput is the bottleneck, and expect a numeric difference from the fp32 path rather than a guaranteed speedup.

## Serialize specs

`TransformSpec` is a frozen value object with dictionary helpers:

```python
payload = [spec.to_dict() for spec in specs]
restored = [TransformSpec.from_dict(item) for item in payload]

assert restored == specs
```

Keep probability in `TransformSpec.prob`; placing `prob` inside `params` is rejected to prevent shadowing.
