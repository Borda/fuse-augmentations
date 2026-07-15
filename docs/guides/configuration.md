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

## Serialize specs

`TransformSpec` is a frozen value object with dictionary helpers:

```python
payload = [spec.to_dict() for spec in specs]
restored = [TransformSpec.from_dict(item) for item in payload]

assert restored == specs
```

Keep probability in `TransformSpec.prob`; placing `prob` inside `params` is rejected to prevent shadowing.
