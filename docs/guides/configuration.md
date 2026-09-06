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

## Letterbox with an exact inverse (`letterbox`)

`letterbox=(height, width)` (or a single `int` for a square) appends an aspect-preserving fit: content is scaled by one ratio `r = min(height_out / H, width_out / W)` and the slack is padded with `fill`. Because the map is a pure scale plus translation, it inverts exactly — a prediction made on the letterboxed canvas maps back to source coordinates with no resampling, which is what makes it usable as an inference preprocessor and not only a training resize.

```python
import torch

from fuse_augmentations import Compose, letterbox_matrix, transform_keypoints
from fuse_augmentations.affine.matrix import inv3x3

augment = Compose.from_params(
    rotation=(20.0, 20.0), letterbox=(32, 32), fill=114.0 / 255.0
)
out, matrix = augment(torch.full((1, 3, 20, 40), 0.8), return_matrix=True)
print(out.shape, len(augment.fusion_plan_descriptors))

forward = letterbox_matrix(
    height_in=20, width_in=40, height_out=32, width_out=32, dtype=torch.float64
)
point = torch.tensor([[[7.5, 3.25]]], dtype=torch.float64)
recovered = transform_keypoints(transform_keypoints(point, forward), inv3x3(forward))
print([round(value, 12) for value in recovered.flatten().tolist()])
```

```
torch.Size([1, 3, 32, 32]) 1
[7.5, 3.25]
```

- **One resample, not two.** The letterbox is a `CROP_RESIZE_FIXED` op, so a geometric run in front of it composes into a single matrix warped once, straight from the source canvas to the letterboxed one — the `1` printed above is the segment count. `return_matrix=True` then hands back the whole chain (`M_letterbox @ M_geo`), not the letterbox alone.
- **Ordering.** The letterbox is placed after the geometry and before any `brightness`/`contrast`, because the crop-resize category is a reorder barrier and a colour op buffered ahead of it would be flushed between the geometric run and the letterbox, splitting the fusion. A colour op in the same call therefore also acts on the pad region.
- **`allow_upscale=False`** caps the ratio at `1.0`, so a source smaller than the canvas is padded rather than magnified.
- **Padding is integer and floor-halved.** An odd slack puts the extra pixel on the right or the bottom. The image uses the reported scale-and-pad matrix; do not assume pixel identity with a separate resize implementation that rounds each axis independently.
- **Rounded before printing.** Exact here means the map inverts without resampling, not that the float64 round trip is bit-identical: composing and inverting leaves the last bits to the linear algebra build in use, so the example rounds rather than printing a bit pattern that differs between PyTorch versions.
- **`letterbox_matrix` / `letterbox_geometry`** expose the same fit as data, computed from sizes alone. An evaluation loop that no longer has the pipeline can recompute the matrix and invert it with `inv3x3`.
- **Standalone letterbox reports its actual-call matrix.** `return_matrix=True` also works without preceding geometry. General random crops do not promise recoverable metadata; image `inverse()` still accepts only one supported fused affine/projective segment. Use coordinate helpers with the paired matrix for letterbox predictions.
- Backend-free mode only; `backend=` raises rather than accepting and ignoring the argument.

## Downscale antialiasing (`antialias`)

`Compose([...], antialias=True)` requires the optional Kornia dependency and raises at construction if it is missing. The default is `False`.

Supported crop-resize and fused geometry-plus-crop segments estimate scales once per warp. A sample shrinking below the filtering threshold on either axis receives its own Gaussian prefilter: axis-aligned scaling uses separate width/height scales; rotation or shear uses the smallest singular value conservatively on both axes. Each sample's kernel support is independent of its batch neighbors. Masks retain their requested interpolation and are not blurred with the image.

This changes the previous batch-wide filtering behavior. It can cost additional filtering calls and host synchronization when enabled; measure your actual batch, device, and crop distribution. It is not a universal antialias switch for every segment or a guarantee of better model accuracy.

## Half-pixel convention (`align_corners`)

Sampling runs with `align_corners=True`, and matrices are normalized with the sandwich derived for that same flag (`2 / (W - 1)` scale, `(W - 1) / 2` offset). Because the two agree, **a pixel-space matrix carries no convention**: the map this package applies is the one an `align_corners=False` implementation — TorchVision, Albumentations, most YOLO data pipelines — would apply for the same matrix. Only the normalization into `[-1, 1]` is convention-bound, and it cancels against the sampling flag.

Practically, for anyone porting matrices in or out:

- A forward pixel matrix (`transform_matrix`, or one you build yourself) transfers unchanged in both directions. This is measured, not assumed: `tests/test_unit/affine/test_coordinate_convention.py` warps against an independently built `align_corners=False` reference for integer, half- and quarter-pixel shifts, up- and downscales, rotation, a non-square canvas, and differing input/output sizes, and matches to float32 rounding.
- Keypoints use the pixel-centre matrix directly; masks use the image grid. Axis-aligned boxes use pixel-edge extents `[0, W] x [0, H]`: their helpers internally conjugate the image matrix by half-pixel translations. Rotated-box centres remain in pixel-centre space; add `0.5` to their corner envelope before comparing it with an edge-space AABB.

Two exceptions, both tested:

- **`padding_mode="reflection"` is convention-bound.** `align_corners=True` reflects about the outer pixel *centres* (OpenCV's `BORDER_REFLECT_101`); `align_corners=False` reflects about the outer pixel *edges* (`BORDER_REFLECT`). The mirrored band differs by a pixel of phase. `"zeros"` and `"border"` agree between conventions; only reflection does not.
- **A canvas thinner than two pixels is refused.** The `True` normalization divides by `L - 1`, which is singular for a one-pixel axis, so a `(H, 1)` or `(1, W)` input raises naming the offending axis rather than warping through an infinite scale.

There is deliberately no `align_corners` parameter. It would change nothing for every case above except reflection padding, and a flag that alters one padding mode while claiming to select a coordinate convention is worse than the documented behaviour it replaces.

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
- **Image only.** Routed masks use their independent scalar `mask_fill`, default `0`; set an appropriate ignore label such as `255` for a segmentation loss. Image `fill` never changes that mask value. Boxes and keypoints are mapped rather than sampled.
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
