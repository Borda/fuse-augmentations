---
title: Capabilities
description: Exact backend, transform, input, fusion, passthrough, and refusal boundaries.
---

# Capabilities

The package supports a precise registry of augmentation transforms; it does not promise every transform or every constructor option exposed by Kornia, TorchVision, or Albumentations. Check both tables below: declarative construction coverage and live-transform fusion coverage are related but not identical.

## Supported data contract

The standard pipeline contract is:

- image input: PyTorch tensor in `(B, C, H, W)` layout;
- internal execution: PyTorch tensors, with path-specific OpenCV/NumPy conversion where required;
- optional image output: channel-last NumPy via `output_backend="numpy"`;
- auxiliary targets: positional tensors declared with `data_keys`;
- supported coordinate systems: pixel-space boxes and keypoints.

An image-only Albumentations pipeline also has a native-style `pipe(image=<HWC ndarray>) -> {"image": ndarray}` path. That path does not accept auxiliary keys. Use tensor inputs with `data_keys` for masks, boxes, or keypoints.

The package is therefore Compose-like for supported tensor pipelines, not a universal behavioral replacement for each backend's native container. PIL input, arbitrary tv-tensors, arbitrary backend dictionaries, and every native Compose option are not part of the general contract.

## Declarative construction matrix

This matrix describes canonical operations accepted by `Compose.from_config`. It should match `Compose.capability_matrix()` for an environment in which all optional backends are installed.

| Canonical operation | Kornia | TorchVision | Albumentations | Native |
| ------------------- | :----: | :---------: | :------------: | :----: |
| `rotation`          |   ✓    |      ✓      |       ✓        |   ✓    |
| `affine`            |   ✓    |      ✓      |       ✓        |   —    |
| `shear`             |   ✓    |      —      |       —        |   ✓    |
| `translate`         |   ✓    |      —      |       —        |   ✓    |
| `hflip`             |   ✓    |      ✓      |       ✓        |   ✓    |
| `vflip`             |   ✓    |      ✓      |       ✓        |   ✓    |
| `scale`             |   ✓    |      ✓      |       ✓        |   ✓    |
| `perspective`       |   ✓    |      ✓      |       ✓        |   —    |
| `rotation90`        |   ✓    |      —      |       ✓        |   —    |
| `normalize`         |   ✓    |      ✓      |       ✓        |   —    |
| `brightness`        |   ✓    |      —      |       —        |   ✓    |
| `contrast`          |   ✓    |      —      |       —        |   ✓    |

An optional backend that is not installed reports an empty capability set for that backend. Query the running environment instead of hard-coding assumptions:

```python
from fuse_augmentations import Compose

print(sorted(Compose.supported_ops("torchvision")))
print(
    {
        backend: len(operations)
        for backend, operations in sorted(Compose.capability_matrix().items())
    }
)
```

<details>
<summary>TorchVision operations and capability counts by backend</summary>

```
['affine', 'hflip', 'normalize', 'perspective', 'rotation', 'scale', 'vflip']
{'albumentations': 8, 'kornia': 12, 'native': 8, 'torchvision': 7}
```

</details>

`SUPPORTED_OPS` is the global canonical vocabulary. Membership in that global set does not mean every backend can build the operation.

## Live-transform registry and execution coverage

These are the registered transform classes recognized when live backend objects are passed to `Compose`. Rows are primitives named by the most common class name across backends; a check means the backend registers that class name, otherwise the cell shows the backend-specific class or a note.

=== "Geometry"

    | Primitive              | Kornia             | TorchVision v1/v2  | Albumentations               |
    | ---------------------- | ------------------ | ------------------ | ---------------------------- |
    | `RandomRotation`       | ✓                  | ✓ (`expand=False`) | `Rotate`, `SafeRotate`       |
    | `RandomAffine`         | ✓                  | ✓                  | `Affine`, `ShiftScaleRotate` |
    | `RandomShear`          | ✓                  | —                  | —                            |
    | `RandomTranslate`      | ✓                  | —                  | —                            |
    | `RandomHorizontalFlip` | ✓                  | ✓                  | `HorizontalFlip`             |
    | `RandomVerticalFlip`   | ✓                  | ✓                  | `VerticalFlip`               |
    | `RandomRotate90`       | `RandomRotation90` | —                  | ✓                            |
    | `D4`                   | —                  | —                  | ✓                            |
    | `Transpose`            | —                  | —                  | ✓                            |
    | `RandomPerspective`    | ✓                  | ✓                  | `Perspective`                |
    | `RandomResizedCrop`    | ✓                  | ✓                  | ✓ (see note below)           |

    Albumentations `RandomResizedCrop`: on the NumPy/cv2 path it is never geo-crop fused; it is native passthrough for image-only pipelines but routes to a standalone crop-resize segment when `data_keys` auxiliary targets are present.

=== "Color"

    | Primitive          | Kornia                  | TorchVision v1/v2       | Albumentations                            |
    | ------------------ | ----------------------- | ----------------------- | ----------------------------------------- |
    | `RandomBrightness` | ✓                       | via `ColorJitter`       | via `RandomBrightnessContrast`            |
    | `RandomContrast`   | ✓                       | via `ColorJitter`       | via `RandomBrightnessContrast`            |
    | `ColorJitter`      | ✓ (brightness/contrast) | ✓ (brightness/contrast) | —                                         |
    | `Normalize`        | ✓ (3-channel RGB)       | ✓ (3-channel RGB)       | ✓ (standard affine mode, 3-channel RGB)   |
    | `RandomGamma`      | ✓                       | —                       | ✓                                         |
    | `RandomSolarize`   | ✓                       | ✓                       | `Solarize`                                |
    | `RandomPosterize`  | ✓                       | ✓                       | `Posterize`                               |
    | `RandomEqualize`   | ✓ (float tensors only)  | ✓                       | `Equalize` (unmasked, `by_channels=True`) |

Important parameter limits:

- TorchVision `RandomRotation(expand=True)` is refused by the fused engine.
- TorchVision and Kornia `ColorJitter` fuse only when saturation and hue are identity.
- Fused normalization requires three RGB mean/std values.
- Albumentations normalization fuses only for its standard affine mode; image-statistics modes remain native passthrough.
- Padding mode and interpolation are segment-level choices for a fused run; per-transform differences inside one run are not preserved.

## Backend-free construction

“Backend-free” means no Kornia, TorchVision, or Albumentations dependency is needed. PyTorch remains required.

`Compose.from_params` directly supports:

- rotation;
- uniform or per-axis scale;
- x/y shear;
- x/y pixel translation;
- horizontal and vertical flip;
- brightness and contrast.

The native declarative backend exposes the canonical operations `rotation`, `scale`, `shear`, `translate`, `hflip`, `vflip`, `brightness`, and `contrast`. It does not currently construct affine bundles, perspective, rotation90, normalize, or crop-resize.

## What passes through

Unregistered transforms from a recognized backend are normally classified as barriers and executed through that backend's native transform call. Examples include spatial kernels and nonlinear color operations.

| Operation kind                                                                                           | Behavior                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nonlinear pointwise, such as saturation/hue                                                              | Native passthrough; geometric fusion stops at the operation unless reordering moves it                                                                                                                                                                                             |
| Spatial kernel other than Gaussian blur, such as Kornia sharpness                                        | Native passthrough and a segment boundary                                                                                                                                                                                                                                          |
| Gaussian blur                                                                                            | Folds with adjacent Gaussian blurs and, when the following affine does not downscale, commutes past it so that affine collapses to one warp; otherwise native passthrough and a segment boundary. See [How fusion works](how-fusion-works.md#segmentation-comes-before-execution). |
| Named coordinate-changing distortion on the finite refusal list, such as elastic/grid/optical distortion | Image-only passthrough; raises when auxiliary targets would become misaligned                                                                                                                                                                                                      |
| Unsupported `ColorJitter` saturation/hue                                                                 | The pending color run falls back to native passthrough rather than dropping nonlinear components                                                                                                                                                                                   |

Passthrough means “call the native transform on the package's supported data representation,” not bit-for-bit transparency for every native workflow.

### Albumentations value-domain warning

The standard tensor pipeline sends Albumentations passthrough transforms HWC float32 arrays in `[0, 1]`. Albumentations operations that assume uint8 `[0, 255]`—including some noise, fog, and compression transforms—can produce the wrong magnitude or a no-op without raising. Apply those transforms outside this tensor pipeline, or use only transforms documented as float-safe.

## What raises or refuses

| Situation                                                                                                                   | Result                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| A normal `Compose` contains no recognized backend transform                                                                 | Construction raises `ValueError`                                                          |
| An unknown foreign callable is mixed into an Albumentations pipeline                                                        | Construction raises because the dict call convention is unsafe                            |
| TorchVision `RandomRotation(expand=True)`                                                                                   | Construction raises `ValueError`                                                          |
| A passthrough whose exact class name appears on the finite coordinate-changing refusal list executes with auxiliary targets | Forward raises `ValueError` to prevent silent target misalignment                         |
| Native Albumentations NumPy call includes auxiliary keyword keys                                                            | Call raises `NotImplementedError`; use tensor `data_keys` instead                         |
| `from_config` receives an unsupported operation                                                                             | Default `on_unsupported="raise"` aggregates failures; `"warn_skip"` explicitly drops them |
| Native construction requests perspective, normalize, rotation90, or crop-resize                                             | Construction raises as unsupported                                                        |

## Multi-backend limits

Kornia, TorchVision, and Albumentations transforms can share one pipeline. Consecutive transforms from different backends cannot share a fused segment. Each backend boundary therefore limits the attainable warp reduction.

Unknown transforms in a mixed pipeline also need special care. Tensor-call-compatible unknown operations may pass through beside Kornia or TorchVision, but unknown operations are rejected beside Albumentations because its native transform call expects a dictionary.

## Auxiliary targets

Supported `data_keys` are:

| Key         | Shape          | Behavior                                                            |
| ----------- | -------------- | ------------------------------------------------------------------- |
| `input`     | `(B, C, H, W)` | Image; must be first                                                |
| `mask`      | `(B, C, H, W)` | Nearest by default; bilinear requires floating point                |
| `bbox_xyxy` | `(B, N, 4)`    | Pixel-space corners; rotations return an enclosing axis-aligned box |
| `bbox_xywh` | `(B, N, 4)`    | Converted through xyxy internally                                   |
| `keypoints` | `(B, N, 2)`    | Pixel-space homogeneous transform                                   |

Known kernel and pointwise passthrough operations leave auxiliary geometry unchanged. A passthrough raises only when its exact class name appears on the current coordinate-changing refusal list. Unknown spatial transforms are not structurally detected and may still run on the image only; treat every `Unknown ... SPATIAL_KERNEL barrier` warning as unsafe with `data_keys`.

## Experimental extension point

`fuse_augmentations._backend.register_adapter` and the `fuse_augmentations.adapters` entry-point group are experimental internals. In the current implementation, third-party entries participate in backend detection, but a default third-party `Backend.UNKNOWN` entry is not an end-to-end `Compose` execution path. Do not build a production integration around this mechanism until the project publishes and tests a complete registration → construction → forward contract.

## Related pages

- [How fusion works](how-fusion-works.md)
- [Core API](../reference/core.md)
- [Configuration API](../reference/configuration.md)
- [Advanced API](../reference/advanced.md)
