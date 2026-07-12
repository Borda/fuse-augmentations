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

print(Compose.supported_ops("torchvision"))
print(Compose.capability_matrix())
```

`SUPPORTED_OPS` is the global canonical vocabulary. Membership in that global set does not mean every backend can build the operation.

## Live-transform registry and execution coverage

These are the registered transform classes recognized when live backend objects are passed to `Compose`.

| Backend           | Affine and exact geometry                                                                                                            | Projective          | Linear color                                                                                        | Crop-resize                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Kornia            | `RandomRotation`, `RandomAffine`, `RandomShear`, `RandomTranslate`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation90` | `RandomPerspective` | `RandomBrightness`, `RandomContrast`, brightness/contrast-only `ColorJitter`, 3-channel `Normalize` | `RandomResizedCrop`                                                                                              |
| TorchVision v1/v2 | `RandomRotation`, `RandomAffine`, `RandomHorizontalFlip`, `RandomVerticalFlip`                                                       | `RandomPerspective` | Brightness/contrast-only `ColorJitter`, 3-channel `Normalize`                                       | `RandomResizedCrop`                                                                                              |
| Albumentations    | `Affine`, `Rotate`, `SafeRotate`, `ShiftScaleRotate`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `D4`, `Transpose`          | `Perspective`       | `RandomBrightnessContrast`, standard 3-channel `Normalize`                                          | `RandomResizedCrop` is registered but remains native passthrough on the NumPy/cv2 path; it is not geo-crop fused |

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

| Operation kind                                                                                           | Behavior                                                                                         |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Nonlinear pointwise, such as saturation/hue                                                              | Native passthrough; geometric fusion stops at the operation unless reordering moves it           |
| Spatial kernel, such as blur                                                                             | Native passthrough and a segment boundary                                                        |
| Named coordinate-changing distortion on the finite refusal list, such as elastic/grid/optical distortion | Image-only passthrough; raises when auxiliary targets would become misaligned                    |
| Unsupported `ColorJitter` saturation/hue                                                                 | The pending color run falls back to native passthrough rather than dropping nonlinear components |

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
