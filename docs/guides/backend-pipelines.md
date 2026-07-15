---
title: Build native and backend-specific augmentation pipelines
description: Use the native builder or registered Kornia, TorchVision, and Albumentations transforms through the package's BCHW tensor contract.
---

# Backend pipelines

`Compose` accepts registered transform objects from Kornia, TorchVision, and Albumentations, but it does not inherit every native container's input or numerical contract. The common safe interface is a batch-first BCHW torch tensor.

## Native builder

Use `from_params` when you need rotation, scale, shear, translation, horizontal or vertical flips, brightness, or contrast without an optional augmentation backend:

```python
from fuse_augmentations import Compose, ReorderPolicy

augment = Compose.from_params(
    rotation=(-20.0, 20.0),
    shear_x=(-5.0, 5.0),
    brightness=0.1,
    reorder=ReorderPolicy.NONE,
)
```

PyTorch is still mandatory; “native” means independent of an augmentation backend, not framework-free.

## Kornia

```python
import kornia.augmentation as K
import torch

from fuse_augmentations import Compose, ReorderPolicy

augment = Compose(
    [
        K.RandomRotation(degrees=15.0, p=0.8),
        K.RandomAffine(degrees=0.0, scale=(0.9, 1.1), p=0.7),
        K.RandomHorizontalFlip(p=0.5),
    ],
    reorder=ReorderPolicy.NONE,
)

output = augment(torch.rand(8, 3, 224, 224))
```

Kornia has the broadest declarative capability matrix in the current package. Registered transforms still use package-level segment interpolation and padding semantics.

## TorchVision

```python
import torch
import torchvision.transforms.v2 as T

from fuse_augmentations import Compose

augment = Compose(
    [
        T.RandomRotation(degrees=15.0),
        T.RandomAffine(degrees=0.0, scale=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
    ]
)

output = augment(torch.rand(8, 3, 224, 224))
```

The fused path requires BCHW input. A native TorchVision `Compose` may accept CHW tensors or PIL images; this package does not generally do so. `RandomRotation(expand=True)` is explicitly unsupported, and center/fill/interpolation behavior is not pixel-equivalent to native TorchVision.

## Albumentations

The cross-backend tensor path accepts BCHW float tensors:

<!--phmdoctest-share-names-->

```python
import albumentations as A
import torch

from fuse_augmentations import Compose

augment = Compose(
    [A.Rotate(limit=15, p=0.8), A.HorizontalFlip(p=0.5)],
    execution="cv2",
)

output = augment(torch.rand(4, 3, 224, 224))
```

`execution="cv2"` is the default CPU path. `execution="torch"` applies the composed matrices through a batched torch sampling grid and can remain on a torch device, but its border and subpixel numerics differ from OpenCV.

An image-only HWC NumPy compatibility path also exists:

```python
import numpy as np

image = np.zeros((224, 224, 3), dtype=np.uint8)
result = augment(image=image)
assert result["image"].shape == image.shape
```

This is not a full Albumentations dictionary replacement: NumPy masks, boxes, keypoints, labels, and processor behavior are not supported through that special path. Tensor passthrough converts images into float32 `[0, 1]`; native Albumentations transforms that expect uint8 ranges can behave incorrectly.

## Mixed backends

```python
import kornia.augmentation as K
import torchvision.transforms.v2 as T

from fuse_augmentations import Compose

augment = Compose(
    [
        T.RandomRotation(10.0),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomBrightness(brightness=(0.9, 1.1), p=1.0),
    ]
)
```

Each transform is routed to its registered adapter. A backend change is a hard segment boundary: matrices from different backends are not combined into one warp. Inspect `fusion_plan_descriptors` to make the cost visible.

## Passthrough warnings are contracts

An unregistered transform becomes a passthrough barrier when the adapter can safely call it. That does not mean the operation is transparent or target-safe.

!!! danger Never ignore an `Unknown ... SPATIAL_KERNEL barrier` warning in a pipeline with `data_keys`. Unsupported spatial transforms can modify only the image and leave auxiliary targets stale. Replace the transform with a registered operation, split the pipeline and route every target yourself, or do not use this package for that pipeline.
