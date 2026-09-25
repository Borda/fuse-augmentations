---
title: Generate and augment
description: Feed synthetic samples from synth_datasets into a fused augmentation pipeline, with the array layout and box coordinate conventions the two packages already share.
---

# Generate and augment

The `vision-synth` distribution ships two independent packages: `synth_datasets` generates labelled samples without `torch`, and `fused_transforms` fuses compatible augmentations on BCHW tensors. This page connects them.

Neither package imports the other, so nothing here is automatic — you own the conversion. What makes it short is that the two contracts already line up:

| Generated                                                          | Expected by the pipeline          | Conversion                                                                                    |
| ------------------------------------------------------------------ | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `Sample.image` — HWC `uint8` NumPy                                 | BCHW floating tensor              | Stack the samples to BHWC, then `NumpyToTorchConverter` transposes and divides `uint8` by 255 |
| `Annotation.bbox_xyxy` — pixel-edge `(x_min, y_min, x_max, y_max)` | `bbox_xyxy` in pixel-edge space   | Stack into `(B, N, 4)` floating tensor; no offset needed                                      |
| `Annotation.polygon`, `obb_corners` — pixel-centre                 | `keypoints` in pixel-centre space | Reshape to `(B, N, 2)`; no offset needed                                                      |

The half-pixel offset between the two conventions is the usual source of silent misalignment. Both packages already document which space each field uses, and they agree — so a box goes across without adjustment, and so does a point field.

## A generated batch through a fused pipeline

This example generates four single-object detection samples, converts them, and routes image and boxes through one fused geometric segment.

```python
import numpy as np
import torch

from fused_transforms import Compose, NumpyToTorchConverter, ReorderPolicy
from synth_datasets import SyntheticConfig, SyntheticGenerator

config = SyntheticConfig(
    img_size=96,
    task="detection",
    class_mode="shape",
    min_objects=1,
    max_objects=1,
    min_size_ratio=0.25,
    max_size_ratio=0.35,
)
samples = list(SyntheticGenerator(config).generate(4, seed=11))

images = NumpyToTorchConverter().convert(np.stack([sample.image for sample in samples]))
boxes = torch.tensor([[sample.annotations[0].bbox_xyxy] for sample in samples], dtype=torch.float32)

torch.manual_seed(3)
augment = Compose.from_params(
    rotation=(-10.0, 10.0),
    scale=(0.9, 1.1),
    hflip_p=0.5,
    reorder=ReorderPolicy.NONE,
    data_keys=["input", "bbox_xyxy"],
)
augmented_images, augmented_boxes = augment(images, boxes)

assert images.shape == (4, 3, 96, 96)
assert images.dtype == torch.float32
assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
assert augmented_images.shape == images.shape
assert augmented_boxes.shape == boxes.shape

print(augment.fusion_plan)
```

<details>
<summary>Fusion plan for the rotation, scale, and flip pipeline</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
```

</details>

`Compose.from_params` needs no optional augmentation backend, so this example runs with `pip install "vision-synth[torch]"` alone. Swap in Kornia, TorchVision, or Albumentations transform objects when your training pipeline already uses them — see [Backend pipelines](../guides/backend-pipelines.md).

## What still belongs to your code

- **Ragged batches.** The example fixes one object per image so the boxes stack. Real generated scenes vary in object count. Use [`augment_detection_batch`](detection-and-keypoints.md#augment-a-ragged-detector-batch) for one target mapping per image, or pad and mask in your collate function.
- **Labels and filtering.** The pipeline moves coordinates. It does not clip boxes, drop degenerate ones, or carry `class_id` — `augment_detection_batch` clips and filters, the plain `data_keys` path does not. See [Auxiliary targets](../guides/auxiliary-targets.md).
- **Keypoint visibility.** Generated visibility flags follow COCO (`2` visible, `1` occluded, `0` clipped away). Augmentation does not recompute them after a warp moves a point off-canvas.
- **Which transforms are safe with targets.** Only registered spatial transforms may run alongside `data_keys`; an unknown or unclassified spatial transform is rejected before any segment executes. Read [Known limitations](../known-limitations.md) before training on this path.

## When not to bother

Generated samples are already varied: `background`, `degrade`, `distractors`, `occluders`, and random per-shape rotation are all generator-side knobs, and they are baked once rather than resampled per epoch. If you want variation rather than resampling economy, change the generator configuration instead of adding an augmentation pipeline — see [Customization and extension](../datasets/customization.md).

Fused augmentation earns its place when you need fresh geometry on every epoch from a fixed exported set, or when the same pipeline must also run on real images later.
