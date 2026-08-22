---
title: Fused augmentation for detection and keypoints
description: Transform bounding boxes and keypoints as dense tensors, then apply the clipping, filtering, and keep-mask postprocessing the package deliberately leaves to you.
---

# Detection and keypoints

The package applies matrix math to coordinate tensors. It is not a detection framework's target processor, and the difference matters more here than anywhere else in these docs.

What it does: transforms `(B, N, 4)` boxes and `(B, N, 2)` keypoints through the same fused geometry as the image.

What it does not do: clip coordinates to the image, drop boxes that left the frame, shrink boxes to their visible part, remove degenerate areas, propagate class labels alongside a filtered box tensor, accept a ragged `N` per image, or update keypoint visibility flags.

Skip that postprocessing and you train on boxes that sit outside the image, on one-pixel slivers, and on labels that no longer line up with their coordinates.

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(11)

detect = Compose.from_params(
    rotation=(30.0, 30.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(1, 3, 128, 128)
boxes = torch.tensor(
    [[[10.0, 10.0, 40.0, 40.0], [90.0, 90.0, 126.0, 126.0], [0.0, 0.0, 3.0, 3.0]]]
)

image_out, boxes_out = detect(images, boxes)

print(tuple(boxes_out.shape))
```

<details>
<summary>Box tensor shape after a fixed 30-degree rotation</summary>

```
(1, 3, 4)
```

</details>

Three boxes in, three boxes out — including the one now rotated off the canvas and the three-pixel sliver. The tensor shape is preserved because filtering is your decision, not the package's.

## The postprocessing you own

```python
labels = torch.tensor([[1, 2, 3]])

clipped = boxes_out.clone()
clipped[..., 0::2] = clipped[..., 0::2].clamp(0, 127)
clipped[..., 1::2] = clipped[..., 1::2].clamp(0, 127)

widths = clipped[..., 2] - clipped[..., 0]
heights = clipped[..., 3] - clipped[..., 1]
keep = (widths > 2) & (heights > 2)

kept_boxes = clipped[keep]
kept_labels = labels[keep]

print(int(keep.sum()))
print(tuple(kept_boxes.shape), tuple(kept_labels.shape))
```

<details>
<summary>Surviving box count and the aligned label tensor</summary>

```
2
(2, 4) (2,)
```

</details>

The important line is `labels[keep]`. The same boolean mask has to be applied to every per-instance array you carry — classes, track identifiers, crowd flags, scores — or your labels silently shift by one instance for the rest of training.

Once boxes are filtered per image, `N` differs across the batch, so the result is a ragged list rather than a padded tensor. Re-pad it, or collate it the way your detector expects, before it reaches the model.

## Rotation inflates axis-aligned boxes

A rotated rectangle is not a rectangle. The package returns the axis-aligned bounding box of the rotated corners, which is strictly larger than the original for any rotation that is not a multiple of 90 degrees. Repeated augmentation therefore grows boxes over epochs if you feed outputs back as inputs.

[Test-time augmentation](test-time-augmentation.md) measures that inflation on a round trip.

## Keypoints

Keypoints transform as `(B, N, 2)` coordinates. Visibility is not part of that tensor, so a keypoint pushed outside the image keeps whatever flag it had. Recompute visibility from the transformed coordinates and the output extent yourself — the standard COCO convention of `2` visible, `1` occluded, `0` unlabelled is not something the package can infer.
