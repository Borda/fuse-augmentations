---
title: Test-time augmentation with an invertible fused segment
description: Augment at inference, map predictions back to the original frame with the paired forward matrix, and measure where the inverse is exact.
---

# Test-time augmentation

Test-time augmentation predicts on several transformed copies of an image and merges the results. The merge only works if every prediction is first mapped back to the original frame, and that is what `inverse` does for a single fused segment.

Pass the matrix returned by the exact same forward call. `transform_matrix` is mutable compatibility state, so reading it afterwards is unsafe under concurrency, and a matrix from a different call produces silently wrong geometry with no error.

<!--phmdoctest-share-names-->

```python
import torch

from fuse_augmentations import Compose, ReorderPolicy

torch.manual_seed(13)

shift = Compose.from_params(
    translate_x=(6.0, 6.0),
    translate_y=(-4.0, -4.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

images = torch.rand(1, 3, 128, 128)
boxes = torch.tensor([[[20.0, 20.0, 60.0, 60.0]]])

(augmented, augmented_boxes), matrix = shift(images, boxes, return_matrix=True)
restored, restored_boxes = shift.inverse(augmented, augmented_boxes, matrix=matrix)

print(tuple(restored.shape), bool(torch.allclose(restored_boxes, boxes, atol=1e-3)))
```

<details>
<summary>Round-tripped image shape and exact box recovery under translation</summary>

```
(1, 3, 128, 128) True
```

</details>

Under translation, scaling, and flips the box round trip is exact, because an axis-aligned transform maps a rectangle to a rectangle. The image recovers to sampling precision rather than bit-exactly: the inverse applies a second `grid_sample`, and two interpolations are not the identity.

## Rotation does not round trip for boxes

```python
torch.manual_seed(19)

rotate = Compose.from_params(
    rotation=(25.0, 25.0),
    data_keys=["input", "bbox_xyxy"],
    reorder=ReorderPolicy.NONE,
)

(rotated, rotated_boxes), rotate_matrix = rotate(images, boxes, return_matrix=True)
_, recovered_boxes = rotate.inverse(rotated, rotated_boxes, matrix=rotate_matrix)


def area(box: torch.Tensor) -> torch.Tensor:
    """Return the axis-aligned area of an xyxy box tensor."""
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


print(bool((area(recovered_boxes) > area(boxes)).all().item()))
print(round(float(area(recovered_boxes).item() / area(boxes).item()), 1))
```

<details>
<summary>Box area grows across a rotate-then-invert round trip</summary>

```
True
3.1
```

</details>

The forward rotation replaces the box with the axis-aligned hull of its rotated corners, and the inverse rotation takes the hull of *that* larger rectangle. Two hull operations at 25 degrees inflate the area by roughly three times. Merging boxes across rotated test-time views therefore needs oriented boxes or mask-based merging, not this round trip.

Masks and keypoints do not have this problem: they are sampled or mapped pointwise, so they recover to sampling precision under any supported transform.

## One segment only

`inverse` supports a single fused affine or projective segment — a chain already fused into that one segment is fine, but a pipeline that splits into geometry, color, and geometry again is not invertible this way. Check `fusion_plan` before designing around it, and see [Introspection](../guides/introspection.md) for the mechanics.
