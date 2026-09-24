"""End-to-end detector contract for the ragged-to-dense augmentation boundary."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fuse_augmentations import Compose
from fuse_augmentations._compat import _TORCHVISION_AVAILABLE
from fuse_augmentations.detection import augment_detection_batch

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign


pytestmark = pytest.mark.integration


class _TinyBackbone(nn.Module):
    """Provide one feature map for a CPU-sized Faster R-CNN regression step."""

    out_channels = 8

    def __init__(self) -> None:
        """Build a minimal feature extractor without pretrained weights."""
        super().__init__()
        self.conv = nn.Conv2d(3, self.out_channels, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the single feature-map level Faster R-CNN consumes."""
        return {"0": self.conv(images)}


def _tiny_fasterrcnn() -> FasterRCNN:
    """Build a download-free Faster R-CNN that can consume 32-pixel targets."""
    backbone = _TinyBackbone()
    anchors = AnchorGenerator(sizes=((16,),), aspect_ratios=((1.0,),))
    pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=2, sampling_ratio=1)
    return FasterRCNN(
        backbone,
        num_classes=3,
        min_size=32,
        max_size=32,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        rpn_anchor_generator=anchors,
        box_roi_pool=pooler,
        rpn_pre_nms_top_n_train=16,
        rpn_post_nms_top_n_train=8,
        rpn_batch_size_per_image=16,
        box_batch_size_per_image=16,
    )


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision detection models are optional")
def test_detection_boundary_filters_middle_instance_and_runs_a_real_training_step():
    """A translated-away middle instance cannot desynchronize labels, crowds, or areas."""
    images = torch.rand(2, 3, 32, 32)
    targets = [
        {
            "boxes": torch.tensor([[10.0, 4.0, 16.0, 10.0], [0.0, 12.0, 4.0, 18.0], [24.0, 20.0, 30.0, 28.0]]),
            "labels": torch.tensor([1, 2, 1]),
            "area": torch.tensor([99.0, 99.0, 99.0]),
            "iscrowd": torch.tensor([0, 1, 0]),
            "image_id": torch.tensor([40]),
        },
        {
            "boxes": torch.empty(0, 4),
            "labels": torch.empty(0, dtype=torch.int64),
            "area": torch.empty(0),
            "iscrowd": torch.empty(0, dtype=torch.int64),
            "image_id": torch.tensor([41]),
        },
    ]
    originals = [{key: value.clone() for key, value in target.items()} for target in targets]
    pipe = Compose.from_params(translate_x=(-8.0, -8.0), data_keys=["input", "bbox_xyxy"])

    augmented_images, augmented_targets = augment_detection_batch(pipe, images, targets)

    assert torch.equal(augmented_targets[0]["boxes"], torch.tensor([[2.0, 4.0, 8.0, 10.0], [16.0, 20.0, 22.0, 28.0]]))
    assert torch.equal(augmented_targets[0]["labels"], torch.tensor([1, 1]))
    assert torch.equal(augmented_targets[0]["iscrowd"], torch.tensor([0, 0]))
    assert torch.equal(augmented_targets[0]["area"], torch.tensor([36.0, 48.0]))
    assert torch.equal(augmented_targets[0]["image_id"], torch.tensor([40]))
    assert augmented_targets[1]["boxes"].shape == (0, 4)
    assert augmented_targets[1]["labels"].shape == (0,)
    assert all(
        torch.equal(target[key], originals[index][key]) for index, target in enumerate(targets) for key in target
    )

    model = _tiny_fasterrcnn()
    losses = model(list(augmented_images), augmented_targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert total_loss.requires_grad
    assert model.backbone.conv.weight.grad is not None


def test_detection_boundary_rejects_unsupported_per_instance_fields():
    """A field without the shared keep-mask contract cannot be silently dropped."""
    pipe = Compose.from_params(data_keys=["input", "bbox_xyxy"])
    images = torch.rand(1, 3, 32, 32)
    targets = [
        {"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]), "labels": torch.tensor([1]), "track_ids": torch.tensor([7])}
    ]

    with pytest.raises(ValueError, match=r"unsupported.*track_ids"):
        augment_detection_batch(pipe, images, targets)


def test_detection_boundary_requires_the_declared_box_pipeline():
    """An image-only pipeline cannot transform packed boxes by accident."""
    images = torch.rand(1, 3, 32, 32)
    targets = [{"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]), "labels": torch.tensor([1])}]

    with pytest.raises(ValueError, match=r"data_keys"):
        augment_detection_batch(Compose.from_params(), images, targets)


def test_detection_boundary_rejects_box_dtype_mismatch_across_the_batch():
    """Dense packing cannot safely route boxes with a different dtype from the image batch."""
    pipe = Compose.from_params(data_keys=["input", "bbox_xyxy"])
    images = torch.rand(2, 3, 32, 32)
    targets = [
        {"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[2.0, 2.0, 9.0, 9.0]], dtype=torch.float64), "labels": torch.tensor([1])},
    ]

    with pytest.raises(ValueError, match=r"boxes.*dtype"):
        augment_detection_batch(pipe, images, targets)


def test_detection_boundary_rejects_non_sequence_targets_and_complex_crowd_flags():
    """Boundary type checks fail before packing can hide an unsupported target value."""
    pipe = Compose.from_params(data_keys=["input", "bbox_xyxy"])
    images = torch.rand(1, 3, 32, 32)

    with pytest.raises(TypeError, match=r"targets must be a sequence"):
        augment_detection_batch(pipe, images, None)  # type: ignore[arg-type]

    targets = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
            "labels": torch.tensor([1]),
            "iscrowd": torch.tensor([1 + 0j]),
        }
    ]
    with pytest.raises(ValueError, match=r"iscrowd.*integer or bool"):
        augment_detection_batch(pipe, images, targets)
