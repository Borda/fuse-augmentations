"""Small downstream classification and segmentation training workflows.

The detection batch boundary, letterbox inference map, and ordered-pose inverse already have dedicated integration
coverage. These tests add the remaining model contracts: an image-only classifier can train and infer after ``Compose``,
and a segmentation model can train from a routed hard-label mask with an ignore border.

"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fuse_augmentations import Compose

pytestmark = pytest.mark.integration


class _TinyClassifier(nn.Module):
    """Classify an image batch without external weights or downloads."""

    def __init__(self) -> None:
        """Build a minimal spatial encoder and class head."""
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Linear(4, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return one three-class logit vector per image."""
        features = torch.relu(self.encoder(image))
        return self.head(features.mean(dim=(-2, -1)))


class _TinySegmenter(nn.Module):
    """Produce two-class pixel logits without external weights or downloads."""

    def __init__(self) -> None:
        """Build a small fully convolutional segmentation head."""
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return two logits per spatial pixel."""
        return self.head(torch.relu(self.encoder(image)))


def test_classifier_training_and_eval_accept_compose_output() -> None:
    """A classifier consumes augmented images, backpropagates, and exposes finite eval logits."""
    torch.manual_seed(701)
    image = torch.rand(2, 3, 8, 8, requires_grad=True)
    labels = torch.tensor([0, 2])
    pipe = Compose.from_params(translate_x=(1.0, 1.0))
    model = _TinyClassifier()

    logits = model(pipe(image))
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (2, 3)
    assert torch.isfinite(loss)
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()

    model.eval()
    with torch.no_grad():
        inference_logits = model(pipe(image.detach()))
    assert inference_logits.shape == (2, 3)
    assert torch.isfinite(inference_logits).all()


def test_segmentation_training_ignores_routed_mask_fill_border() -> None:
    """A translated uint8 mask maps independently and its ignore border cannot change loss."""
    torch.manual_seed(702)
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    mask[..., 2:6, 3:7] = 1
    pipe = Compose.from_params(
        translate_x=(2.0, 2.0),
        data_keys=["input", "mask"],
        mask_fill=255,
    )
    expected_mask = torch.full_like(mask, 255)
    expected_mask[..., 2:] = mask[..., :-2]

    augmented_image, augmented_mask = pipe(image, mask)

    assert torch.equal(augmented_mask, expected_mask)
    assert torch.equal(augmented_mask[..., :2], torch.full_like(augmented_mask[..., :2], 255))

    model = _TinySegmenter()
    logits = model(augmented_image)
    targets = augmented_mask[:, 0].to(dtype=torch.long)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(logits, targets)
    padded = targets == 255
    changed_logits = logits.detach().clone()
    changed_logits[:, 0] = changed_logits[:, 0].masked_fill(padded, 1_000.0)
    changed_logits[:, 1] = changed_logits[:, 1].masked_fill(padded, -1_000.0)
    changed_loss = criterion(changed_logits, targets)
    loss.backward()

    torch.testing.assert_close(changed_loss, loss.detach(), atol=0.0, rtol=0.0)
    assert torch.isfinite(loss)
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()
