"""Regression tests for range-aware color-segment clipping."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations.affine.segment import FusedColorSegment
from fuse_augmentations.types import TransformCategory


class _AffineColor:
    """Small deterministic pointwise transform used to isolate clamp behavior."""

    _category = TransformCategory.POINTWISE_LINEAR
    p = 1.0

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta


class _AffineAdapter:
    """Adapter exposing fixed scalar affine color matrices for clamp tests."""

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the pointwise-linear category."""
        return TransformCategory.POINTWISE_LINEAR

    @staticmethod
    def sample_params(transform: object, shape: tuple[int, ...], device: torch.device) -> dict[str, torch.Tensor]:
        """Return the input batch size without introducing randomness."""
        return {"_batch_size": torch.tensor([shape[0]], device=device)}

    @staticmethod
    def build_color_matrix(
        transform: _AffineColor,
        params: dict[str, torch.Tensor],
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the fixed scalar affine matrix."""
        del mean
        batch_size = int(params.get("_batch_size", torch.tensor([1])).item())
        matrix = torch.eye(4, dtype=torch.float32).expand(batch_size, -1, -1).clone()
        matrix[:, :3, :3] *= transform.alpha
        matrix[:, :3, 3] = transform.beta
        return matrix

    @staticmethod
    def call_nonfused(transform: _AffineColor, image: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Apply one native-style clamped affine operation."""
        del kwargs
        return (transform.alpha * image + transform.beta).clamp(0.0, 1.0)


def _native_chain(image: torch.Tensor, transforms: list[_AffineColor]) -> torch.Tensor:
    """Apply the reference chain with a clamp after every operation."""
    for transform in transforms:
        image = _AffineAdapter.call_nonfused(transform, image)
    return image


def test_per_op_parity_clamps_escaping_intermediate() -> None:
    """The parity policy matches native per-operation clipping on a range escape."""
    transforms = [_AffineColor(alpha=2.0, beta=0.0), _AffineColor(alpha=0.5, beta=0.2)]
    image = torch.tensor([[[[0.25, 0.75]], [[0.25, 0.75]], [[0.25, 0.75]]]], dtype=torch.float32)
    segment = FusedColorSegment(transforms, _AffineAdapter(), clip_policy="per_op_parity")

    actual = segment(image)
    expected = _native_chain(image, transforms)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_final_policy_preserves_single_composed_matmul_behavior() -> None:
    """The default policy composes first and clamps only the final result."""
    transforms = [_AffineColor(alpha=2.0, beta=0.0), _AffineColor(alpha=0.5, beta=0.2)]
    image = torch.tensor([[[[0.25, 0.75]], [[0.25, 0.75]], [[0.25, 0.75]]]], dtype=torch.float32)
    segment = FusedColorSegment(transforms, _AffineAdapter())

    expected = (image + 0.2).clamp(0.0, 1.0)
    torch.testing.assert_close(segment(image), expected, atol=1e-6, rtol=1e-6)
    assert segment.clip_policy == "final"


def test_invalid_clip_policy_is_rejected() -> None:
    """The segment constructor rejects unknown policy strings."""
    with pytest.raises(ValueError, match="clip policy"):
        FusedColorSegment([_AffineColor(1.0, 0.0)], _AffineAdapter(), clip_policy="unknown")
