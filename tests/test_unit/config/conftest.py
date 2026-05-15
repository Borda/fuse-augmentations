"""Shared fixtures for config-layer unit tests."""

import pytest
import torch


@pytest.fixture
def image16x16_batch1() -> torch.Tensor:
    """Return a (1, 3, 16, 16) float32 image tensor."""
    return torch.rand(1, 3, 16, 16)


@pytest.fixture
def image16x16_batch2() -> torch.Tensor:
    """Return a (2, 3, 16, 16) float32 image tensor."""
    return torch.rand(2, 3, 16, 16)


@pytest.fixture
def image32x32_batch2() -> torch.Tensor:
    """Return a (2, 3, 32, 32) float32 image tensor."""
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def image32x32_batch4() -> torch.Tensor:
    """Return a (4, 3, 32, 32) float32 image tensor."""
    return torch.rand(4, 3, 32, 32)


@pytest.fixture
def image64x64_batch2() -> torch.Tensor:
    """Return a (2, 3, 64, 64) float32 image tensor."""
    return torch.rand(2, 3, 64, 64)
