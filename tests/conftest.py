"""Shared pytest fixtures for fuse-augmentations test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset all random seeds before each test for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for tests (GPU tests use gpu marker)."""
    return torch.device("cpu")


@pytest.fixture
def img_batch() -> torch.Tensor:
    """Return a (4, 3, 64, 64) float32 image batch on CPU."""
    torch.manual_seed(0)
    return torch.rand(4, 3, 64, 64)


@pytest.fixture
def img_single() -> torch.Tensor:
    """Return a (1, 3, 64, 64) float32 image on CPU."""
    torch.manual_seed(0)
    return torch.rand(1, 3, 64, 64)
