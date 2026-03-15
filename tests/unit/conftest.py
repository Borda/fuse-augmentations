"""Unit-test fixtures — pure torch, no external backend required."""

import pytest
import torch


@pytest.fixture()
def identity3() -> torch.Tensor:
    """Return a (1, 3, 3) identity matrix."""
    return torch.eye(3).unsqueeze(0)


@pytest.fixture()
def identity3_batch() -> torch.Tensor:
    """Return a (4, 3, 3) batch of identity matrices."""
    return torch.eye(3).unsqueeze(0).expand(4, -1, -1).clone()
