"""Shared pytest fixtures for fuse-augmentations test suite."""

import contextlib
import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset all random seeds (torch, numpy, stdlib random, CUDA) before each test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(autouse=True)
def disable_interactive_prompts():
    """Keep test runs non-interactive across dependency versions."""
    old_breakpoint = os.environ.get("PYTHONBREAKPOINT")
    os.environ["PYTHONBREAKPOINT"] = "0"

    restore_kornia = None
    with contextlib.suppress(ImportError):
        from kornia.config import InstallationMode, kornia_config

        restore_kornia = kornia_config.lazyloader.installation_mode
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE

    try:
        yield
    finally:
        if old_breakpoint is None:
            os.environ.pop("PYTHONBREAKPOINT", None)
        else:
            os.environ["PYTHONBREAKPOINT"] = old_breakpoint

        if restore_kornia is not None:
            from kornia.config import kornia_config

            kornia_config.lazyloader.installation_mode = restore_kornia


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for tests (GPU tests use gpu marker)."""
    return torch.device("cpu")


@pytest.fixture
def img_batch() -> torch.Tensor:
    """Return a (4, 3, 64, 64) float32 image batch on CPU."""
    return torch.rand(4, 3, 64, 64)


@pytest.fixture
def img_single() -> torch.Tensor:
    """Return a (1, 3, 64, 64) float32 image on CPU."""
    return torch.rand(1, 3, 64, 64)
