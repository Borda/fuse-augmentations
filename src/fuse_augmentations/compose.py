"""Compatibility import surface for fused augmentation pipelines.

The implementation lives in :mod:`fuse_augmentations.pipeline`. This module
forwards both public and historical private attributes so existing imports and
pickle payloads that reference ``fuse_augmentations.compose`` remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass

from fuse_augmentations import pipeline as _pipeline
from fuse_augmentations.affine.segment import _OpaqueBorderModeTransform  # noqa: F401

__all__ = [name for name in dir(_pipeline) if not name.startswith("_")] + ["dataclass"]


def __getattr__(name: str) -> object:
    """Forward compatibility lookups to the implementation module."""
    return getattr(_pipeline, name)


def __dir__() -> list[str]:
    """Return the compatibility surface together with module metadata."""
    return sorted(set(globals()) | set(dir(_pipeline)))
