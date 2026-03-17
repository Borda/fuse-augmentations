"""Fuse augmentation transforms into a single interpolation pass.

Both ``import fuse_augmentations`` and ``import fuse_aug`` expose the same
public API. All implementation lives here; ``fuse_aug`` re-exports via star.

Example:
    >>> from fuse_augmentations import Compose
    >>> pipe = Compose([])
    >>> pipe.__class__.__name__
    'FusedAffineCompose'
"""

from __future__ import annotations

import os

from fuse_augmentations.__about__ import *  # noqa: F403
from fuse_augmentations._compose import (
    AugmentationSequential,
    Compose,
    FusedAffineCompose,
)
from fuse_augmentations._segment import FusedAffineSegment, build_segments
from fuse_augmentations._types import (
    InterpolationMode,
    PaddingMode,
    ReorderPolicy,
    TransformAdapter,
    TransformCategory,
)

__all__ = [
    "AugmentationSequential",
    "Compose",
    "FusedAffineCompose",
    "FusedAffineSegment",
    "InterpolationMode",
    "PaddingMode",
    "ReorderPolicy",
    "TransformAdapter",
    "TransformCategory",
    "build_segments",
]

_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)
