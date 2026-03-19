"""Fuse augmentation transforms into a single interpolation pass.

Both ``import fuse_augmentations`` and ``import fuse_aug`` expose the same
public API. All implementation lives here; ``fuse_aug`` re-exports via star.

Example:
    >>> from fuse_augmentations import Compose
    >>> pipe = Compose([])
    >>> pipe.__class__.__name__
    'FusedCompose'

"""

from __future__ import annotations

import os

from fuse_augmentations.__about__ import *  # noqa: F403
from fuse_augmentations._compose import (
    AugmentationSequential,
    Compose,
    FusedCompose,
)
from fuse_augmentations._targets import (
    transform_bbox_xywh,
    transform_bbox_xyxy,
    transform_keypoints,
    transform_mask,
)
from fuse_augmentations._types import (
    InterpolationMode,
    PaddingMode,
    ReorderPolicy,
    TransformAdapter,
    TransformCategory,
)
from fuse_augmentations.affine._segment import ExactSegment, FusedAffineSegment, build_segments

__all__ = [
    "AugmentationSequential",
    "Compose",
    "ExactSegment",
    "FusedAffineSegment",
    "FusedCompose",
    "InterpolationMode",
    "PaddingMode",
    "ReorderPolicy",
    "TransformAdapter",
    "TransformCategory",
    "build_segments",
    "transform_bbox_xywh",
    "transform_bbox_xyxy",
    "transform_keypoints",
    "transform_mask",
]

_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)
