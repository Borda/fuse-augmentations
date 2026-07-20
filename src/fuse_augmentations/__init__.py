"""Fuse augmentation transforms into a single interpolation pass.

Both ``import fuse_augmentations`` and ``import fuse_aug`` expose the same
public API. All implementation lives here; ``fuse_aug`` re-exports via star.

Examples:
    ```pycon
    >>> from fuse_augmentations import Compose
    >>> pipe = Compose([])
    >>> pipe.__class__.__name__
    'FusedCompose'

    ```

"""

from __future__ import annotations

import os

from fuse_augmentations.__about__ import *  # noqa: F403
from fuse_augmentations.affine.segment import (
    CropResizeSegment,
    ExactAffineSegment,
    FusedAffineSegment,
    FusedColorSegment,
    FusedLUTSegment,
    ProjectiveSegment,
    build_segments,
)
from fuse_augmentations.converters import NumpyToTorchConverter, TorchToNumpyConverter

# Import from the implementation module (not the ``compose`` compatibility
# shim, whose runtime ``__getattr__`` forwarding is invisible to static doc
# tooling such as griffe/mkdocstrings). ``compose`` stays a valid import and
# pickle path for historical payloads.
from fuse_augmentations.pipeline import (
    AugmentationSequential,
    Compose,
    FusedCompose,
)
from fuse_augmentations.targets import (
    transform_bbox_xywh,
    transform_bbox_xyxy,
    transform_keypoints,
    transform_mask,
)
from fuse_augmentations.types import (
    BackendConverter,
    ClipPolicyStr,
    InterpolationMode,
    PaddingMode,
    RandomnessPolicy,
    ReorderPolicy,
    SegmentDescriptor,
    TransformAdapter,
    TransformCategory,
    TransformSpec,
)

__all__ = [
    "AugmentationSequential",
    "BackendConverter",
    "ClipPolicyStr",
    "Compose",
    "CropResizeSegment",
    "ExactAffineSegment",
    "FusedAffineSegment",
    "FusedColorSegment",
    "FusedCompose",
    "FusedLUTSegment",
    "InterpolationMode",
    "NumpyToTorchConverter",
    "PaddingMode",
    "ProjectiveSegment",
    "RandomnessPolicy",
    "ReorderPolicy",
    "SegmentDescriptor",
    "TorchToNumpyConverter",
    "TransformAdapter",
    "TransformCategory",
    "TransformSpec",
    "build_segments",
    "transform_bbox_xywh",
    "transform_bbox_xyxy",
    "transform_keypoints",
    "transform_mask",
]

_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)
