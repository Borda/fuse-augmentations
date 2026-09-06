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
from typing import Any

from fuse_augmentations.__about__ import *  # noqa: F403
from fuse_augmentations.affine.matrix import (
    LetterboxGeometry,
    letterbox_geometry,
    letterbox_matrix,
)
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
from fuse_augmentations.detection import augment_detection_batch

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
    clip_bbox_xyxy,
    corners_to_rboxes,
    instance_keep_mask,
    mirror_rboxes,
    orientation_reversed,
    permute_keypoint_pairs,
    rbox_envelopes,
    rboxes_to_corners,
    shift_rboxes,
    transform_bbox_xywh,
    transform_bbox_xyxy,
    transform_keypoints,
    transform_mask,
    transform_rboxes,
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
    "LetterboxGeometry",
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
    "augment_detection_batch",
    "build_segments",
    "clip_bbox_xyxy",
    "corners_to_rboxes",
    "generate_dataset",  # noqa: F405 - provided lazily via module __getattr__
    "instance_keep_mask",
    "letterbox_geometry",
    "letterbox_matrix",
    "mirror_rboxes",
    "orientation_reversed",
    "permute_keypoint_pairs",
    "rbox_envelopes",
    "rboxes_to_corners",
    "shift_rboxes",
    "transform_bbox_xywh",
    "transform_bbox_xyxy",
    "transform_keypoints",
    "transform_mask",
    "transform_rboxes",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module-level lazy attribute access
    """Lazily expose the dataset-generation facade, deferring the datasets subpackage and Pillow until first use."""
    if name == "generate_dataset":
        from fuse_augmentations.data import generate_dataset

        return generate_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)
