"""Deprecated alias for :mod:`fuse_augmentations.data.geometry`.

The analytic shape geometry moved to :mod:`fuse_augmentations.data.geometry` when the shape
vocabulary was split into its geometric and animal halves. This module re-exports the names the
released ``fuse_augmentations.data.shapes`` published, so code written against that import path keeps
working, and warns at import time to point at the new one.

The shim will be removed in a future release; new code should import from
:mod:`fuse_augmentations.data.geometry` directly::

    from fuse_augmentations.data.geometry import shape_polygon

Only the geometry half is forwarded. ``animal_keypoints`` is deliberately absent: it is an
animal-only API and now lives in :mod:`fuse_augmentations.data.animals`, so forwarding it from a
module named for shape geometry would point users at the wrong home for it.

"""

from __future__ import annotations

import warnings

from fuse_augmentations.data.geometry import (
    CIRCLE_POINTS,
    GEOMETRIC_SHAPES,
    RECT_ASPECT,
    bbox_iou,
    polygon_to_bbox_xyxy,
    polygon_to_obb,
    rotate_polygon,
    shape_polygon,
)

__all__ = [
    "CIRCLE_POINTS",
    "GEOMETRIC_SHAPES",
    "RECT_ASPECT",
    "bbox_iou",
    "polygon_to_bbox_xyxy",
    "polygon_to_obb",
    "rotate_polygon",
    "shape_polygon",
]

# Emitted after the re-exports so the module is fully usable by the time the importer sees the
# warning: a caller who filters DeprecationWarning into an error still gets a working module in the
# non-error case, and one who does not gets the pointer without losing the import.
warnings.warn(
    "fuse_augmentations.data.shapes is deprecated; import from fuse_augmentations.data.geometry "
    "instead. This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
