"""``split_reason="backend_boundary"`` fires for adjacent fused↔fused backend changes (XB-2).

Two adjacent fused geometric segments from different backends (e.g. ``kornia.RandomAffine``
then ``A.Affine``) cannot be composed into one warp; the break is a genuine backend-driven
split. ``fusion_plan_descriptors`` previously computed ``split_reason`` only for
crop/passthrough segments, so this common mixed boundary reported ``None`` despite the
documented contract. The second fused segment of such a pair now carries
``split_reason="backend_boundary"``.

Requires kornia and albumentations.

"""

from __future__ import annotations

import pytest

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required"),
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]


def test_fused_to_fused_backend_change_marks_boundary():
    """A kornia fused run followed by an albumentations fused run marks the second as a backend boundary."""
    pipe = Compose([kornia_aug.RandomAffine(degrees=30, p=1.0), albu.Affine(rotate=(20, 20), p=1.0)])
    descriptors = pipe.fusion_plan_descriptors
    assert [d.kind for d in descriptors] == ["fused", "fused"]
    assert descriptors[0].backend == "KorniaAdapter"
    assert descriptors[1].backend == "AlbumentationsAdapter"
    # First segment opens the run (no predecessor); second is the split point.
    assert descriptors[0].split_reason is None
    assert descriptors[1].split_reason == "backend_boundary"


def test_single_backend_fused_run_has_no_boundary():
    """A single-backend fused run reports no split_reason — there is no backend change to flag."""
    pipe = Compose([kornia_aug.RandomAffine(degrees=30, p=1.0), kornia_aug.RandomRotation(degrees=15, p=1.0)])
    for descriptor in pipe.fusion_plan_descriptors:
        assert descriptor.split_reason is None
