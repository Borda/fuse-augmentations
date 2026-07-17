"""``fusion_plan`` surfaces the hidden CPU round-trip of Albumentations cv2 warps (XB-3).

Albumentations fused/projective segments on the default ``"cv2"`` execution strategy warp
each sample with OpenCV, copying to host and back on every call. On a non-CPU pipeline this
device round-trip is a poison pill, but ``fusion_plan`` previously rendered them as a plain
``fused(...)`` with no marker. They now carry the same ``" [CPU passthrough]"`` marker as a
passthrough segment; switching to ``execution="torch"`` keeps the warp on-device and drops it.

A meta device is used to force the non-CPU branch without requiring CUDA/MPS.

Requires albumentations.

"""

from __future__ import annotations

import pytest

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")

_MARKER = "[CPU passthrough]"


def test_albu_cv2_fused_marked_on_non_cpu_pipeline():
    """An Albumentations cv2 fused segment carries the device-round-trip marker on a non-CPU pipeline."""
    pipe = Compose([albu.Affine(rotate=(20, 20), p=1.0)], execution="cv2")
    pipe.to("meta")
    plan = pipe.fusion_plan
    assert plan.startswith("fused(")
    assert _MARKER in plan


def test_albu_torch_fused_not_marked_on_non_cpu_pipeline():
    """The torch execution strategy warps on-device, so no marker is added even on a non-CPU pipeline."""
    pipe = Compose([albu.Affine(rotate=(20, 20), p=1.0)], execution="torch")
    pipe.to("meta")
    assert _MARKER not in pipe.fusion_plan


def test_albu_cv2_fused_not_marked_on_cpu_pipeline():
    """On a CPU pipeline no round-trip marker is emitted — the string stays unchanged."""
    pipe = Compose([albu.Affine(rotate=(20, 20), p=1.0)], execution="cv2")
    assert _MARKER not in pipe.fusion_plan
