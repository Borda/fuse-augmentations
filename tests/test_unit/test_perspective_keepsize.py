"""Fused ``Perspective(keep_size=False)`` raises instead of silently keeping the canvas (ADP-2).

Albumentations ``Perspective`` with ``keep_size=False`` returns a crop of different spatial
dimensions. The batched shape-preserving fusion engine cannot reproduce that in a single warp:
the fused matrix path previously warped into the original ``HxW`` canvas with no error, silently
diverging from native. The fused projective path now raises ``NotImplementedError`` for
``keep_size=False``; ``keep_size=True`` (whose output resize folds into the homography) is
unaffected.

Requires albumentations.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 32, 32)`` float32 image batch."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 32, 32, generator=gen)


def test_perspective_keep_size_false_raises():
    """A fused ``Perspective(keep_size=False)`` raises NotImplementedError rather than silently mis-shaping."""
    pipe = Compose([albu.Perspective(scale=(0.05, 0.1), keep_size=False, p=1.0)])
    with pytest.raises(NotImplementedError, match="keep_size=False"):
        pipe(_image())


def test_perspective_keep_size_true_preserves_shape():
    """A fused ``Perspective(keep_size=True)`` runs and preserves the input spatial shape."""
    pipe = Compose([albu.Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0)])
    out = pipe(_image())
    assert out.shape == (2, 3, 32, 32)
