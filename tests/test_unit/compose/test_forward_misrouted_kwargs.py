"""Regression tests for misrouted ``image=`` keyword calls on tensor pipelines (CORE-3).

A single-tensor pipeline invoked as ``pipe(image=<tensor>)`` previously fell through to
``forward()`` and produced an opaque ``TypeError`` ("unexpected keyword argument") whose exact
wording depended on the adapter/backend. An early guard now raises a clear, backend-independent
message telling the caller to pass the image positionally.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations.compose import FusedCompose


class TestMisroutedImageKeyword:
    """A tensor pipeline called with ``image=<tensor>`` raises a clear, backend-independent error."""

    def test_image_keyword_raises_clear_typeerror(self):
        """``pipe(image=tensor)`` on a native tensor pipeline raises ``TypeError`` mentioning 'positionally'."""
        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0))
        with pytest.raises(TypeError, match=r"positionally"):
            pipe(image=torch.rand(2, 3, 16, 16))

    def test_arbitrary_keyword_raises_clear_typeerror(self):
        """A non-'image' data keyword is also reported clearly rather than as an opaque TypeError."""
        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0))
        with pytest.raises(TypeError, match=r"keyword argument"):
            pipe(mask=torch.rand(2, 3, 16, 16))

    def test_positional_call_works(self):
        """The same pipeline accepts the image positionally and returns the expected shape."""
        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0))
        out = pipe(torch.rand(2, 3, 16, 16))
        assert out.shape == (2, 3, 16, 16)
