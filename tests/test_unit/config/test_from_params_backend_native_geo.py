"""Regression tests for per-axis geometry in ``from_params`` with ``backend=`` set (CORE-2).

The per-axis kwargs ``scale_x``/``scale_y``, ``shear_x``/``shear_y`` and
``translate_x``/``translate_y`` are supported only in backend-free mode (``backend=None``). When any
backend is set -- including ``backend="native"``, which is the same direct-parameter engine as
``backend=None`` -- they are rejected with an actionable ``ValueError`` pointing to ``backend=None``.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations.compose import FusedCompose


class TestFromParamsNativePerAxisRestriction:
    """``backend="native"`` rejects per-axis shear/translate/scale with an actionable message."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"shear_x": (-10.0, 10.0)}, id="shear_x"),
            pytest.param({"shear_y": (-10.0, 10.0)}, id="shear_y"),
            pytest.param({"translate_x": (-3.0, 3.0)}, id="translate_x"),
            pytest.param({"translate_y": (-3.0, 3.0)}, id="translate_y"),
            pytest.param({"scale_x": (0.9, 1.1)}, id="scale_x"),
            pytest.param({"scale_y": (0.9, 1.1)}, id="scale_y"),
        ],
    )
    def test_native_per_axis_raises_pointing_to_backend_none(self, kwargs):
        """Per-axis geometry with ``backend="native"`` raises ``ValueError`` naming ``backend=None``."""
        with pytest.raises(ValueError, match=r"backend=None"):
            FusedCompose.from_params(backend="native", **kwargs)

    def test_native_rotation_still_supported(self):
        """A symmetric op (rotation) with ``backend="native"`` builds a working pipeline (no regression)."""
        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0), backend="native")
        assert isinstance(pipe, FusedCompose)


class TestFromParamsBackendFreePerAxis:
    """``backend=None`` (backend-free) supports per-axis shear/translate/scale directly."""

    def test_backend_free_shear_translate_builds_and_runs(self):
        """``from_params(shear_x, translate_y)`` in backend-free mode builds a working pipeline."""
        pipe = FusedCompose.from_params(shear_x=(-10.0, 10.0), translate_y=(-3.0, 3.0))
        out = pipe(torch.rand(2, 3, 16, 16))
        assert out.shape == (2, 3, 16, 16)

    def test_backend_free_per_axis_scale_builds(self):
        """``from_params(scale_x, scale_y)`` in backend-free mode builds a working pipeline."""
        pipe = FusedCompose.from_params(scale_x=(0.9, 1.1), scale_y=(0.8, 1.2))
        out = pipe(torch.rand(2, 3, 16, 16))
        assert out.shape == (2, 3, 16, 16)
