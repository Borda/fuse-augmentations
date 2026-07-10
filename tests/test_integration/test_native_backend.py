"""Integration coverage for the opt-in zero-dependency native backend."""

from __future__ import annotations

import builtins

import pytest
import torch

from fuse_augmentations import Compose, TransformSpec


def _optional_import_guard(name: str, *args: object, **kwargs: object) -> object:
    """Reject optional backend imports while allowing the native path to run."""
    if name == "kornia" or name.startswith(("kornia.", "torchvision", "albumentations")):
        raise ModuleNotFoundError(name)
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


_ORIGINAL_IMPORT = builtins.__import__


def test_native_from_config_uses_no_optional_backend_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native construction and execution work when optional imports are hidden."""
    monkeypatch.setattr(builtins, "__import__", _optional_import_guard)
    specs = [
        TransformSpec(operation="rotation", params={"degrees": (0.0, 0.0)}),
        TransformSpec(operation="hflip", params={}, prob=1.0),
    ]
    pipe = Compose.from_config(specs, backend="native")
    image = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 255.0

    output, matrix = pipe(image, return_matrix=True)

    torch.testing.assert_close(output, image.flip(dims=[3]))
    assert matrix is not None
    assert matrix.dtype == image.dtype


def test_native_capabilities_and_color_reserved_names() -> None:
    """Native capabilities expose cheap geometric and linear color operations only."""
    capabilities = Compose.supported_ops("native")
    assert {"rotation", "scale", "shear", "translate", "hflip", "vflip"} <= capabilities
    assert {"brightness", "contrast"} <= capabilities
    assert "perspective" not in capabilities

    pipe = Compose.from_config(
        [TransformSpec(operation="brightness", params={"factor": (1.0, 1.0)})],
        backend="native",
    )
    image = torch.rand(2, 3, 8, 8)
    output = pipe(image)
    torch.testing.assert_close(output, image)
    assert pipe.fusion_plan.startswith("color(")


def test_native_geometric_matrix_matches_direct_torch_reference() -> None:
    """A fixed native rotation has the expected pixel-space matrix and shape."""
    pipe = Compose.from_config(
        [TransformSpec(operation="rotation", params={"degrees": (15.0, 15.0)})],
        backend="native",
    )
    output, matrix = pipe(torch.rand(2, 3, 12, 10), return_matrix=True)

    assert output.shape == (2, 3, 12, 10)
    assert matrix is not None
    assert matrix.shape == (2, 3, 3)
    assert torch.isfinite(matrix).all()
