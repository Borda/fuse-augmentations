"""Demo use-case tests for meta-config: TransformSpec, from_config, from_params expansion.

These tests document the intended API contract and serve as regression guards.
"""

from __future__ import annotations

import json

import pytest
import torch

# ---------------------------------------------------------------------------
# TransformSpec frozen dataclass
# ---------------------------------------------------------------------------


def test_transform_spec_construction_and_equality() -> None:
    """TransformSpec holds operation, params, prob; equality is value-based."""
    from fuse_augmentations import TransformSpec

    spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
    assert spec.operation == "rotation"
    assert spec.params == {"degrees": (-30.0, 30.0)}
    assert spec.prob == 0.8


def test_transform_spec_default_p() -> None:
    """TransformSpec.prob defaults to 1.0 when omitted."""
    from fuse_augmentations import TransformSpec

    spec = TransformSpec(operation="hflip", params={})
    assert spec.prob == 1.0


def test_transform_spec_json_round_trip() -> None:
    """to_dict() / from_dict() round-trips through json.dumps / json.loads."""
    from fuse_augmentations import TransformSpec

    original = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
    d = original.to_dict()
    json_str = json.dumps(d)
    loaded = TransformSpec.from_dict(json.loads(json_str))
    assert loaded == original


def test_transform_spec_is_frozen() -> None:
    """TransformSpec is immutable (frozen dataclass)."""
    from fuse_augmentations import TransformSpec

    spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)})
    with pytest.raises((TypeError, AttributeError)):
        spec.operation = "hflip"  # type: ignore[misc]


def test_transform_spec_exported() -> None:
    """TransformSpec is importable from top-level fuse_augmentations and fuse_aug."""
    import fuse_aug
    import fuse_augmentations

    assert hasattr(fuse_augmentations, "TransformSpec")
    assert hasattr(fuse_aug, "TransformSpec")
    assert "TransformSpec" in fuse_augmentations.__all__


# ---------------------------------------------------------------------------
# Backend resolver
# ---------------------------------------------------------------------------


def test_resolver_returns_type_for_known_op() -> None:
    """resolve_op('rotation', 'kornia') returns a callable class (not None)."""
    from fuse_augmentations._resolver import resolve_op

    kornia = pytest.importorskip("kornia")  # noqa: F841
    cls = resolve_op("rotation", "kornia")
    assert callable(cls)


def test_resolver_raises_for_unknown_op() -> None:
    """resolve_op raises ValueError for unknown operation names."""
    from fuse_augmentations._resolver import resolve_op

    with pytest.raises(ValueError, match="unknown"):
        resolve_op("nonexistent_op_xyz", "kornia")


def test_resolver_raises_for_unknown_backend() -> None:
    """resolve_op raises ValueError for unknown backend strings."""
    from fuse_augmentations._resolver import resolve_op

    with pytest.raises(ValueError, match="unknown backend"):
        resolve_op("rotation", "unknown_backend_xyz")


# ---------------------------------------------------------------------------
# Compose.from_config() classmethod
# ---------------------------------------------------------------------------


def test_from_config_kornia_rotation_hflip() -> None:
    """from_config produces a working pipeline for rotation + hflip on Kornia."""
    pytest.importorskip("kornia")
    from fuse_augmentations import Compose, TransformSpec

    specs = [
        TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8),
        TransformSpec(operation="hflip", params={}, prob=0.5),
    ]
    pipe = Compose.from_config(specs, backend="kornia")
    x = torch.rand(2, 3, 64, 64)
    out = pipe(x)
    assert out.shape == torch.Size([2, 3, 64, 64])


def test_from_config_empty_specs() -> None:
    """from_config with empty specs returns identity pipeline."""
    from fuse_augmentations import Compose

    pipe = Compose.from_config([], backend="kornia")
    x = torch.rand(2, 3, 32, 32)
    out = pipe(x)
    assert out.shape == torch.Size([2, 3, 32, 32])
    assert torch.allclose(out, x)


def test_from_config_produces_fused_segment() -> None:
    """from_config with two geometric ops fuses them into a single segment."""
    pytest.importorskip("kornia")
    from fuse_augmentations import Compose, TransformSpec

    specs = [
        TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}),
        TransformSpec(operation="hflip", params={}),
    ]
    pipe = Compose.from_config(specs, backend="kornia")
    x = torch.rand(2, 3, 64, 64)
    pipe(x)  # trigger a forward pass to populate fusion_plan
    assert pipe.n_warps_saved >= 1, f"Expected fusion, got plan: {pipe.fusion_plan}"


def test_from_config_torchvision() -> None:
    """from_config works with backend='torchvision'."""
    pytest.importorskip("torchvision")
    from fuse_augmentations import Compose, TransformSpec

    specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
    pipe = Compose.from_config(specs, backend="torchvision")
    x = torch.rand(2, 3, 32, 32)
    out = pipe(x)
    assert out.shape == torch.Size([2, 3, 32, 32])


# ---------------------------------------------------------------------------
# from_params(specs=...) keyword-only overload
# ---------------------------------------------------------------------------


def test_from_params_specs_overload_backend_free() -> None:
    """from_params(specs=[...]) works without any backend import."""
    from fuse_augmentations import Compose, TransformSpec

    specs = [
        TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0),
        TransformSpec(operation="hflip", params={}, prob=0.5),
    ]
    pipe = Compose.from_params(specs=specs)
    x = torch.rand(2, 3, 64, 64)
    out = pipe(x)
    assert out.shape == torch.Size([2, 3, 64, 64])


def test_from_params_existing_api_unchanged() -> None:
    """Existing from_params keyword-only API still works."""
    from fuse_augmentations import Compose

    pipe = Compose.from_params(rotation=(-30.0, 30.0), hflip_p=0.5)
    x = torch.rand(2, 3, 64, 64)
    out = pipe(x)
    assert out.shape == torch.Size([2, 3, 64, 64])


def test_from_params_specs_and_kwargs_are_mutually_exclusive() -> None:
    """Passing both specs and keyword params raises ValueError."""
    from fuse_augmentations import Compose, TransformSpec

    specs = [TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)})]
    with pytest.raises(ValueError, match="mutually exclusive"):
        Compose.from_params(specs=specs, rotation=(-30.0, 30.0))


# ---------------------------------------------------------------------------
# Per-transform probability in from_params(specs=...)
# ---------------------------------------------------------------------------


def test_p_zero_transform_never_applied() -> None:
    """Albu spec with prob=0.0 is never applied — output matches identity-rotated input."""
    from fuse_augmentations import Compose, TransformSpec

    # Only a large rotation (prob=0.0) — should never change image
    specs = [TransformSpec(operation="rotation", params={"degrees": (-90.0, 90.0)}, prob=0.0)]
    pipe = Compose.from_params(specs=specs)
    x = torch.rand(2, 3, 64, 64)
    out = pipe(x)
    assert torch.allclose(out, x, atol=1e-5), "prob=0.0 transform should never be applied"


def test_p_one_transform_always_applied_for_flip() -> None:
    """Albu hflip spec with prob=1.0 is always applied — output != input for non-symmetric images."""
    from fuse_augmentations import Compose, TransformSpec

    torch.manual_seed(42)
    specs = [TransformSpec(operation="hflip", params={}, prob=1.0)]
    pipe = Compose.from_params(specs=specs)
    x = torch.rand(2, 3, 32, 32)
    out = pipe(x)
    # hflip with prob=1.0 must produce flipped output
    assert torch.allclose(out, x.flip(dims=[3])), "prob=1.0 hflip must always flip"
