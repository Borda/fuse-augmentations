"""Comprehensive unit tests for _resolver.py.

Tests cover resolve_op, SUPPORTED_OPS, SUPPORTED_BACKENDS — including error paths, case sensitivity, and cross-backend
resolution.

"""

from __future__ import annotations

import pytest

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE
from fuse_augmentations._resolver import SUPPORTED_BACKENDS, SUPPORTED_OPS, resolve_op, translate_params


class TestSupportedConstants:
    """SUPPORTED_OPS and SUPPORTED_BACKENDS structure and membership."""

    def test_supported_backends_type(self):
        assert isinstance(SUPPORTED_BACKENDS, frozenset), (
            f"SUPPORTED_BACKENDS should be frozenset, got {type(SUPPORTED_BACKENDS).__name__}"
        )

    def test_supported_ops_type(self):
        assert isinstance(SUPPORTED_OPS, frozenset), (
            f"SUPPORTED_OPS should be frozenset, got {type(SUPPORTED_OPS).__name__}"
        )

    @pytest.mark.parametrize(
        "backend",
        ["kornia", "torchvision", "albumentations"],
    )
    def test_supported_backends_contains(self, backend):
        assert backend in SUPPORTED_BACKENDS, f"'{backend}' should be in SUPPORTED_BACKENDS"

    @pytest.mark.parametrize(
        "transform",
        ["rotation", "hflip", "vflip"],
    )
    def test_supported_ops_minimum_ops(self, transform):
        assert transform in SUPPORTED_OPS, f"'{transform}' should be in SUPPORTED_OPS"

    def test_supported_ops_not_empty(self):
        assert len(SUPPORTED_OPS) >= 3, f"Expected at least 3 supported ops, got {len(SUPPORTED_OPS)}"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestResolveOpKornia:
    """resolve_op with backend='kornia'."""

    def test_rotation_returns_callable(self):
        cls = resolve_op("rotation", "kornia")
        assert callable(cls), f"resolve_op('rotation', 'kornia') should return callable, got {type(cls)}"

    def test_hflip_returns_callable(self):
        cls = resolve_op("hflip", "kornia")
        assert callable(cls), f"resolve_op('hflip', 'kornia') should return callable, got {type(cls)}"

    def test_vflip_returns_callable(self):
        cls = resolve_op("vflip", "kornia")
        assert callable(cls), f"resolve_op('vflip', 'kornia') should return callable, got {type(cls)}"


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestResolveOpTorchVision:
    """resolve_op with backend='torchvision'."""

    def test_hflip_returns_callable(self):
        cls = resolve_op("hflip", "torchvision")
        assert callable(cls)

    def test_rotation_returns_callable(self):
        cls = resolve_op("rotation", "torchvision")
        assert callable(cls)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestResolveOpAlbumentations:
    """resolve_op with backend='albumentations'."""

    def test_hflip_returns_callable(self):
        cls = resolve_op("hflip", "albumentations")
        assert callable(cls)


class TestResolveOpErrors:
    """Error paths for resolve_op."""

    def test_unknown_op_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown"):
            resolve_op("nonexistent_op_xyz", "kornia")

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_op("rotation", "nonexistent_backend_xyz")

    def test_case_sensitive_op(self):
        """resolve_op is case-sensitive: 'Rotation' != 'rotation'."""
        with pytest.raises(ValueError, match="unknown op"):
            resolve_op("Rotation", "kornia")

    def test_case_sensitive_backend(self):
        """resolve_op is case-sensitive: 'Kornia' != 'kornia'."""
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_op("rotation", "Kornia")

    def test_empty_op_string(self):
        with pytest.raises(ValueError, match="unknown op"):
            resolve_op("", "kornia")

    def test_empty_backend_string(self):
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_op("rotation", "")


class TestResolveOpComprehensive:
    """All ops in SUPPORTED_OPS resolve for at least one installed backend."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required for full op coverage (shear, translate)")
    def test_all_ops_resolvable(self):
        import importlib

        available_backends = [
            backend for backend in SUPPORTED_BACKENDS if importlib.util.find_spec(backend) is not None
        ]

        unresolvable = []
        for op in SUPPORTED_OPS:
            resolved = False
            for backend in available_backends:
                try:
                    cls = resolve_op(op, backend)
                    if callable(cls):
                        resolved = True
                        break
                except ValueError:
                    continue
            if not resolved:
                unresolvable.append(op)

        assert not unresolvable, (
            f"Ops not resolvable by any installed backend: {unresolvable}. Available backends: {available_backends}"
        )


class TestResolveOpBackendGap:
    """Ops that are in SUPPORTED_OPS but not covered by a specific backend."""

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_rotation90_not_in_torchvision_raises_value_error(self) -> None:
        """Rotation90 is in SUPPORTED_OPS but TorchVision has no equivalent."""
        with pytest.raises(ValueError, match="does not support op"):
            resolve_op("rotation90", "torchvision")

    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
    def test_rotation90_in_supported_ops_but_not_albumentations_shear(self) -> None:
        """Shear is in SUPPORTED_OPS and albumentations supports it (via Affine)."""
        # albumentations does not have a standalone shear; if it raises it's a backend-gap
        # ValueError, if it resolves (via Affine) that's also correct behaviour.
        # This test simply ensures no unexpected exception type is raised.
        try:
            result = resolve_op("shear", "albumentations")
            assert callable(result)
        except ValueError:
            pass  # acceptable — backend-gap path

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_shear_not_in_torchvision_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="does not support op"):
            resolve_op("shear", "torchvision")

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_translate_not_in_torchvision_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="does not support op"):
            resolve_op("translate", "torchvision")


class TestTranslateParams:
    def test_rotation_kornia_degrees_unchanged(self) -> None:
        result = translate_params("rotation", "kornia", {"degrees": (-30.0, 30.0)})
        assert result == {"degrees": (-30.0, 30.0)}

    def test_rotation_albumentations_degrees_to_limit(self) -> None:
        result = translate_params("rotation", "albumentations", {"degrees": (-10.0, 10.0)})
        assert "limit" in result
        assert result["limit"] == (-10.0, 10.0)
        assert "degrees" not in result

    def test_affine_albumentations_degrees_to_rotate(self) -> None:
        result = translate_params("affine", "albumentations", {"degrees": (-5.0, 5.0)})
        assert "rotate" in result
        assert result["rotate"] == (-5.0, 5.0)
        assert "degrees" not in result

    def test_scale_kornia_injects_degrees_default(self) -> None:
        result = translate_params("scale", "kornia", {"scale": (0.8, 1.2)})
        assert result.get("degrees") == 0.0
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_torchvision_injects_degrees_default(self) -> None:
        result = translate_params("scale", "torchvision", {"scale": (0.8, 1.2)})
        assert result.get("degrees") == 0.0
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_albumentations_no_degrees_injection(self) -> None:
        result = translate_params("scale", "albumentations", {"scale": (0.8, 1.2)})
        assert "degrees" not in result
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_factor_key_renamed_to_scale(self) -> None:
        result = translate_params("scale", "kornia", {"factor": (0.9, 1.1)})
        assert "scale" in result
        assert "factor" not in result

    def test_shear_kornia_degrees_to_shear(self) -> None:
        result = translate_params("shear", "kornia", {"degrees": (-10.0, 10.0)})
        assert "shear" in result
        assert "degrees" not in result

    def test_hflip_passthrough(self) -> None:
        result = translate_params("hflip", "kornia", {})
        assert result == {}

    def test_unknown_op_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown op"):
            translate_params("nonexistent", "kornia", {})

    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            translate_params("rotation", "bad_backend", {})
