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
        """SUPPORTED_BACKENDS is a frozenset (immutable, hashable registry)."""
        assert isinstance(SUPPORTED_BACKENDS, frozenset), (
            f"SUPPORTED_BACKENDS should be frozenset, got {type(SUPPORTED_BACKENDS).__name__}"
        )

    def test_supported_ops_type(self):
        """SUPPORTED_OPS is a frozenset (immutable, hashable registry)."""
        assert isinstance(SUPPORTED_OPS, frozenset), (
            f"SUPPORTED_OPS should be frozenset, got {type(SUPPORTED_OPS).__name__}"
        )

    @pytest.mark.parametrize(
        "backend",
        ["kornia", "torchvision", "albumentations"],
    )
    def test_supported_backends_contains(self, backend):
        """Every recognised backend name is present in SUPPORTED_BACKENDS."""
        assert backend in SUPPORTED_BACKENDS, f"'{backend}' should be in SUPPORTED_BACKENDS"

    @pytest.mark.parametrize(
        "transform",
        ["rotation", "hflip", "vflip"],
    )
    def test_supported_ops_minimum_ops(self, transform):
        """Core geometric ops (rotation, hflip, vflip) are registered in SUPPORTED_OPS."""
        assert transform in SUPPORTED_OPS, f"'{transform}' should be in SUPPORTED_OPS"

    def test_supported_ops_not_empty(self):
        """SUPPORTED_OPS contains at least the three baseline ops (sanity guard against empty registry)."""
        assert len(SUPPORTED_OPS) >= 3, f"Expected at least 3 supported ops, got {len(SUPPORTED_OPS)}"


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestResolveOpKornia:
    """resolve_op with backend='kornia'."""

    @pytest.mark.parametrize("op", ["rotation", "hflip", "vflip"])
    def test_op_returns_callable(self, op):
        """Common geometric ops resolve to a callable on kornia backend."""
        cls = resolve_op(op, "kornia")
        assert callable(cls), f"resolve_op({op!r}, 'kornia') should return callable, got {type(cls)}"


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
class TestResolveOpTorchVision:
    """resolve_op with backend='torchvision'."""

    @pytest.mark.parametrize("op", ["hflip", "rotation"])
    def test_op_returns_callable(self, op):
        """Common geometric ops resolve to a callable on torchvision backend."""
        cls = resolve_op(op, "torchvision")
        assert callable(cls), f"resolve_op({op!r}, 'torchvision') should return callable, got {type(cls)}"


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
class TestResolveOpAlbumentations:
    """resolve_op with backend='albumentations'."""

    def test_hflip_returns_callable(self):
        """Hflip resolves to a callable on albumentations backend."""
        cls = resolve_op("hflip", "albumentations")
        assert callable(cls)


class TestResolveOpErrors:
    """Error paths for resolve_op."""

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param("nonexistent_op_xyz", id="unknown"),
            pytest.param("Rotation", id="wrong_case"),
            pytest.param("", id="empty"),
        ],
    )
    def test_bad_op_raises_value_error(self, op):
        """Unknown, wrong-case, or empty op strings raise ValueError."""
        with pytest.raises(ValueError, match="unknown"):
            resolve_op(op, "kornia")

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param("nonexistent_backend_xyz", id="unknown"),
            pytest.param("Kornia", id="wrong_case"),
            pytest.param("", id="empty"),
        ],
    )
    def test_bad_backend_raises_value_error(self, backend):
        """Unknown, wrong-case, or empty backend strings raise ValueError."""
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_op("rotation", backend)


class TestResolveOpComprehensive:
    """All ops in SUPPORTED_OPS resolve for at least one installed backend."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required for full op coverage (shear, translate)")
    def test_all_ops_resolvable(self):
        """Every op in SUPPORTED_OPS resolves to a callable on at least one installed backend.

        Guards against silently registering an op in SUPPORTED_OPS without wiring it to any backend implementation — a
        broken registry that would only surface at runtime when a user requests the unwired op.

        """
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
    @pytest.mark.parametrize("op", ["rotation90", "shear", "translate"])
    def test_op_not_in_torchvision_raises(self, op) -> None:
        """Ops valid globally but missing in torchvision raise ValueError."""
        with pytest.raises(ValueError, match="does not support op"):
            resolve_op(op, "torchvision")

    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
    def test_shear_albumentations_resolves_or_raises_backend_gap(self) -> None:
        """Shear resolves via Affine or raises ValueError for backend-gap — both valid outcomes."""
        try:
            result = resolve_op("shear", "albumentations")
        except ValueError:
            return  # backend-gap path — valid
        except Exception as exc:
            pytest.fail(f"Unexpected exception type raised: {exc!r}")
        assert callable(result), f"Expected callable, got {result!r}"


class TestTranslateParams:
    """translate_params normalises op kwargs across backend-specific parameter names."""

    def test_rotation_kornia_degrees_unchanged(self) -> None:
        """Kornia uses 'degrees' natively, so rotation params pass through unchanged."""
        result = translate_params("rotation", "kornia", {"degrees": (-30.0, 30.0)})
        assert result == {"degrees": (-30.0, 30.0)}

    def test_rotation_albumentations_degrees_to_limit(self) -> None:
        """Albumentations rotation expects 'limit', so 'degrees' is renamed to 'limit'."""
        result = translate_params("rotation", "albumentations", {"degrees": (-10.0, 10.0)})
        assert "limit" in result
        assert result["limit"] == (-10.0, 10.0)
        assert "degrees" not in result

    def test_affine_albumentations_degrees_to_rotate(self) -> None:
        """Albumentations Affine expects 'rotate' for the angle range, so 'degrees' is renamed."""
        result = translate_params("affine", "albumentations", {"degrees": (-5.0, 5.0)})
        assert "rotate" in result
        assert result["rotate"] == (-5.0, 5.0)
        assert "degrees" not in result

    def test_scale_kornia_injects_degrees_default(self) -> None:
        """Kornia RandomAffine requires 'degrees'; translate_params injects 0.0 for scale-only ops.

        Kornia's RandomAffine has 'degrees' as a required positional argument, but a scale-only op semantically does no
        rotation. The translator injects degrees=0.0 to satisfy the Kornia signature without changing pipeline
        semantics.

        """
        result = translate_params("scale", "kornia", {"scale": (0.8, 1.2)})
        assert result.get("degrees") == 0.0
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_torchvision_injects_degrees_default(self) -> None:
        """TorchVision RandomAffine also requires 'degrees'; translator injects 0.0 for scale-only ops."""
        result = translate_params("scale", "torchvision", {"scale": (0.8, 1.2)})
        assert result.get("degrees") == 0.0
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_albumentations_no_degrees_injection(self) -> None:
        """Albumentations Affine accepts scale without a degrees argument, so no injection occurs."""
        result = translate_params("scale", "albumentations", {"scale": (0.8, 1.2)})
        assert "degrees" not in result
        assert result.get("scale") == (0.8, 1.2)

    def test_scale_factor_key_renamed_to_scale(self) -> None:
        """User-facing 'factor' alias is renamed to the backend-expected 'scale' key."""
        result = translate_params("scale", "kornia", {"factor": (0.9, 1.1)})
        assert "scale" in result
        assert "factor" not in result

    def test_shear_kornia_degrees_to_shear(self) -> None:
        """Kornia shear takes 'shear', not 'degrees'; translator renames the key."""
        result = translate_params("shear", "kornia", {"degrees": (-10.0, 10.0)})
        assert "shear" in result
        assert "degrees" not in result

    def test_hflip_passthrough(self) -> None:
        """Hflip has no tunable parameters; empty kwargs pass through unchanged."""
        result = translate_params("hflip", "kornia", {})
        assert result == {}

    def test_unknown_op_raises_value_error(self) -> None:
        """Unknown op name raises ValueError mentioning 'unknown op'."""
        with pytest.raises(ValueError, match="unknown op"):
            translate_params("nonexistent", "kornia", {})

    def test_unknown_backend_raises_value_error(self) -> None:
        """Unknown backend name raises ValueError mentioning 'unknown backend'."""
        with pytest.raises(ValueError, match="unknown backend"):
            translate_params("rotation", "bad_backend", {})
