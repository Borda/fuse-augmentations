"""Comprehensive unit tests for _resolver.py (Phase C.2).

Tests cover resolve_op, SUPPORTED_OPS, SUPPORTED_BACKENDS — including
error paths, case sensitivity, and cross-backend resolution.
"""

from __future__ import annotations

import pytest


class TestSupportedConstants:
    """SUPPORTED_OPS and SUPPORTED_BACKENDS structure and membership."""

    def test_supported_backends_type(self):
        from fuse_augmentations._resolver import SUPPORTED_BACKENDS

        assert isinstance(SUPPORTED_BACKENDS, frozenset), (
            f"SUPPORTED_BACKENDS should be frozenset, got {type(SUPPORTED_BACKENDS).__name__}"
        )

    def test_supported_ops_type(self):
        from fuse_augmentations._resolver import SUPPORTED_OPS

        assert isinstance(SUPPORTED_OPS, frozenset), (
            f"SUPPORTED_OPS should be frozenset, got {type(SUPPORTED_OPS).__name__}"
        )

    @pytest.mark.parametrize(
        "backend",
        ["kornia", "torchvision", "albumentations"],
    )
    def test_supported_backends_contains(self, backend):
        from fuse_augmentations._resolver import SUPPORTED_BACKENDS

        assert backend in SUPPORTED_BACKENDS, f"'{backend}' should be in SUPPORTED_BACKENDS"

    @pytest.mark.parametrize(
        "op",
        ["rotation", "hflip", "vflip"],
    )
    def test_supported_ops_minimum_ops(self, op):
        from fuse_augmentations._resolver import SUPPORTED_OPS

        assert op in SUPPORTED_OPS, f"'{op}' should be in SUPPORTED_OPS"

    def test_supported_ops_not_empty(self):
        from fuse_augmentations._resolver import SUPPORTED_OPS

        assert len(SUPPORTED_OPS) >= 3, f"Expected at least 3 supported ops, got {len(SUPPORTED_OPS)}"


class TestResolveOpKornia:
    """resolve_op with backend='kornia'."""

    @pytest.fixture(autouse=True)
    def _skip_without_kornia(self):
        pytest.importorskip("kornia")

    def test_rotation_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("rotation", "kornia")
        assert callable(cls), f"resolve_op('rotation', 'kornia') should return callable, got {type(cls)}"

    def test_hflip_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("hflip", "kornia")
        assert callable(cls), f"resolve_op('hflip', 'kornia') should return callable, got {type(cls)}"

    def test_vflip_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("vflip", "kornia")
        assert callable(cls), f"resolve_op('vflip', 'kornia') should return callable, got {type(cls)}"


class TestResolveOpTorchVision:
    """resolve_op with backend='torchvision'."""

    @pytest.fixture(autouse=True)
    def _skip_without_torchvision(self):
        pytest.importorskip("torchvision")

    def test_hflip_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("hflip", "torchvision")
        assert callable(cls)

    def test_rotation_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("rotation", "torchvision")
        assert callable(cls)


class TestResolveOpAlbumentations:
    """resolve_op with backend='albumentations'."""

    @pytest.fixture(autouse=True)
    def _skip_without_albumentations(self):
        pytest.importorskip("albumentations")

    def test_hflip_returns_callable(self):
        from fuse_augmentations._resolver import resolve_op

        cls = resolve_op("hflip", "albumentations")
        assert callable(cls)


class TestResolveOpErrors:
    """Error paths for resolve_op."""

    def test_unknown_op_raises_value_error(self):
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError, match="unknown"):
            resolve_op("nonexistent_op_xyz", "kornia")

    def test_unknown_backend_raises_value_error(self):
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError):
            resolve_op("rotation", "nonexistent_backend_xyz")

    def test_case_sensitive_op(self):
        """resolve_op is case-sensitive: 'Rotation' != 'rotation'."""
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError):
            resolve_op("Rotation", "kornia")

    def test_case_sensitive_backend(self):
        """resolve_op is case-sensitive: 'Kornia' != 'kornia'."""
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError):
            resolve_op("rotation", "Kornia")

    def test_empty_op_string(self):
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError):
            resolve_op("", "kornia")

    def test_empty_backend_string(self):
        from fuse_augmentations._resolver import resolve_op

        with pytest.raises(ValueError):
            resolve_op("rotation", "")


class TestResolveOpComprehensive:
    """All ops in SUPPORTED_OPS resolve for at least one installed backend."""

    def test_all_ops_resolvable(self):
        from fuse_augmentations._resolver import SUPPORTED_BACKENDS, SUPPORTED_OPS, resolve_op

        available_backends = []
        for backend in SUPPORTED_BACKENDS:
            try:
                __import__(backend if backend != "albumentations" else "albumentations")
                available_backends.append(backend)
            except ImportError:
                continue

        if not available_backends:
            pytest.skip("No backends installed")

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
            f"Ops not resolvable by any installed backend: {unresolvable}. "
            f"Available backends: {available_backends}"
        )
