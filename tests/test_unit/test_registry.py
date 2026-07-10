"""Tests for the pluggable adapter registry.

Covers registration, capability flags on the built-in adapters, longest-prefix detection routing to a registered
third-party adapter, lazy entry-point discovery (via a monkeypatched group), and backwards-compatible ``isinstance``.

"""

from __future__ import annotations

import warnings

import pytest

from fuse_augmentations import _backend
from fuse_augmentations._backend import (
    ADAPTERS_ENTRY_POINT_GROUP,
    Backend,
    _Entry,
    adapter_capabilities,
    detect_backend,
    detect_backends_per_transform,
    register_adapter,
)
from fuse_augmentations.adapters import AlbumentationsAdapter, KorniaAdapter, TorchVisionAdapter
from fuse_augmentations.types import TransformAdapter


@pytest.fixture
def clean_registry():
    """Snapshot and restore the process-global registry + entry-point guard around a test."""
    saved_registry = dict(_backend._ADAPTER_REGISTRY)
    saved_loaded = _backend._ENTRYPOINTS_LOADED
    try:
        yield
    finally:
        _backend._ADAPTER_REGISTRY.clear()
        _backend._ADAPTER_REGISTRY.update(saved_registry)
        _backend._ENTRYPOINTS_LOADED = saved_loaded


class _DummyAdapter:
    """Minimal adapter-like object with a capabilities flag for registry tests."""

    capabilities = frozenset({"rotation", "hflip"})
    sampling_semantics = "per_sample"


class TestBuiltinRegistration:
    """The three built-in adapters self-register with the expected backend, prefix, and capabilities."""

    def test_builtins_registered(self):
        """All three built-in adapter names are present in the registry."""
        assert {"kornia", "torchvision", "albumentations"} <= set(_backend._ADAPTER_REGISTRY)

    @pytest.mark.parametrize(
        ("name", "backend", "prefix"),
        [
            ("kornia", Backend.KORNIA, "kornia."),
            ("torchvision", Backend.TORCHVISION, "torchvision."),
            ("albumentations", Backend.ALBUMENTATIONS, "albumentations."),
        ],
    )
    def test_builtin_entry_fields(self, name, backend, prefix):
        """Each built-in registers its backend enum and module prefix."""
        entry = _backend._ADAPTER_REGISTRY[name]
        assert entry.backend is backend
        assert prefix in entry.prefixes

    @pytest.mark.parametrize(
        ("adapter_cls", "expected_ops", "expected_semantics"),
        [
            pytest.param(
                KorniaAdapter,
                {
                    "rotation",
                    "affine",
                    "shear",
                    "translate",
                    "hflip",
                    "vflip",
                    "scale",
                    "perspective",
                    "rotation90",
                    "brightness",
                    "contrast",
                },
                "per_batch",
                id="kornia",
            ),
            pytest.param(
                TorchVisionAdapter,
                {"rotation", "affine", "hflip", "vflip", "scale", "perspective"},
                "per_batch",
                id="torchvision",
            ),
            pytest.param(
                AlbumentationsAdapter,
                {"rotation", "affine", "hflip", "vflip", "scale", "perspective", "rotation90"},
                "per_sample",
                id="albumentations",
            ),
        ],
    )
    def test_capabilities_and_semantics_present(self, adapter_cls, expected_ops, expected_semantics):
        """capabilities/sampling_semantics are present on all 3 built-in adapters."""
        assert adapter_cls.capabilities == frozenset(expected_ops)
        assert adapter_cls.sampling_semantics == expected_semantics

    def test_torchvision_lacks_rotation90(self):
        """Rotation90 is a known TorchVision gap and must not appear in its capabilities."""
        assert "rotation90" not in TorchVisionAdapter.capabilities


class TestIsinstanceBackwardsCompat:
    """Adding optional Protocol members must not break runtime_checkable isinstance."""

    @pytest.mark.parametrize("adapter_cls", [KorniaAdapter, TorchVisionAdapter, AlbumentationsAdapter])
    def test_isinstance_still_true(self, adapter_cls):
        """Each built-in adapter is still a TransformAdapter instance."""
        assert isinstance(adapter_cls(), TransformAdapter)


class TestAdapterCapabilitiesHelper:
    """adapter_capabilities getattr helper defaults gracefully."""

    def test_missing_member_defaults_empty(self):
        """An object without capabilities yields an empty frozenset (not an error)."""
        assert adapter_capabilities(object()) == frozenset()

    def test_reads_declared_capabilities(self):
        """A declared capabilities set is returned as a frozenset."""
        assert adapter_capabilities(_DummyAdapter()) == frozenset({"rotation", "hflip"})


class TestRegisterAdapterRouting:
    """A registered dummy adapter with a fake prefix routes detection to its backend."""

    def test_dummy_prefix_detected(self, clean_registry):
        """register_adapter + a fake prefix makes detect_backends_per_transform route to that entry."""

        class _FakeTransform:
            pass

        _FakeTransform.__module__ = "dummypkg.transforms"
        register_adapter("dummy", _DummyAdapter(), "dummypkg.", backend=Backend.UNKNOWN)

        entry = _backend._ADAPTER_REGISTRY["dummy"]
        assert isinstance(entry, _Entry)
        assert entry.capabilities == frozenset({"rotation", "hflip"})

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a matched prefix must NOT warn
            result = detect_backends_per_transform([_FakeTransform()])
        assert result == [Backend.UNKNOWN]

    def test_longest_prefix_wins(self, clean_registry):
        """When two prefixes both match, the longer (more specific) one determines the backend."""

        class _SubTransform:
            pass

        _SubTransform.__module__ = "vendor.aug.geometric"
        register_adapter("vendor", _DummyAdapter(), "vendor.", backend=Backend.UNKNOWN)
        register_adapter("vendor_geo", _DummyAdapter(), "vendor.aug.", backend=Backend.KORNIA)

        assert detect_backend([_SubTransform()]) is Backend.KORNIA


class TestLazyEntryPoints:
    """Entry points load lazily on the first detection miss, in isolation, and never at import."""

    def test_not_loaded_at_import(self, clean_registry):
        """A prefix hit resolvable from the built-in registry must not trigger entry-point scanning."""
        _backend._ENTRYPOINTS_LOADED = False

        class _KorniaLike:
            pass

        _KorniaLike.__module__ = "kornia.augmentation"
        detect_backend([_KorniaLike()])
        assert _backend._ENTRYPOINTS_LOADED is False

    def test_loaded_on_miss(self, clean_registry, monkeypatch):
        """A detection miss scans the entry-point group exactly once and registers the discovered adapter."""
        _backend._ENTRYPOINTS_LOADED = False

        registered = {"count": 0}

        def _fake_register():
            registered["count"] += 1
            register_adapter("late", _DummyAdapter(), "latepkg.", backend=Backend.UNKNOWN)

        class _FakeEP:
            name = "late"

            @staticmethod
            def load():
                return _fake_register

        def _fake_entry_points(*, group):
            assert group == ADAPTERS_ENTRY_POINT_GROUP
            return [_FakeEP()]

        monkeypatch.setattr("importlib.metadata.entry_points", _fake_entry_points)

        class _LateTransform:
            pass

        _LateTransform.__module__ = "latepkg.transforms"

        # First call misses the built-in registry, triggers lazy load, then resolves via the late adapter.
        assert detect_backend([_LateTransform()]) is Backend.UNKNOWN
        assert registered["count"] == 1
        assert _backend._ENTRYPOINTS_LOADED is True

        # A second detection must not re-scan the group.
        detect_backend([_LateTransform()])
        assert registered["count"] == 1

    def test_broken_entry_point_isolated(self, clean_registry, monkeypatch):
        """A failing entry point is warned and skipped without breaking detection."""
        _backend._ENTRYPOINTS_LOADED = False

        class _BadEP:
            name = "broken"

            @staticmethod
            def load():
                raise RuntimeError("boom")

        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: [_BadEP()])

        class _Unknown:
            pass

        _Unknown.__module__ = "totally.unknown"

        with pytest.warns(UserWarning, match="broken"):
            result = detect_backends_per_transform([_Unknown()])
        assert result == [None]
