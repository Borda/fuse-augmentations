"""Unit tests for SegmentDescriptor dataclass."""

from __future__ import annotations

import json

import pytest

from fuse_augmentations import SegmentDescriptor


class TestSegmentDescriptorImport:
    def test_importable_from_top_level(self):
        """SegmentDescriptor is importable from the top-level fuse_augmentations package and is a class."""
        import fuse_augmentations

        assert hasattr(fuse_augmentations, "SegmentDescriptor")
        assert isinstance(fuse_augmentations.SegmentDescriptor, type)

    def test_in_all(self):
        """SegmentDescriptor is listed in fuse_augmentations.__all__ as part of the public API surface."""
        import fuse_augmentations

        assert "SegmentDescriptor" in fuse_augmentations.__all__


class TestSegmentDescriptorConstruction:
    def test_minimal_construction(self):
        """SegmentDescriptor constructs with required fields and defaults backend to None."""
        descriptor = SegmentDescriptor(kind="fused", transforms=("RandomRotation",), n_warps_saved=0)
        assert descriptor.kind == "fused"
        assert descriptor.transforms == ("RandomRotation",)
        assert descriptor.n_warps_saved == 0
        assert descriptor.backend is None

    def test_with_backend(self):
        """SegmentDescriptor stores the optional backend field when explicitly provided."""
        descriptor = SegmentDescriptor(
            kind="exact",
            transforms=("HFlip", "VFlip"),
            n_warps_saved=2,
            backend="KorniaAdapter",
        )
        assert descriptor.backend == "KorniaAdapter"

    def test_machine_readable_reasons_default_none(self):
        """The additive barrier/split_reason/refused fields default to None when omitted."""
        descriptor = SegmentDescriptor(kind="fused", transforms=("Rotate",), n_warps_saved=0)
        assert descriptor.barrier is None
        assert descriptor.split_reason is None
        assert descriptor.refused is None

    def test_machine_readable_reasons_stored(self):
        """SegmentDescriptor stores the optional machine-readable reason fields when provided."""
        descriptor = SegmentDescriptor(
            kind="passthrough",
            transforms=("GaussianBlur",),
            n_warps_saved=0,
            barrier="spatial_kernel",
            split_reason="backend_boundary",
            refused="not_fusible",
        )
        assert descriptor.barrier == "spatial_kernel"
        assert descriptor.split_reason == "backend_boundary"
        assert descriptor.refused == "not_fusible"

    def test_frozen_immutable(self):
        """SegmentDescriptor is frozen so attribute mutation raises AttributeError or TypeError.

        Immutability is required because descriptors are emitted as deterministic plan summaries consumed by downstream
        tools; allowing post-construction mutation would silently invalidate plan equality checks and serialised output.

        """
        descriptor = SegmentDescriptor(kind="fused", transforms=("Rotate",), n_warps_saved=1)
        with pytest.raises((AttributeError, TypeError)):
            descriptor.kind = "exact"  # type: ignore[misc]

    def test_slots(self):
        """SegmentDescriptor defines __slots__ to prevent ad-hoc attribute assignment and reduce memory."""
        assert hasattr(SegmentDescriptor, "__slots__")


class TestSegmentDescriptorToDict:
    def test_to_dict_has_all_fields(self):
        """to_dict() emits every public field of SegmentDescriptor and nothing else."""
        descriptor = SegmentDescriptor(
            kind="fused",
            transforms=("Rotate", "Scale"),
            n_warps_saved=1,
            backend="KorniaAdapter",
        )
        result = descriptor.to_dict()
        assert set(result.keys()) == {
            "kind",
            "transforms",
            "n_warps_saved",
            "backend",
            "barrier",
            "split_reason",
            "refused",
        }

    def test_to_dict_json_serializable(self):
        """to_dict() output round-trips through json.dumps and json.loads with all values preserved.

        JSON-compatible output is required because plan descriptors are persisted to disk and shared across tooling that
        does not import fuse_augmentations directly.

        """
        descriptor = SegmentDescriptor(kind="projective", transforms=("Perspective",), n_warps_saved=0)
        serialized = json.dumps(descriptor.to_dict())
        restored = json.loads(serialized)
        assert restored["kind"] == "projective"
        assert restored["transforms"] == ["Perspective"]
        assert restored["n_warps_saved"] == 0
        assert restored["backend"] is None

    def test_to_dict_transforms_is_list_not_tuple(self):
        """to_dict() converts tuple to list for JSON compatibility."""
        descriptor = SegmentDescriptor(kind="exact", transforms=("HFlip",), n_warps_saved=1)
        result = descriptor.to_dict()
        assert isinstance(result["transforms"], list)

    @pytest.mark.parametrize("kind", ["fused", "exact", "projective", "passthrough"])
    def test_all_kind_values_round_trip(self, kind):
        """Every valid SegmentDescriptor.kind value survives a JSON round-trip without loss."""
        descriptor = SegmentDescriptor(kind=kind, transforms=("SomeTransform",), n_warps_saved=0)
        serialized = json.dumps(descriptor.to_dict())
        restored = json.loads(serialized)
        assert restored["kind"] == kind
