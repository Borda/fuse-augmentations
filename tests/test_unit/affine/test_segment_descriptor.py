"""Unit tests for SegmentDescriptor dataclass."""

from __future__ import annotations

import json

import pytest

from fuse_augmentations import SegmentDescriptor


class TestSegmentDescriptorImport:
    def test_importable_from_top_level(self):
        import fuse_augmentations

        assert hasattr(fuse_augmentations, "SegmentDescriptor")
        assert isinstance(fuse_augmentations.SegmentDescriptor, type)

    def test_in_all(self):
        import fuse_augmentations

        assert "SegmentDescriptor" in fuse_augmentations.__all__


class TestSegmentDescriptorConstruction:
    def test_minimal_construction(self):
        descriptor = SegmentDescriptor(kind="fused", transforms=("RandomRotation",), n_warps_saved=0)
        assert descriptor.kind == "fused"
        assert descriptor.transforms == ("RandomRotation",)
        assert descriptor.n_warps_saved == 0
        assert descriptor.backend is None

    def test_with_backend(self):
        descriptor = SegmentDescriptor(
            kind="exact",
            transforms=("HFlip", "VFlip"),
            n_warps_saved=2,
            backend="KorniaAdapter",
        )
        assert descriptor.backend == "KorniaAdapter"

    def test_frozen_immutable(self):
        descriptor = SegmentDescriptor(kind="fused", transforms=("Rotate",), n_warps_saved=1)
        with pytest.raises((AttributeError, TypeError)):
            descriptor.kind = "exact"  # type: ignore[misc]

    def test_slots(self):
        assert hasattr(SegmentDescriptor, "__slots__")


class TestSegmentDescriptorToDict:
    def test_to_dict_has_all_fields(self):
        descriptor = SegmentDescriptor(
            kind="fused",
            transforms=("Rotate", "Scale"),
            n_warps_saved=1,
            backend="KorniaAdapter",
        )
        result = descriptor.to_dict()
        assert set(result.keys()) == {"kind", "transforms", "n_warps_saved", "backend"}

    def test_to_dict_json_serializable(self):
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
        descriptor = SegmentDescriptor(kind=kind, transforms=("SomeTransform",), n_warps_saved=0)
        serialized = json.dumps(descriptor.to_dict())
        restored = json.loads(serialized)
        assert restored["kind"] == kind
