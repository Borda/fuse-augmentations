"""Unit tests for SegmentDescriptor dataclass."""

from __future__ import annotations

import json

import pytest


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
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(kind="fused", transforms=("RandomRotation",), n_warps_saved=0)
        assert d.kind == "fused"
        assert d.transforms == ("RandomRotation",)
        assert d.n_warps_saved == 0
        assert d.backend is None

    def test_with_backend(self):
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(
            kind="exact",
            transforms=("HFlip", "VFlip"),
            n_warps_saved=2,
            backend="KorniaAdapter",
        )
        assert d.backend == "KorniaAdapter"

    def test_frozen_immutable(self):
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(kind="fused", transforms=("Rotate",), n_warps_saved=1)
        with pytest.raises((AttributeError, TypeError)):
            d.kind = "exact"  # type: ignore[misc]

    def test_slots(self):
        from fuse_augmentations import SegmentDescriptor

        assert hasattr(SegmentDescriptor, "__slots__")


class TestSegmentDescriptorToDict:
    def test_to_dict_has_all_fields(self):
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(
            kind="fused",
            transforms=("Rotate", "Scale"),
            n_warps_saved=1,
            backend="KorniaAdapter",
        )
        result = d.to_dict()
        assert set(result.keys()) == {"kind", "transforms", "n_warps_saved", "backend"}

    def test_to_dict_json_serializable(self):
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(kind="projective", transforms=("Perspective",), n_warps_saved=0)
        serialized = json.dumps(d.to_dict())
        restored = json.loads(serialized)
        assert restored["kind"] == "projective"
        assert restored["transforms"] == ["Perspective"]
        assert restored["n_warps_saved"] == 0
        assert restored["backend"] is None

    def test_to_dict_transforms_is_list_not_tuple(self):
        """to_dict() converts tuple to list for JSON compatibility."""
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(kind="exact", transforms=("HFlip",), n_warps_saved=1)
        result = d.to_dict()
        assert isinstance(result["transforms"], list)

    @pytest.mark.parametrize("kind", ["fused", "exact", "projective", "passthrough"])
    def test_all_kind_values_round_trip(self, kind):
        from fuse_augmentations import SegmentDescriptor

        d = SegmentDescriptor(kind=kind, transforms=("SomeTransform",), n_warps_saved=0)
        serialized = json.dumps(d.to_dict())
        restored = json.loads(serialized)
        assert restored["kind"] == kind
