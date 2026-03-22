"""Comprehensive unit tests for TransformSpec frozen dataclass (Phase C.1).

Tests cover construction, defaults, immutability, serialization round-trips, equality, export visibility, edge cases
with nested params and boundary p values.

"""

from __future__ import annotations

import json

import pytest


class TestTransformSpecConstruction:
    """TransformSpec construction with explicit and default fields."""

    def test_all_fields_explicit(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        assert spec.op == "rotation"
        assert spec.params == {"degrees": (-30.0, 30.0)}
        assert spec.p == 0.8

    def test_default_p_is_one(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="hflip", params={})
        assert spec.p == 1.0, f"Expected default p=1.0, got {spec.p}"

    def test_empty_params_dict(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="vflip", params={})
        assert spec.params == {}

    def test_nested_params_structure(self):
        from fuse_augmentations import TransformSpec

        nested = {"scale": (0.8, 1.2), "translate": {"x": 0.1, "y": -0.2}}
        spec = TransformSpec(op="affine", params=nested, p=0.9)
        assert spec.params == nested
        assert spec.params["translate"]["x"] == 0.1  # type: ignore[index]

    @pytest.mark.parametrize(
        "p_value",
        [0.0, 0.5, 1.0],
        ids=["p=0.0", "p=0.5", "p=1.0"],
    )
    def test_p_boundary_values(self, p_value):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={}, p=p_value)
        assert spec.p == p_value


class TestTransformSpecFrozen:
    """TransformSpec is immutable (frozen dataclass)."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("op", "hflip"),
            ("params", {"new": True}),
            ("p", 0.5),
        ],
        ids=["op", "params", "p"],
    )
    def test_assignment_raises(self, field, value):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        with pytest.raises((TypeError, AttributeError)):
            setattr(spec, field, value)

    def test_params_mapping_is_read_only(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        with pytest.raises(TypeError):
            spec.params["degrees"] = (-10.0, 10.0)  # type: ignore[index]


class TestTransformSpecSerialization:
    """to_dict() / from_dict() serialization and round-trips."""

    def test_to_dict_keys(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        d = spec.to_dict()
        assert set(d.keys()) == {"op", "params", "p"}

    def test_to_dict_values(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        d = spec.to_dict()
        assert d["op"] == "rotation"
        assert d["params"] == {"degrees": [-30.0, 30.0]}
        assert d["p"] == 0.8

    def test_round_trip_identity(self):
        from fuse_augmentations import TransformSpec

        original = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        restored = TransformSpec.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_all_fields(self):
        from fuse_augmentations import TransformSpec

        d = {"op": "hflip", "params": {"mirror": True}, "p": 0.5}
        spec = TransformSpec.from_dict(d)
        assert spec.op == "hflip"
        assert spec.params == {"mirror": True}
        assert spec.p == 0.5

    def test_from_dict_missing_p_defaults(self):
        from fuse_augmentations import TransformSpec

        d = {"op": "rotation", "params": {"degrees": (-10.0, 10.0)}}
        spec = TransformSpec.from_dict(d)
        assert spec.p == 1.0, f"Missing 'p' should default to 1.0, got {spec.p}"

    def test_from_dict_missing_params_defaults_to_empty(self):
        from fuse_augmentations import TransformSpec

        d = {"op": "hflip"}
        spec = TransformSpec.from_dict(d)
        assert spec.params == {}, f"Missing 'params' should default to empty dict, got {spec.params}"

    def test_json_full_round_trip(self):
        from fuse_augmentations import TransformSpec

        original = TransformSpec(op="scale", params={"factor": (0.8, 1.2)}, p=0.9)
        json_str = json.dumps(original.to_dict())
        restored = TransformSpec.from_dict(json.loads(json_str))
        assert restored == original

    def test_json_round_trip_with_nested_params(self):
        from fuse_augmentations import TransformSpec

        original = TransformSpec(
            op="affine",
            params={"scale": [0.8, 1.2], "translate": {"x": 0.1}},
            p=0.75,
        )
        json_str = json.dumps(original.to_dict())
        restored = TransformSpec.from_dict(json.loads(json_str))
        # After JSON round-trip, tuples become lists — compare via to_dict()
        assert restored.to_dict() == json.loads(json_str)

    def test_from_dict_preserves_non_range_lists(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec.from_dict(
            {
                "op": "affine",
                "params": {"padding": [1, 2, 3], "degrees": [-10.0, 10.0]},
                "p": 0.5,
            },
        )
        assert spec.params["padding"] == [1, 2, 3]
        assert spec.params["degrees"] == (-10.0, 10.0)

    def test_from_dict_backend_specific_key_stays_list(self) -> None:
        import json

        from fuse_augmentations import TransformSpec

        spec = TransformSpec.from_dict({"op": "rotation", "params": {"limit": [-10, 10]}, "p": 1.0})
        assert spec.params["limit"] == [-10, 10], f"Expected limit to remain a list, got {spec.params['limit']!r}"
        reloaded = TransformSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert reloaded.params["limit"] == [-10, 10]

    def test_from_dict_extra_keys_behavior(self):
        """from_dict with extra unknown keys should raise TypeError (frozen dataclass rejects unknowns)."""
        from fuse_augmentations import TransformSpec

        d = {"op": "hflip", "params": {}, "p": 1.0, "unknown_key": "surprise"}
        # The implementation may either raise TypeError or ignore extra keys.
        # We test both: if it raises, it should be TypeError; if it succeeds,
        # the extra key should not appear as an attribute.
        try:
            spec = TransformSpec.from_dict(d)
        except TypeError:
            pass  # expected: dataclass rejects extra kwargs
        else:
            assert not hasattr(spec, "unknown_key"), "Extra keys should not become attributes"


class TestTransformSpecEquality:
    """Value-based equality for TransformSpec."""

    def test_equal_specs(self):
        from fuse_augmentations import TransformSpec

        a = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        b = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        assert a == b

    @pytest.mark.parametrize(
        "other_kwargs",
        [
            {"op": "hflip", "params": {"degrees": (-30.0, 30.0)}, "p": 0.8},
            {"op": "rotation", "params": {"degrees": (-10.0, 10.0)}, "p": 0.8},
            {"op": "rotation", "params": {"degrees": (-30.0, 30.0)}, "p": 0.5},
        ],
        ids=["different_op", "different_params", "different_p"],
    )
    def test_unequal_specs(self, other_kwargs):
        from fuse_augmentations import TransformSpec

        a = TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=0.8)
        b = TransformSpec(**other_kwargs)
        assert a != b

    def test_not_equal_to_non_spec(self):
        from fuse_augmentations import TransformSpec

        spec = TransformSpec(op="rotation", params={}, p=1.0)
        assert spec != "rotation"
        assert spec != 42
        assert spec != None  # noqa: E711


class TestTransformSpecExport:
    """TransformSpec is accessible from top-level package and fuse_aug alias."""

    def test_in_all(self):
        import fuse_augmentations

        assert "TransformSpec" in fuse_augmentations.__all__

    def test_importable_from_fuse_augmentations(self):
        import fuse_augmentations

        assert hasattr(fuse_augmentations, "TransformSpec")

    def test_importable_from_fuse_aug(self):
        import fuse_aug

        assert hasattr(fuse_aug, "TransformSpec")

    def test_same_class_both_packages(self):
        import fuse_aug
        import fuse_augmentations

        assert fuse_augmentations.TransformSpec is fuse_aug.TransformSpec


class TestTransformSpecFromDictValidation:
    """from_dict() validates p bounds and type errors."""

    def test_from_dict_invalid_p_type_raises(self) -> None:
        """float('not_a_float') raises ValueError precisely — not TypeError."""
        from fuse_augmentations import TransformSpec

        with pytest.raises(ValueError, match="could not convert"):
            TransformSpec.from_dict({"op": "rotation", "params": {}, "p": "not_a_float"})

    def test_from_dict_p_out_of_range_raises(self) -> None:
        from fuse_augmentations import TransformSpec

        with pytest.raises(ValueError, match="p must be in"):
            TransformSpec.from_dict({"op": "rotation", "params": {}, "p": -0.1})

        with pytest.raises(ValueError, match="p must be in"):
            TransformSpec.from_dict({"op": "rotation", "params": {}, "p": 1.5})

    def test_from_dict_missing_op_key_raises(self) -> None:
        """Missing 'op' key raises KeyError — 'op' is the only required field."""
        from fuse_augmentations import TransformSpec

        with pytest.raises(KeyError):
            TransformSpec.from_dict({})

        with pytest.raises(KeyError):
            TransformSpec.from_dict({"params": {"degrees": (-10.0, 10.0)}, "p": 1.0})
