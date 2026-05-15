"""Comprehensive unit tests for TransformSpec frozen dataclass.

Tests cover construction, defaults, immutability, serialization round-trips, equality, export visibility, edge cases
with nested params and boundary prob values.

"""

from __future__ import annotations

import json

import pytest

import fuse_aug
import fuse_augmentations
from fuse_augmentations import TransformSpec


class TestTransformSpecConstruction:
    """TransformSpec construction with explicit and default fields."""

    def test_all_fields_explicit(self):
        """All three fields (operation, params, prob) are stored correctly."""
        spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        assert spec.operation == "rotation"
        assert spec.params == {"degrees": (-30.0, 30.0)}
        assert spec.prob == 0.8

    def test_default_prob_is_one(self):
        """TransformSpec.prob defaults to 1.0 when omitted."""
        spec = TransformSpec(operation="hflip", params={})
        assert spec.prob == 1.0, f"Expected default prob=1.0, got {spec.prob}"

    def test_empty_params_dict(self):
        """Params={} is accepted and stored as empty mapping."""
        spec = TransformSpec(operation="vflip", params={})
        assert spec.params == {}

    def test_nested_params_structure(self):
        """Nested params dict is stored and accessible as-is."""
        nested = {"scale": (0.8, 1.2), "translate": {"x": 0.1, "y": -0.2}}
        spec = TransformSpec(operation="affine", params=nested, prob=0.9)
        assert spec.params == nested
        assert spec.params["translate"]["x"] == 0.1  # type: ignore[index]

    @pytest.mark.parametrize(
        "prob_value",
        [
            pytest.param(0.0, id="prob=0.0"),
            pytest.param(0.5, id="prob=0.5"),
            pytest.param(1.0, id="prob=1.0"),
        ],
    )
    def test_prob_boundary_values(self, prob_value):
        """Boundary prob values [0.0, 0.5, 1.0] are accepted."""
        spec = TransformSpec(operation="rotation", params={}, prob=prob_value)
        assert spec.prob == prob_value


class TestTransformSpecFrozen:
    """TransformSpec is immutable (frozen dataclass)."""

    @pytest.mark.parametrize(
        "field,value",
        [
            pytest.param("operation", "hflip", id="operation"),
            pytest.param("params", {"new": True}, id="params"),
            pytest.param("prob", 0.5, id="prob"),
        ],
    )
    def test_assignment_raises(self, field, value):
        """Assigning to any field raises TypeError or AttributeError."""
        spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        with pytest.raises((TypeError, AttributeError)):
            setattr(spec, field, value)

    def test_params_mapping_is_read_only(self):
        """Params mapping does not support item assignment."""
        spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        with pytest.raises(TypeError):
            spec.params["degrees"] = (-10.0, 10.0)  # type: ignore[index]


class TestTransformSpecSerialization:
    """to_dict() / from_dict() serialization and round-trips."""

    def test_to_dict_keys(self):
        """to_dict() returns dict with 'operation', 'params', 'prob' keys."""
        spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        spec_dict = spec.to_dict()
        assert set(spec_dict.keys()) == {"operation", "params", "prob"}

    def test_to_dict_values(self):
        """to_dict() values match the constructed spec fields."""
        spec = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        spec_dict = spec.to_dict()
        assert spec_dict["operation"] == "rotation"
        assert spec_dict["params"] == {"degrees": [-30.0, 30.0]}
        assert spec_dict["prob"] == 0.8

    def test_round_trip_identity(self):
        """from_dict(to_dict(spec)) == spec."""
        original = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        restored = TransformSpec.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_all_fields(self):
        """from_dict with all keys produces correct spec."""
        spec_dict = {"operation": "hflip", "params": {"mirror": True}, "prob": 0.5}
        spec = TransformSpec.from_dict(spec_dict)
        assert spec.operation == "hflip"
        assert spec.params == {"mirror": True}
        assert spec.prob == 0.5

    def test_from_dict_missing_prob_defaults(self):
        """Missing 'prob' key in dict defaults to 1.0."""
        spec_dict = {"operation": "rotation", "params": {"degrees": (-10.0, 10.0)}}
        spec = TransformSpec.from_dict(spec_dict)
        assert spec.prob == 1.0, f"Missing 'prob' should default to 1.0, got {spec.prob}"

    def test_from_dict_missing_params_defaults_to_empty(self):
        """Missing 'params' key in dict defaults to empty dict."""
        spec_dict = {"operation": "hflip"}
        spec = TransformSpec.from_dict(spec_dict)
        assert spec.params == {}, f"Missing 'params' should default to empty dict, got {spec.params}"

    def test_json_full_round_trip(self):
        """Full JSON serialization round-trip preserves spec equality."""
        original = TransformSpec(operation="scale", params={"factor": (0.8, 1.2)}, prob=0.9)
        json_str = json.dumps(original.to_dict())
        restored = TransformSpec.from_dict(json.loads(json_str))
        assert restored == original

    def test_json_round_trip_with_nested_params(self):
        """Nested params survive JSON round-trip (tuples become lists)."""
        original = TransformSpec(
            operation="affine",
            params={"scale": [0.8, 1.2], "translate": {"x": 0.1}},
            prob=0.75,
        )
        json_str = json.dumps(original.to_dict())
        restored = TransformSpec.from_dict(json.loads(json_str))
        assert restored.to_dict() == json.loads(json_str)

    def test_from_dict_preserves_non_range_lists(self):
        """Non-2-element lists are preserved as lists, not converted to tuples."""
        spec = TransformSpec.from_dict(
            {
                "operation": "affine",
                "params": {"padding": [1, 2, 3], "degrees": [-10.0, 10.0]},
                "prob": 0.5,
            },
        )
        assert spec.params["padding"] == [1, 2, 3]
        assert spec.params["degrees"] == (-10.0, 10.0)

    def test_from_dict_backend_specific_key_stays_list(self) -> None:
        """Two-element list with non-float-range values remains a list after round-trip."""
        spec = TransformSpec.from_dict({"operation": "rotation", "params": {"limit": [-10, 10]}, "prob": 1.0})
        assert spec.params["limit"] == [-10, 10], f"Expected limit to remain a list, got {spec.params['limit']!r}"
        reloaded = TransformSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert reloaded.params["limit"] == [-10, 10]

    def test_from_dict_extra_keys_behavior(self):
        """from_dict with extra unknown keys should raise TypeError (frozen dataclass rejects unknowns)."""
        spec_dict = {"operation": "hflip", "params": {}, "prob": 1.0, "unknown_key": "surprise"}
        try:
            spec = TransformSpec.from_dict(spec_dict)
        except TypeError:
            pass
        else:
            assert not hasattr(spec, "unknown_key"), "Extra keys should not become attributes"


class TestTransformSpecEquality:
    """Value-based equality for TransformSpec."""

    def test_equal_specs(self):
        """Two specs with identical fields compare equal."""
        spec_a = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        spec_b = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        assert spec_a == spec_b

    @pytest.mark.parametrize(
        "other_kwargs",
        [
            pytest.param(
                {"operation": "hflip", "params": {"degrees": (-30.0, 30.0)}, "prob": 0.8},
                id="different_operation",
            ),
            pytest.param(
                {"operation": "rotation", "params": {"degrees": (-10.0, 10.0)}, "prob": 0.8},
                id="different_params",
            ),
            pytest.param(
                {"operation": "rotation", "params": {"degrees": (-30.0, 30.0)}, "prob": 0.5},
                id="different_prob",
            ),
        ],
    )
    def test_unequal_specs(self, other_kwargs):
        """Specs differing in any field compare unequal."""
        spec_a = TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=0.8)
        spec_b = TransformSpec(**other_kwargs)
        assert spec_a != spec_b

    def test_not_equal_to_non_spec(self):
        """TransformSpec does not equal str, int, or None."""
        spec = TransformSpec(operation="rotation", params={}, prob=1.0)
        assert spec != "rotation"
        assert spec != 42
        assert spec != None  # noqa: E711


class TestTransformSpecExport:
    """TransformSpec is accessible from top-level package and fuse_aug alias."""

    def test_in_all(self):
        """'TransformSpec' appears in fuse_augmentations.__all__."""
        assert "TransformSpec" in fuse_augmentations.__all__

    def test_importable_from_fuse_augmentations(self):
        """TransformSpec is an attribute of the fuse_augmentations module."""
        assert hasattr(fuse_augmentations, "TransformSpec")

    def test_importable_from_fuse_aug(self):
        """TransformSpec is an attribute of the fuse_aug alias module."""
        assert hasattr(fuse_aug, "TransformSpec")

    def test_same_class_both_packages(self):
        """fuse_augmentations.TransformSpec and fuse_aug.TransformSpec are the same class."""
        assert fuse_augmentations.TransformSpec is fuse_aug.TransformSpec


class TestTransformSpecFromDictValidation:
    """from_dict() validates prob bounds and type errors."""

    def test_from_dict_invalid_prob_type_raises(self) -> None:
        """Non-float string for 'prob' raises ValueError."""
        with pytest.raises(ValueError, match="could not convert"):
            TransformSpec.from_dict({"operation": "rotation", "params": {}, "prob": "not_a_float"})

    def test_from_dict_prob_out_of_range_raises(self) -> None:
        """Prob outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="prob must be in"):
            TransformSpec.from_dict({"operation": "rotation", "params": {}, "prob": -0.1})

        with pytest.raises(ValueError, match="prob must be in"):
            TransformSpec.from_dict({"operation": "rotation", "params": {}, "prob": 1.5})

    def test_from_dict_missing_operation_key_raises(self) -> None:
        """Missing 'operation' key raises KeyError — 'operation' is the only required field."""
        with pytest.raises(KeyError):
            TransformSpec.from_dict({})

        with pytest.raises(KeyError):
            TransformSpec.from_dict({"params": {"degrees": (-10.0, 10.0)}, "prob": 1.0})
