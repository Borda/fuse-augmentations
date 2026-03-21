"""Tests for fuse_augmentations._types."""

import pytest
import torch

from fuse_augmentations._types import (
    InterpolationMode,
    PaddingMode,
    ReorderPolicy,
    TransformAdapter,
    TransformCategory,
)


def _full_adapter_methods():
    """Return a dict of all six TransformAdapter method implementations."""
    return {
        "category": lambda self, transform: TransformCategory.SPATIAL_KERNEL,
        "sample_params": lambda self, transform, input_shape, device: {},
        "build_matrix": lambda self, transform, params, H, W: torch.eye(3).unsqueeze(0),
        "exact_flip_dims": lambda self, transform: [],
        "exact_apply": lambda self, transform, image: image,
        "call_nonfused": lambda self, transform, image, **kwargs: image,
    }


class TestTransformCategory:
    """TransformCategory enum -- membership and string values."""

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (TransformCategory.GEOMETRIC_INTERP, "geometric_interp"),
            (TransformCategory.GEOMETRIC_EXACT, "geometric_exact"),
            (TransformCategory.POINTWISE, "pointwise"),
            (TransformCategory.SPATIAL_KERNEL, "spatial_kernel"),
            (TransformCategory.PROJECTIVE, "projective"),
        ],
    )
    def test_value(self, member, expected_value):
        """Enum member has the expected string value."""
        assert member.value == expected_value, f"{member.name}.value should be {expected_value!r}"

    def test_exactly_five_members(self):
        """TransformCategory has exactly 5 members."""
        assert len(TransformCategory) == 5, f"Expected 5 members, got {len(TransformCategory)}"

    def test_member_names(self):
        """TransformCategory member names match the spec."""
        assert {m.name for m in TransformCategory} == {
            "GEOMETRIC_INTERP",
            "GEOMETRIC_EXACT",
            "POINTWISE",
            "SPATIAL_KERNEL",
            "PROJECTIVE",
        }

    @pytest.mark.parametrize("member", list(TransformCategory))
    def test_value_is_str(self, member):
        """Each TransformCategory member value is a string."""
        assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value).__name__}"


class TestReorderPolicy:
    """ReorderPolicy enum -- membership and string values."""

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (ReorderPolicy.NONE, "none"),
            (ReorderPolicy.POINTWISE, "pointwise"),
            (ReorderPolicy.AGGRESSIVE, "aggressive"),
        ],
    )
    def test_value(self, member, expected_value):
        """Enum member has the expected string value."""
        assert member.value == expected_value, f"{member.name}.value should be {expected_value!r}"

    def test_exactly_three_members(self):
        """ReorderPolicy has exactly 3 members."""
        assert len(ReorderPolicy) == 3, f"Expected 3 members, got {len(ReorderPolicy)}"

    def test_member_names(self):
        """ReorderPolicy member names match the spec."""
        assert {m.name for m in ReorderPolicy} == {"NONE", "POINTWISE", "AGGRESSIVE"}

    @pytest.mark.parametrize("member", list(ReorderPolicy))
    def test_value_is_str(self, member):
        """Each ReorderPolicy member value is a string."""
        assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value).__name__}"


class TestInterpolationMode:
    """InterpolationMode IntEnum -- values, ordering, and int usability."""

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (InterpolationMode.NEAREST, 0),
            (InterpolationMode.BILINEAR, 1),
            (InterpolationMode.BICUBIC, 2),
        ],
    )
    def test_int_value(self, member, expected_int):
        """Enum member has the expected integer value."""
        assert int(member) == expected_int, f"{member.name} should equal {expected_int}"

    def test_ordering_nearest_lt_bilinear_lt_bicubic(self):
        """NEAREST < BILINEAR < BICUBIC ordering holds."""
        assert InterpolationMode.NEAREST < InterpolationMode.BILINEAR < InterpolationMode.BICUBIC

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (InterpolationMode.NEAREST, 0),
            (InterpolationMode.BILINEAR, 1),
            (InterpolationMode.BICUBIC, 2),
        ],
    )
    def test_usable_as_int_in_arithmetic(self, member, expected_int):
        """IntEnum member + 0 equals its integer value."""
        # IntEnum members must be interchangeable with plain ints.
        result = member + 0
        assert result == expected_int, f"{member.name} + 0 should equal {expected_int}"

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (InterpolationMode.NEAREST, 0),
            (InterpolationMode.BILINEAR, 1),
            (InterpolationMode.BICUBIC, 2),
        ],
    )
    def test_usable_as_list_index(self, member, expected_int):
        """IntEnum member can be used as a list index."""
        seq = ["a", "b", "c"]
        assert seq[member] == seq[expected_int], (
            f"Indexing with {member.name} should behave identically to indexing with {expected_int}"
        )


class TestPaddingMode:
    """PaddingMode IntEnum -- values, ordering, and int usability."""

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (PaddingMode.ZEROS, 0),
            (PaddingMode.BORDER, 1),
            (PaddingMode.REFLECTION, 2),
        ],
    )
    def test_int_value(self, member, expected_int):
        """Enum member has the expected integer value."""
        assert int(member) == expected_int, f"{member.name} should equal {expected_int}"

    def test_ordering_zeros_lt_border_lt_reflection(self):
        """ZEROS < BORDER < REFLECTION ordering holds."""
        assert PaddingMode.ZEROS < PaddingMode.BORDER < PaddingMode.REFLECTION

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (PaddingMode.ZEROS, 0),
            (PaddingMode.BORDER, 1),
            (PaddingMode.REFLECTION, 2),
        ],
    )
    def test_usable_as_int_in_arithmetic(self, member, expected_int):
        """IntEnum member + 0 equals its integer value."""
        result = member + 0
        assert result == expected_int, f"{member.name} + 0 should equal {expected_int}"

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (PaddingMode.ZEROS, 0),
            (PaddingMode.BORDER, 1),
            (PaddingMode.REFLECTION, 2),
        ],
    )
    def test_usable_as_list_index(self, member, expected_int):
        """IntEnum member can be used as a list index."""
        seq = ["a", "b", "c"]
        assert seq[member] == seq[expected_int], (
            f"Indexing with {member.name} should behave identically to indexing with {expected_int}"
        )


class TestTransformAdapterProtocol:
    """TransformAdapter @runtime_checkable Protocol -- isinstance contract."""

    def test_conforming_class_passes_isinstance(self):
        """A class implementing all six methods satisfies TransformAdapter."""
        methods = _full_adapter_methods()
        _DummyAdapter = type("_DummyAdapter", (), methods)
        assert isinstance(_DummyAdapter(), TransformAdapter), (
            "An object with all six required methods should be an instance of TransformAdapter"
        )

    @pytest.mark.parametrize(
        "missing_method",
        ["category", "sample_params", "build_matrix", "exact_flip_dims", "exact_apply", "call_nonfused"],
    )
    def test_missing_any_one_method_fails_isinstance(self, missing_method):
        """Dropping any single required method makes the object fail the Protocol check."""
        methods = _full_adapter_methods()
        del methods[missing_method]
        _Incomplete = type("_Incomplete", (), methods)
        assert not isinstance(_Incomplete(), TransformAdapter), (
            f"Object missing '{missing_method}' should not be an instance of TransformAdapter"
        )

    @pytest.mark.parametrize("non_adapter", [42, "hello", None, [], {}])
    def test_non_adapter_objects_fail_isinstance(self, non_adapter):
        """Non-adapter objects (int, str, None, list, dict) fail isinstance."""
        assert not isinstance(non_adapter, TransformAdapter), (
            f"{non_adapter!r} should not satisfy the TransformAdapter protocol"
        )
