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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_adapter_methods():
    """Return a dict of all four TransformAdapter method implementations."""
    return {
        "category": lambda self, transform: TransformCategory.SPATIAL_KERNEL,
        "sample_params": lambda self, transform, input_shape, device: {},
        "build_matrix": lambda self, transform, params, H, W: torch.eye(3).unsqueeze(0),
        "call_nonfused": lambda self, transform, image, **kwargs: image,
    }


# ---------------------------------------------------------------------------
# TransformCategory
# ---------------------------------------------------------------------------


class TestTransformCategory:
    """TransformCategory enum — membership and string values."""

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (TransformCategory.GEOMETRIC_INTERP, "geometric_interp"),
            (TransformCategory.GEOMETRIC_EXACT, "geometric_exact"),
            (TransformCategory.POINTWISE, "pointwise"),
            (TransformCategory.SPATIAL_KERNEL, "spatial_kernel"),
        ],
    )
    def test_value(self, member, expected_value):
        assert member.value == expected_value, f"{member.name}.value should be {expected_value!r}"

    def test_exactly_four_members(self):
        assert len(TransformCategory) == 4, f"Expected 4 members, got {len(TransformCategory)}"

    def test_member_names(self):
        assert {m.name for m in TransformCategory} == {
            "GEOMETRIC_INTERP",
            "GEOMETRIC_EXACT",
            "POINTWISE",
            "SPATIAL_KERNEL",
        }

    @pytest.mark.parametrize("member", list(TransformCategory))
    def test_value_is_str(self, member):
        assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value).__name__}"


# ---------------------------------------------------------------------------
# ReorderPolicy
# ---------------------------------------------------------------------------


class TestReorderPolicy:
    """ReorderPolicy enum — membership and string values."""

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (ReorderPolicy.NONE, "none"),
            (ReorderPolicy.POINTWISE, "pointwise"),
            (ReorderPolicy.AGGRESSIVE, "aggressive"),
        ],
    )
    def test_value(self, member, expected_value):
        assert member.value == expected_value, f"{member.name}.value should be {expected_value!r}"

    def test_exactly_three_members(self):
        assert len(ReorderPolicy) == 3, f"Expected 3 members, got {len(ReorderPolicy)}"

    def test_member_names(self):
        assert {m.name for m in ReorderPolicy} == {"NONE", "POINTWISE", "AGGRESSIVE"}

    @pytest.mark.parametrize("member", list(ReorderPolicy))
    def test_value_is_str(self, member):
        assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value).__name__}"


# ---------------------------------------------------------------------------
# InterpolationMode
# ---------------------------------------------------------------------------


class TestInterpolationMode:
    """InterpolationMode IntEnum — values, ordering, and int usability."""

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (InterpolationMode.NEAREST, 0),
            (InterpolationMode.BILINEAR, 1),
            (InterpolationMode.BICUBIC, 2),
        ],
    )
    def test_int_value(self, member, expected_int):
        assert int(member) == expected_int, f"{member.name} should equal {expected_int}"

    def test_ordering_nearest_lt_bilinear_lt_bicubic(self):
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
        seq = ["a", "b", "c"]
        assert seq[member] == seq[expected_int], (
            f"Indexing with {member.name} should behave identically to indexing with {expected_int}"
        )


# ---------------------------------------------------------------------------
# PaddingMode
# ---------------------------------------------------------------------------


class TestPaddingMode:
    """PaddingMode IntEnum — values, ordering, and int usability."""

    @pytest.mark.parametrize(
        "member, expected_int",
        [
            (PaddingMode.ZEROS, 0),
            (PaddingMode.BORDER, 1),
            (PaddingMode.REFLECTION, 2),
        ],
    )
    def test_int_value(self, member, expected_int):
        assert int(member) == expected_int, f"{member.name} should equal {expected_int}"

    def test_ordering_zeros_lt_border_lt_reflection(self):
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
        seq = ["a", "b", "c"]
        assert seq[member] == seq[expected_int], (
            f"Indexing with {member.name} should behave identically to indexing with {expected_int}"
        )


# ---------------------------------------------------------------------------
# TransformAdapter Protocol
# ---------------------------------------------------------------------------


class TestTransformAdapterProtocol:
    """TransformAdapter @runtime_checkable Protocol — isinstance contract."""

    def test_conforming_class_passes_isinstance(self):
        """A class implementing all four methods satisfies TransformAdapter."""
        methods = _full_adapter_methods()
        _DummyAdapter = type("_DummyAdapter", (), methods)
        assert isinstance(_DummyAdapter(), TransformAdapter), (
            "An object with all four required methods should be an instance of TransformAdapter"
        )

    @pytest.mark.parametrize("missing_method", ["category", "sample_params", "build_matrix", "call_nonfused"])
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
        assert not isinstance(non_adapter, TransformAdapter), (
            f"{non_adapter!r} should not satisfy the TransformAdapter protocol"
        )
