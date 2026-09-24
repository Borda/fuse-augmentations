"""Smoke tests for ExactAffineSegment public availability."""

from __future__ import annotations


class TestExactAffineSegmentImport:
    """ExactAffineSegment must be importable from the top-level package."""

    def test_exact_affine_segment_importable(self):
        """ExactAffineSegment is importable from the top-level fused_transforms package and is a class."""
        import fused_transforms

        assert hasattr(fused_transforms, "ExactAffineSegment"), (
            "fused_transforms.ExactAffineSegment must remain importable from the top-level package."
        )
        assert isinstance(fused_transforms.ExactAffineSegment, type), "ExactAffineSegment exists but is not a class"

    def test_exact_affine_segment_in_all(self):
        """ExactAffineSegment is listed in fused_transforms.__all__ as part of the public API surface."""
        import fused_transforms

        assert "ExactAffineSegment" in fused_transforms.__all__, (
            f"'ExactAffineSegment' missing from __all__: {fused_transforms.__all__}"
        )
