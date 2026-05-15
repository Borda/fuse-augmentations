"""Smoke tests for ExactAffineSegment public availability."""

from __future__ import annotations


class TestExactAffineSegmentImport:
    """ExactAffineSegment must be importable from the top-level package."""

    def test_exact_affine_segment_importable(self):
        """ExactAffineSegment is importable from the top-level fuse_augmentations package and is a class."""
        import fuse_augmentations

        assert hasattr(fuse_augmentations, "ExactAffineSegment"), (
            "fuse_augmentations.ExactAffineSegment must remain importable from the top-level package."
        )
        assert isinstance(fuse_augmentations.ExactAffineSegment, type), "ExactAffineSegment exists but is not a class"

    def test_exact_affine_segment_in_all(self):
        """ExactAffineSegment is listed in fuse_augmentations.__all__ as part of the public API surface."""
        import fuse_augmentations

        assert "ExactAffineSegment" in fuse_augmentations.__all__, (
            f"'ExactAffineSegment' missing from __all__: {fuse_augmentations.__all__}"
        )
