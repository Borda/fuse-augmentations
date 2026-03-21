"""Smoke tests for ExactAffineSegment public availability."""

from __future__ import annotations


class TestExactAffineSegmentImport:
    """ExactAffineSegment must be importable from the top-level package."""

    def test_exact_affine_segment_importable(self):
        import fuse_augmentations

        assert hasattr(fuse_augmentations, "ExactAffineSegment"), (
            "fuse_augmentations.ExactAffineSegment must remain importable from the top-level package."
        )
        assert isinstance(fuse_augmentations.ExactAffineSegment, type), "ExactAffineSegment exists but is not a class"

    def test_exact_affine_segment_in_all(self):
        import fuse_augmentations

        assert "ExactAffineSegment" in fuse_augmentations.__all__, (
            f"'ExactAffineSegment' missing from __all__: {fuse_augmentations.__all__}"
        )
