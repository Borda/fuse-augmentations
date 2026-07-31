"""Public-API surface tests for the datasets subpackage."""

from __future__ import annotations


def test_top_level_facade_is_accessible():
    """`fuse_augmentations.generate_dataset` resolves via the lazy module getattr."""
    import fuse_augmentations

    assert callable(fuse_augmentations.generate_dataset)


def test_subpackage_exports_public_names():
    """The datasets subpackage re-exports the documented public API."""
    from fuse_augmentations.data import (
        CocoWriter,
        SyntheticConfig,
        SyntheticGenerator,
        YoloWriter,
        generate_dataset,
        get_writer,
    )

    assert all(callable(obj) for obj in (generate_dataset, get_writer, SyntheticGenerator, SyntheticConfig))
    assert CocoWriter.__name__ == "CocoWriter"
    assert YoloWriter.__name__ == "YoloWriter"
