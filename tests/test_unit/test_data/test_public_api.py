"""Public-API surface tests for the datasets subpackage."""

from __future__ import annotations


def test_top_level_facade_is_accessible() -> None:
    """`fuse_augmentations.generate_dataset` resolves via the lazy module getattr."""
    import fuse_augmentations

    assert callable(fuse_augmentations.generate_dataset)


def test_subpackage_exports_public_names() -> None:
    """The datasets subpackage re-exports the documented public API.

    `animal_keypoints` and `DEFAULT_SHAPES` are part of that surface — both are documented public names (the first
    carries its own `Examples:` block, the second is the default of `SyntheticConfig.shapes`), so reaching them must not
    require importing a submodule the facade otherwise hides.

    """
    from fuse_augmentations.data import (
        DEFAULT_SHAPES,
        CocoWriter,
        SyntheticConfig,
        SyntheticGenerator,
        YoloWriter,
        animal_keypoints,
        generate_dataset,
        get_writer,
    )

    assert all(
        callable(obj) for obj in (generate_dataset, get_writer, SyntheticGenerator, SyntheticConfig, animal_keypoints)
    )
    assert CocoWriter.__name__ == "CocoWriter"
    assert YoloWriter.__name__ == "YoloWriter"
    assert isinstance(DEFAULT_SHAPES, tuple)
    assert DEFAULT_SHAPES


def test_all_exported_names_resolve() -> None:
    """Every name in `__all__` is reachable as an attribute of the subpackage.

    `__all__` is hand-maintained alongside the imports that populate it; a name added to one but not the other would
    still import cleanly and only fail for a caller doing `from fuse_augmentations.data import *` or introspecting the
    module, so this checks the two stay in sync directly.

    """
    import fuse_augmentations.data as data

    missing = [name for name in data.__all__ if not hasattr(data, name)]
    assert missing == []
