"""Public-API surface tests for the datasets subpackage."""

from __future__ import annotations


def test_top_level_facade_is_accessible() -> None:
    """`fuse_augmentations.generate_dataset` resolves via the lazy module getattr."""
    import fuse_augmentations

    assert callable(fuse_augmentations.generate_dataset)


def test_subpackage_exports_public_names() -> None:
    """The datasets subpackage re-exports the documented public API.

    The facade carries what a dataset-building caller needs — the generator, the config, the writers, the vocabulary
    helpers — and no longer carries five names per shape family. Family specifics like `animal_keypoints` now come from
    the family's own module, which is what stops this surface growing every time a family is added.

    """
    from fuse_augmentations.data import (
        ALL_SHAPES,
        DEFAULT_SHAPES,
        SHAPE_FAMILIES,
        CocoWriter,
        SyntheticConfig,
        SyntheticGenerator,
        YoloWriter,
        generate_dataset,
        get_writer,
        shape_outline,
    )
    from fuse_augmentations.data.animals import animal_keypoints

    assert all(
        callable(obj)
        for obj in (generate_dataset, get_writer, SyntheticGenerator, SyntheticConfig, shape_outline, animal_keypoints)
    )
    assert len(ALL_SHAPES) == sum(len(family.members) for family in SHAPE_FAMILIES)
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


def test_every_registered_family_derives_from_the_shape_base() -> None:
    """`Shape` is the shared base class, so the registry is the only place a family is named.

    `Shape` used to be a hand-written `PrimitiveShape | AnimalShape | SymbolShape | LetterShape` union that had to be
    extended alongside `SHAPE_FAMILIES`. Missing that second edit left the new family drawable but invisible to
    `SyntheticConfig`'s validation, which rejects anything failing `isinstance(value, Shape)` — so the family's own
    members would have been refused as if they were bare strings. Deriving every family enum from one base makes the two
    impossible to desynchronize, and this asserts the property the union used to provide by hand.

    """
    from fuse_augmentations.data import ALL_SHAPES, SHAPE_FAMILIES, Shape

    assert all(issubclass(family.member_type, Shape) for family in SHAPE_FAMILIES)
    assert all(isinstance(shape, Shape) for shape in ALL_SHAPES)


def test_the_shape_base_still_rejects_a_bare_string() -> None:
    """A shape *value* is not a `Shape`, even though the str mixin makes it compare equal to one.

    This is the check `SyntheticConfig._validate_vocabulary` relies on to catch `shapes=("duck",)`. Under the `str`
    mixin `"duck" == AnimalShape.DUCK` and both hash alike, so a membership or equality test would let the string
    through and it would only fail much later, in a registry lookup keyed by `type(shape)`.

    """
    from fuse_augmentations.data import Shape
    from fuse_augmentations.data.animals import AnimalShape

    assert AnimalShape.DUCK == "duck"
    assert not isinstance("duck", Shape)
