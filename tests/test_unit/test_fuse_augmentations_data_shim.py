"""Back-compat coverage for the ``fuse_augmentations.data`` legacy alias.

The synthetic-dataset generator lives in :mod:`synth_datasets` now (see ``tests/test_unit/test_synth_datasets/``, which
is torch-free and exercises the generator directly). This module only checks that the ``fuse_augmentations.data`` re-
export shim still works -- it necessarily imports ``fuse_augmentations``, which requires the ``torch`` extra, so it is
not part of the torch-free test selection.

"""

from __future__ import annotations


def test_top_level_facade_is_accessible() -> None:
    """`fuse_augmentations.generate_dataset` resolves via the lazy module getattr."""
    import fuse_augmentations

    assert callable(fuse_augmentations.generate_dataset)


def test_legacy_data_module_reexports_synth_datasets() -> None:
    """`fuse_augmentations.data` re-exports the exact same objects as `synth_datasets`."""
    import fuse_augmentations.data as legacy
    import synth_datasets

    assert legacy.__all__ == synth_datasets.__all__
    assert legacy.generate_dataset is synth_datasets.generate_dataset
    assert legacy.SyntheticGenerator is synth_datasets.SyntheticGenerator


def test_all_exported_names_resolve() -> None:
    """Every name in `__all__` is reachable as an attribute of the legacy alias module.

    `__all__` is hand-maintained on the `synth_datasets` side alongside the imports that populate it;
    a name added to one but not the other would still import cleanly and only fail for a caller doing
    `from fuse_augmentations.data import *` or introspecting the module, so this checks the two stay
    in sync directly.

    """
    import fuse_augmentations.data as data

    missing = [name for name in data.__all__ if not hasattr(data, name)]
    assert missing == []
