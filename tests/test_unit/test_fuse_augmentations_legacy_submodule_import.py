"""Pin the failure mode for a removed `fuse_augmentations.data.<submodule>` legacy import path.

The author accepted this as a plain break with no deprecation shim (see the F2 review finding): the synthetic-dataset
generator moved to the standalone `synth_datasets` package, and `fuse_augmentations/data/__init__.py`'s docstring states
it "does not preserve former submodule paths such as `fuse_augmentations.data.geometry`" without saying what a caller
who tries anyway actually hits. This pins the actual, intended contract — `ModuleNotFoundError`, not an `AttributeError`
from a half-resolved lazy alias or a silent partial import — since `src/fuse_augmentations/data/` holds only
`__init__.py`, has no submodule files, and defines no `__getattr__` for submodule names.

"""

from __future__ import annotations

import pytest


def test_importing_a_removed_legacy_submodule_raises_module_not_found_error() -> None:
    """`import fuse_augmentations.data.geometry` raises `ModuleNotFoundError`, the accepted F2 contract.

    Before the move, `fuse_augmentations.data.geometry` was a real submodule; today the whole `data` package is one file
    that only re-exports `synth_datasets`'s top-level names via `import *`, so Python's own submodule-resolution
    machinery is what raises here, not any bespoke error handling in this codebase. Pinning it guards against a well-
    meaning follow-up (e.g. a `__getattr__` added to `fuse_augmentations.data` for the top-level names) quietly starting
    to swallow submodule lookups too and turning this into an `AttributeError` or a silent `None` instead.

    """
    with pytest.raises(ModuleNotFoundError, match=r"fuse_augmentations\.data\.geometry"):
        import fuse_augmentations.data.geometry  # noqa: F401 - import is the act under test
