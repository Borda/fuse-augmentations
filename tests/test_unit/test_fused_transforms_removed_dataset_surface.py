"""Pin the failure mode for the removed `fused_transforms.data` package and `generate_dataset` alias.

Both were migration affordances left behind when the synthetic-dataset generator moved to the standalone
`synth_datasets` package, and both are gone as of the rebranding release: `src/fused_transforms/data/` no longer exists,
and the module-level `__getattr__` that lazily resolved `generate_dataset` was deleted with it. Callers import from
`synth_datasets` directly.

These tests pin the errors a caller on the old path actually hits, so the removal can't silently regrow. The risk is a
well-meaning follow-up reintroducing a `__getattr__` on the package for convenience and turning a loud
`ModuleNotFoundError`/`AttributeError` into a half-resolved alias or a silent `None`.

"""

from __future__ import annotations

import pytest


def test_importing_the_removed_data_package_raises_module_not_found_error() -> None:
    """`import fused_transforms.data` raises `ModuleNotFoundError`.

    Before the move, `fused_transforms.data` was the generator's home; after the move it survived for one release as a
    star re-export of `synth_datasets`. Now there is no such subpackage at all, so Python's own submodule resolution is
    what raises — no bespoke error handling in this codebase is involved.

    """
    with pytest.raises(ModuleNotFoundError, match=r"fused_transforms\.data"):
        import fused_transforms.data  # noqa: F401 - import is the act under test


def test_importing_a_removed_legacy_submodule_raises_module_not_found_error() -> None:
    """`import fused_transforms.data.geometry` raises `ModuleNotFoundError`.

    The deepest form of the old path. It failed while the one-file `data` facade still existed, because that facade re-
    exported only top-level names and had no submodule files; it fails now for the plainer reason that its parent
    package is gone. Pinned separately from the parent case so a reintroduced `data` package cannot quietly make the
    submodule path resolve again.

    """
    with pytest.raises(ModuleNotFoundError, match=r"fused_transforms\.data"):
        import fused_transforms.data.geometry  # noqa: F401 - import is the act under test


def test_generate_dataset_is_no_longer_a_top_level_attribute() -> None:
    """`fused_transforms.generate_dataset` raises `AttributeError` and is absent from `__all__`.

    The alias reached `synth_datasets.generate_dataset` through a module-level `__getattr__`, so its removal is only
    observable through attribute access — a stale `__all__` entry would otherwise keep advertising a name that no
    longer resolves, and `from fused_transforms import *` would fail at import time rather than here.

    """
    import fused_transforms

    assert "generate_dataset" not in fused_transforms.__all__
    with pytest.raises(AttributeError, match="generate_dataset"):
        _ = fused_transforms.generate_dataset
