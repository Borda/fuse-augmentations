"""Package-level re-export facade for :mod:`synth_datasets`.

The synthetic detection / segmentation / OBB / keypoint dataset generator used to live here. It has
moved to the standalone top-level package :mod:`synth_datasets`, which installs and imports without
the ``torch`` extra. This module re-exports the package-level public API for migration, but it does
not preserve former submodule paths such as ``fuse_augmentations.data.geometry``; import those from
``synth_datasets`` instead.

Prefer ``import synth_datasets`` for new code, especially when only dataset generation is needed:
importing ``fuse_augmentations.data`` still runs the parent :mod:`fuse_augmentations` package's
``__init__``, which eagerly imports the torch-dependent augmentation stack and therefore requires
the ``torch`` extra (``pip install "vision-synth[torch]"``). ``import synth_datasets`` never
touches that package and needs no such extra.

Examples:
    ```pycon
    >>> import tempfile
    >>> from fuse_augmentations.data import generate_dataset
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     counts = generate_dataset(tmp, num_images=10, fmt="yolo", task="detection", seed=0)
    ...     sorted(counts)
    ['test', 'train', 'val']

    ```

"""

from __future__ import annotations

from synth_datasets import *  # noqa: F403 - re-exports the identical public surface, see module docstring
from synth_datasets import SyntheticIterableDataset as SyntheticIterableDataset  # synth_datasets drops this name
from synth_datasets import __all__ as __all__  # base surface, extended below with the one legacy name above

# `SyntheticIterableDataset` is kept resolvable via synth_datasets' lazy `__getattr__` but left out of its
# `__all__` so `from synth_datasets import *` stays torch-free; this shim already requires torch to import
# at all, so restoring the legacy name here is free. All elements below are strings at runtime -- ruff's
# static check just can't see through the splat of an imported `__all__` list.
__all__ = [*__all__, "SyntheticIterableDataset"]  # noqa: PLE0604
