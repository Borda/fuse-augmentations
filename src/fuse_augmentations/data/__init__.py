"""Backward-compatible alias for :mod:`synth_datasets`.

The synthetic detection / segmentation / OBB / keypoint dataset generator used to live here. It has
moved to the standalone top-level package :mod:`synth_datasets`, which installs and imports without
the ``torch`` extra. This module now only re-exports that package's public API so existing
``fuse_augmentations.data`` imports keep working.

Prefer ``import synth_datasets`` for new code, especially when only dataset generation is needed:
importing ``fuse_augmentations.data`` still runs the parent :mod:`fuse_augmentations` package's
``__init__``, which eagerly imports the torch-dependent augmentation stack and therefore requires
the ``torch`` extra (``pip install "fuse-augmentations[torch]"``). ``import synth_datasets`` never
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
from synth_datasets import __all__ as __all__  # keep this module's __all__ in lockstep, never hand-duplicated
