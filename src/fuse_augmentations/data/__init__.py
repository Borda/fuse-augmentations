"""Synthetic detection / segmentation / OBB / keypoint dataset generation.

Draw colored shapes on a canvas and export **COCO** or **YOLO** datasets for detection,
segmentation, oriented-bounding-box, or keypoint tasks. This is a standalone generation utility: no
dataset loaders, no model, no training loop.

The shape vocabulary is assembled from independent families —
:mod:`~fuse_augmentations.data.primitives` (analytic), :mod:`~fuse_augmentations.data.animals`
(traced silhouettes), :mod:`~fuse_augmentations.data.symbols`, and
:mod:`~fuse_augmentations.data.letters` (stroke figures) — registered in
:mod:`~fuse_augmentations.data.families`. Reach for a family's own module when you want its
specifics; this namespace exports the pieces a dataset-building caller needs.

This module does not *itself* import torch: :class:`SyntheticIterableDataset` is the only
torch-dependent name here and is resolved lazily on first attribute access. Note that importing it
still pulls torch in today, because the parent :mod:`fuse_augmentations` package imports the
augmentation stack eagerly — roughly 440 ms of the ~480 ms an ``import fuse_augmentations.data``
costs. Making the saving visible needs the same treatment there.

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

import importlib
import itertools
from typing import TYPE_CHECKING, Any

from fuse_augmentations.data.config import (
    ClassEntry,
    ClassMode,
    ClassVocabulary,
    Color,
    Fill,
    OutputFormat,
    SplitRatios,
    SyntheticConfig,
    Task,
    class_id,
    class_names,
    class_vocabulary,
)
from fuse_augmentations.data.families import (
    ALL_SHAPES,
    DEFAULT_SHAPES,
    SHAPE_FAMILIES,
    Shape,
    ShapeFamily,
    family_of,
    keypoint_schema_for,
    shape_outline,
)
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.keypoints import KeypointSchema
from fuse_augmentations.data.sample import Annotation, Sample
from fuse_augmentations.data.writers import CocoWriter, DatasetWriter, YoloWriter, get_writer, register_writer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fuse_augmentations.data.datasets import SyntheticIterableDataset

__all__ = [
    "ALL_SHAPES",
    "DEFAULT_SHAPES",
    "SHAPE_FAMILIES",
    "Annotation",
    "ClassEntry",
    "ClassMode",
    "ClassVocabulary",
    "CocoWriter",
    "Color",
    "DatasetWriter",
    "Fill",
    "KeypointSchema",
    "OutputFormat",
    "Sample",
    "Shape",
    "ShapeFamily",
    "SplitRatios",
    "SyntheticConfig",
    "SyntheticGenerator",
    "SyntheticIterableDataset",
    "Task",
    "YoloWriter",
    "class_id",
    "class_names",
    "class_vocabulary",
    "family_of",
    "generate_dataset",
    "get_writer",
    "keypoint_schema_for",
    "register_writer",
    "shape_outline",
]

#: Names resolved on first access rather than at import. :class:`SyntheticIterableDataset` is here
#: to keep ``import fuse_augmentations.data`` free of torch — measured at ~440 ms of the ~480 ms this
#: package used to cost, imposed on every caller including the many who only write a dataset to disk.
_LAZY: dict[str, str] = {"SyntheticIterableDataset": "fuse_augmentations.data.datasets"}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module-level attribute access is untyped by nature
    """Resolve a lazily-imported public name on first access (:pep:`562`).

    Args:
        name: The attribute being looked up on this module.

    Returns:
        The resolved object. Names in :data:`_LAZY` are exported here like any other; they are
        simply imported late.

    Raises:
        AttributeError: If ``name`` is neither exported nor deferred.

    """
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def _assign_splits(num_images: int, split_ratios: SplitRatios) -> dict[str, int]:
    """Return per-split image counts summing exactly to ``num_images``.

    Args:
        num_images: Total images to distribute.
        split_ratios: Fractions per split.

    Returns:
        Ordered ``split -> count`` mapping whose values sum to ``num_images``.

    """
    ratios = split_ratios.to_dict()
    counts = {name: round(frac * num_images) for name, frac in ratios.items()}
    first = next(iter(counts))
    counts[first] += num_images - sum(counts.values())
    return {name: count for name, count in counts.items() if count > 0}


def generate_dataset(
    output_dir: str | Path,
    num_images: int,
    fmt: OutputFormat | str = OutputFormat.COCO,
    split_ratios: SplitRatios | None = None,
    seed: int | None = None,
    config: SyntheticConfig | None = None,
    **config_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
) -> dict[str, int]:
    """Generate a synthetic dataset on disk and return per-split image counts.

    Everything about an image's *content* — task, class mode, shapes, colors, size — is a
    :class:`SyntheticConfig` field, reachable either through ``config`` or through ``config_kwargs``.
    Only ``fmt`` and ``split_ratios``, which describe the on-disk layout rather than the pixels, are
    parameters here.

    ``task`` and ``class_mode`` used to be parameters *as well as* config fields. The task in
    particular had two owners — the generator read it off the config, the writer off the argument —
    so passing both meant a cross-check, a ``None`` sentinel to distinguish "not supplied" from a
    default, and a paragraph of documentation about which won. Giving the config sole ownership
    deleted all three; ``task="keypoints"`` still works, it simply arrives as a config field.

    Args:
        output_dir: Destination directory (created if absent).
        num_images: Total number of images to generate across all splits.
        fmt: Output layout, ``"coco"`` or ``"yolo"`` (or an :class:`OutputFormat`).
        split_ratios: Train/val/test fractions; defaults to 70/20/10.
        seed: Seed for reproducible generation; ``None`` uses fresh entropy.
        config: Full :class:`SyntheticConfig`. When given, ``config_kwargs`` must be empty — a
            config already says everything they would.
        **config_kwargs: :class:`SyntheticConfig` fields (``task``, ``class_mode``, ``img_size``,
            ``shapes``, ``colors``, …) used to build the config when ``config`` is not supplied.

    Returns:
        Ordered mapping of split name to the number of images written.

    Raises:
        ValueError: If ``num_images`` is not a positive integer, or if both a ``config`` and
            ``config_kwargs`` were supplied.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from fuse_augmentations.data import generate_dataset
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     generate_dataset(tmp, num_images=10, fmt="coco", task="segmentation",
        ...                      img_size=64, seed=1)
        {'train': 7, 'val': 2, 'test': 1}

        ```

    """
    if num_images < 1:
        raise ValueError(f"num_images must be a positive integer, got {num_images}")
    if config is not None and config_kwargs:
        raise ValueError(
            f"pass either a config or its fields as keywords, not both; got config plus {sorted(config_kwargs)}"
        )
    fmt = OutputFormat(fmt)
    split_ratios = split_ratios or SplitRatios()
    config = config if config is not None else SyntheticConfig(**config_kwargs)

    counts = _assign_splits(num_images, split_ratios)

    # Stream a single lazy sample source through per-split islice views so only one Sample is
    # materialized at a time. The writer must consume the splits in insertion order and exactly once
    # each; DatasetWriter.write documents that contract, since these views share one iterator.
    generator = SyntheticGenerator(config)
    sample_stream = generator.generate(num_images, seed=seed)
    splits: dict[str, Iterable[Sample]] = {
        split: itertools.islice(sample_stream, count) for split, count in counts.items()
    }

    # ``SyntheticConfig`` already guarantees a single keypoint-bearing family under Task.KEYPOINTS,
    # so this is None only for tasks that never read it.
    writer: DatasetWriter = get_writer(
        fmt,
        config.task,
        class_vocabulary(config.class_mode, config.shapes, config.colors),
        keypoint_schema=keypoint_schema_for(config.shapes),
    )
    writer.write(splits, output_dir)
    return counts
