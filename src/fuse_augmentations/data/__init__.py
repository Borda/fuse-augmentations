"""Synthetic detection / segmentation / OBB / keypoint dataset generation.

Draw colored shapes on a canvas and export **COCO** or **YOLO** datasets for
detection, segmentation, oriented-bounding-box, or keypoint tasks. This is a standalone
generation utility: no dataset loaders, no model, no training loop.

Examples:
    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from fuse_augmentations.data import generate_dataset
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     counts = generate_dataset(tmp, num_images=10, fmt="yolo", task="detection", seed=0)
    ...     sorted(counts)
    ['test', 'train', 'val']

    ```

"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from fuse_augmentations.data.animals import (
    ANIMAL_KEYPOINT_NAMES,
    ANIMAL_KEYPOINT_SKELETON,
    AnimalShape,
    animal_keypoints,
    animal_shapes,
)
from fuse_augmentations.data.config import (
    DEFAULT_SHAPES,
    ClassMode,
    Color,
    OutputFormat,
    Shape,
    SplitRatios,
    SyntheticConfig,
    Task,
    class_id_of,
    class_names,
)
from fuse_augmentations.data.datasets import SyntheticIterableDataset
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.geometry import GeomShape
from fuse_augmentations.data.sample import Annotation, Sample
from fuse_augmentations.data.writers import CocoWriter, DatasetWriter, YoloWriter, get_writer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "ANIMAL_KEYPOINT_NAMES",
    "ANIMAL_KEYPOINT_SKELETON",
    "DEFAULT_SHAPES",
    "AnimalShape",
    "Annotation",
    "ClassMode",
    "CocoWriter",
    "Color",
    "GeomShape",
    "OutputFormat",
    "Sample",
    "Shape",
    "SplitRatios",
    "SyntheticConfig",
    "SyntheticGenerator",
    "SyntheticIterableDataset",
    "Task",
    "YoloWriter",
    "animal_keypoints",
    "animal_shapes",
    "class_id_of",
    "class_names",
    "generate_dataset",
    "get_writer",
]


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
    task: Task | str | None = None,
    class_mode: ClassMode | str = ClassMode.SHAPE,
    split_ratios: SplitRatios | None = None,
    seed: int | None = None,
    config: SyntheticConfig | None = None,
    **config_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
) -> dict[str, int]:
    """Generate a synthetic dataset on disk and return per-split image counts.

    Args:
        output_dir: Destination directory (created if absent).
        num_images: Total number of images to generate across all splits.
        fmt: Output layout, ``"coco"`` or ``"yolo"`` (or an :class:`OutputFormat`).
        task: ``"detection"``, ``"segmentation"``, ``"obb"``, or ``"keypoints"`` (or a
            :class:`Task`). ``None`` (the default) adopts :attr:`SyntheticConfig.task` — the
            supplied ``config``'s own task, or the config default when no ``config`` is given — so
            a keypoints config never has to restate its task here. ``"keypoints"`` additionally
            requires every entry of :attr:`SyntheticConfig.shapes` to be an :class:`AnimalShape` —
            the default geometric shapes have no keypoint table, and pairing them with it raises
            :class:`ValueError`.
        class_mode: ``"shape"``, ``"color"``, or ``"shape_color"`` (or a :class:`ClassMode`).
        split_ratios: Train/val/test fractions; defaults to 70/20/10.
        seed: Seed for reproducible generation; ``None`` uses fresh entropy.
        config: Full :class:`SyntheticConfig`; when given, ``class_mode`` and ``config_kwargs``
            are ignored in favor of the config's own fields. ``task`` is **not** ignored: an
            explicitly supplied one is cross-checked against :attr:`SyntheticConfig.task` and must
            agree with it, because the generator reads the task off the config while the writer
            reads the argument. Omitting ``task`` adopts the config's own task instead.
        **config_kwargs: Extra :class:`SyntheticConfig` fields (e.g. ``img_size``)
            used only when ``config`` is not supplied.

    Returns:
        Ordered mapping of split name to the number of images written.

    Raises:
        ValueError: If ``num_images`` is not a positive integer, or if an explicitly supplied
            ``task`` disagrees with the :attr:`SyntheticConfig.task` of a supplied ``config``.
            An omitted ``task`` can never conflict, since it takes its value from the config.

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
    fmt = OutputFormat(fmt)
    # ``None`` is the "adopt whatever the config says" sentinel and must survive normalization:
    # ``Task(None)`` raises, and folding it into a concrete default here would resurrect the very
    # clash the sentinel removes -- a keypoints config plus an omitted argument reported as a
    # conflict against a default the caller never passed.
    task = None if task is None else Task(task)
    split_ratios = split_ratios or SplitRatios()
    if config is None:
        # ``class_mode`` feeds only the config built here; when a ``config`` is supplied it is
        # ignored, so normalize (and validate) it lazily to avoid rejecting a valid ``config`` call.
        # ``task`` cannot be treated that way: the generator reads it off the config and the writer
        # off the argument, so it is forwarded here -- one task, two readers. An omitted task is
        # left out of the call rather than restated, keeping SyntheticConfig the only place that
        # names the default.
        mode = ClassMode(class_mode)
        config = (
            SyntheticConfig(class_mode=mode, **config_kwargs)
            if task is None
            else SyntheticConfig(class_mode=mode, task=task, **config_kwargs)
        )
    elif task is not None and config.task is not task:
        # Both sides are normalized ``Task`` members by now, so identity is the exact test. Left
        # unchecked, a mismatch splits the two readers: e.g. a KEYPOINTS writer over a DETECTION
        # config emits a landmark block that is all-zero and visibility-0 for every object.
        raise ValueError(
            f"task={task.value!r} conflicts with config.task={config.task.value!r}; "
            "pass the task on the config, or omit the task argument"
        )
    # Every path above leaves the config carrying the task, so reading it back is what keeps the
    # writer below on the same member the generator uses -- including when the argument was omitted.
    task = config.task

    counts = _assign_splits(num_images, split_ratios)

    # Stream a single lazy sample source through per-split islice views so only one
    # Sample is materialized at a time; the writer consumes each split in order.
    generator = SyntheticGenerator(config)
    sample_stream = generator.generate(num_images, seed=seed)
    splits: dict[str, Iterable[Sample]] = {
        split: itertools.islice(sample_stream, count) for split, count in counts.items()
    }

    writer: DatasetWriter = get_writer(fmt, task, class_names(config.class_mode))
    writer.write(splits, output_dir)
    return counts
