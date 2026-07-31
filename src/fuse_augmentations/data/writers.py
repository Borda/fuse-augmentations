"""Serialize generated samples to COCO or YOLO dataset layouts.

Both writers consume the same format-agnostic
:class:`~fuse_augmentations.data.sample.Sample` objects and select which
annotation fields to emit based on the requested
:class:`~fuse_augmentations.data.config.Task`.

COCO layout (Roboflow-style)::

    <out>/<split>/img_000000.jpg
    <out>/<split>/_annotations.coco.json

YOLO layout (Ultralytics-style)::

    <out>/images/<split>/img_000000.jpg
    <out>/labels/<split>/img_000000.txt
    <out>/data.yaml

COCO has no native oriented-box field, so for :attr:`Task.OBB` the four corners are
stored as a 4-point ``segmentation`` polygon alongside the axis-aligned ``bbox``.

"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from fuse_augmentations.data.config import OutputFormat, Task

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from fuse_augmentations.data.sample import Annotation, Sample

_IMAGE_STEM = "img_{index:06d}"


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive ``[lo, hi]`` range."""
    return max(lo, min(hi, value))


def _clamp_flat(flat: list[float], img_w: float, img_h: float) -> list[float]:
    """Clamp a flat ``[x1, y1, ...]`` coordinate list to the image extent."""
    return [_clamp(v, 0.0, img_w) if i % 2 == 0 else _clamp(v, 0.0, img_h) for i, v in enumerate(flat)]


def _save_image(image: NDArray[Any], path: Path) -> None:
    """Write an RGB ``uint8`` array to ``path`` as JPEG."""
    Image.fromarray(image).save(path, quality=95)


class DatasetWriter(ABC):
    """Base class for dataset serializers.

    Args:
        task: The annotation task determining which fields are emitted.
        class_names: Ordered class vocabulary; list index is the class id.

    """

    def __init__(self, task: Task, class_names: list[str]) -> None:
        """Store the task and class vocabulary."""
        self.task = task
        self.class_names = class_names

    @abstractmethod
    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write all splits under ``output_dir``.

        Args:
            splits: Mapping of split name to its samples.
            output_dir: Destination root directory (created if absent).

        """


class CocoWriter(DatasetWriter):
    """Write a COCO-format dataset, one JSON per split.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.writers import CocoWriter
        >>> from fuse_augmentations.data.config import Task
        >>> CocoWriter(Task.DETECTION, ["square"]).task.value
        'detection'

        ```

    """

    def _annotation_dict(self, ann: Annotation, ann_id: int, image_id: int, img_w: int, img_h: int) -> dict[str, Any]:
        """Build one COCO annotation record, clamping geometry to the image extent."""
        x1 = _clamp(ann.bbox_xyxy[0], 0, img_w)
        y1 = _clamp(ann.bbox_xyxy[1], 0, img_h)
        x2 = _clamp(ann.bbox_xyxy[2], 0, img_w)
        y2 = _clamp(ann.bbox_xyxy[3], 0, img_h)
        width, height = x2 - x1, y2 - y1
        record = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann.class_id + 1,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        if self.task is Task.SEGMENTATION:
            record["segmentation"] = [_clamp_flat(ann.polygon, img_w, img_h)]
        elif self.task is Task.OBB:
            record["segmentation"] = [_clamp_flat(ann.obb_corners, img_w, img_h)]
        return record

    def _coco_doc(self, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap image and annotation records into a COCO document with categories."""
        categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(self.class_names)]
        return {
            "info": {"description": "fuse-augmentations synthetic dataset"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Stream each split to ``<output_dir>/<split>/`` in a single pass over its samples.

        Images are written as they are produced; only lightweight COCO metadata (no pixels) accumulates in memory, so
        arbitrarily large datasets stay bounded.

        """
        output_dir = Path(output_dir)
        for split, samples in splits.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            ann_id = 1
            for image_id, sample in enumerate(samples):
                stem = _IMAGE_STEM.format(index=image_id)
                _save_image(sample.image, split_dir / f"{stem}.jpg")
                images.append({
                    "id": image_id,
                    "file_name": f"{stem}.jpg",
                    "width": sample.width,
                    "height": sample.height,
                })
                for ann in sample.annotations:
                    annotations.append(self._annotation_dict(ann, ann_id, image_id, sample.width, sample.height))
                    ann_id += 1
            doc = self._coco_doc(images, annotations)
            (split_dir / "_annotations.coco.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


class YoloWriter(DatasetWriter):
    """Write a YOLO-format dataset with normalized labels and a ``data.yaml``.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.writers import YoloWriter
        >>> from fuse_augmentations.data.config import Task
        >>> YoloWriter(Task.OBB, ["square"]).task.value
        'oriented_bounding_boxes'

        ```

    """

    def _label_row(self, ann: Annotation, width: int, height: int) -> str:
        """Format one YOLO label row for the writer's task, clamping to ``[0, 1]``."""
        if self.task is Task.DETECTION:
            x1, y1, x2, y2 = ann.bbox_xyxy
            coords = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
        else:
            flat = ann.polygon if self.task is Task.SEGMENTATION else ann.obb_corners
            coords = [v / width if i % 2 == 0 else v / height for i, v in enumerate(flat)]
        coords = [_clamp(c, 0.0, 1.0) for c in coords]
        return " ".join([str(ann.class_id), *(f"{c:.6f}" for c in coords)])

    def _write_split(self, split: str, samples: Iterable[Sample], output_dir: Path) -> None:
        """Write images and label files for one split."""
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            stem = _IMAGE_STEM.format(index=index)
            _save_image(sample.image, img_dir / f"{stem}.jpg")
            rows = [self._label_row(ann, sample.width, sample.height) for ann in sample.annotations]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _data_yaml(self, splits: dict[str, Iterable[Sample]]) -> str:
        """Build the ``data.yaml`` contents referencing present splits."""
        lines = ["path: .", f"train: images/{'train' if 'train' in splits else next(iter(splits))}"]
        if "val" in splits:
            lines.append("val: images/val")
        if "test" in splits:
            lines.append("test: images/test")
        lines.append(f"nc: {len(self.class_names)}")
        lines.append("names:")
        lines.extend(f"  {i}: {name}" for i, name in enumerate(self.class_names))
        return "\n".join(lines) + "\n"

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write images, labels, and ``data.yaml`` under ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, samples in splits.items():
            self._write_split(split, samples, output_dir)
        (output_dir / "data.yaml").write_text(self._data_yaml(splits), encoding="utf-8")


def get_writer(fmt: OutputFormat, task: Task, class_names: list[str]) -> DatasetWriter:
    """Return the writer for an output format.

    Args:
        fmt: Target :class:`~fuse_augmentations.data.config.OutputFormat`.
        task: Annotation task to emit.
        class_names: Ordered class vocabulary.

    Returns:
        A concrete :class:`DatasetWriter`.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.config import OutputFormat, Task
        >>> from fuse_augmentations.data.writers import get_writer
        >>> type(get_writer(OutputFormat.YOLO, Task.DETECTION, ["square"])).__name__
        'YoloWriter'

        ```

    """
    if fmt is OutputFormat.COCO:
        return CocoWriter(task, class_names)
    return YoloWriter(task, class_names)
