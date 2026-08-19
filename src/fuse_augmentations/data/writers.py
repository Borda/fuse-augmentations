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

For :attr:`Task.KEYPOINTS` both writers emit the landmark block in addition to the box: COCO gains
per-category ``keypoints``/``skeleton`` plus per-annotation ``keypoints``/``num_keypoints``, and
YOLO appends ``x y v`` triples to the detection row and declares ``kpt_shape`` in ``data.yaml``. An
annotation that carries no landmarks — one generated for a different task and then handed to a
keypoint writer — is written as an all-zero, visibility-``0`` ("not labeled") table rather than a
short record, so every row and record still matches the schema the task declares.

"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_NAMES, ANIMAL_KEYPOINT_SKELETON
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


def _keypoint_triples(ann: Annotation, img_w: float, img_h: float) -> list[tuple[float, float, int]]:
    """Return one ``(x, y, visibility)`` landmark triple per keypoint, clamped to the image.

    Args:
        ann: The annotation to read landmarks from.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        One triple per name in :data:`~fuse_augmentations.data.animals.ANIMAL_KEYPOINT_NAMES`, in that
        order. A visible point is clamped to the image extent like every other coordinate field; an
        invisible one keeps the zeroed placeholder coordinates rather than being clamped into a
        spurious corner position. An annotation without landmarks yields the all-zero, "not
        labeled" table — see the module docstring.

    """
    if ann.keypoints is None:
        return [(0.0, 0.0, 0)] * len(ANIMAL_KEYPOINT_NAMES)
    return [
        (_clamp(x, 0.0, img_w), _clamp(y, 0.0, img_h), visibility) if visibility > 0 else (0.0, 0.0, visibility)
        for x, y, visibility in ann.keypoints
    ]


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
        elif self.task is Task.KEYPOINTS:
            triples = _keypoint_triples(ann, img_w, img_h)
            record["keypoints"] = [value for triple in triples for value in triple]
            record["num_keypoints"] = sum(1 for *_, visibility in triples if visibility > 0)
        return record

    def _categories(self) -> list[dict[str, Any]]:
        """Build the category records, adding the keypoint schema for the pose task."""
        categories: list[dict[str, Any]] = [
            {"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(self.class_names)
        ]
        if self.task is Task.KEYPOINTS:
            for category in categories:
                category["keypoints"] = list(ANIMAL_KEYPOINT_NAMES)
                # COCO skeleton edges are 1-based indices into the category's own keypoint list.
                category["skeleton"] = [[i + 1, j + 1] for i, j in ANIMAL_KEYPOINT_SKELETON]
        return categories

    def _coco_doc(self, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap image and annotation records into a COCO document with categories."""
        categories = self._categories()
        return {
            "info": {"description": "fuse-augmentations synthetic dataset"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Stream each split to ``<output_dir>/<split>/`` in a single pass over its samples.

        Image pixels are written as they are produced and never held in memory. The COCO schema,
        however, emits one JSON document per split, so lightweight per-image and per-annotation
        metadata records (no pixels) accumulate for the duration of the split and are serialized
        once the split is exhausted: memory is O(n) in the split's image and annotation counts, not
        constant. For a constant-memory path use the YOLO writer (one label file per image) or the
        in-memory :class:`~fuse_augmentations.data.datasets.SyntheticIterableDataset`.

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

    @staticmethod
    def _box_coords(ann: Annotation, width: int, height: int) -> list[float]:
        """Return the normalized ``cx cy w h`` of the box clipped to the image extent."""
        # Clamp corners to the image extent before deriving cx/cy/w/h so an edge-crossing box's
        # label matches its clipped visible box (consistent with CocoWriter._annotation_dict).
        x1 = _clamp(ann.bbox_xyxy[0], 0.0, width)
        y1 = _clamp(ann.bbox_xyxy[1], 0.0, height)
        x2 = _clamp(ann.bbox_xyxy[2], 0.0, width)
        y2 = _clamp(ann.bbox_xyxy[3], 0.0, height)
        return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]

    @staticmethod
    def _keypoint_tokens(ann: Annotation, width: int, height: int) -> list[str]:
        """Return the trailing ``x y v`` tokens of a pose row, three per landmark.

        Coordinates are normalized and clamped like every other coordinate; the visibility flag is an index into COCO's
        scale, so it is written as a plain integer and never normalized.

        """
        tokens: list[str] = []
        for x, y, visibility in _keypoint_triples(ann, float(width), float(height)):
            tokens += [f"{_clamp(x / width, 0.0, 1.0):.6f}", f"{_clamp(y / height, 0.0, 1.0):.6f}", str(visibility)]
        return tokens

    def _label_row(self, ann: Annotation, width: int, height: int) -> str:
        """Format one YOLO label row for the writer's task, clamping coordinates to ``[0, 1]``."""
        if self.task in (Task.DETECTION, Task.KEYPOINTS):
            # A pose row is a detection row plus the landmark block (Ultralytics' order).
            coords = self._box_coords(ann, width, height)
        else:
            flat = ann.polygon if self.task is Task.SEGMENTATION else ann.obb_corners
            coords = [v / width if i % 2 == 0 else v / height for i, v in enumerate(flat)]
        tokens = [str(ann.class_id), *(f"{_clamp(c, 0.0, 1.0):.6f}" for c in coords)]
        if self.task is Task.KEYPOINTS:
            tokens += self._keypoint_tokens(ann, width, height)
        return " ".join(tokens)

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
        if not splits:  # defensive backstop; write() is the primary guard
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
        lines = ["path: .", f"train: images/{'train' if 'train' in splits else next(iter(splits))}"]
        if "val" in splits:
            lines.append("val: images/val")
        if "test" in splits:
            lines.append("test: images/test")
        lines.append(f"nc: {len(self.class_names)}")
        if self.task is Task.KEYPOINTS:
            # Ultralytics carries one dataset-wide (num_keypoints, dims) shape, hence one shared
            # landmark schema for every class; dims is 3 because each point ships its visibility.
            lines.append(f"kpt_shape: [{len(ANIMAL_KEYPOINT_NAMES)}, 3]")
        lines.append("names:")
        lines.extend(f"  {i}: {name}" for i, name in enumerate(self.class_names))
        return "\n".join(lines) + "\n"

    def write(self, splits: dict[str, Iterable[Sample]], output_dir: str | Path) -> None:
        """Write images, labels, and ``data.yaml`` under ``output_dir``.

        Raises:
            ValueError: If ``splits`` is empty (checked before any output directory is created).

        """
        if not splits:
            raise ValueError("YoloWriter requires at least one split, got an empty mapping")
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
