"""End-to-end facade tests: all format x task combos, splits, and determinism."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import yaml

from fuse_augmentations.data import SplitRatios, generate_dataset
from fuse_augmentations.data.animals import AnimalShape
from fuse_augmentations.data.letters import LetterShape
from fuse_augmentations.data.symbols import SymbolShape

_FORMATS = ["coco", "yolo"]
_TASKS = ["detection", "segmentation", "obb"]

#: Shape families whose ids are *not* a prefix of the full `Shape` vocabulary — the four geometric
#: shapes come first, so a geometric-only run's narrowed ids coincide with its global ones and can
#: never catch a narrowing seam. Every other family starts at a non-zero offset (animals at 4,
#: symbols at 16, letters at 23) and does catch it.
_NON_PREFIX_FAMILIES = [
    pytest.param((AnimalShape.GIRAFFE, AnimalShape.DUCK), id="animals"),
    pytest.param((SymbolShape.KITE,), id="symbols"),
    pytest.param((LetterShape.X, LetterShape.I), id="letters"),
]


@pytest.mark.parametrize(("fmt", "task"), list(itertools.product(_FORMATS, _TASKS)))
def test_generate_all_combos(tmp_path: Path, fmt: str, task: str) -> None:
    """Dataset generation works for all format and task combinations."""
    out = tmp_path / f"{fmt}_{task}"
    counts = generate_dataset(out, num_images=10, fmt=fmt, task=task, img_size=64, seed=0)
    assert sum(counts.values()) == 10
    assert out.exists()


def test_split_counts_follow_ratios(tmp_path: Path) -> None:
    """Generated dataset split counts match default ratios."""
    counts = generate_dataset(tmp_path, num_images=10, fmt="coco", seed=0)
    assert counts == {"train": 7, "val": 2, "test": 1}


def test_custom_split_ratios(tmp_path: Path) -> None:
    """Custom split ratios are respected in dataset generation."""
    counts = generate_dataset(tmp_path, num_images=10, fmt="yolo", split_ratios=SplitRatios(0.5, 0.5, 0.0), seed=0)
    assert counts == {"train": 5, "val": 5}


def test_coco_file_counts_match(tmp_path: Path) -> None:
    """COCO format file counts match metadata."""
    counts = generate_dataset(tmp_path, num_images=10, fmt="coco", seed=1)
    for split, count in counts.items():
        assert len(list((tmp_path / split).glob("*.jpg"))) == count


def test_configured_shapes_scope_coco_categories(tmp_path: Path) -> None:
    """COCO categories use the shape family configured through the facade."""
    from fuse_augmentations.data import DEFAULT_SHAPES

    generate_dataset(tmp_path, num_images=1, fmt="coco", shapes=DEFAULT_SHAPES, img_size=32, seed=0)
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())

    assert [(category["id"], category["name"]) for category in doc["categories"]] == [
        (index, shape.value) for index, shape in enumerate(DEFAULT_SHAPES, start=1)
    ]


@pytest.mark.parametrize("shapes", _NON_PREFIX_FAMILIES)
@pytest.mark.parametrize("task", ["detection", "segmentation", "obb", "keypoints"])
def test_coco_annotation_category_ids_resolve_against_the_written_categories(
    tmp_path: Path, shapes: tuple, task: str
) -> None:
    """Every COCO annotation's `category_id` is declared in the same document's `categories`.

    The invariant a COCO reader relies on: building a `category_id -> name` map from `categories`
    and looking up each annotation must never raise. It broke once already, when the writer started
    narrowing its category list to the run's own `shapes` while the generator kept stamping ids into
    the full 49-shape vocabulary — a giraffes-only dataset declared category 1 and annotated with
    id 7. Only a non-prefix family catches it, hence the parametrization: the geometric shapes lead
    the vocabulary, so their narrowed and global ids coincide.

    """
    generate_dataset(tmp_path, num_images=6, fmt="coco", task=task, shapes=shapes, img_size=96, seed=17)

    drawn = set()
    for split_dir in sorted(p for p in tmp_path.iterdir() if p.is_dir()):
        doc = json.loads((split_dir / "_annotations.coco.json").read_text())
        by_id = {category["id"]: category["name"] for category in doc["categories"]}
        assert doc["annotations"], f"{split_dir.name} drew no objects to check"
        drawn.update(by_id[ann["category_id"]] for ann in doc["annotations"])
    assert drawn <= {shape.value for shape in shapes}


@pytest.mark.parametrize("shapes", _NON_PREFIX_FAMILIES)
def test_yolo_label_class_indices_resolve_against_data_yaml(tmp_path: Path, shapes: tuple) -> None:
    """Every YOLO label row's leading class index is one `data.yaml` declares.

    The YOLO half of the same seam: `nc`/`names` narrow to the run's own `shapes`, so a row carrying
    a full-vocabulary id points past the end of the declared names and any loader either crashes or
    silently mislabels. Checked on non-prefix families for the reason spelled out on the COCO twin.

    """
    generate_dataset(tmp_path, num_images=6, fmt="yolo", shapes=shapes, img_size=96, seed=17)

    names = yaml.safe_load((tmp_path / "data.yaml").read_text())["names"]
    assert set(names.values()) == {shape.value for shape in shapes}

    rows = [line for path in tmp_path.rglob("*.txt") for line in path.read_text().splitlines() if line]
    assert rows, "run drew no objects to check"
    assert {int(row.split()[0]) for row in rows} <= set(names)


def test_yolo_file_counts_match(tmp_path: Path) -> None:
    """YOLO format file counts match metadata."""
    counts = generate_dataset(tmp_path, num_images=10, fmt="yolo", seed=1)
    for split, count in counts.items():
        assert len(list((tmp_path / "images" / split).glob("*.jpg"))) == count


def test_deterministic_labels(tmp_path: Path) -> None:
    """Dataset generation is deterministic with fixed seed."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(a, num_images=6, fmt="yolo", task="obb", img_size=64, seed=99)
    generate_dataset(b, num_images=6, fmt="yolo", task="obb", img_size=64, seed=99)
    label_a = (a / "labels" / "train" / "img_000000.txt").read_text()
    label_b = (b / "labels" / "train" / "img_000000.txt").read_text()
    assert label_a == label_b


def test_enum_and_string_args_equivalent(tmp_path: Path) -> None:
    """Enum and string arguments produce equivalent results."""
    from fuse_augmentations.data import ClassMode, OutputFormat, Task

    a = tmp_path / "str"
    b = tmp_path / "enum"
    counts_str = generate_dataset(a, num_images=5, fmt="coco", task="detection", class_mode="color", seed=3)
    counts_enum = generate_dataset(
        b, num_images=5, fmt=OutputFormat.COCO, task=Task.DETECTION, class_mode=ClassMode.COLOR, seed=3
    )
    assert counts_str == counts_enum
    # Split counts alone miss normalization defects; compare the generated COCO annotations too.
    for split in counts_str:
        doc_str = json.loads((a / split / "_annotations.coco.json").read_text())
        doc_enum = json.loads((b / split / "_annotations.coco.json").read_text())
        assert doc_str["categories"] == doc_enum["categories"]
        assert doc_str["annotations"] == doc_enum["annotations"]
        assert doc_str["images"] == doc_enum["images"]


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_rejects_non_positive_num_images(tmp_path: Path, bad: int) -> None:
    """Non-positive num_images raises ValueError."""
    with pytest.raises(ValueError, match="num_images"):
        generate_dataset(tmp_path, num_images=bad, fmt="coco", seed=0)


def test_supplied_config_ignores_invalid_class_mode(tmp_path: Path) -> None:
    """Invalid class_mode is ignored when config is supplied."""
    from fuse_augmentations.data import ClassMode, SyntheticConfig

    config = SyntheticConfig(img_size=64, class_mode=ClassMode.COLOR)
    # An invalid class_mode string must be ignored (not normalized) when a config is supplied.
    counts = generate_dataset(tmp_path, num_images=5, fmt="coco", class_mode="not-a-real-mode", config=config, seed=0)
    assert sum(counts.values()) == 5
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    assert len(doc["categories"]) == 3  # config's COLOR vocabulary, not the ignored class_mode


def test_task_reaches_the_generator_not_only_the_writer(tmp_path: Path) -> None:
    """A facade `task=` configures the generator too, so the landmark block holds real points."""
    from fuse_augmentations.data import ANIMAL_KEYPOINT_NAMES, AnimalShape

    counts = generate_dataset(
        tmp_path, num_images=4, fmt="coco", task="keypoints", shapes=(AnimalShape.DUCK,), img_size=64, seed=0
    )
    assert sum(counts.values()) == 4
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    # While the task reached the writer only, the generator computed no landmarks and every
    # annotation carried a full-length but all-zero, visibility-0 block instead of real points.
    assert all(len(ann["keypoints"]) == 3 * len(ANIMAL_KEYPOINT_NAMES) for ann in doc["annotations"])
    assert any(ann["num_keypoints"] > 0 for ann in doc["annotations"])


def test_rejects_a_task_conflicting_with_the_supplied_config(tmp_path: Path) -> None:
    """A `task=` disagreeing with config's own task is refused."""
    from fuse_augmentations.data import AnimalShape, SyntheticConfig, Task

    config = SyntheticConfig(img_size=64, task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK,))
    with pytest.raises(ValueError, match=r"conflicts with config\.task"):
        generate_dataset(tmp_path, num_images=5, fmt="coco", task="detection", config=config, seed=0)


def test_omitted_task_adopts_the_supplied_config_task(tmp_path: Path) -> None:
    """Omitted `task=` adopts the config's task instead of clashing with the default.

    A caller who sets `task=Task.KEYPOINTS` on the config and omits the facade argument used to hit the conflict error
    above -- raised against a default they never passed, and telling them to do exactly what they had already done.
    Omission must instead defer to the config.

    """
    from fuse_augmentations.data import ANIMAL_KEYPOINT_NAMES, AnimalShape, SyntheticConfig, Task

    config = SyntheticConfig(img_size=64, task=Task.KEYPOINTS, shapes=(AnimalShape.DUCK,))
    counts = generate_dataset(tmp_path, num_images=4, fmt="coco", config=config, seed=0)
    assert sum(counts.values()) == 4
    doc = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    # The keypoints writer ran (there is a landmark block at all) and the generator agreed with it
    # (the block holds real, visible points rather than the all-zero filler of a detection config).
    assert all(len(ann["keypoints"]) == 3 * len(ANIMAL_KEYPOINT_NAMES) for ann in doc["annotations"])
    assert any(ann["num_keypoints"] > 0 for ann in doc["annotations"])


def test_keypoints_task_rejects_the_default_geometric_shapes(tmp_path: Path) -> None:
    """The keypoints task refuses the default shapes, which have no landmark table."""
    with pytest.raises(ValueError, match="keypoint table"):
        generate_dataset(tmp_path, num_images=5, fmt="coco", task="keypoints", img_size=64, seed=0)
