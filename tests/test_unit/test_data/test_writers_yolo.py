"""YOLO writer tests: normalized labels, row shape per task, and data.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_NAMES, AnimalShape
from fuse_augmentations.data.config import SyntheticConfig, Task, class_names
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.landmarks import KeypointSchema
from fuse_augmentations.data.sample import Annotation, Sample
from fuse_augmentations.data.symbols import SYMBOL_KEYPOINT_SCHEMA, SymbolShape
from fuse_augmentations.data.writers import YoloWriter


def _write(
    tmp_path: Path,
    writer_task: Task,
    splits: tuple[str, ...] = ("train", "val"),
    count: int = 3,
    seed: int = 9,
    keypoint_schema: KeypointSchema | None = None,
    **config_kwargs: object,
) -> tuple:  # type: ignore[type-arg]
    """Generate a seeded dataset, write it as YOLO, and return the samples per split plus the class names.

    Every knob the individual tests vary — splits, sample count, seed, and any `SyntheticConfig` field such as `shapes`,
    `img_size`, or `task` — is a keyword here, so a test states only what makes it different from the others. The
    writer's task is positional and separate from the config's: the writer formats whatever the samples already carry,
    so a detection-configured stream can still be written out as segmentation or OBB. `keypoint_schema` is only relevant
    to `Task.KEYPOINTS` runs and defaults to `YoloWriter`'s own default (the animal schema) when omitted.

    """
    settings = {"img_size": 96, "min_objects": 2, "max_objects": 4, **config_kwargs}
    config = SyntheticConfig(**settings)
    generator = SyntheticGenerator(config)
    rng = np.random.default_rng(seed)
    data = {split: [generator.sample(rng) for _ in range(count)] for split in splits}
    names = class_names(config.class_mode)
    writer_kwargs = {} if keypoint_schema is None else {"keypoint_schema": keypoint_schema}
    YoloWriter(writer_task, names, **writer_kwargs).write(data, tmp_path)
    return data, names


def _rows(tmp_path: Path, split: str = "train") -> list[str]:
    """Read and return non-empty rows from label files."""
    out: list[str] = []
    for txt in sorted((tmp_path / "labels" / split).glob("*.txt")):
        out.extend(r for r in txt.read_text().splitlines() if r)
    return out


@pytest.mark.parametrize(("task", "tokens"), [(Task.DETECTION, 5), (Task.SEGMENTATION, None), (Task.OBB, 9)])
def test_row_token_counts(tmp_path: Path, task: Task, tokens: int | None) -> None:
    """YOLO rows have expected token counts per task."""
    _write(tmp_path, task)
    for row in _rows(tmp_path):
        parts = row.split()
        if tokens is None:  # segmentation: cls + 2n polygon coords, even coord count
            assert len(parts) >= 7
            assert (len(parts) - 1) % 2 == 0
        else:
            assert len(parts) == tokens


@pytest.mark.parametrize("task", list(Task))
def test_all_coords_normalized(tmp_path: Path, task: Task) -> None:
    """All coordinates in YOLO labels are normalized to [0, 1]."""
    _write(tmp_path, task)
    for row in _rows(tmp_path):
        coords = [float(v) for v in row.split()[1:]]
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_class_id_is_zero_based_integer(tmp_path: Path) -> None:
    """Class IDs are zero-based integers."""
    _write(tmp_path, Task.DETECTION)
    for row in _rows(tmp_path):
        cls = row.split()[0]
        assert cls.isdigit()
        assert 0 <= int(cls) < 4


@pytest.mark.parametrize("task", [Task.DETECTION, Task.SEGMENTATION, Task.OBB])
def test_data_yaml_contents(tmp_path: Path, task: Task) -> None:
    """data.yaml contains expected structure and contents."""
    _, names = _write(tmp_path, task)
    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["nc"] == len(names)
    assert list(doc["names"].values()) == names
    assert doc["train"] == "images/train"
    assert doc["val"] == "images/val"
    # A stray kpt_shape on a non-pose dataset would make Ultralytics reject it as malformed.
    assert "kpt_shape" not in doc


def test_data_yaml_omits_absent_split(tmp_path: Path) -> None:
    """data.yaml omits missing splits."""
    _write(tmp_path, Task.DETECTION, splits=("train",))
    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert "val" not in doc
    assert "test" not in doc


def test_image_label_parity(tmp_path: Path) -> None:
    """Image and label files have one-to-one correspondence."""
    _write(tmp_path, Task.DETECTION)
    for split in ("train", "val"):
        imgs = {p.stem for p in (tmp_path / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (tmp_path / "labels" / split).glob("*.txt")}
        assert imgs == lbls
        assert len(imgs) == 3


def test_write_rejects_empty_splits(tmp_path: Path) -> None:
    """Empty splits dict is rejected."""
    names = class_names(SyntheticConfig().class_mode)
    with pytest.raises(ValueError, match="at least one split"):
        YoloWriter(Task.DETECTION, names).write({}, tmp_path)
    assert list(tmp_path.iterdir()) == []  # guarded before any partial output is created


@pytest.mark.parametrize(("task", "tokens"), [(Task.DETECTION, 5), (Task.SEGMENTATION, None), (Task.OBB, 9)])
def test_animal_shape_rows_keep_their_task_format(tmp_path: Path, task: Task, tokens: int | None) -> None:
    """A giraffe-only dataset emits well-formed, normalized YOLO rows for every task.

    Animal outlines are concave and carry many more vertices than a square; this is the regression guard that the writer
    stays purely format-agnostic and needed no special casing.

    """
    _, names = _write(
        tmp_path, task, splits=("train",), seed=31, img_size=192, max_objects=3, shapes=(AnimalShape.GIRAFFE,)
    )

    rows = _rows(tmp_path)
    assert rows
    for row in rows:
        parts = row.split()
        assert int(parts[0]) == names.index(AnimalShape.GIRAFFE.value)
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
        if tokens is None:  # segmentation: cls + the full outline, an even coord count
            assert len(parts) - 1 >= 2 * 15
            assert (len(parts) - 1) % 2 == 0
        else:
            assert len(parts) == tokens


def test_detection_label_clamps_edge_crossing_box(tmp_path: Path) -> None:
    """Detection labels clamp boxes that cross image edges."""
    # bbox crosses the left/top edges; the YOLO label must describe the clipped visible box.
    ann = Annotation(
        class_id=0,
        class_name="square",
        polygon=[-20.0, 10.0, 40.0, 10.0, 40.0, 50.0, -20.0, 50.0],
        bbox_xyxy=(-20.0, 10.0, 40.0, 50.0),
        obb_corners=[-20.0, 10.0, 40.0, 10.0, 40.0, 50.0, -20.0, 50.0],
    )
    sample = Sample(image=np.zeros((100, 100, 3), dtype=np.uint8), annotations=[ann], width=100, height=100)
    YoloWriter(Task.DETECTION, class_names(SyntheticConfig().class_mode)).write({"train": [sample]}, tmp_path)
    cx, cy, w, h = (float(v) for v in (tmp_path / "labels" / "train" / "img_000000.txt").read_text().split()[1:])
    # Clipped box is (0, 10, 40, 50): cx=0.2, cy=0.3, w=0.4, h=0.4 — not the unclipped cx=0.1, w=0.6.
    assert (cx, cy, w, h) == pytest.approx((0.2, 0.3, 0.4, 0.4))


def test_keypoints_row_has_fifty_three_tokens(tmp_path: Path) -> None:
    """A pose row is `cls cx cy w h` (5) plus 16 `x y v` triples (48) = 53 tokens, always.

    Even a whale's absent hind-leg points still emit their zeroed `0.000000 0.000000 0` triples, and every
    other triple's normalized coordinates and visibility flag must match the sample's own `ann.keypoints`
    table — the schema is dataset-wide and fixed-width, so a short row or a mismatched value is never valid.

    """
    img_size = 192
    hind_names = {"hind_knee_left", "hind_knee_right", "hind_limb_left", "hind_limb_right"}
    data, _ = _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=4,
        img_size=img_size,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=(AnimalShape.WHALE,),
    )

    rows = _rows(tmp_path)
    anns = [ann for sample in data["train"] for ann in sample.annotations]
    assert rows
    assert len(rows) == len(anns)
    for row, ann in zip(rows, anns, strict=True):
        tokens = row.split()
        assert len(tokens) == 53
        triples = {name: tokens[5 + 3 * i : 8 + 3 * i] for i, name in enumerate(ANIMAL_KEYPOINT_NAMES)}
        for name in hind_names:
            assert triples[name] == ["0.000000", "0.000000", "0"]
        for i, name in enumerate(ANIMAL_KEYPOINT_NAMES):
            if name in hind_names:
                continue
            x_tok, y_tok, v_tok = triples[name]
            kp_x, kp_y, kp_v = ann.keypoints[i]
            assert int(v_tok) == kp_v
            if kp_v > 0:
                assert float(x_tok) == pytest.approx(kp_x / img_size, abs=1e-3)
                assert float(y_tok) == pytest.approx(kp_y / img_size, abs=1e-3)


def test_keypoints_writer_task_with_non_keypoints_config_emits_the_all_zero_row(tmp_path: Path) -> None:
    """A KEYPOINTS-task writer over a plain detection-configured stream emits a fully-zeroed landmark block.

    The writer's `task` and the config's `task` are independent (see `_write`'s docstring): the config here stays at the
    default `Task.DETECTION` with the four geometric shapes, so `SyntheticGenerator` never computes a landmark table and
    every `ann.keypoints` is `None`. `_keypoint_triples`' `ann.keypoints is None` fallback is what has to still emit a
    well-formed 53-token row with a zeroed landmark block, instead of raising or writing a short row.

    """
    data, _names = _write(tmp_path, Task.KEYPOINTS)
    anns = [ann for sample in data["train"] for ann in sample.annotations]
    assert anns
    assert all(ann.keypoints is None for ann in anns)

    rows = _rows(tmp_path)
    assert rows
    for row in rows:
        tokens = row.split()
        assert len(tokens) == 53
        assert tokens[5:] == ["0.000000", "0.000000", "0"] * 16


def test_data_yaml_declares_kpt_shape_sixteen_three(tmp_path: Path) -> None:
    """`data.yaml` declares `kpt_shape: [16, 3]` for the keypoints task, the shared dataset-wide schema."""
    _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=0,
        img_size=192,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=tuple(AnimalShape),
    )

    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["kpt_shape"] == [16, 3]


def test_data_yaml_declares_an_identity_flip_idx(tmp_path: Path) -> None:
    """`data.yaml` declares `flip_idx` as the identity permutation over the sixteen landmarks.

    The schema's `left`/`right` names are viewer-relative — `left` is the limb nearer the viewer, not the animal's
    anatomical left — so horizontally flipping a side profile never turns a near limb into a far one. Identity is the
    correct mapping rather than a placeholder, and emitting it is what distinguishes "flipping re-indexes nothing here"
    from "nobody declared a mapping"; a wrong pairing here would silently mislabel every flip-augmented pose.

    """
    _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=0,
        img_size=192,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=tuple(AnimalShape),
    )

    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["flip_idx"] == list(range(len(ANIMAL_KEYPOINT_NAMES)))


def test_symbol_keypoints_row_has_twenty_six_tokens(tmp_path: Path) -> None:
    """A symbol pose row is `cls cx cy w h` (5) plus 7 `x y v` triples (21) = 26 tokens, always.

    Mirrors `test_keypoints_row_has_fifty_three_tokens` for the 7-slot symbol schema: a symbol whose outline uses
    fewer than all 7 slots (every symbol but house/cross) still emits its absent slots' zeroed `0.000000 0.000000 0`
    triples, keeping the row fixed-width.

    """
    img_size = 128
    data, _ = _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=4,
        img_size=img_size,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=(SymbolShape.KITE,),
        keypoint_schema=SYMBOL_KEYPOINT_SCHEMA,
    )

    rows = _rows(tmp_path)
    anns = [ann for sample in data["train"] for ann in sample.annotations]
    assert rows
    assert len(rows) == len(anns)
    for row, ann in zip(rows, anns, strict=True):
        tokens = row.split()
        assert len(tokens) == 26
        for i in range(len(SYMBOL_KEYPOINT_SCHEMA.names)):
            x_tok, y_tok, v_tok = tokens[5 + 3 * i : 8 + 3 * i]
            kp_x, kp_y, kp_v = ann.keypoints[i]
            assert int(v_tok) == kp_v
            if kp_v > 0:
                assert float(x_tok) == pytest.approx(kp_x / img_size, abs=1e-3)
                assert float(y_tok) == pytest.approx(kp_y / img_size, abs=1e-3)


def test_data_yaml_declares_kpt_shape_seven_three_for_symbols(tmp_path: Path) -> None:
    """`data.yaml` declares `kpt_shape: [7, 3]` for a symbol-family keypoints run."""
    _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=0,
        img_size=128,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=tuple(SymbolShape),
        keypoint_schema=SYMBOL_KEYPOINT_SCHEMA,
    )

    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["kpt_shape"] == [7, 3]


def test_data_yaml_declares_the_symbol_flip_idx(tmp_path: Path) -> None:
    """`data.yaml` declares `flip_idx` as the symbol schema's left/right swap, not the animals' identity mapping.

    Unlike the animal schema, every symbol is bilaterally symmetric about its own vertical axis, so a horizontal flip
    genuinely exchanges `flank_left`/`flank_right` (indices 3, 4) and `base_left`/`base_right` (indices 5, 6) — see
    `SYMBOL_KEYPOINT_FLIP_IDX`.

    """
    _write(
        tmp_path,
        Task.KEYPOINTS,
        splits=("train",),
        count=2,
        seed=0,
        img_size=128,
        max_objects=2,
        task=Task.KEYPOINTS,
        shapes=tuple(SymbolShape),
        keypoint_schema=SYMBOL_KEYPOINT_SCHEMA,
    )

    doc = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert doc["flip_idx"] == [0, 1, 2, 4, 3, 6, 5]
