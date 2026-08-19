"""Annotation construction-time validation of the landmark table."""

from __future__ import annotations

import pytest

from fuse_augmentations.data.animals import ANIMAL_KEYPOINT_NAMES
from fuse_augmentations.data.sample import Annotation

_BOX = (0.0, 0.0, 10.0, 10.0)
_POLYGON = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]


def _annotation(keypoints: tuple[tuple[float, float, int], ...] | None) -> Annotation:
    """Build an annotation whose only interesting field is its landmark table."""
    return Annotation(
        class_id=4,
        class_name="duck",
        polygon=_POLYGON,
        bbox_xyxy=_BOX,
        obb_corners=_POLYGON,
        keypoints=keypoints,
    )


def _valid_table(visibility: int = 2) -> tuple[tuple[float, float, int], ...]:
    """Return a full-length table with every landmark carrying the same visibility flag."""
    return tuple((1.0, 2.0, visibility) for _ in ANIMAL_KEYPOINT_NAMES)


def test_accepts_a_full_length_table() -> None:
    """A 16-triple table with valid visibilities is stored verbatim.

    This is the shape the generator emits for every `Task.KEYPOINTS` object, so the guard must let the normal case
    through untouched — a validator that also rewrote or reordered the table would break the landmark-to-name
    correspondence it exists to protect.

    """
    table = _valid_table()

    ann = _annotation(table)

    assert ann.keypoints == table
    assert len(ann.keypoints or ()) == len(ANIMAL_KEYPOINT_NAMES)


def test_accepts_no_table_at_all() -> None:
    """`keypoints=None` stays the documented "not a keypoints task" state and is not validated.

    Every non-keypoint task constructs annotations this way, so `None` must not be mistaken for a zero-length table and
    rejected.

    """
    assert _annotation(None).keypoints is None


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single-triple"),
        pytest.param(len(ANIMAL_KEYPOINT_NAMES) - 1, id="one-short"),
        pytest.param(len(ANIMAL_KEYPOINT_NAMES) + 1, id="one-too-many"),
    ],
)
def test_rejects_a_table_of_the_wrong_length(count: int) -> None:
    """A table that is not exactly one triple per schema name raises at construction.

    A wrong-length table is read positionally downstream — a COCO `keypoints` array of any length still parses and a
    YOLO pose row is split by position — so every landmark past the gap would be silently attributed to the wrong name
    rather than failing.

    """
    with pytest.raises(ValueError, match="must hold exactly"):
        _annotation(tuple((1.0, 2.0, 2) for _ in range(count)))


@pytest.mark.parametrize(
    "visibility",
    [
        pytest.param(3, id="above-range"),
        pytest.param(-1, id="negative"),
        pytest.param(255, id="pixel-value"),
        pytest.param(1.0, id="float-equal-to-a-valid-flag"),
        pytest.param(True, id="bool-equal-to-a-valid-flag"),
    ],
)
def test_rejects_a_visibility_outside_the_coco_flags(visibility: float | bool) -> None:
    """A visibility that is not one of COCO's 0/1/2 flags raises and names the offending landmark.

    Visibility is an index into a three-value scale, not a magnitude; a stray value passes straight through the writers
    into the emitted dataset, where a consumer reads it as an unknown flag. `1.0` and `True` are included because both
    compare equal to the int `1` in a plain membership check, so a validator that only checks membership would let them
    through — `True` would then reach the YOLO writer and serialize as the word "True" instead of a numeric flag.

    """
    with pytest.raises(ValueError, match="visibility"):
        _annotation(_valid_table(visibility))


@pytest.mark.parametrize("visibility", [0, 1, 2])
def test_accepts_every_coco_visibility_flag(visibility: int) -> None:
    """All three COCO flags are valid, including the `1` the generator never emits.

    The generator only produces 0 and 2, but 1 ("labeled but not visible") is part of the COCO schema; rejecting it
    would refuse a legitimate table built by hand or by a downstream tool.

    """
    assert _annotation(_valid_table(visibility)).keypoints == _valid_table(visibility)


def test_names_the_offending_landmark_in_the_error() -> None:
    """The error identifies which landmark carries the bad flag, not just that one does.

    With 16 same-shaped triples, an error that only says "a visibility is wrong" leaves the caller to bisect the table
    by hand.

    """
    table = (*_valid_table()[:2], (1.0, 2.0, 9), *_valid_table()[3:])

    with pytest.raises(ValueError, match=ANIMAL_KEYPOINT_NAMES[2]):
        _annotation(table)
