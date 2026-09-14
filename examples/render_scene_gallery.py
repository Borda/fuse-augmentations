"""Render static contact sheets for the scene knobs: backgrounds, degradations, clutter, bands.

``render_shape_reference.py`` pictures the *vocabulary* — one upright shape per file — and
``animate_synthetic_dataset.py`` pictures the *tasks* — the same stream under four overlays. Neither
pictures the knobs that decide how hard a sample is to read, which is what this script adds: one
contact sheet per knob group, every tile drawn from the same seed so the only thing that changes
between tiles is the knob named under it.

That shared seed is the point of the backgrounds sheet rather than a convenience. Every background
draws from a side stream rather than from the placement stream (see
:attr:`~fuse_augmentations.data.backgrounds.Background.consumes_randomness`), so switching modes
cannot move an object — the sheet shows the same shapes in the same places on six different
canvases, which is that guarantee as a picture instead of as a test name.

The sheets are plain images with no box, polygon, or keypoint overlay. The overlays belong to the
task previews; drawing them here would put the annotation between the reader and the pixels, and the
pixels are the subject.

Rendering uses Pillow and numpy (both base dependencies); the CLI below uses ``fire``, which ships
in the ``cli`` extra:

    pip install "fuse-augmentations[cli]"

Write every sheet (the four files under ``docs/assets/datasets/``):
    python examples/render_scene_gallery.py

Write one sheet:
    python examples/render_scene_gallery.py --sheets backgrounds

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fuse_augmentations.data import (
    JPEG,
    ColorCast,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    GradientBackground,
    ImageBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    Quantize,
    SolidBackground,
    SyntheticConfig,
    SyntheticGenerator,
    TextureBackground,
    Vignette,
)
from fuse_augmentations.data.letters import LetterShape

if TYPE_CHECKING:
    from collections.abc import Callable

    from fuse_augmentations.data.backgrounds import Background
    from fuse_augmentations.data.degradations import Degradation

#: Side length of one rendered tile. Smaller than the 320-pixel task animations because a sheet
#: shows six to eight tiles side by side and the whole row has to stay legible in a docs column.
_TILE = 176
_LABEL_HEIGHT = 20
#: Baseline step for a label that wraps onto more than one line.
_LINE_HEIGHT = 12
_GAP = 6
_PAPER = (255, 255, 255)
_INK = (32, 32, 32)
#: What ``ImageFont.load_default`` returns: the bundled bitmap font on a Pillow built without
#: FreeType, and a scalable one otherwise. Both are drawable; only the type differs.
_Font = ImageFont.FreeTypeFont | ImageFont.ImageFont
#: One seed for every sheet, so a tile in the degradations sheet is the same scene as a tile in the
#: clutter sheet with its knob turned off.
_SEED = 3

#: The backgrounds sheet, in the order the prose introduces them. ``ImageBackground`` is absent: it
#: reads files the package does not ship, so it gets its own sheet built on a synthesized directory.
_BACKGROUNDS: tuple[tuple[str, Background], ...] = (
    ("SolidBackground()", SolidBackground()),
    ("GradientBackground()", GradientBackground()),
    ("GradientBackground(radial=True)", GradientBackground(radial=True)),
    ("NoiseBackground(sigma=24)", NoiseBackground(sigma=24.0)),
    ("ImpulseNoiseBackground()", ImpulseNoiseBackground()),
    ("TextureBackground()", TextureBackground()),
)

#: The degradations sheet. Each is shown alone, at a strength chosen to be visible at this tile size
#: rather than at its default, so the sheet reads as "what this knob does" rather than "what this
#: default looks like" — the label carries the argument that produced the tile.
_DEGRADATIONS: tuple[tuple[str, Degradation | None], ...] = (
    ("none", None),
    ("GaussianNoise(sigma=16)", GaussianNoise(sigma=16.0)),
    ("GaussianBlur(radius=1.5)", GaussianBlur(radius=1.5)),
    ("JPEG(quality=15)", JPEG(quality=15)),
    ("Contrast(factor=0.45)", Contrast(factor=0.45)),
    ("ColorCast(gain=(1.3, 1.0, 0.7))", ColorCast(gain=(1.3, 1.0, 0.7))),
    ("Vignette(strength=0.7)", Vignette(strength=0.7)),
    ("Quantize(levels=4)", Quantize(levels=4)),
)

#: The clutter sheet: the two knobs that add unlabelled ink, alone and together. Both draw shapes
#: from the same process as the targets, which is what the sheet is for — an unlabelled square is
#: not distinguishable from a labelled one by any "is this a shape" test.
_CLUTTER: tuple[tuple[str, dict[str, int]], ...] = (
    ("neither", {}),
    ("distractors=6", {"distractors": 6}),
    ("occluders=3", {"occluders": 3}),
    ("distractors=6, occluders=3", {"distractors": 6, "occluders": 3}),
)

#: The three bands from ``docs/datasets/difficulty.md``, kept as the keyword arguments that page's
#: table lists so the picture and the prose cannot drift apart. Changing a band there means changing
#: it here and re-running this script.
_BANDS: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "easy",
        {
            "background": NoiseBackground(sigma=12.0),
            "min_size_ratio": 0.10,
            "max_size_ratio": 0.30,
        },
    ),
    (
        "moderate",
        {
            "background": TextureBackground(frequency=8.0),
            "min_size_ratio": 0.08,
            "max_size_ratio": 0.25,
            "distractors": 3,
            "degrade": (GaussianBlur(radius=0.5), JPEG(quality=75)),
        },
    ),
    (
        "hard",
        {
            "background": TextureBackground(frequency=8.0),
            "shapes": tuple(LetterShape),
            "min_size_ratio": 0.03,
            "max_size_ratio": 0.25,
            "min_objects": 4,
            "max_objects": 8,
            "distractors": 6,
            "degrade": (GaussianBlur(radius=0.5), JPEG(quality=75)),
        },
    ),
)


def _render_tile(**overrides: object) -> Image.Image:
    """Return the first sample of a stream at :data:`_SEED`, as a tile-sized image.

    Every knob not named in ``overrides`` keeps its package default, so a tile differs from its neighbour by exactly
    what the label says.

    """
    defaults: dict[str, object] = {"img_size": _TILE, "max_objects": 4}
    config = SyntheticConfig(**{**defaults, **overrides})
    sample = next(iter(SyntheticGenerator(config).generate(1, seed=_SEED)))
    return Image.fromarray(sample.image)


def _wrap(label: str, draw: ImageDraw.ImageDraw, font: _Font) -> list[str]:
    """Split one label into lines no wider than a tile.

    Labels are constructor calls rather than prose, so they break at the argument separator and
    nowhere else: ``ImageBackground(folder, grayscale=True)`` reads as two lines, while a single
    long argument stays on one line and is allowed to overhang rather than be cut mid-token.

    """
    lines: list[str] = []
    for piece in label.split(", "):
        candidate = f"{lines[-1]}, {piece}" if lines else piece
        if lines and draw.textlength(candidate, font=font) <= _TILE:
            lines[-1] = candidate
        else:
            if lines:
                # The separator stays on the line it broke after, so a wrapped call still reads as
                # one argument list rather than as two unrelated fragments.
                lines[-1] += ","
            lines.append(piece)
    return lines


def _contact_sheet(tiles: list[tuple[str, Image.Image]], columns: int) -> Image.Image:
    """Lay tiles out on a white sheet, each with its label centred underneath.

    Every cell is as tall as the sheet's longest label needs, so a wrapped label pushes the whole row down rather than
    overprinting the tile beneath it.

    """
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = ImageFont.load_default()
    wrapped = [(_wrap(label, measure, font), tile) for label, tile in tiles]
    label_height = _LABEL_HEIGHT + _LINE_HEIGHT * (max(len(lines) for lines, _ in wrapped) - 1)

    rows = -(-len(tiles) // columns)
    cell_height = _TILE + label_height
    sheet = Image.new(
        "RGB",
        (columns * _TILE + (columns - 1) * _GAP, rows * cell_height + (rows - 1) * _GAP),
        _PAPER,
    )
    draw = ImageDraw.Draw(sheet)
    for index, (lines, tile) in enumerate(wrapped):
        left = (index % columns) * (_TILE + _GAP)
        top = (index // columns) * (cell_height + _GAP)
        sheet.paste(tile, (left, top))
        for line_index, line in enumerate(lines):
            width = draw.textlength(line, font=font)
            draw.text(
                (left + max(0.0, (_TILE - width) / 2.0), top + _TILE + 5 + line_index * _LINE_HEIGHT),
                line,
                fill=_INK,
                font=font,
            )
    return sheet


def _backgrounds_sheet() -> Image.Image:
    """Every canvas mode under one placement stream: six procedural, then two read from files."""
    procedural = [(label, _render_tile(background=background)) for label, background in _BACKGROUNDS]
    # The pictures the last two tiles crop from exist only for the length of this call.
    with tempfile.TemporaryDirectory() as folder:
        return _contact_sheet(procedural + _photographic_tiles(Path(folder)), columns=4)


def _photographic_tiles(folder: Path) -> list[tuple[str, Image.Image]]:
    """Return the two ``ImageBackground`` tiles, drawn from a directory built in ``folder``.

    The pictures are synthesized rather than shipped: the point of the tile is that the canvas comes
    from a file, which a generated file makes as well as a photograph would, and a binary added to
    the repository only to be pasted into one contact sheet would be carried forever.

    """
    grid = np.linspace(0.0, 2.0 * np.pi, 240, dtype=np.float64)
    x, y = np.meshgrid(grid, grid)
    for index, (level, period) in enumerate(((70.0, 3.0), (95.0, 5.0), (120.0, 8.0))):
        # Banded, mid-grey and low-saturation on purpose: a crop has to read as a backdrop that a
        # coloured shape still stands out against, which a bright or saturated one does not.
        texture = level + 26.0 * np.sin(period * x) * np.cos(period * y) + 14.0 * np.sin(2.0 * period * y)
        rgb = np.stack([texture + 10.0, texture, texture - 8.0], axis=-1)
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(folder / f"picture_{index}.png")
    return [
        ("ImageBackground(folder)", _render_tile(background=ImageBackground(folder))),
        (
            "ImageBackground(folder, grayscale=True)",
            _render_tile(background=ImageBackground(folder, grayscale=True)),
        ),
    ]


def _degradations_sheet() -> Image.Image:
    """One scene, first clean and then under each degradation on its own."""
    tiles = [
        (label, _render_tile(degrade=() if degradation is None else (degradation,)))
        for label, degradation in _DEGRADATIONS
    ]
    return _contact_sheet(tiles, columns=4)


def _clutter_sheet() -> Image.Image:
    """The two unlabelled-ink knobs, alone and together."""
    tiles = [(label, _render_tile(**knobs)) for label, knobs in _CLUTTER]
    return _contact_sheet(tiles, columns=4)


def _difficulty_sheet() -> Image.Image:
    """The three documented bands at the canvas size their table was measured on.

    Rendered at 256 pixels and then fitted to the tile, because the bands are size ratios: 0.03 of a 256-pixel canvas is
    the eight-pixel glyph the hard band is named for, and rendering the band at the tile size instead would quietly make
    that glyph five pixels.

    """
    tiles = []
    for label, knobs in _BANDS:
        tile = _render_tile(img_size=256, **knobs).resize((_TILE, _TILE), Image.Resampling.NEAREST)
        tiles.append((label, tile))
    return _contact_sheet(tiles, columns=3)


#: One builder per sheet name, so ``--sheets`` validates against the same mapping that renders.
_BUILDERS: dict[str, Callable[[], Image.Image]] = {
    "backgrounds": _backgrounds_sheet,
    "degradations": _degradations_sheet,
    "clutter": _clutter_sheet,
    "difficulty": _difficulty_sheet,
}


def main(output_dir: str = "docs/assets/datasets", sheets: str = "all", image_format: str = "webp") -> None:
    """Write one contact sheet per knob group.

    Args:
        output_dir: Directory for the generated sheets, written as ``gallery-<sheet>.<format>``.
        sheets: One of ``backgrounds``, ``degradations``, ``clutter``, ``difficulty``, or ``all``.
        image_format: Still-image container, ``webp`` or ``png``.

    Examples:
        >>> callable(main)
        True

    """
    assert image_format in ("webp", "png"), f"--image_format must be 'webp' or 'png', got {image_format!r}"
    assert sheets in (*_BUILDERS, "all"), f"--sheets must be one of {(*_BUILDERS, 'all')}, got {sheets!r}"
    wanted = tuple(_BUILDERS) if sheets == "all" else (sheets,)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        sheet = _BUILDERS[name]()
        path = out_dir / f"gallery-{name}.{image_format}"
        # Lossless because a sheet is mostly flat fills with hard edges and small text, which is what
        # a lossy encoder spends its budget destroying; the files land near the animated previews'.
        sheet.save(path, lossless=True) if image_format == "webp" else sheet.save(path)
        print(f"wrote {path}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
