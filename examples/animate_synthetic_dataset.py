"""Animate synthetic-dataset previews, one looping clip per annotation task.

The static companion (three PNGs under ``docs/assets/datasets``) showed a single
frame per task. This script renders a short looping animation instead: it draws a
fixed, seeded stream of synthetic images and, for every task, cycles through them
showing each image first bare and then with its yellow annotation overlay drawn
back on — axis-aligned boxes for detection, filled-shape polygons for
segmentation, and oriented boxes for OBB.

Because all three tasks share the same seeded sample stream, the clips line up
image-for-image: the same shapes appear in each, differing only in the label type
overlaid, so the three annotation representations can be compared directly.

Frames are written as an animated WebP (smaller than GIF, matching the existing
WebP assets); ``--image_format gif`` is available as a fallback.

``--shapes`` picks which vocabulary is drawn: ``geometric`` (square, rectangle,
triangle, circle) writes ``<task>.webp``, and ``animals`` (the eight side-profile
silhouettes) writes ``animals-<task>.webp`` so both sets can live side by side in
the docs. The animal scene uses fewer, larger objects, because a snake or a giraffe
needs more pixels than a square to stay readable at preview size.

Render every task (detection, segmentation, obb):
    python examples/animate_synthetic_dataset.py

Render the animal-shape previews:
    python examples/animate_synthetic_dataset.py --shapes animals

Render one task:
    python examples/animate_synthetic_dataset.py --task obb --img_size 320 --num_images 6

"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fuse_augmentations.data import SyntheticConfig, SyntheticGenerator
from fuse_augmentations.data.config import DEFAULT_SHAPES, Shape
from fuse_augmentations.data.sample import Annotation, Sample

TASKS = ("detection", "segmentation", "obb")
#: Drawable vocabulary per ``--shapes`` choice; ``animals`` is everything the enum gained
#: on top of the original four, so a new animal is picked up without editing this script.
SHAPE_SETS = {
    "geometric": DEFAULT_SHAPES,
    "animals": tuple(s for s in Shape if s not in DEFAULT_SHAPES),
}
#: Scene knobs per vocabulary. ``geometric`` reproduces the original clips byte-for-byte;
#: ``animals`` trades object count for object size so each silhouette stays legible.
SCENES = {
    "geometric": {"min_objects": 5, "max_objects": 7, "min_size_ratio": 0.1, "max_size_ratio": 0.3},
    "animals": {"min_objects": 3, "max_objects": 5, "min_size_ratio": 0.2, "max_size_ratio": 0.42},
}
#: Filename prefix per vocabulary, keeping the pre-existing asset names untouched.
PREFIXES = {"geometric": "", "animals": "animals-"}
_OVERLAY_RGB = (255, 255, 0)  # yellow annotation overlay, matching the static previews
_SUPERSAMPLE = 2  # render overlays at 2x then downscale so diagonal edges read smooth
_BARE_MS = 500  # hold the un-annotated image briefly before the label appears
_LABELLED_MS = 1300  # hold the annotated image long enough to read the overlay


def _pairs(flat: list[float], scale: float) -> list[tuple[float, float]]:
    """Turn a flat ``[x1, y1, x2, y2, ...]`` list into scaled ``(x, y)`` point tuples."""
    return [(flat[i] * scale, flat[i + 1] * scale) for i in range(0, len(flat), 2)]


def _draw_annotation(draw: ImageDraw.ImageDraw, ann: Annotation, task: str, scale: float, width: int) -> None:
    """Draw one annotation's overlay for the given task onto ``draw``."""
    if task == "detection":
        x1, y1, x2, y2 = (v * scale for v in ann.bbox_xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=_OVERLAY_RGB, width=width)
        return
    points = _pairs(ann.polygon if task == "segmentation" else ann.obb_corners, scale)
    draw.line([*points, points[0]], fill=_OVERLAY_RGB, width=width, joint="curve")


def _render_frame(sample: Sample, task: str, out_size: int, annotate: bool) -> Image.Image:
    """Render one sample as a PIL frame, optionally with the task overlay drawn on."""
    canvas = Image.fromarray(sample.image).resize((out_size * _SUPERSAMPLE,) * 2, Image.Resampling.NEAREST)
    if annotate:
        scale = out_size * _SUPERSAMPLE / sample.width
        draw = ImageDraw.Draw(canvas)
        for ann in sample.annotations:
            _draw_annotation(draw, ann, task, scale, width=_SUPERSAMPLE * 2)
    return canvas.resize((out_size, out_size), Image.Resampling.LANCZOS)


def _build_frames(samples: list[Sample], task: str, out_size: int) -> tuple[list[Image.Image], list[int]]:
    """Build the ordered (image, duration) frame lists cycling bare then annotated per sample."""
    images: list[Image.Image] = []
    durations: list[int] = []
    for sample in samples:
        images.append(_render_frame(sample, task, out_size, annotate=False))
        durations.append(_BARE_MS)
        images.append(_render_frame(sample, task, out_size, annotate=True))
        durations.append(_LABELLED_MS)
    return images, durations


def _save_animation(images: list[Image.Image], durations: list[int], output_path: Path, image_format: str) -> None:
    """Write frames as an animated WebP or GIF loop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {
        "save_all": True,
        "append_images": images[1:],
        "duration": durations,
        "loop": 0,
    }
    if image_format == "webp":
        save_kwargs["quality"] = 90
        save_kwargs["method"] = 6
    else:
        save_kwargs["disposal"] = 2
        save_kwargs["optimize"] = True
    images[0].save(output_path, format=image_format.upper(), **save_kwargs)


def render(
    samples: list[Sample], task: str, output_dir: Path, out_size: int, image_format: str, prefix: str = ""
) -> Path:
    """Render one task's animation from a shared sample stream and return its path."""
    images, durations = _build_frames(samples, task, out_size)
    output_path = output_dir / f"{prefix}{task}.{image_format}"
    _save_animation(images, durations, output_path, image_format)
    return output_path


def main(
    output_dir: str = "docs/assets/datasets",
    task: str = "all",
    img_size: int = 320,
    num_images: int = 6,
    seed: int = 0,
    image_format: str = "webp",
    shapes: str = "geometric",
) -> None:
    """Generate looping preview animations for the synthetic-dataset annotation tasks.

    Args:
        output_dir: Directory for the generated animations.
        task: One of ``detection``, ``segmentation``, ``obb``, or ``all``.
        img_size: Output canvas side length in pixels.
        num_images: Number of distinct images cycled in each clip.
        seed: Seed for the shared sample stream (byte-identical output).
        image_format: Animation container, ``webp`` or ``gif``.
        shapes: Drawable vocabulary, ``geometric`` or ``animals``; the latter writes
            ``animals-<task>`` files so both preview sets can coexist.

    Examples:
        >>> callable(main)
        True

    """
    assert image_format in ("webp", "gif"), f"--image_format must be 'webp' or 'gif', got {image_format!r}"
    assert shapes in SHAPE_SETS, f"--shapes must be one of {tuple(SHAPE_SETS)}, got {shapes!r}"
    tasks = TASKS if task == "all" else (task,)
    assert all(t in TASKS for t in tasks), f"--task must be one of {TASKS} or 'all', got {task!r}"

    # One shared, seeded stream so every task clip shows the same shapes, only the overlay differs.
    config = SyntheticConfig(img_size=img_size, rotate=True, shapes=SHAPE_SETS[shapes], **SCENES[shapes])
    samples = list(SyntheticGenerator(config).generate(num_images, seed=seed))

    out_dir = Path(output_dir)
    for a_task in tasks:
        print(f"wrote {render(samples, a_task, out_dir, img_size, image_format, PREFIXES[shapes])}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
