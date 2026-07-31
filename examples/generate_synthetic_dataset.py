"""Generate synthetic shape/color datasets in COCO and YOLO layouts.

Produces small labelled datasets of colored shapes (square, rectangle, triangle,
circle) for detection, segmentation, and oriented-bounding-box (OBB) tasks, in both
COCO and YOLO formats. Rendering uses Pillow (a base dependency); the CLI below uses
``fire``, which ships in the ``cli`` extra:

    pip install "fuse-augmentations[cli]"

Run (writes every format x task combo under ./synthetic_out):
    python examples/generate_synthetic_dataset.py

Custom run:
    python examples/generate_synthetic_dataset.py --outdir /tmp/shapes --num-images 50 \
        --img-size 512 --class-mode shape_color --seed 7

"""

from __future__ import annotations

from pathlib import Path

from fuse_augmentations.data import generate_dataset

FORMATS = ("coco", "yolo")
TASKS = ("detection", "segmentation", "obb")


def main(
    outdir: str = "synthetic_out",
    num_images: int = 20,
    img_size: int = 256,
    class_mode: str = "shape",
    seed: int = 0,
) -> None:
    """Write every format x task combo under ``outdir``.

    Args:
        outdir: Root directory for the generated datasets.
        num_images: Number of images per format/task combination.
        img_size: Square canvas size in pixels.
        class_mode: One of ``shape``, ``color``, ``shape_color``.
        seed: Random seed for byte-identical output.

    Examples:
        >>> main(outdir="/tmp/shapes", num_images=2, img_size=64)  # doctest: +SKIP

    """
    root = Path(outdir)
    for fmt in FORMATS:
        for task in TASKS:
            out = root / f"{fmt}_{task}"
            counts = generate_dataset(
                out,
                num_images=num_images,
                fmt=fmt,
                task=task,
                class_mode=class_mode,
                img_size=img_size,
                seed=seed,
            )
            print(f"{fmt:5} / {task:12} -> {out}  splits={counts}")

    print(f"\nDone. Explore the tree under: {root.resolve()}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
