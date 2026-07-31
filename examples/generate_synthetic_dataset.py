"""Generate synthetic shape/color datasets in COCO and YOLO layouts.

Produces small labelled datasets of colored shapes (square, rectangle, triangle,
circle) for detection, segmentation, and oriented-bounding-box (OBB) tasks, in both
COCO and YOLO formats. Requires the optional Pillow dependency:

    pip install 'fuse-augmentations[synthetic]'

Run (writes every format x task combo under ./synthetic_out):
    python examples/generate_synthetic_dataset.py

Custom run:
    python examples/generate_synthetic_dataset.py --outdir /tmp/shapes --num-images 50 \
        --img-size 512 --class-mode shape_color --seed 7

"""

from __future__ import annotations

import argparse
from pathlib import Path

from fuse_augmentations.data import generate_dataset

FORMATS = ("coco", "yolo")
TASKS = ("detection", "segmentation", "obb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("synthetic_out"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--class-mode", default="shape", choices=("shape", "color", "shape_color"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for fmt in FORMATS:
        for task in TASKS:
            out = args.outdir / f"{fmt}_{task}"
            counts = generate_dataset(
                out,
                num_images=args.num_images,
                fmt=fmt,
                task=task,
                class_mode=args.class_mode,
                img_size=args.img_size,
                seed=args.seed,
            )
            print(f"{fmt:5} / {task:12} -> {out}  splits={counts}")

    print(f"\nDone. Explore the tree under: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
