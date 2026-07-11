"""Benchmark augmentation pipelines: native backend compose vs. fuse-augmentations.

Compares wall-clock time per call for ten augmentation sequences across Albumentations,
Kornia, and TorchVision, running each pipeline in both *native* and *fused* mode
(60 variants total).  Sequences are defined in ``SEQUENCE_BANK`` as ``AugSequence``
dataclass entries — add or remove rows there to change what gets benchmarked.

Sequences
---------
Keys follow ``<group>_<number>_<name>`` so ``sorted(SEQUENCE_BANK)`` gives the intended
display order.  Group letters: ``a`` = single-op baselines, ``b`` = geometric sequences,
``c`` = colour sequences, ``d`` = realistic mixed (geo + colour).

b01_geom_2 … b04_geom_5
    Pure geometric chains of 2-5 ops (Rotate, HFlip, Shear, VFlip) - each consecutive
    affine op saves one ``grid_sample`` call when fused.
c01_color_2 … c03_color_4
    Colour-only chains of 2-4 ops (Brightness, Contrast, Saturation, Hue) - fuse-aug
    merges consecutive pointwise ops into a single matrix-multiply pass.
d01_mixed_g3c2 … d05_mixed_g5c4
    Mixed geo+colour chains of 5-9 ops (gN = geo count, cN = colour count).
    Reorder variants ``__pw`` / ``__agr`` show how policy recovers fusion by regrouping ops.
a01_rotate … a05_shear
    Single-op affine baselines (Rotate, HFlip, VFlip, Scale, Shear) — no fusion possible;
    these measure the raw fuse-aug wrapper overhead vs. native compose.

Usage
-----
Run as a script::

    python experiments/bench_augmentation_pipelines.py

Open as a Jupyter notebook (requires jupytext)::

    jupytext --to notebook experiments/bench_augmentation_pipelines.py
    jupyter lab experiments/bench_augmentation_pipelines.ipynb

Results are saved to ``experiments/results/benchmark_results.json``.
Visual sanity figures are saved to ``experiments/results/visual_<seq>.png``.

Notes:
-----
*  Albumentations benchmarks — both native and fused — operate on HWC ``uint8`` NumPy arrays
   via the Albumentations dict-input API (``pipeline(image=ndarray)``), matching the real
   PyTorch training workflow where Albumentations runs CPU-side before ``ToTensorV2``.
   Kornia and TorchVision use BCHW ``float32`` tensors.
*  ``batch_size=1`` throughout — Albumentations is single-image natively.
*  Visual figures seed both torch and numpy RNGs identically for native and fused so they
   draw the same random parameters; the ``max|native-fused|`` annotation in each subplot
   confirms equivalence (diff = interpolation error only, not random draw difference).
"""

# %% [markdown]
# # Augmentation Pipeline Benchmark
#
# Compares **native backend compose** vs. **fuse-augmentations** across
# Albumentations, Kornia, and TorchVision -- 10 sequences x 3 backends x 2 modes
# = **60 pipeline variants**.
#
# fuse-aug merges consecutive fusible transforms (affine, flip, colour) into a
# single `grid_sample` or matrix-multiply call where possible, reducing the number
# of warp operations.  Single-op baselines (sequences 06-10) show the raw wrapper
# overhead when no fusion is possible.
#
# **Sequences** live in `SEQUENCE_BANK` — keys follow `<group>_<number>_<name>` so
# `sorted()` gives the display order.  Group `a` = single-op baselines,
# `b` = geometric chains, `c` = colour chains, `d` = realistic mixed (geo + colour).
#
# **Visual sanity check** (cell 3): native and fused rows use the same RNG seed so
# they draw identical random parameters.  The `max|native-fused|` annotation in
# each subplot confirms equivalence — any nonzero diff is interpolation error only.

# %% ── setup  Install dependencies (run once) ────────────────────────────────
# !pip install -e ".[all,benchmark]" matplotlib

# %% ── 0  Imports and configuration ──────────────────────────────────────────
from __future__ import annotations

import copy
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as tv
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from fuse_aug import Compose as FuseCompose
from fuse_aug import ReorderPolicy

log = logging.getLogger(__name__)

NUM_WARMUP: int = 20
NUM_REPEATS: int = 100
IMAGE_HEIGHT: int = 256
IMAGE_WIDTH: int = 256

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Torch {torch.__version__} | image: {IMAGE_HEIGHT}x{IMAGE_WIDTH} | warmup={NUM_WARMUP} repeats={NUM_REPEATS}")

# %% [markdown]
# ## 1 — Synthetic scene image
#
# White background with a red grid and four coloured shapes (one per quadrant):
# - top-left = blue triangle
# - top-right = green circle
# - bottom-left = orange rectangle
# - bottom-right = purple star
#
# The scene exercises both colour and spatial augmentations clearly.

# %% ── 1  Synthetic scene image ───────────────────────────────────────────────


def _draw_triangle(image: np.ndarray, row0: int, row1: int, col0: int, col1: int, pad: int, color: np.ndarray) -> None:
    """Fill a triangle (apex top-centre, base at bottom) into a quadrant of ``image``."""
    apex_row, apex_col = row0 + pad, (col0 + col1) // 2
    base_row = row1 - pad
    base_col0, base_col1 = col0 + pad, col1 - pad
    for row in range(apex_row, base_row + 1):
        progress = (row - apex_row) / max(base_row - apex_row, 1)
        left = int(apex_col + progress * (base_col0 - apex_col))
        right = int(apex_col + progress * (base_col1 - apex_col))
        image[row, left : right + 1] = color


def _draw_circle(image: np.ndarray, row0: int, row1: int, col0: int, col1: int, pad: int, color: np.ndarray) -> None:
    """Fill a circle centred in a quadrant of ``image``."""
    center_row, center_col = (row0 + row1) // 2, (col0 + col1) // 2
    radius = min(row1 - row0, col1 - col0) // 2 - pad
    rows_grid, cols_grid = np.ogrid[row0:row1, col0:col1]
    mask = (rows_grid - center_row) ** 2 + (cols_grid - center_col) ** 2 <= radius**2
    image[row0:row1, col0:col1][mask] = color


def _draw_rectangle(image: np.ndarray, row0: int, row1: int, col0: int, col1: int, pad: int, color: np.ndarray) -> None:
    """Fill a rectangle (inset by extra padding) into a quadrant of ``image``."""
    rect_pad = pad + pad // 2
    image[row0 + rect_pad : row1 - rect_pad, col0 + rect_pad : col1 - rect_pad] = color


def _draw_star(image: np.ndarray, row0: int, row1: int, col0: int, col1: int, pad: int, color: np.ndarray) -> None:
    """Fill a 5-pointed star centred in a quadrant of ``image`` using scanline fill."""
    center_row, center_col = (row0 + row1) // 2, (col0 + col1) // 2
    outer_radius = min(row1 - row0, col1 - col0) // 2 - pad
    inner_radius = outer_radius // 2
    num_points = 5
    angles_outer = [np.pi / 2 + 2 * np.pi * idx / num_points for idx in range(num_points)]
    angles_inner = [angle + np.pi / num_points for angle in angles_outer]
    all_cols = [center_col + int(outer_radius * np.cos(angle)) for angle in angles_outer] + [
        center_col + int(inner_radius * np.cos(angle)) for angle in angles_inner
    ]
    all_rows = [center_row - int(outer_radius * np.sin(angle)) for angle in angles_outer] + [
        center_row - int(inner_radius * np.sin(angle)) for angle in angles_inner
    ]
    point_order = [idx for idx_pt in range(num_points) for idx in (idx_pt, idx_pt + num_points)]
    poly_rows = [all_rows[idx] for idx in point_order]
    poly_cols = [all_cols[idx] for idx in point_order]
    min_row, max_row = min(poly_rows), max(poly_rows)
    num_poly_points = len(poly_rows)
    for row in range(max(row0, min_row), min(row1, max_row + 1)):
        x_intersects = []
        for idx in range(num_poly_points):
            row_a, row_b = poly_rows[idx], poly_rows[(idx + 1) % num_poly_points]
            col_a, col_b = poly_cols[idx], poly_cols[(idx + 1) % num_poly_points]
            if (row_a <= row < row_b) or (row_b <= row < row_a):
                x_intersects.append(int(col_a + (row - row_a) / (row_b - row_a) * (col_b - col_a)))
        if len(x_intersects) >= 2:
            x_intersects.sort()
            image[row, x_intersects[0] : x_intersects[-1] + 1] = color


def _synthetic_scene(width: int, height: int) -> np.ndarray:
    """Render a synthetic test scene as an HWC ``uint8`` RGB image.

    White background, red grid lines dividing the image into four quadrants,
    one coloured shape per quadrant: blue triangle (top-left), green circle
    (top-right), orange rectangle (bottom-left), purple star (bottom-right).

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        NumPy array of shape ``(height, width, 3)`` with ``dtype=uint8``.

    """
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    image[center_y - 1 : center_y + 1, :] = [220, 20, 20]  # horizontal grid line
    image[:, center_x - 1 : center_x + 1] = [220, 20, 20]  # vertical grid line

    pad = width // 16
    _draw_triangle(image, 0, center_y, 0, center_x, pad, np.array([30, 100, 220], dtype=np.uint8))
    _draw_circle(image, 0, center_y, center_x, width, pad, np.array([40, 180, 60], dtype=np.uint8))
    _draw_rectangle(image, center_y, height, 0, center_x, pad, np.array([230, 130, 20], dtype=np.uint8))
    _draw_star(image, center_y, height, center_x, width, pad, np.array([140, 40, 200], dtype=np.uint8))
    return image


torch.manual_seed(42)
image_np: np.ndarray = _synthetic_scene(IMAGE_WIDTH, IMAGE_HEIGHT)  # HWC uint8
image_tensor: torch.Tensor = (
    torch.from_numpy(image_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
)  # (1, C, H, W) float32

print(
    f"Scene: {image_np.shape} uint8 | tensor: {tuple(image_tensor.shape)} float32"
    f" [{image_tensor.min():.3f}, {image_tensor.max():.3f}]"
)

# %% [markdown]
# ## 2 — Pipeline definitions
#
# For each (sequence, backend) pair we register both a *native* and a *fused* runner.
# Runners are zero-argument callables that apply the pipeline to the pre-allocated image.
#
# - **Native**: backend's own sequential compose class (`K.AugmentationSequential` /
#   `tv.Compose` / `A.Compose`) — each transform runs its own `grid_sample` / per-pixel operation.
# - **Fused**: `fuse_aug.Compose` wrapping the same transforms — consecutive fusible ops
#   are grouped and their affine matrices composed before a single `grid_sample` call.

# %% ── 2  Pipeline definitions ────────────────────────────────────────────────

_ENTRIES: list[dict] = []  # populated by _register() below


def _register(seq: str, backend: str, native_pipe, fused_pipe, *, albu_native: bool = False) -> None:
    """Add native + fused runners for one (sequence, backend) pair.

    Args:
        seq: Sequence name (e.g. ``"geo_3"``).
        backend: One of ``"kornia"``, ``"torchvision"``, ``"albumentations"``.
        native_pipe: Instantiated native compose pipeline.
        fused_pipe: Instantiated ``FuseCompose`` pipeline (wrapping same transforms).
        albu_native: When ``True``, the native runner feeds ``image_np`` (HWC uint8) to
            ``native_pipe`` via the Albumentations keyword API.

    """
    if albu_native:

        def run_native(_p=native_pipe):
            return _p(image=image_np)["image"]

    else:

        def run_native(_p=native_pipe, _img=image_tensor):
            return _p(_img)

    if albu_native:

        def run_fused(_p=fused_pipe):
            return _p(image=image_np)["image"]

    else:

        def run_fused(_p=fused_pipe, _img=image_tensor):
            return _p(_img)

    # One warm-up call to populate fusion_plan / n_warps_saved.
    try:
        run_fused()
        fused_meta = {
            "fusion_plan": str(fused_pipe.fusion_plan),
            "n_warps_saved": int(fused_pipe.n_warps_saved),
        }
    except Exception as exc:
        print(f"  ⚠ {seq}/{backend}/fused metadata unavailable: {exc}")
        fused_meta = {}

    _ENTRIES.append({"seq": seq, "backend": backend, "mode": "native", "runner": run_native, "meta": {}})
    _ENTRIES.append({"seq": seq, "backend": backend, "mode": "fused", "runner": run_fused, "meta": fused_meta})


# ── AugSequence dataclass ────────────────────────────────────────────────────


@dataclass
class AugSequence:
    """Parallel transform lists for each backend, an optional reorder policy, and a display label.

    Each ``aug_*`` field holds the transforms for one backend in the same logical order.
    The benchmark loop builds native and fused pipelines from these lists
    automatically — no per-backend boilerplate needed.

    Args:
        aug_albu: Albumentations transforms.
        aug_kornia: Kornia transforms.
        aug_tv: TorchVision transforms.
        reorder: Reorder policy passed to ``FuseCompose`` (default: no reorder).
        label: Human-readable description shown in figure titles and tables.

    """

    aug_albu: list
    aug_kornia: list
    aug_tv: list
    reorder: ReorderPolicy | None = None
    label: str = ""


# ── d-group ordered pool — interleaved geo/colour, 9 ops ─────────────────────
# Alternating geo/colour is the worst case for naive fusion: every colour op
# breaks the consecutive-geo run, so reordering has maximum effect on this group.
# Strict-prefix slices keep shorter chains comparable to longer ones.
_D_POOL_ALB: list = [
    A.Rotate(limit=15, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
    A.VerticalFlip(p=0.5),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0, p=1.0),
    A.Rotate(limit=10, p=1.0),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=1.0),
    A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}, p=1.0),
]
_D_POOL_K: list = [
    K.RandomRotation(15.0),
    K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
    K.RandomVerticalFlip(p=0.5),
    K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
    K.RandomRotation(10.0),
    K.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.05, p=1.0),
    K.RandomAffine(degrees=0, shear=(-5.0, 5.0)),
]
_D_POOL_TV: list = [
    tv.RandomRotation(15),
    tv.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0),
    tv.RandomHorizontalFlip(0.5),
    tv.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.0, hue=0.0),
    tv.RandomVerticalFlip(0.5),
    tv.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.0),
    tv.RandomRotation(10),
    tv.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.05),
    tv.RandomAffine(degrees=0, shear=5),
]

# ── Sequence bank ─────────────────────────────────────────────────────────────
# Each entry is one benchmark sequence.  Add or remove rows here to change what
# gets benchmarked — the registration loop below handles the rest.

SEQUENCE_BANK: dict[str, AugSequence] = {
    # Keys follow "<group>_<number>_<name>" so sorted() gives the intended display order.
    # Group letters: a = single-op baselines,  b = geometric sequences,
    #                c = colour sequences,      d = realistic mixed (geo + colour).
    # ── a: single-op baselines (no fusion possible; measures wrapper overhead) ──
    "a01_rotate": AugSequence(
        aug_albu=[A.Rotate(limit=30, p=1.0)],
        aug_kornia=[K.RandomRotation(30.0)],
        aug_tv=[tv.RandomRotation(30)],
        label="Rotate  [single op]",
    ),
    "a02_hflip": AugSequence(
        aug_albu=[A.HorizontalFlip(p=1.0)],
        aug_kornia=[K.RandomHorizontalFlip(p=1.0)],
        aug_tv=[tv.RandomHorizontalFlip(1.0)],
        label="HorizontalFlip  [single op]",
    ),
    "a03_vflip": AugSequence(
        aug_albu=[A.VerticalFlip(p=1.0)],
        aug_kornia=[K.RandomVerticalFlip(p=1.0)],
        aug_tv=[tv.RandomVerticalFlip(1.0)],
        label="VerticalFlip  [single op]",
    ),
    "a04_scale": AugSequence(
        aug_albu=[A.Affine(scale=(0.8, 1.2), p=1.0)],
        aug_kornia=[K.RandomAffine(0, scale=(0.8, 1.2))],
        aug_tv=[tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
        label="Affine(scale)  [single op]",
    ),
    "a05_shear": AugSequence(
        aug_albu=[A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0)],
        aug_kornia=[K.RandomAffine(0, shear=(-10.0, 10.0))],
        aug_tv=[tv.RandomAffine(degrees=0, shear=10)],
        label="Affine(shear)  [single op]",
    ),
    # ── b: geometric sequences (2-5 ops) ──────────────────────────────────────
    "b01_geom_2": AugSequence(
        aug_albu=[A.Rotate(limit=30, p=1.0), A.HorizontalFlip(p=0.5)],
        aug_kornia=[K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5)],
        aug_tv=[tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5)],
        label="Rotate + HFlip",
    ),
    "b02_geom_3": AugSequence(
        aug_albu=[A.Rotate(limit=30), A.HorizontalFlip(p=0.5), A.Affine(scale=(0.8, 1.2))],
        aug_kornia=[K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5), K.RandomAffine(0, scale=(0.8, 1.2))],
        aug_tv=[tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5), tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
        label="Rotate + HFlip + Scale",
    ),
    "b03_geom_4": AugSequence(
        aug_albu=[
            A.Rotate(limit=10),
            A.HorizontalFlip(p=0.5),
            A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}),
            A.VerticalFlip(p=0.5),
        ],
        aug_kornia=[
            K.RandomRotation(10.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(0, shear=(-5.0, 5.0)),
            K.RandomVerticalFlip(p=0.5),
        ],
        aug_tv=[
            tv.RandomRotation(10),
            tv.RandomHorizontalFlip(0.5),
            tv.RandomAffine(degrees=0, shear=5),
            tv.RandomVerticalFlip(0.5),
        ],
        label="Rotate + HFlip + Shear + VFlip",
    ),
    "b04_geom_5": AugSequence(
        aug_albu=[
            A.Rotate(limit=10),
            A.HorizontalFlip(p=0.5),
            A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=5),
        ],
        aug_kornia=[
            K.RandomRotation(10.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(0, shear=(-5.0, 5.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(5.0),
        ],
        aug_tv=[
            tv.RandomRotation(10),
            tv.RandomHorizontalFlip(0.5),
            tv.RandomAffine(degrees=0, shear=5),
            tv.RandomVerticalFlip(0.5),
            tv.RandomRotation(5),
        ],
        label="Rotate + HFlip + Shear + VFlip + Rotate",
    ),
    # ── b05: 5 pure-warp ops (no flips) — demonstrates fixed-cost architecture ─
    "b05_geom_5_warp": AugSequence(
        aug_albu=[
            A.Rotate(limit=30, p=1.0),
            A.Affine(scale=(0.8, 1.2), p=1.0),
            A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0),
            A.Rotate(limit=15, p=1.0),
        ],
        aug_kornia=[
            K.RandomRotation(30.0, p=1.0),
            K.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
            K.RandomAffine(0, shear=(-10.0, 10.0), p=1.0),
            K.RandomAffine(0, translate=(0.1, 0.1), p=1.0),
            K.RandomRotation(15.0, p=1.0),
        ],
        aug_tv=[
            tv.RandomRotation(30),
            tv.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            tv.RandomAffine(degrees=0, shear=10),
            tv.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            tv.RandomRotation(15),
        ],
        label="Rotate + Scale + Shear + Translate + Rotate [5 warps]",
    ),
    # ── c: colour sequences (2-4 ops) ─────────────────────────────────────────
    "c01_color_2": AugSequence(
        aug_albu=[
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
        ],
        aug_kornia=[
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
        ],
        aug_tv=[
            tv.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.0, hue=0.0),
        ],
        label="Brightness + Contrast",
    ),
    "c02_color_3": AugSequence(
        aug_albu=[
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0, p=1.0),
        ],
        aug_kornia=[
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
            K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
        ],
        aug_tv=[
            tv.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.0, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.0),
        ],
        label="Brightness + Contrast + Saturation",
    ),
    "c03_color_4": AugSequence(
        aug_albu=[
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=1.0),
        ],
        aug_kornia=[
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
            K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
            K.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.05, p=1.0),
        ],
        aug_tv=[
            tv.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.2, saturation=0.0, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.0),
            tv.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.05),
        ],
        label="Brightness + Contrast + Saturation + Hue",
    ),
    # ── d: realistic mixed (geo + colour), 5-9 ops ────────────────────────────
    # Strict-prefix slices of _D_POOL_* so each N-op chain is a sub-sequence of
    # the (N+1)-op chain — isolating the added op's contribution across lengths.
    # POINTWISE and AGGRESSIVE reorder variants are registered by the sweep loop
    # below, keeping SEQUENCE_BANK free of duplicated transform definitions.
    **{
        f"d0{i}_mixed_g{(n + 1) // 2}c{n // 2}": AugSequence(
            aug_albu=_D_POOL_ALB[:n],
            aug_kornia=_D_POOL_K[:n],
            aug_tv=_D_POOL_TV[:n],
            label=f"{n}-op geo+colour ({(n + 1) // 2}G/{n // 2}C)",
        )
        for i, n in enumerate(range(5, 10), start=1)
    },
}


def _register_seq(seq_name: str, aug_seq: AugSequence, *, reorder: ReorderPolicy | None = None) -> None:
    """Register all three backend variants for one (sequence, reorder-policy) pair.

    Args:
        seq_name: Key stored in every entry; callers append a policy suffix for
            reorder variants so they appear as separate rows in results.
        aug_seq: Transform lists and optional base reorder policy.
        reorder: Explicit policy override; falls back to ``aug_seq.reorder`` when
            ``None`` so sequences with a built-in policy still use it.

    """
    effective_reorder = reorder if reorder is not None else aug_seq.reorder
    for backend, transforms, is_albu in [
        ("albumentations", aug_seq.aug_albu, True),
        ("kornia", aug_seq.aug_kornia, False),
        ("torchvision", aug_seq.aug_tv, False),
    ]:
        try:
            native_tfms = copy.deepcopy(transforms)
            if is_albu:
                native = A.Compose(native_tfms)
            elif backend == "kornia":
                native = K.AugmentationSequential(*native_tfms)
            else:
                native = tv.Compose(native_tfms)
            fused_kwargs = {"reorder": effective_reorder} if effective_reorder is not None else {}
            fused = FuseCompose(copy.deepcopy(transforms), **fused_kwargs)
            _register(seq_name, backend, native, fused, albu_native=is_albu)
        except Exception as exc:  # noqa: PERF203
            print(f"  ⚠ {seq_name}/{backend} skipped: {exc}")


# ── Registration ─────────────────────────────────────────────────────────────
# The interleaved geo/colour pattern is the worst case for naive fusion — every
# colour op breaks a consecutive geo segment.  Sweeping all non-default policies
# shows how much each reorder strategy recovers, using the same pipeline each time.
# Keys get a policy suffix so variants sort alongside their NONE baseline.
_D_REORDER_POLICIES: list[tuple[str, ReorderPolicy]] = [
    ("pw", ReorderPolicy.POINTWISE),
    ("agr", ReorderPolicy.AGGRESSIVE),
]
_d_keys = [k for k in SEQUENCE_BANK if k.startswith("d")]

for _seq_name, _aug_seq in SEQUENCE_BANK.items():
    _register_seq(_seq_name, _aug_seq)

for _policy_name, _policy in _D_REORDER_POLICIES:
    for _seq_name in _d_keys:
        _register_seq(f"{_seq_name}__{_policy_name}", SEQUENCE_BANK[_seq_name], reorder=_policy)


class SeededRunner:
    """Zero-argument callable wrapper that resets both RNGs to a fixed seed before every call.

    Wrapping the native and fused runners for the same (sequence, backend) pair with
    identical seeds guarantees that both pipelines draw **identical random parameters**
    on each forward pass.  Visual differences then come only from interpolation quality,
    not from different random draws.

    Args:
        runner: The zero-argument callable to wrap (typically a pipeline runner closure).
        seed: Integer seed applied to both ``torch.manual_seed`` and ``np.random.seed``
            before each invocation.  Must be in ``[0, 2**32 - 1]`` for NumPy
            compatibility.

    Notes:
        * Albumentations transforms draw from **numpy random**; kornia / torchvision
          draw from **torch RNG**.  Resetting both ensures all backends are covered.
        * This wrapper is used in the visual sanity-check cell only.  The benchmark
          cell intentionally omits seeding so that wall-clock timings reflect realistic,
          varied-parameter workloads rather than a fixed single-image scenario.

    """

    def __init__(self, runner, seed: int) -> None:
        self._runner = runner
        self._seed = int(seed) & 0xFFFF_FFFF  # clamp to uint32 for numpy compatibility

    def __call__(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        return self._runner()


print(f"\nRegistered {len(_ENTRIES)} pipeline variants across {len({e['seq'] for e in _ENTRIES})} sequences.")
for entry in _ENTRIES:
    plan = entry["meta"].get("fusion_plan", "")
    warps = entry["meta"].get("n_warps_saved", "")
    suffix = f"  →  {plan}  [{warps} warps saved]" if plan else ""
    print(f"  {entry['seq']:25s} {entry['backend']:16s} {entry['mode']:6s}{suffix}")

# %% [markdown]
# ## 3 — Visual sanity check
#
# One 2x3 figure per augmentation sequence.
# Top row = native compose output, bottom row = fused output.
# Columns = albumentations | kornia | torchvision.

# %% ── 3  Visual sanity check ─────────────────────────────────────────────────

_BACKENDS_ORDER = ["albumentations", "kornia", "torchvision"]
_SEQS_ORDER = sorted(SEQUENCE_BANK)


def _to_hwc(img) -> np.ndarray:
    """Convert a tensor (B,C,H,W) or numpy HWC array to a displayable HWC float32 array."""
    if isinstance(img, torch.Tensor):
        return img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    # Check dtype explicitly — max()-based heuristic silently corrupts float arrays
    # that happen to contain any pixel above 1.0.
    src = np.asarray(img)
    arr = src.astype(np.float32)
    return arr / 255.0 if src.dtype == np.uint8 else arr


def _show_saved(path: Path) -> None:
    """Close the active matplotlib figure and display the saved PNG in Jupyter.

    In a plain Python script context this is a silent no-op — the PNG is
    already on disk.  In a Jupyter notebook (``ipynb`` via jupytext) the saved
    file is rendered inline via ``IPython.display.Image`` so the notebook
    always shows the raster result rather than a re-drawn matplotlib canvas.

    Args:
        path: Path to the PNG file that was just saved with ``plt.savefig``.

    """
    plt.close("all")
    try:
        from IPython.display import Image as _IPImage
        from IPython.display import display as _ipy_display

        _ipy_display(_IPImage(str(path)))
    except ImportError:
        pass  # running as a plain script — PNG saved, nothing to display


# Index entries for quick lookup: (seq, backend, mode) -> runner
_runner_idx = {(e["seq"], e["backend"], e["mode"]): e["runner"] for e in _ENTRIES}

for seq in _SEQS_ORDER:
    backends_present = [b for b in _BACKENDS_ORDER if (seq, b, "native") in _runner_idx]
    if not backends_present:
        continue

    n_cols = len(backends_present)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7), squeeze=False)
    fig.suptitle(f"{seq}:  {SEQUENCE_BANK[seq].label or seq}", fontsize=10, fontweight="bold")

    # Store native HWC arrays so the fused row can show |native - fused| diff.
    # Both rows use the same per-(seq, backend) seed via SeededRunner, so they
    # draw **identical random parameters** — the diff measures only interpolation
    # quality, not randomness.
    _native_arr: dict[str, np.ndarray] = {}

    for col, backend in enumerate(backends_present):
        # Stable per-(seq, backend) seed — not Python hash() which is
        # PYTHONHASHSEED-randomised.  SeededRunner resets both torch and
        # numpy RNGs before every call, so native and fused draw identical
        # random parameters.
        _vis_seed = int.from_bytes((seq + backend).encode(), "little") & 0xFFFF_FFFF

        for row, mode in enumerate(["native", "fused"]):
            runner = _runner_idx.get((seq, backend, mode))
            ax = axes[row, col]
            ax.axis("off")
            if runner is None:
                ax.set_title(f"{backend}\n{mode}\n(N/A)", fontsize=7)
                continue
            arr = _to_hwc(SeededRunner(runner, _vis_seed)())
            ax.imshow(arr)

            title_extra = ""
            if mode == "native":
                _native_arr[backend] = arr
            else:
                # fused row: show warps saved + max pixel diff vs. native.
                # A small diff (< ~0.03) confirms the pipelines are equivalent:
                # same random params, difference is interpolation only.
                entry_meta = next(
                    (
                        e["meta"]
                        for e in _ENTRIES
                        if e["seq"] == seq and e["backend"] == backend and e["mode"] == "fused"
                    ),
                    {},
                )
                w = entry_meta.get("n_warps_saved")
                warps_str = f"\n({w} warps saved)" if w is not None else ""
                if backend in _native_arr and arr.shape == _native_arr[backend].shape:
                    max_diff = float(np.abs(arr - _native_arr[backend]).max())
                    title_extra = f"{warps_str}\nmax|native-fused|={max_diff:.3f}"
                else:
                    title_extra = warps_str
            ax.set_title(f"{backend} / {mode}{title_extra}", fontsize=7)

    # Row labels
    for row, label in enumerate(["native", "fused"]):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"visual_{seq}.png"
    plt.savefig(out_path, dpi=96, bbox_inches="tight")
    _show_saved(out_path)  # close fig + show saved PNG inline when running as .ipynb
    print(f"  Saved {out_path.name}")

# %% ── 4  Benchmark function ──────────────────────────────────────────────────


def benchmark(runner, num_warmup: int = NUM_WARMUP, num_repeats: int = NUM_REPEATS) -> tuple[dict, list]:
    """Time a zero-argument callable with warmup runs.

    Args:
        runner: Zero-argument callable to benchmark (should apply a pipeline to image).
        num_warmup: Number of warm-up calls (excluded from timing).
        num_repeats: Number of timed calls.

    Returns:
        A ``(stats, raw_times_ms)`` tuple where ``stats`` is a dict with keys
        ``mean``, ``std``, ``min``, ``max``, ``median`` (all in milliseconds) and
        ``raw_times_ms`` is the list of per-iteration timings.

    """
    for _ in range(num_warmup):
        runner()
    times_ms: list[float] = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        runner()
        times_ms.append((time.perf_counter() - t0) * 1_000)
    arr = np.asarray(times_ms)
    stats = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }
    return stats, times_ms


# %% ── 5  Run benchmarks ──────────────────────────────────────────────────────

print(f"\nRunning {len(_ENTRIES)} benchmarks  (warmup={NUM_WARMUP}, repeats={NUM_REPEATS}) …\n")
results: list[dict] = []

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    redirect_stdout=True,
) as _progress:
    _bench_task = _progress.add_task("Benchmarking", total=len(_ENTRIES))
    for entry in _ENTRIES:
        label = f"{entry['seq']:25s} {entry['backend']:16s} {entry['mode']:6s}"
        try:
            stats, raw = benchmark(entry["runner"])
            log.debug("  %s  %7.3f +/- %6.3f ms", label, stats["mean"], stats["std"])
            results.append({
                "sequence_name": entry["seq"],
                "backend": entry["backend"],
                "mode": entry["mode"],
                "timing_ms": stats,
                "raw_times_ms": raw,
                **entry["meta"],
            })
        except Exception as exc:
            log.debug("  %s  ERROR: %s", label, exc)
        _progress.advance(_bench_task)

# %% [markdown]
# ## 6 — Results table
#
# Wide format: rows = sequences, column groups = backend (alb | kornia | tv),
# each group has two sub-columns: native ms and fused ms.

# %% ── 6  Display results table ───────────────────────────────────────────────

_by_key: dict[tuple, dict] = {(r["sequence_name"], r["backend"], r["mode"]): r for r in results}
# Table order includes __pointwise / __aggressive variants; visual section uses only base sequences.
_TABLE_SEQS_ORDER = sorted({r["sequence_name"] for r in results})

_COL_BACKENDS = ["albumentations", "kornia", "torchvision"]
_COL_ABBREV = {"albumentations": "alb", "kornia": "kornia", "torchvision": "tv"}


def _boost_symbol(ratio: float) -> str:
    """Return a colorized rich-markup symbol indicating whether fusion is beneficial.

    Args:
        ratio: native_ms / fused_ms.  Values >1 mean fused is faster.

    Returns:
        "[green]v[/green]" when fused is faster (ratio > 1.0),
        "[blue]~[/blue]" when roughly equal (0.9 <= ratio <= 1.0),
        "[red]x[/red]" when fused is noticeably slower (ratio < 0.9).

    """
    if ratio > 1.0:
        return "[green]v[/green]"
    if ratio >= 0.9:
        return "[blue]~[/blue]"
    return "[red]x[/red]"


# ── build rich table ──────────────────────────────────────────────────────────
# Leftmost "Sequence" column, then one grouped column per backend with three
# sub-columns each: native (ms), fused (ms), boost (ratio + colorized symbol).
_table = Table(
    title="Augmentation pipeline benchmark",
    box=box.SIMPLE_HEAVY,
    show_header=True,
    header_style="bold",
)
_table.add_column("Sequence", style="bold", no_wrap=True, min_width=20)
for b in _COL_BACKENDS:
    _table.add_column("", style="dim", no_wrap=True, width=1)  # vertical divider before each group
    _table.add_column(f"[bold cyan]{_COL_ABBREV[b]}[/bold cyan]\n[dim]native[/dim]", justify="right", min_width=7)
    _table.add_column("\n[dim]fused[/dim]", justify="right", min_width=7)
    _table.add_column("\n[dim]boost[/dim]", justify="right", min_width=8)

# ── data rows ─────────────────────────────────────────────────────────────────
for seq in _TABLE_SEQS_ORDER:
    cells = [seq]
    for b in _COL_BACKENDS:
        cells.append("│")
        nat = _by_key.get((seq, b, "native"))
        fus = _by_key.get((seq, b, "fused"))
        if nat and fus:
            n_ms = nat["timing_ms"]["mean"]
            f_ms = fus["timing_ms"]["mean"]
            ratio = n_ms / f_ms if f_ms > 0 else None
            boost = f"{ratio:.2f} {_boost_symbol(ratio)}" if ratio is not None else "N/A"
            cells.extend([f"{n_ms:.3f}", f"{f_ms:.3f}", boost])
        else:
            cells.extend(["N/A", "N/A", "---"])
    _table.add_row(*cells)

# width=200 renders the full table regardless of actual terminal width — no truncation.
Console(width=200).print(_table)

print(f"Values in ms (mean over {NUM_REPEATS} repeats, batch_size=1).  boost = native/fused (>1 = fused faster).")
print("Note: alb native runs on HWC uint8 NumPy; all fused/kornia/tv run on BCHW float32 tensor.")

# %% ── 7  Save JSON ───────────────────────────────────────────────────────────


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


output = {
    "metadata": {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "image_shape": list(image_tensor.shape),
        "num_warmup": NUM_WARMUP,
        "num_repeats": NUM_REPEATS,
        "package_versions": {
            "fuse_augmentations": _pkg_version("fuse-augmentations"),
            "albumentations": _pkg_version("albumentations"),
            "kornia": _pkg_version("kornia"),
            "torchvision": _pkg_version("torchvision"),
        },
    },
    "results": results,
}

out_path = RESULTS_DIR / "benchmark_results.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nResults saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
