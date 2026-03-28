"""Benchmark augmentation pipelines: native backend compose vs. fuse-augmentations.

Compares wall-clock time per call for ten augmentation sequences across Albumentations,
Kornia, and TorchVision, running each pipeline in both *native* and *fused* mode
(60 variants total).  Sequences are defined in ``SEQUENCE_BANK`` as ``AugSequence``
dataclass entries — add or remove rows there to change what gets benchmarked.

Sequences
---------
Keys follow ``<group>_<number>_<name>`` so ``sorted(SEQUENCE_BANK)`` gives the intended
display order.  Group letters: ``a`` = single-op baselines, ``b`` = multi-op sequences.

b1_geo_3
    Rotate + HFlip + Affine(scale) — 3-op geometric chain, saves 2 ``grid_sample`` calls.
b2_color_2
    RandomBrightness + RandomContrast — colour fusion into one matrix multiply.
b3_mixed_interleaved
    Rotate -> ColorJitter -> HFlip — ``POINTWISE`` reorder policy bubbles the colour op
    after the geometric chain so both geometric ops fuse.
b4_geo_5
    Rotate -> HFlip -> Shear -> VFlip -> Rotate — 5-op chain, saves 4 ``grid_sample`` calls.
b5_realistic
    Rotate -> HFlip -> GaussianBlur -> BrightnessContrast -> Rotate — blur acts as a fusion
    barrier; fuse-aug produces two partial segments (partial fusion).
a1_rotate … a5_shear
    Single-op affine baselines (Rotate, HFlip, VFlip, Scale, Shear) — no fusion possible;
    these measure the raw fuse-aug wrapper overhead vs. native compose.

Usage
-----
Run as a script::

    python examples/bench_augmentation_pipelines.py

Open as a Jupyter notebook (requires jupytext)::

    jupytext --to notebook examples/bench_augmentation_pipelines.py
    jupyter lab examples/bench_augmentation_pipelines.ipynb

Results are saved to ``examples/results/benchmark_results.json``.
Visual sanity figures are saved to ``examples/results/visual_<seq>.png``.

Notes
-----
*  Albumentations **native** benchmarks operate on HWC ``uint8`` NumPy arrays — the format
   Albumentations expects natively.  All other pipelines (including fuse-aug wrapping
   Albumentations transforms) use BCHW ``float32`` tensors.  Timing is informative but not
   strictly apples-to-apples for Albumentations native vs. fused.
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
# group `b` = multi-op fusion sequences.  Add more groups as needed.
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
from tqdm.auto import tqdm

from fuse_aug import Compose as FuseCompose
from fuse_aug import ReorderPolicy

NUM_WARMUP: int = 5
NUM_REPEATS: int = 50
IMAGE_H: int = 256
IMAGE_W: int = 256

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Torch {torch.__version__} | image: {IMAGE_H}x{IMAGE_W} | warmup={NUM_WARMUP} repeats={NUM_REPEATS}")

# %% ── 1  Synthetic scene image ───────────────────────────────────────────────
#
# White background with a red grid and four coloured shapes (one per quadrant):
#   top-left    = blue triangle
#   top-right   = green circle
#   bottom-left = orange rectangle
#   bottom-right= purple star
# The scene exercises both colour and spatial augmentations clearly.


def _draw_triangle(img: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int, color: np.ndarray) -> None:
    """Fill a triangle (apex top-centre, base at bottom) into a quadrant of ``img``."""
    apex_r, apex_c = r0 + pad, (c0 + c1) // 2
    base_r = r1 - pad
    base_c0, base_c1 = c0 + pad, c1 - pad
    for row in range(apex_r, base_r + 1):
        t = (row - apex_r) / max(base_r - apex_r, 1)
        left = int(apex_c + t * (base_c0 - apex_c))
        right = int(apex_c + t * (base_c1 - apex_c))
        img[row, left : right + 1] = color


def _draw_circle(img: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int, color: np.ndarray) -> None:
    """Fill a circle centred in a quadrant of ``img``."""
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    radius = min(r1 - r0, c1 - c0) // 2 - pad
    rows_g, cols_g = np.ogrid[r0:r1, c0:c1]
    mask = (rows_g - cr) ** 2 + (cols_g - cc) ** 2 <= radius**2
    img[r0:r1, c0:c1][mask] = color


def _draw_rectangle(img: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int, color: np.ndarray) -> None:
    """Fill a rectangle (inset by extra padding) into a quadrant of ``img``."""
    rpad = pad + pad // 2
    img[r0 + rpad : r1 - rpad, c0 + rpad : c1 - rpad] = color


def _draw_star(img: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int, color: np.ndarray) -> None:
    """Fill a 5-pointed star centred in a quadrant of ``img`` using scanline fill."""
    sr, sc = (r0 + r1) // 2, (c0 + c1) // 2
    outer = min(r1 - r0, c1 - c0) // 2 - pad
    inner = outer // 2
    n_pts = 5
    angles_o = [np.pi / 2 + 2 * np.pi * k / n_pts for k in range(n_pts)]
    angles_i = [a + np.pi / n_pts for a in angles_o]
    all_cols = [sc + int(outer * np.cos(a)) for a in angles_o] + [sc + int(inner * np.cos(a)) for a in angles_i]
    all_rows = [sr - int(outer * np.sin(a)) for a in angles_o] + [sr - int(inner * np.sin(a)) for a in angles_i]
    order = [idx for k in range(n_pts) for idx in (k, k + n_pts)]
    poly_r = [all_rows[i] for i in order]
    poly_c = [all_cols[i] for i in order]
    min_r, max_r = min(poly_r), max(poly_r)
    n = len(poly_r)
    for row in range(max(r0, min_r), min(r1, max_r + 1)):
        xs = []
        for i in range(n):
            r_a, r_b = poly_r[i], poly_r[(i + 1) % n]
            c_a, c_b = poly_c[i], poly_c[(i + 1) % n]
            if (r_a <= row < r_b) or (r_b <= row < r_a):
                xs.append(int(c_a + (row - r_a) / (r_b - r_a) * (c_b - c_a)))
        if len(xs) >= 2:
            xs.sort()
            img[row, xs[0] : xs[-1] + 1] = color


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
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cy, cx = height // 2, width // 2
    img[cy - 1 : cy + 1, :] = [220, 20, 20]  # horizontal grid line
    img[:, cx - 1 : cx + 1] = [220, 20, 20]  # vertical grid line

    pad = width // 16
    _draw_triangle(img, 0, cy, 0, cx, pad, np.array([30, 100, 220], dtype=np.uint8))
    _draw_circle(img, 0, cy, cx, width, pad, np.array([40, 180, 60], dtype=np.uint8))
    _draw_rectangle(img, cy, height, 0, cx, pad, np.array([230, 130, 20], dtype=np.uint8))
    _draw_star(img, cy, height, cx, width, pad, np.array([140, 40, 200], dtype=np.uint8))
    return img


torch.manual_seed(42)
image_np: np.ndarray = _synthetic_scene(IMAGE_W, IMAGE_H)  # HWC uint8
image_tensor: torch.Tensor = (
    torch.from_numpy(image_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
)  # (1, C, H, W) float32

print(
    f"Scene: {image_np.shape} uint8 | tensor: {tuple(image_tensor.shape)} float32"
    f" [{image_tensor.min():.3f}, {image_tensor.max():.3f}]"
)

# %% ── 2  Pipeline definitions ────────────────────────────────────────────────
#
# For each (sequence, backend) pair we register both a *native* and a *fused* runner.
# Runners are zero-argument callables that apply the pipeline to the pre-allocated image.
#
# Native:  backend's own sequential compose class (K.AugmentationSequential / tv.Compose
#          / A.Compose) — each transform runs its own grid_sample / pixel op in turn.
# Fused:   fuse_aug.Compose wrapping the same transforms — consecutive fusible ops are
#          grouped and their affine matrices composed before a single grid_sample call.

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


# ── Sequence bank ─────────────────────────────────────────────────────────────
# Each entry is one benchmark sequence.  Add or remove rows here to change what
# gets benchmarked — the registration loop below handles the rest.

SEQUENCE_BANK: dict[str, AugSequence] = {
    # Keys follow "<group>_<number>_<name>" so sorted() gives the intended display order.
    # Group letters: a = single-op baselines, b = multi-op fusion sequences.
    # Add more groups (c, d, ...) for future categories.
    # ── b: multi-op fusion sequences ─────────────────────────────────────────
    "b01_geo_3": AugSequence(
        aug_albu=[A.Rotate(limit=30), A.HorizontalFlip(p=0.5), A.Affine(scale=(0.8, 1.2))],
        aug_kornia=[K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5), K.RandomAffine(0, scale=(0.8, 1.2))],
        aug_tv=[tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5), tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
        label="Rotate + HFlip + Affine(scale)",
    ),
    "b02_color_2": AugSequence(
        aug_albu=[A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)],
        aug_kornia=[K.RandomBrightness(brightness=(0.8, 1.2), p=1.0), K.RandomContrast(contrast=(0.8, 1.2), p=1.0)],
        aug_tv=[tv.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0)],
        label="RandomBrightness + RandomContrast",
    ),
    "b03_mixed_interleaved": AugSequence(
        aug_albu=[
            A.Rotate(limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HorizontalFlip(p=0.5),
        ],
        aug_kornia=[
            K.RandomRotation(15.0),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=1.0),
            K.RandomHorizontalFlip(p=0.5),
        ],
        aug_tv=[
            tv.RandomRotation(15),
            tv.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            tv.RandomHorizontalFlip(0.5),
        ],
        reorder=ReorderPolicy.POINTWISE,
        label="Rotate -> ColorJitter -> HFlip  [reorder policy]",
    ),
    "b04_geo_5": AugSequence(
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
        label="Rotate -> HFlip -> Shear -> VFlip -> Rotate",
    ),
    "b05_realistic": AugSequence(
        aug_albu=[
            A.Rotate(limit=15),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            A.Rotate(limit=5),
        ],
        aug_kornia=[
            K.RandomRotation(15.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomGaussianBlur((5, 5), (0.1, 2.0), p=1.0),
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            K.RandomRotation(5.0),
        ],
        aug_tv=[
            tv.RandomRotation(15),
            tv.RandomHorizontalFlip(0.5),
            tv.GaussianBlur(kernel_size=5),
            tv.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0),
            tv.RandomRotation(5),
        ],
        label="Rotate -> HFlip -> GaussianBlur -> BrightnessContrast -> Rotate",
    ),
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
}

# ── Registration loop ─────────────────────────────────────────────────────────
# For each (sequence, backend) pair: build a native compose and a FuseCompose,
# then register both as timed runners.  deepcopy ensures native and fused use
# independent transform instances even though both read from the same bank list.


for _seq_name, _aug_seq in tqdm(SEQUENCE_BANK.items(), desc="Registering", unit="seq"):
    for _backend, _transforms, _is_albu in [
        ("albumentations", _aug_seq.aug_albu, True),
        ("kornia", _aug_seq.aug_kornia, False),
        ("torchvision", _aug_seq.aug_tv, False),
    ]:
        try:
            _tfms_native = copy.deepcopy(_transforms)
            if _is_albu:
                _native = A.Compose(_tfms_native)
            elif _backend == "kornia":
                _native = K.AugmentationSequential(*_tfms_native)
            else:
                _native = tv.Compose(_tfms_native)
            _fused_kwargs = {"reorder": _aug_seq.reorder} if _aug_seq.reorder is not None else {}
            _fused = FuseCompose(_transforms, **_fused_kwargs)
            _register(_seq_name, _backend, _native, _fused, albu_native=_is_albu)
        except Exception as exc:  # noqa: PERF203
            print(f"  ⚠ {_seq_name}/{_backend} skipped: {exc}")


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

    Examples:
        >>> seeded = SeededRunner(lambda: None, seed=42)
        >>> seeded._seed
        42

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

# %% ── 3  Visual sanity check ─────────────────────────────────────────────────
# One 2x3 figure per augmentation sequence.
# Top row = native compose output, bottom row = fused output.
# Columns = albumentations | kornia | torchvision.

_BACKENDS_ORDER = ["albumentations", "kornia", "torchvision"]
# Sequence display order comes from sorted SEQUENCE_BANK keys (numeric prefix guarantees it).
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

for entry in tqdm(_ENTRIES, desc="Benchmarking", unit="pipeline"):
    label = f"{entry['seq']:25s} {entry['backend']:16s} {entry['mode']:6s}"
    try:
        stats, raw = benchmark(entry["runner"])
        tqdm.write(f"  {label}  {stats['mean']:7.3f} +/- {stats['std']:6.3f} ms")
        results.append({
            "sequence_name": entry["seq"],
            "backend": entry["backend"],
            "mode": entry["mode"],
            "timing_ms": stats,
            "raw_times_ms": raw,
            **entry["meta"],
        })
    except Exception as exc:
        tqdm.write(f"  {label}  ERROR: {exc}")

# %% ── 6  Display results table ───────────────────────────────────────────────
# Wide format: rows = sequences, column groups = backend (alb | kornia | tv),
# each group has two sub-columns: native ms and fused ms.

_by_key: dict[tuple, dict] = {(r["sequence_name"], r["backend"], r["mode"]): r for r in results}

_COL_BACKENDS = ["albumentations", "kornia", "torchvision"]
_COL_ABBREV = {"albumentations": "alb", "kornia": "kornia", "torchvision": "tv"}
_W_SEQ = 20  # sequence name column width
_W_VAL = 7  # ms value column width

# ── header ───────────────────────────────────────────────────────────────────
_sep = "─" * (_W_SEQ + 1 + len(_COL_BACKENDS) * (_W_VAL * 2 + 3))
print("\n" + _sep)

# row 1: backend group labels, centred over their two sub-columns
header1 = f"{'Sequence':<{_W_SEQ}}"
for b in _COL_BACKENDS:
    group_w = _W_VAL * 2 + 3  # "native" + " " + "fused" + spacing
    header1 += " " + f"{_COL_ABBREV[b]:^{group_w}}"
print(header1)

# row 2: sub-column labels
header2 = f"{'':^{_W_SEQ}}"
for _ in _COL_BACKENDS:
    header2 += f" {'native':>{_W_VAL}} {'fused':>{_W_VAL}}  "
print(header2)
print(_sep)

# ── data rows ─────────────────────────────────────────────────────────────────
for seq in _SEQS_ORDER:
    row = f"{seq:<{_W_SEQ}}"
    for b in _COL_BACKENDS:
        nat = _by_key.get((seq, b, "native"))
        fus = _by_key.get((seq, b, "fused"))
        if nat and fus:
            row += f" {nat['timing_ms']['mean']:>{_W_VAL}.2f} {fus['timing_ms']['mean']:>{_W_VAL}.2f}  "
        else:
            row += f" {'N/A':>{_W_VAL}} {'N/A':>{_W_VAL}}  "
    print(row)

print(_sep)
print("Values in ms (mean over 50 repeats, batch_size=1).")
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
