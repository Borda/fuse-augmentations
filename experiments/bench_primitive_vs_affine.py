# %% [markdown]
# # Primitive vs Affine Benchmark
#
# For each backend (**alb** / **kornia** / **tv**) and each geometric primitive, this
# notebook compares the **dedicated primitive transform** against the **backend's own
# generic Affine** doing the same spatial effect.
#
# | column | meaning |
# |--------|---------|
# | `prims` | dedicated primitive class (e.g. `A.Rotate`, `K.RandomHorizontalFlip`) |
# | `affine` | backend's generic Affine encoding the same spatial effect |
# | `ratio` | `affine / prims` — ≈ 1.0 means routing through Affine is free |
#
# ## Key questions
#
# **Q1 — Is `backend.Affine(op) ≈ backend.Primitive(op)`?**
# Yes → fuse-aug can route single ops through Affine at no N=1 penalty and saves
# N−1 warp calls for any chain of length N.
# No → fuse-aug keeps `ExactAffineSegment` shortcuts for that specific op.
#
# **Q2 — How much does fusing an N-op primitive chain into one Affine save?**
# The fusion section measures a realistic pipeline of N separate primitive calls vs one
# combined Affine — the exact saving fuse-aug delivers.
#
# ## Backends
#
# - **alb** — Albumentations, HWC uint8 NumPy (native format)
# - **kornia** — BCHW float32 tensor
# - **tv** — TorchVision v2, BCHW float32 tensor
#
# `—` in a cell = that backend has no dedicated primitive / cannot express the op via Affine.
#
# ## Usage
#
# ```bash
# python experiments/bench_primitive_vs_affine.py
# jupytext --to notebook experiments/bench_primitive_vs_affine.py
# jupyter lab experiments/bench_primitive_vs_affine.ipynb
# ```
#
# Results are saved to `experiments/results/bench_primitive_vs_affine.json`.

# %% ── setup  Install dependencies (run once) ────────────────────────────────
# !pip install -e ".[all,benchmark]"

# %% ── 0  Imports and configuration ──────────────────────────────────────────
# ruff: noqa: RUF001, RUF002, RUF003  -- multiplication sign used intentionally as ratio marker
from __future__ import annotations

import dataclasses
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import albumentations as A
import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tv

NUM_WARMUP: int = 20
NUM_REPEATS: int = 100
IMAGE_HEIGHT: int = 256
IMAGE_WIDTH: int = 256

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(
    f"Platform: {platform.system()} {platform.machine()}"
    f"  |  torch {torch.__version__}"
    f"  |  image {IMAGE_HEIGHT}×{IMAGE_WIDTH}"
    f"  |  warmup={NUM_WARMUP}  repeats={NUM_REPEATS}"
)

# %% [markdown]
# ## Input images
#
# Seeds are fixed so every transform draws the same random parameters on every run,
# ensuring a fair comparison across backends.
# Albumentations receives HWC uint8 NumPy arrays; Kornia and TorchVision receive
# BCHW float32 tensors.

# %% ── 1  Input images ────────────────────────────────────────────────────────

np.random.seed(0)
torch.manual_seed(0)

image_np: np.ndarray = np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
image_tensor: torch.Tensor = torch.rand(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)  # BCHW float32

print(f"alb input  : {image_np.shape}  {image_np.dtype}")
print(f"tensor input: {tuple(image_tensor.shape)}  {image_tensor.dtype}")

# %% ── 2  Helpers ─────────────────────────────────────────────────────────────


def _bench(
    func: Callable[[], object],
    n_warmup: int = NUM_WARMUP,
    n_repeats: int = NUM_REPEATS,
) -> float:
    """Return mean wall-clock time per call in milliseconds."""
    for _ in range(n_warmup):
        func()
    t_start = time.perf_counter()
    for _ in range(n_repeats):
        func()
    return (time.perf_counter() - t_start) * 1_000 / n_repeats


def _alb(transform: A.BasicTransform) -> Callable[[], object]:
    """Wrap an alb transform into a zero-arg callable on ``image_np``."""

    def _run(_transform: A.BasicTransform = transform) -> np.ndarray:
        return _transform(image=image_np)["image"]  # type: ignore[return-value]

    return _run


def _tensor(transform: object) -> Callable[[], object]:
    """Wrap a kornia/TV transform into a zero-arg callable on ``image_tensor``."""

    def _run(_transform: object = transform) -> torch.Tensor:
        return _transform(image_tensor)  # type: ignore[operator]

    return _run


def _is_notebook() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


# %% [markdown]
# ## Operation catalogue
#
# **`GEO_OPS`** — one row per geometric primitive.
# `*_prims` = dedicated primitive class; `*_affine` = generic Affine doing the same effect.
# `None` = backend has no such variant (shown as `—`).
#
# **`FUSION_OPS`** — realistic N-op chains grown by adding one primitive at a time:
# Rotate → +HFlip → +Shear → +Scale → +Translate → +VFlip (chains 2–6).
# Each operation appears exactly once per chain to avoid bias stacking.
# `*_prims` = N separate primitive calls; `*_affine` = single combined Affine (fuse-aug output).
# `k_affine` / `tv_affine` = `None` for all chains: `RandomAffine` cannot encode a flip.

# %% ── 3  Operation catalogue ─────────────────────────────────────────────────


@dataclasses.dataclass(kw_only=True)
class BenchOp:
    """One benchmark row: a named ``op`` with an optional callable per backend × kind.

    ``*_prims`` = dedicated primitive (or N×separate for fusion ops).
    ``*_affine`` = generic Affine doing the same effect (or 1×combined for fusion ops).
    ``None`` means that backend has no such variant — displayed as ``—`` in the table.
    """

    a_prims: Callable[[], object] | None = None
    a_affine: Callable[[], object] | None = None
    k_prims: Callable[[], object] | None = None
    k_affine: Callable[[], object] | None = None
    tv_prims: Callable[[], object] | None = None
    tv_affine: Callable[[], object] | None = None
    name: str


GEO_OPS: list[BenchOp] = [
    BenchOp(
        name="Rotate 30°",
        a_prims=_alb(A.Rotate(limit=30, p=1.0)),
        a_affine=_alb(A.Affine(rotate=(-30, 30), p=1.0)),
        k_prims=_tensor(K.RandomRotation(30.0)),
        k_affine=_tensor(K.RandomAffine(degrees=30.0)),
        tv_prims=_tensor(tv.RandomRotation(30)),
        tv_affine=_tensor(tv.RandomAffine(degrees=30)),
    ),
    BenchOp(
        name="SafeRotate 30°",  # pads before rotating — alb only
        a_prims=_alb(A.SafeRotate(limit=30, p=1.0)),
        a_affine=_alb(A.Affine(rotate=(-30, 30), p=1.0)),
    ),
    BenchOp(
        name="Rotate90",
        a_prims=_alb(A.RandomRotate90(p=1.0)),
        a_affine=_alb(A.Affine(rotate=90, p=1.0)),
        k_prims=_tensor(K.RandomRotation90(times=(0, 3), p=1.0)),
        k_affine=_tensor(K.RandomAffine(degrees=90.0)),
        tv_prims=_tensor(tv.RandomRotation(degrees=(90, 90))),
        tv_affine=_tensor(tv.RandomAffine(degrees=(90, 90))),
    ),
    BenchOp(
        name="HFlip",
        a_prims=_alb(A.HorizontalFlip(p=1.0)),
        a_affine=_alb(A.Affine(scale={"x": (-1.0, -1.0), "y": (1.0, 1.0)}, p=1.0)),
        k_prims=_tensor(K.RandomHorizontalFlip(p=1.0)),
        tv_prims=_tensor(tv.RandomHorizontalFlip(1.0)),
    ),
    BenchOp(
        name="VFlip",
        a_prims=_alb(A.VerticalFlip(p=1.0)),
        a_affine=_alb(A.Affine(scale={"x": (1.0, 1.0), "y": (-1.0, -1.0)}, p=1.0)),
        k_prims=_tensor(K.RandomVerticalFlip(p=1.0)),
        tv_prims=_tensor(tv.RandomVerticalFlip(1.0)),
    ),
    BenchOp(
        name="Transpose",  # swap H and W axes — alb only, no affine equivalent
        a_prims=_alb(A.Transpose(p=1.0)),
    ),
    BenchOp(
        name="D4",  # 8 orientations (4 rotations × 2 flips) — alb only
        a_prims=_alb(A.D4(p=1.0)),
    ),
    BenchOp(
        name="Translate",  # kornia: RandomTranslate primitive; alb/tv: no dedicated primitive
        a_affine=_alb(A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0)),
        k_prims=_tensor(K.RandomTranslate(translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), p=1.0)),
        k_affine=_tensor(K.RandomAffine(degrees=0, translate=(0.1, 0.1))),
        tv_affine=_tensor(tv.RandomAffine(degrees=0, translate=(0.1, 0.1))),
    ),
    BenchOp(
        name="Scale",  # alb: RandomScale primitive; kornia/tv: no dedicated primitive
        a_prims=_alb(A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0)),
        a_affine=_alb(A.Affine(scale=(0.8, 1.2), p=1.0)),
        k_affine=_tensor(K.RandomAffine(degrees=0, scale=(0.8, 1.2))),
        tv_affine=_tensor(tv.RandomAffine(degrees=0, scale=(0.8, 1.2))),
    ),
    BenchOp(
        name="Shear",  # no dedicated primitive in alb or tv
        a_affine=_alb(A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0)),
        k_prims=_tensor(K.RandomShear(shear=(-10.0, 10.0), p=1.0)),
        k_affine=_tensor(K.RandomAffine(degrees=0, shear=(-10.0, 10.0))),
        tv_affine=_tensor(tv.RandomAffine(degrees=0, shear=10.0)),
    ),
    BenchOp(
        name="ShiftScaleRotate",  # legacy combined primitive — alb only
        a_prims=_alb(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0)),
        a_affine=_alb(
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-30, 30),
                p=1.0,
            )
        ),
    ),
]

# Ordered primitive pools — non-flip ops first so short chains stay flip-free.
# Chain N = pool[0..N-1]; cycles back to pool[0] when N exceeds pool size.
#
#   alb   (4): Rotate → RandomScale → VFlip → HFlip
#   kornia(6): Rotation → Shear → Translate → Rotation90 → VFlip → HFlip
#   tv    (3): Rotation → VFlip → HFlip  (only 1 non-flip dedicated primitive in tv)
#
# a_affine: single combined A.Affine — alb pool has no Rotation90, so all 6 chains expressible.
#   Rotations accumulate as wider range; each flip adds scale_x/y = -1.
# k_affine: chains 2–3 (Rotation+Shear / +Translate, no flip); chains 4+ = None.
# tv_affine: always None — VFlip appears at pool[1], present from chain 2 onwards.
FUSION_OPS: list[BenchOp] = [
    BenchOp(
        name="2-op chain",  # alb: Rotate+Scale | k: Rotation+Shear | tv: Rotation+VFlip
        a_prims=_alb(
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            ])
        ),
        a_affine=_alb(A.Affine(rotate=(-30, 30), scale=(0.8, 1.2), p=1.0)),
        k_prims=_tensor(
            K.AugmentationSequential(
                K.RandomRotation(30.0),
                K.RandomShear(shear=(-10.0, 10.0), p=1.0),
            )
        ),
        k_affine=_tensor(K.RandomAffine(degrees=30, shear=(-10.0, 10.0))),
        tv_prims=_tensor(
            tv.Compose([
                tv.RandomRotation(30),
                tv.RandomVerticalFlip(1.0),
            ])
        ),
    ),
    BenchOp(
        name="3-op chain",  # alb: +VFlip | k: +Translate | tv: +HFlip
        a_prims=_alb(
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                A.VerticalFlip(p=1.0),
            ])
        ),
        a_affine=_alb(A.Affine(rotate=(-30, 30), scale={"x": (0.8, 1.2), "y": (-1.2, -0.8)}, p=1.0)),
        k_prims=_tensor(
            K.AugmentationSequential(
                K.RandomRotation(30.0),
                K.RandomShear(shear=(-10.0, 10.0), p=1.0),
                K.RandomTranslate(translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), p=1.0),
            )
        ),
        k_affine=_tensor(K.RandomAffine(degrees=30, shear=(-10.0, 10.0), translate=(0.1, 0.1))),
        tv_prims=_tensor(
            tv.Compose([
                tv.RandomRotation(30),
                tv.RandomVerticalFlip(1.0),
                tv.RandomHorizontalFlip(1.0),
            ])
        ),
    ),
    BenchOp(
        name="4-op chain",  # alb: +HFlip | k: +Rotation90 | tv: +Rotation (cycle)
        a_prims=_alb(
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                A.VerticalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
            ])
        ),
        a_affine=_alb(A.Affine(rotate=(-30, 30), scale={"x": (-1.2, -0.8), "y": (-1.2, -0.8)}, p=1.0)),
        k_prims=_tensor(
            K.AugmentationSequential(
                K.RandomRotation(30.0),
                K.RandomShear(shear=(-10.0, 10.0), p=1.0),
                K.RandomTranslate(translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), p=1.0),
                K.RandomRotation90(times=(0, 3), p=1.0),
            )
        ),
        tv_prims=_tensor(
            tv.Compose([
                tv.RandomRotation(30),
                tv.RandomVerticalFlip(1.0),
                tv.RandomHorizontalFlip(1.0),
                tv.RandomRotation(30),  # cycle: pool[3 % 3 = 0]
            ])
        ),
    ),
    BenchOp(
        name="5-op chain",  # alb: +Rotate (cycle) | k: +VFlip | tv: +VFlip (cycle)
        a_prims=_alb(
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                A.VerticalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=30, p=1.0),  # cycle: pool[4 % 4 = 0]
            ])
        ),
        a_affine=_alb(A.Affine(rotate=(-60, 60), scale={"x": (-1.2, -0.8), "y": (-1.2, -0.8)}, p=1.0)),
        k_prims=_tensor(
            K.AugmentationSequential(
                K.RandomRotation(30.0),
                K.RandomShear(shear=(-10.0, 10.0), p=1.0),
                K.RandomTranslate(translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), p=1.0),
                K.RandomRotation90(times=(0, 3), p=1.0),
                K.RandomVerticalFlip(p=1.0),
            )
        ),
        tv_prims=_tensor(
            tv.Compose([
                tv.RandomRotation(30),
                tv.RandomVerticalFlip(1.0),
                tv.RandomHorizontalFlip(1.0),
                tv.RandomRotation(30),  # cycle: pool[3 % 3 = 0]
                tv.RandomVerticalFlip(1.0),  # cycle: pool[4 % 3 = 1]
            ])
        ),
    ),
    BenchOp(
        name="6-op chain",  # alb: +Scale (cycle) | k: +HFlip | tv: +HFlip (cycle)
        a_prims=_alb(
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                A.VerticalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=30, p=1.0),  # cycle: pool[4 % 4 = 0]
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),  # cycle: pool[5 % 4 = 1]
            ])
        ),
        a_affine=_alb(A.Affine(rotate=(-60, 60), scale={"x": (-1.44, -0.64), "y": (-1.44, -0.64)}, p=1.0)),
        k_prims=_tensor(
            K.AugmentationSequential(
                K.RandomRotation(30.0),
                K.RandomShear(shear=(-10.0, 10.0), p=1.0),
                K.RandomTranslate(translate_x=(-0.1, 0.1), translate_y=(-0.1, 0.1), p=1.0),
                K.RandomRotation90(times=(0, 3), p=1.0),
                K.RandomVerticalFlip(p=1.0),
                K.RandomHorizontalFlip(p=1.0),
            )
        ),
        tv_prims=_tensor(
            tv.Compose([
                tv.RandomRotation(30),
                tv.RandomVerticalFlip(1.0),
                tv.RandomHorizontalFlip(1.0),
                tv.RandomRotation(30),  # cycle: pool[3 % 3 = 0]
                tv.RandomVerticalFlip(1.0),  # cycle: pool[4 % 3 = 1]
                tv.RandomHorizontalFlip(1.0),  # cycle: pool[5 % 3 = 2]
            ])
        ),
    ),
]

# %% ── 4  Display helpers ──────────────────────────────────────────────────────

_SECTIONS: list[tuple[str, str]] = [
    ("ops", "Geometric primitives"),
    ("fusion", "Multi-op sequence"),
]
_BACKENDS: list[str] = ["alb", "kornia", "tv"]


def _section_pivot(section_key: str) -> pd.DataFrame | None:
    """Return a (op_name) x (backend, prim|affine) pivot for one section; None if empty."""
    rows = [r for r in results if r["section"] == section_key and r["backend"] in _BACKENDS]
    long = [
        {"op_name": r["op_name"], "backend": r["backend"], "kind": kind, "ms": r[col]}
        for r in rows
        for kind, col in (("prims", "prims_ms"), ("affine", "affine_ms"))
        if r[col] is not None
    ]
    if not long:
        return None
    return pd.DataFrame(long).pivot_table(index="op_name", columns=["backend", "kind"], values="ms", aggfunc="first")


def _show_combined_table(title: str) -> None:
    """Display all benchmark results in one unified table (notebook or terminal)."""
    if _is_notebook():
        _nb_combined(title)
    else:
        _term_combined(title)


def _nb_combined(title: str) -> None:
    """Notebook path: pandas Styler with (section, op_name) MultiIndex rows."""
    from IPython.display import display  # type: ignore[import-untyped]

    frames: list[pd.DataFrame] = []
    for section_key, section_label in _SECTIONS:
        df_s = _section_pivot(section_key)
        if df_s is None:
            continue
        df_s.index = pd.MultiIndex.from_tuples(
            [(section_label, op) for op in df_s.index], names=["section", "operation"]
        )
        frames.append(df_s)
    if not frames:
        return

    df = pd.concat(frames)
    bases = pd.MultiIndex.from_product([_BACKENDS, ["prims", "affine"]], names=["backend", "kind"])
    df = df.reindex(columns=bases)

    for backend in _BACKENDS:
        p_col, a_col = (backend, "prims"), (backend, "affine")
        if p_col in df.columns or a_col in df.columns:
            p_s, a_s = df.get(p_col), df.get(a_col)
            df[(backend, "ratio")] = (a_s / p_s) if (p_s is not None and a_s is not None) else float("nan")

    interleaved: list[tuple[str, str]] = [
        (backend, kind)
        for backend in _BACKENDS
        for kind in ("prims", "affine", "ratio")
        if (backend, kind) in df.columns
    ]

    df = df[interleaved].dropna(axis=1, how="all")
    formatters = {
        col: (lambda v, _k=kind: "—" if pd.isna(v) else f"{v:.2f}×")
        if kind == "ratio"
        else (lambda v, _k=kind: "—" if pd.isna(v) else f"{v:.3f}")
        for col in df.columns
        for _, kind in [col]
    }
    display(df.style.format(formatters).set_caption(title))


def _term_combined(title: str) -> None:
    """Terminal path: rich Table with section rows and add_section() separators."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for section_key, section_label in _SECTIONS:
            df_s = _section_pivot(section_key)
            if df_s is not None:
                print(f"\n{section_label}")
                print(df_s.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"))
        return

    n_data = len(_BACKENDS) * 3  # prim + affine + ratio per backend
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("", style="bold", no_wrap=True, min_width=18)

    prev_backend: str | None = None
    for backend in _BACKENDS:
        for kind in ("prims", "affine", "ratio"):
            if backend != prev_backend:
                header = f"[bold cyan]{backend}[/bold cyan]\n[dim]{kind}[/dim]"
                prev_backend = backend
            else:
                header = f"\n[dim]{kind}[/dim]"
            table.add_column(header, justify="right", min_width=5 if kind == "ratio" else 6)

    first = True
    for section_key, section_label in _SECTIONS:
        df_s = _section_pivot(section_key)
        if df_s is None:
            continue
        if not first:
            table.add_section()
        first = False
        table.add_row(f"[dim italic]{section_label}[/dim italic]", *[""] * n_data)
        for op in df_s.index:
            cells = [f"  {op}"]
            for backend in _BACKENDS:
                p_col, a_col = (backend, "prims"), (backend, "affine")
                p_raw = df_s.loc[op, p_col] if p_col in df_s.columns else None
                a_raw = df_s.loc[op, a_col] if a_col in df_s.columns else None
                p = float(p_raw) if p_raw is not None and pd.notna(p_raw) else None
                a = float(a_raw) if a_raw is not None and pd.notna(a_raw) else None
                ratio = (a / p) if (p and a) else None
                cells.append(f"{p:.3f}" if p is not None else "[dim]—[/dim]")
                cells.append(f"{a:.3f}" if a is not None else "[dim]—[/dim]")
                cells.append(f"[yellow]{ratio:.2f}×[/yellow]" if ratio is not None else "[dim]—[/dim]")
            table.add_row(*cells)

    # width=200 renders the full table regardless of actual terminal width — no truncation.
    Console(width=200).print(table)


# %% [markdown]
# ## Benchmark execution
#
# Each callable is warmed up `NUM_WARMUP` times, then timed over `NUM_REPEATS` calls.
# Wall-clock mean in milliseconds is stored per `(section, backend, op_name, kind)`.

# %% ── 5  Run benchmarks ──────────────────────────────────────────────────────

results: list[dict] = []


def _run_ops(section: str, ops: list[BenchOp]) -> None:
    """Bench every (backend, kind) pair in *ops* and append rows to ``results``."""
    for op in ops:
        for backend, native_fn, fused_fn in [
            ("alb", op.a_prims, op.a_affine),
            ("kornia", op.k_prims, op.k_affine),
            ("tv", op.tv_prims, op.tv_affine),
        ]:
            n = _bench(native_fn) if native_fn is not None else None
            f = _bench(fused_fn) if fused_fn is not None else None
            if n is not None or f is not None:
                results.append({
                    "section": section,
                    "backend": backend,
                    "op_name": op.name,
                    "prims_ms": n,
                    "affine_ms": f,
                })


_run_ops("ops", GEO_OPS)

_run_ops("fusion", FUSION_OPS)

_show_combined_table("Primitive vs Affine — all backends")

# %% [markdown]
# ## Summary
#
# **Rotate / Scale / Shear / Translate — ratio ≈ 1.0:**
# Routing single ops through Affine has negligible cost; fuse-aug gains come from
# saving N−1 warp calls for chains of length N.
#
# **HFlip / VFlip — alb ratio >> 1:**
# `alb.HFlip` is a zero-copy `np.flip` on uint8 HWC; `A.Affine(scale_x=-1)`
# uses `cv2.warpAffine` — much slower.  fuse-aug's `ExactAffineSegment` keeps
# `tensor.flip` on float32 BCHW, bypassing the full warp.
#
# **Fusion chains — alb combined Affine is faster than N separate primitives:**
# Even though individual primitives (e.g. HFlip) can be faster than their Affine
# equivalent, the combined-Affine path wins for chains ≥ 3 because it avoids N−1
# separate warp calls.  kornia/tv chains show prims-only cost (no combined Affine
# for flip chains) — use those rows to size the raw pipeline overhead.

# %% ── 6  Save results ────────────────────────────────────────────────────────

out_path = RESULTS_DIR / "bench_primitive_vs_affine.json"
out_path.write_text(
    json.dumps(
        {
            "meta": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "image": f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
                "warmup": NUM_WARMUP,
                "repeats": NUM_REPEATS,
            },
            "results": results,
        },
        indent=2,
    )
)
print()
print(f"Results → {out_path}")
