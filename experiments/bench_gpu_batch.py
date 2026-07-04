"""GPU/batch throughput benchmark for the fuse-augmentations pipeline.

Unlike :mod:`experiments.optimize_score` and
:mod:`experiments.bench_augmentation_pipelines` (both CPU-only, batch=1,
latency-only), this script sweeps **device x batch-size** and reports both
per-batch latency (median + p10/p90) *and* throughput (images/sec) for the
native vs. fused implementations of a representative set of sequences.

Devices are auto-detected: ``cpu`` is always benchmarked; ``cuda`` and ``mps``
are added when available. Timers are wrapped with the correct per-device
synchronisation (``torch.cuda.synchronize`` / ``torch.mps.synchronize``) so the
reported numbers reflect real device execution rather than async dispatch.

Backend I/O conventions match the sibling scripts:

- **Kornia / TorchVision** — BCHW float32 tensors on the target device. Native
  and fused both run the tensor path; batching is native.
- **Albumentations** — native ``A.Compose`` consumes HWC uint8 NumPy (CPU,
  per-image), so a batch of ``B`` is timed as ``B`` sequential calls. The fused
  Albu pipeline runs the same NumPy dict path on CPU; on ``cuda``/``mps`` its
  tensor path is *probed* and either benchmarked or skipped with a recorded
  reason.

Output:

- JSON to ``experiments/results/bench_gpu_batch_<platform>.json``.
- A printed Markdown summary table.

Usage::

    uv run python experiments/bench_gpu_batch.py            # full sweep
    uv run python experiments/bench_gpu_batch.py --quick    # fast smoke run

"""

from __future__ import annotations

import argparse
import copy
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
import torchvision.transforms.v2 as tv

from fuse_aug import Compose as FuseCompose
from fuse_aug import ReorderPolicy

IMAGE_HEIGHT: int = 256
IMAGE_WIDTH: int = 256
RESULTS_DIR: Path = Path(__file__).parent / "results"
QUICK_MAX_BATCH: int = 8  # --quick caps the batch sweep at this size

# Representative subset of the optimize_score.py bank: one single-op baseline,
# a 3-op and two 5-op geometric chains, and two mixed geo+colour sequences.
_GEO_LABELS: tuple[str, ...] = ("a01_rotate", "b02_geom_3", "b04_geom_5", "b05_geom_5_warp")
_MIXED_LABELS: tuple[str, ...] = ("d02_mixed_g3c3__agr", "d03_mixed_g4c3__agr")


@dataclass
class Sequence:
    """A benchmarkable sequence with per-backend transform lists."""

    label: str
    nb_geom: int
    kornia: list
    torchvision: list
    albumentations: list
    reorder: ReorderPolicy | None = None


class SkipCase(Exception):
    """Raised when a (backend, device, mode) combination cannot be benchmarked."""


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def _load_sequences() -> tuple[list[Sequence], str]:
    """Import the representative subset from optimize_score, else build inline.

    Returns:
        The selected sequences and a provenance string (``"imported"`` or
        ``"inline-fallback"``).

    Examples:
        >>> seqs, source = _load_sequences()
        >>> len(seqs) >= 6
        True

    """
    try:
        from optimize_score import _ALBU_CASES, _CASES, _MIXED_AGR_CASES
    except Exception:
        return _fallback_sequences(), "inline-fallback"

    albu_by_label = {label: tfms for label, _nb, tfms in _ALBU_CASES}
    sequences: list[Sequence] = []
    for label, nb_geom, k_tfms, tv_tfms in _CASES:
        if label in _GEO_LABELS:
            sequences.append(Sequence(label, nb_geom, k_tfms, tv_tfms, albu_by_label[label], reorder=None))
    for label, nb_geom, alb_tfms, k_tfms, tv_tfms in _MIXED_AGR_CASES:
        if label in _MIXED_LABELS:
            sequences.append(Sequence(label, nb_geom, k_tfms, tv_tfms, alb_tfms, reorder=ReorderPolicy.AGGRESSIVE))
    return sequences, "imported"


def _fallback_sequences() -> list[Sequence]:
    """Redefine ~6 representative cases inline when optimize_score is unavailable."""
    return [
        Sequence(
            "a01_rotate",
            1,
            [K.RandomRotation(30.0)],
            [tv.RandomRotation(30)],
            [A.Rotate(limit=30, p=1.0)],
        ),
        Sequence(
            "b02_geom_3",
            3,
            [K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5), K.RandomAffine(0, scale=(0.8, 1.2))],
            [tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5), tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
            [A.Rotate(limit=30), A.HorizontalFlip(p=0.5), A.Affine(scale=(0.8, 1.2))],
        ),
        Sequence(
            "b05_geom_5_warp",
            5,
            [
                K.RandomRotation(30.0, p=1.0),
                K.RandomAffine(0, scale=(0.8, 1.2), p=1.0),
                K.RandomAffine(0, shear=(-10.0, 10.0), p=1.0),
                K.RandomAffine(0, translate=(0.1, 0.1), p=1.0),
                K.RandomRotation(15.0, p=1.0),
            ],
            [
                tv.RandomRotation(30),
                tv.RandomAffine(degrees=0, scale=(0.8, 1.2)),
                tv.RandomAffine(degrees=0, shear=10),
                tv.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                tv.RandomRotation(15),
            ],
            [
                A.Rotate(limit=30, p=1.0),
                A.Affine(scale=(0.8, 1.2), p=1.0),
                A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0),
                A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0),
                A.Rotate(limit=15, p=1.0),
            ],
        ),
        Sequence(
            "d02_mixed_g3c3__agr",
            3,
            [
                K.RandomRotation(15.0),
                K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomContrast(contrast=(0.8, 1.2), p=1.0),
                K.RandomVerticalFlip(p=0.5),
                K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
            ],
            [
                tv.RandomRotation(15),
                tv.ColorJitter(brightness=0.2),
                tv.RandomHorizontalFlip(0.5),
                tv.ColorJitter(contrast=0.2),
                tv.RandomVerticalFlip(0.5),
                tv.ColorJitter(saturation=0.2),
            ],
            [
                A.Rotate(limit=15, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
                A.VerticalFlip(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=0, p=1.0),
            ],
            reorder=ReorderPolicy.AGGRESSIVE,
        ),
    ]


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
@dataclass
class Device:
    """A benchmark target device with its synchronisation hook."""

    name: str
    torch_device: torch.device
    sync: Callable[[], None]
    supports_peak_mem: bool = False


def _discover_devices(requested: list[str] | None) -> list[Device]:
    """Return the available devices, optionally filtered by ``requested`` names."""
    devices: list[Device] = [Device("cpu", torch.device("cpu"), lambda: None)]
    if torch.cuda.is_available():
        devices.append(Device("cuda", torch.device("cuda"), torch.cuda.synchronize, supports_peak_mem=True))
    if torch.backends.mps.is_available():
        devices.append(Device("mps", torch.device("mps"), torch.mps.synchronize))
    if requested:
        wanted = set(requested)
        devices = [d for d in devices if d.name in wanted]
    return devices


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _time_thunk(thunk: Callable[[], object], warmup: int, measure: int, sync: Callable[[], None]) -> list[float]:
    """Time ``thunk`` and return per-iteration wall times in ms (device-synced)."""
    for _ in range(warmup):
        thunk()
    sync()
    samples: list[float] = []
    for _ in range(measure):
        start = time.perf_counter()
        thunk()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summarise(samples: list[float], batch: int) -> dict[str, float]:
    """Reduce raw ms samples to median/p10/p90 latency and throughput (img/s)."""
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "p10_ms": float(np.percentile(samples, 10)),
        "p90_ms": float(np.percentile(samples, 90)),
        "throughput_img_s": (batch / median_ms * 1000.0) if median_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Thunk builders per backend
# ---------------------------------------------------------------------------
def _to_device(obj: object, device: torch.device) -> object:
    """Best-effort move of a transform/module to ``device`` (no-op if unsupported)."""
    mover = getattr(obj, "to", None)
    if callable(mover):
        try:
            return mover(device)
        except Exception:
            return obj
    return obj


def _build_fused(tfms: list, reorder: ReorderPolicy | None) -> FuseCompose:
    """Construct a fused pipeline, honouring the sequence reorder policy."""
    if reorder is None:
        return FuseCompose(copy.deepcopy(tfms))
    return FuseCompose(copy.deepcopy(tfms), reorder=reorder)


def _tensor_thunks(seq: Sequence, backend: str, device: Device, batch: int) -> tuple[Callable, Callable]:
    """Build (native, fused) zero-arg thunks for the Kornia/TorchVision tensor path.

    Thunks are returned unprobed: each mode is exercised independently by the
    caller's warmup loop, so a fused-only failure (e.g. the fused MPS matrix
    path) never suppresses the native measurement, and vice versa.

    """
    tfms = getattr(seq, backend)
    image = torch.rand(batch, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device.torch_device)
    if backend == "kornia":
        native = _to_device(K.AugmentationSequential(*copy.deepcopy(tfms)), device.torch_device)
    else:
        native = _to_device(tv.Compose(copy.deepcopy(tfms)), device.torch_device)
    fused = _to_device(_build_fused(tfms, seq.reorder), device.torch_device)

    def native_thunk() -> object:
        return native(image)

    def fused_thunk() -> object:
        return fused(image)

    return native_thunk, fused_thunk


def _albu_thunks(seq: Sequence, device: Device, batch: int) -> tuple[Callable, Callable]:
    """Build (native, fused) thunks for Albumentations on the given device.

    On CPU both modes use the NumPy dict path; a batch of ``B`` is timed as ``B``
    sequential calls. On non-CPU devices native Albu has no path (it raises
    :class:`SkipCase`), and the fused tensor path is exercised (and skipped with
    a recorded reason if unsupported) by the caller's warmup — not eagerly here,
    so it cannot suppress unrelated cases.

    """
    tfms = seq.albumentations
    if device.name == "cpu":
        native = A.Compose(copy.deepcopy(tfms))
        fused = _build_fused(tfms, seq.reorder)
        imgs = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) for _ in range(batch)]

        def native_thunk() -> object:
            return [native(image=im) for im in imgs]

        def fused_thunk() -> object:
            return [fused(image=im) for im in imgs]

        return native_thunk, fused_thunk

    # Non-CPU: native has no GPU path; the fused tensor path is probed via warmup.
    fused = _to_device(_build_fused(tfms, seq.reorder), device.torch_device)
    image = torch.rand(batch, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device.torch_device)

    def native_thunk() -> object:
        raise SkipCase("native Albumentations is NumPy/CPU-bound; no GPU path")

    def fused_thunk() -> object:
        return fused(image)

    return native_thunk, fused_thunk


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------
@dataclass
class CaseKey:
    """Identity of one benchmark case: which sequence, backend, device, batch."""

    seq: Sequence
    backend: str
    device: Device
    batch: int

    def record(self, **extra: Any) -> dict[str, Any]:
        """Return a base result record for this case, merged with ``extra``."""
        base = {
            "sequence": self.seq.label,
            "nb_geom": self.seq.nb_geom,
            "backend": self.backend,
            "mode": "",
            "device": self.device.name,
            "batch": self.batch,
        }
        base.update(extra)
        return base


def _measure_mode(thunk: Callable, case: CaseKey, mode: str, cfg: BenchConfig) -> dict[str, Any]:
    """Run one (mode) thunk and assemble its result record."""
    device = case.device
    record = case.record(mode=mode)
    try:
        if device.supports_peak_mem:
            torch.cuda.reset_peak_memory_stats(device.torch_device)
        samples = _time_thunk(thunk, cfg.warmup, cfg.measure, device.sync)
    except SkipCase as skip:
        record.update(status="skipped", skip_reason=str(skip))
        return record
    except Exception as exc:
        record.update(status="skipped", skip_reason=f"{type(exc).__name__}: {exc}")
        return record
    record.update(status="ok", **_summarise(samples, case.batch))
    if device.supports_peak_mem:
        record["peak_mb"] = torch.cuda.max_memory_allocated(device.torch_device) / (1024 * 1024)
    return record


def _run_case(case: CaseKey, cfg: BenchConfig) -> list[dict[str, Any]]:
    """Benchmark both native and fused modes for one case."""
    try:
        if case.backend == "albumentations":
            native_thunk, fused_thunk = _albu_thunks(case.seq, case.device, case.batch)
        else:
            native_thunk, fused_thunk = _tensor_thunks(case.seq, case.backend, case.device, case.batch)
    except SkipCase as skip:
        reason = str(skip)
        return [case.record(mode=mode, status="skipped", skip_reason=reason) for mode in ("native", "fused")]
    return [
        _measure_mode(native_thunk, case, "native", cfg),
        _measure_mode(fused_thunk, case, "fused", cfg),
    ]


@dataclass
class BenchConfig:
    """Benchmark run configuration."""

    devices: list[Device]
    batch_sizes: list[int]
    warmup: int
    measure: int
    quick: bool = False
    backends: tuple[str, ...] = ("kornia", "torchvision", "albumentations")
    sequences: list[Sequence] = field(default_factory=list)


def run_benchmark(cfg: BenchConfig) -> list[dict[str, Any]]:
    """Execute the full device x batch x sequence x backend sweep."""
    results: list[dict[str, Any]] = []
    for device in cfg.devices:
        for batch in cfg.batch_sizes:
            for seq in cfg.sequences:
                for backend in cfg.backends:
                    print(f"  · {device.name}/b{batch}/{seq.label}/{backend}", flush=True)
                    results.extend(_run_case(CaseKey(seq, backend, device, batch), cfg))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pkg_version(name: str) -> str:
    """Return the installed version of ``name`` or ``"unknown"``."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _platform_slug() -> str:
    """Filesystem-safe ``<system>_<machine>`` slug for the output filename."""
    return f"{platform.system().lower()}_{platform.machine().lower()}"


def _build_metadata(cfg: BenchConfig, source: str) -> dict[str, Any]:
    """Assemble the JSON metadata block."""
    device_info: dict[str, str] = {}
    for device in cfg.devices:
        if device.name == "cuda":
            device_info["cuda"] = torch.cuda.get_device_name(device.torch_device)
        elif device.name == "mps":
            device_info["mps"] = f"{platform.processor() or 'apple'} (Metal)"
        else:
            device_info["cpu"] = platform.processor() or platform.machine()
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "devices": device_info,
        "image_shape": [3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "batch_sizes": cfg.batch_sizes,
        "warmup": cfg.warmup,
        "measure": cfg.measure,
        "quick": cfg.quick,
        "sequence_source": source,
        "package_versions": {
            "fuse_augmentations": _pkg_version("fuse-augmentations"),
            "albumentations": _pkg_version("albumentations"),
            "kornia": _pkg_version("kornia"),
            "torchvision": _pkg_version("torchvision"),
        },
    }


def _fused_ratio(results: list[dict[str, Any]], key: tuple) -> float | None:
    """Return native/fused median ratio for a (seq, backend, device, batch) key."""
    seq, backend, device, batch = key
    lookup = {
        (r["sequence"], r["backend"], r["device"], r["batch"], r["mode"]): r for r in results if r.get("status") == "ok"
    }
    native = lookup.get((seq, backend, device, batch, "native"))
    fused = lookup.get((seq, backend, device, batch, "fused"))
    if not native or not fused or fused["median_ms"] <= 0:
        return None
    return native["median_ms"] / fused["median_ms"]


def _markdown_table(results: list[dict[str, Any]]) -> str:
    """Render the results as a Markdown summary table sorted by run order."""
    header = (
        "| sequence | backend | mode | device | batch | median ms | p10 | p90 | img/s | peak MB | boost | notes |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['sequence']} | {r['backend']} | {r['mode']} | {r['device']} | {r['batch']} "
                f"| — | — | — | — | — | — | skip: {r.get('skip_reason', '')} |"
            )
            continue
        boost = ""
        if r["mode"] == "fused":
            ratio = _fused_ratio(results, (r["sequence"], r["backend"], r["device"], r["batch"]))
            boost = f"{ratio:.2f}x" if ratio is not None else ""
        peak = f"{r['peak_mb']:.1f}" if "peak_mb" in r else "—"
        lines.append(
            f"| {r['sequence']} | {r['backend']} | {r['mode']} | {r['device']} | {r['batch']} "
            f"| {r['median_ms']:.3f} | {r['p10_ms']:.3f} | {r['p90_ms']:.3f} "
            f"| {r['throughput_img_s']:.0f} | {peak} | {boost} | |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=None, help="Subset of devices (cpu cuda mps).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32], help="Batch sizes to sweep.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (>=10 recommended).")
    parser.add_argument("--measure", type=int, default=30, help="Measured iterations (>=30 recommended).")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: fewer iters, batch 1+8 only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the GPU/batch benchmark and write JSON + Markdown outputs."""
    import json

    args = _parse_args(argv)
    torch.manual_seed(0)
    np.random.seed(0)

    warmup, measure = args.warmup, args.measure
    batch_sizes = args.batch_sizes
    if args.quick:
        warmup, measure = min(warmup, 5), min(measure, 10)
        batch_sizes = [b for b in batch_sizes if b <= QUICK_MAX_BATCH] or [1, QUICK_MAX_BATCH]

    sequences, source = _load_sequences()
    cfg = BenchConfig(
        devices=_discover_devices(args.devices),
        batch_sizes=batch_sizes,
        warmup=warmup,
        measure=measure,
        quick=args.quick,
        sequences=sequences,
    )

    print(
        f"Torch {torch.__version__} | devices={[d.name for d in cfg.devices]} "
        f"| batches={batch_sizes} | warmup={warmup} measure={measure} "
        f"| sequences={len(sequences)} ({source})"
    )
    results = run_benchmark(cfg)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"bench_gpu_batch_{_platform_slug()}.json"
    out_path.write_text(json.dumps({"metadata": _build_metadata(cfg, source), "results": results}, indent=2))

    table = _markdown_table(results)
    print("\n" + table)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skip = len(results) - n_ok
    print(f"\nResults: {n_ok} ok, {n_skip} skipped → {out_path}")


if __name__ == "__main__":
    main()
