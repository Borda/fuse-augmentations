"""Peak-memory and allocation-count benchmark for the fuse-augmentations pipeline.

Where :mod:`experiments.bench_gpu_batch` measures *speed* (latency + throughput),
this script measures *memory*: the peak resident allocation and the number of
distinct allocation events per pipeline, comparing the **fused** implementation
against the **native** backend chain. The hypothesis under test is that fusing an
N-op geometric chain into a single ``grid_sample`` both lowers the peak (no chain
of intermediate warped tensors) and cuts allocation count (fewer transient
buffers).

Sequences and backend I/O conventions mirror :mod:`experiments.bench_gpu_batch`
(imported from :mod:`experiments.optimize_score` when available, else an inline
fallback): Kornia/TorchVision run the BCHW float32 tensor path; Albumentations
runs the HWC uint8 NumPy dict path on CPU (a batch of ``B`` is ``B`` sequential
calls) and its opt-in ``execution="torch"`` tensor path on GPU/MPS.

Per-device memory counter (profiler memory support is uneven across backends, so
each device uses the counter that is actually reliable there):

- **cpu** — ``torch.profiler`` with ``profile_memory=True`` gives the authoritative
  peak (reconstructed as the running-total maximum over the allocation/free
  timeline) and the allocation count (number of positive timeline entries). Because
  the torch allocator does not see NumPy/cv2 buffers used by native Albumentations,
  a :mod:`tracemalloc` peak is reported alongside as a Python-heap cross-check, and
  the process ``ru_maxrss`` delta is recorded as a coarse resident-set sanity bound.
- **mps** — profiler memory events are weak on Metal, so peak is the
  ``torch.mps.current_allocated_memory()`` delta across the timed region (with
  ``torch.mps.driver_allocated_memory()`` reported as the driver-side figure). The
  allocation count is a best-effort read of the profiler's CPU-side timeline and is
  flagged as approximate.
- **cuda** — ``reset_peak_memory_stats`` + ``max_memory_allocated`` for peak;
  profiler timeline for the allocation count. (Code-complete but unexercised on the
  darwin/arm64 development host — no NVIDIA device.)

Output:

- Aligned text table: sequence x backend x mode -> peak MB, alloc count, plus the
  fused/native ratio for each metric (``<1`` means fused uses less).
- Optional ``--json`` dump to ``experiments/results/bench_memory_<platform>.json``.

Usage::

    uv run python experiments/bench_memory.py            # full sweep
    uv run python experiments/bench_memory.py --quick    # fast smoke subset
    uv run python experiments/bench_memory.py --json      # also write JSON

"""

from __future__ import annotations

import argparse
import copy
import gc
import platform
import resource
import sys
import tracemalloc
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
from torch.profiler import ProfilerActivity, profile

from fuse_aug import Compose as FuseCompose
from fuse_aug import ReorderPolicy

IMAGE_HEIGHT: int = 256
IMAGE_WIDTH: int = 256
RESULTS_DIR: Path = Path(__file__).parent / "results"
_BYTES_PER_MB: float = 1024.0 * 1024.0

# Representative subset (mirrors bench_gpu_batch): a geometric chain, a longer
# geometric warp chain, two mixed geo+colour sequences, and a crop-resize probe.
_GEO_LABELS: tuple[str, ...] = ("b02_geom_3", "b04_geom_5", "b05_geom_5_warp")
_MIXED_LABELS: tuple[str, ...] = ("d02_mixed_g3c3__agr", "d03_mixed_g4c3__agr")
_CROP_TARGET: tuple[int, int] = (192, 192)

# --quick trims to one geometric chain, one mixed sequence, and the crop probe.
_QUICK_LABELS: frozenset[str] = frozenset({"b02_geom_3", "d02_mixed_g3c3__agr", "e01_geo_crop_fuse"})
_QUICK_BATCHES: tuple[int, ...] = (1, 8)
_FULL_BATCHES: tuple[int, ...] = (1, 8)


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
# Sequence construction (shared with bench_gpu_batch conventions)
# ---------------------------------------------------------------------------
def _load_sequences() -> tuple[list[Sequence], str]:
    """Import the representative subset from optimize_score, else build inline.

    Returns:
        The selected sequences and a provenance string (``"imported"`` or
        ``"inline-fallback"``).

    Examples:
        >>> seqs, source = _load_sequences()
        >>> len(seqs) >= 4
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
    sequences.append(_crop_fusion_sequence())
    return sequences, "imported"


def _crop_fusion_sequence() -> Sequence:
    """Build the geo+RandomResizedCrop probe (fuses to one warp on the torch path)."""
    h_out, w_out = _CROP_TARGET
    return Sequence(
        "e01_geo_crop_fuse",
        2,
        [K.RandomRotation(20.0, p=1.0), K.RandomResizedCrop((h_out, w_out), scale=(0.6, 1.0))],
        [tv.RandomRotation(20), tv.RandomResizedCrop((h_out, w_out), scale=(0.6, 1.0))],
        [A.Rotate(limit=20, p=1.0), A.RandomResizedCrop(size=(h_out, w_out), scale=(0.6, 1.0), p=1.0)],
    )


def _fallback_sequences() -> list[Sequence]:
    """Redefine the representative cases inline when optimize_score is unavailable."""
    return [
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
        _crop_fusion_sequence(),
    ]


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
@dataclass
class Device:
    """A benchmark target device and which memory counter it uses."""

    name: str
    torch_device: torch.device
    sync: Callable[[], None]
    counter: str  # human label for the peak-memory source used on this device


def _discover_devices(requested: list[str] | None) -> list[Device]:
    """Return the available devices, optionally filtered by ``requested`` names."""
    devices: list[Device] = [Device("cpu", torch.device("cpu"), lambda: None, counter="torch-profiler timeline")]
    if torch.cuda.is_available():
        devices.append(
            Device("cuda", torch.device("cuda"), torch.cuda.synchronize, counter="cuda max_memory_allocated")
        )
    if torch.backends.mps.is_available():
        mps_dev = Device("mps", torch.device("mps"), torch.mps.synchronize, counter="mps current_allocated delta")
        devices.append(mps_dev)
    if requested:
        wanted = set(requested)
        devices = [d for d in devices if d.name in wanted]
    return devices


# ---------------------------------------------------------------------------
# Thunk builders per backend (mirrors bench_gpu_batch I/O conventions)
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
    """Build (native, fused) zero-arg thunks for the Kornia/TorchVision tensor path."""
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
    """Build (native, fused) thunks for Albumentations on the given device."""
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

    # Non-CPU: native Albu is NumPy/CPU-bound (no GPU path); the fused tensor path
    # opts into execution="torch" so it warps natively on device.
    reorder = seq.reorder
    fused_pipe = (
        FuseCompose(copy.deepcopy(tfms), execution="torch")
        if reorder is None
        else FuseCompose(copy.deepcopy(tfms), reorder=reorder, execution="torch")
    )
    fused = _to_device(fused_pipe, device.torch_device)
    image = torch.rand(batch, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device.torch_device)

    def native_thunk() -> object:
        raise SkipCase("native Albumentations is NumPy/CPU-bound; no GPU path")

    def fused_thunk() -> object:
        return fused(image)

    return native_thunk, fused_thunk


# ---------------------------------------------------------------------------
# Memory measurement per device
# ---------------------------------------------------------------------------
@dataclass
class MemSample:
    """One memory measurement: peak MB, allocation count, plus optional cross-checks."""

    peak_mb: float
    alloc_count: int
    tracemalloc_peak_mb: float | None = None
    rss_delta_mb: float | None = None
    driver_mb: float | None = None
    approx_allocs: bool = False


def _timeline_stats(prof: profile) -> tuple[float, int]:
    """Reconstruct (peak bytes, alloc count) from the profiler memory timeline.

    The timeline is a stream of ``(ts, action, key, nbytes)`` records; ``nbytes``
    is positive for allocations and negative for frees. The running total's maximum
    is the peak, and the number of positive records is the allocation count.

    """
    try:
        timeline = prof._memory_profile().timeline
    except Exception:
        return 0.0, 0
    running = 0
    peak = 0
    n_alloc = 0
    for _ts, _action, _key, nbytes in timeline:
        running += nbytes
        if nbytes > 0:
            n_alloc += 1
        peak = max(peak, running)
    return float(peak), n_alloc


def _measure_cpu(thunk: Callable, warmup: int) -> MemSample:
    """CPU peak + alloc count via torch profiler, with tracemalloc and RSS cross-checks."""
    for _ in range(warmup):
        thunk()
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        thunk()
    _tm_cur, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes, n_alloc = _timeline_stats(prof)
    rss_delta = max(rss_after - rss_before, 0)  # macOS reports ru_maxrss in bytes
    return MemSample(
        peak_mb=peak_bytes / _BYTES_PER_MB,
        alloc_count=n_alloc,
        tracemalloc_peak_mb=tm_peak / _BYTES_PER_MB,
        rss_delta_mb=rss_delta / _BYTES_PER_MB,
    )


def _measure_mps(thunk: Callable, warmup: int) -> MemSample:
    """MPS resident footprint via current_allocated_memory delta; allocs from profiler (approx).

    Metal has no ``reset_peak_memory_stats`` in this torch build, so a true transient
    peak is not exposed. Instead the thunk output is held live and
    ``current_allocated_memory`` is read while it is still referenced — the delta over
    an emptied-cache baseline is the *resident* device footprint of the result plus any
    tensors the pipeline keeps alive. ``driver_allocated_memory`` (the Metal driver's
    total reservation) is reported alongside as an upper bound. The allocation count
    comes from the profiler's CPU-side timeline and is flagged approximate — Metal
    kernel-level allocations are not fully represented there.

    """
    for _ in range(warmup):
        thunk()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        out = thunk()
        torch.mps.synchronize()
    footprint = torch.mps.current_allocated_memory() - base
    driver = torch.mps.driver_allocated_memory()
    del out  # release only after the counter has been read
    _peak_bytes, n_alloc = _timeline_stats(prof)
    return MemSample(
        peak_mb=max(footprint, 0) / _BYTES_PER_MB,
        alloc_count=n_alloc,
        driver_mb=driver / _BYTES_PER_MB,
        approx_allocs=True,
    )


def _measure_cuda(thunk: Callable, device: torch.device, warmup: int) -> MemSample:
    """CUDA peak via max_memory_allocated; alloc count from profiler timeline."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True
    ) as prof:
        thunk()
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    _peak_bytes, n_alloc = _timeline_stats(prof)
    return MemSample(peak_mb=peak / _BYTES_PER_MB, alloc_count=n_alloc)


def _measure(thunk: Callable, device: Device, warmup: int) -> MemSample:
    """Dispatch to the device-appropriate memory counter."""
    if device.name == "cuda":
        return _measure_cuda(thunk, device.torch_device, warmup)
    if device.name == "mps":
        return _measure_mps(thunk, warmup)
    return _measure_cpu(thunk, warmup)


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


def _record_from_sample(case: CaseKey, mode: str, sample: MemSample) -> dict[str, Any]:
    """Assemble an ``ok`` result record from a memory sample."""
    record = case.record(mode=mode, status="ok")
    record["peak_mb"] = sample.peak_mb
    record["alloc_count"] = sample.alloc_count
    if sample.tracemalloc_peak_mb is not None:
        record["tracemalloc_peak_mb"] = sample.tracemalloc_peak_mb
    if sample.rss_delta_mb is not None:
        record["rss_delta_mb"] = sample.rss_delta_mb
    if sample.driver_mb is not None:
        record["driver_mb"] = sample.driver_mb
    if sample.approx_allocs:
        record["approx_allocs"] = True
    return record


def _measure_mode(thunk: Callable, case: CaseKey, mode: str, warmup: int) -> dict[str, Any]:
    """Run one mode thunk under the device memory counter, capturing skips."""
    try:
        sample = _measure(thunk, case.device, warmup)
    except SkipCase as skip:
        return case.record(mode=mode, status="skipped", skip_reason=str(skip))
    except Exception as exc:
        return case.record(mode=mode, status="skipped", skip_reason=f"{type(exc).__name__}: {exc}")
    return _record_from_sample(case, mode, sample)


def _run_case(case: CaseKey, warmup: int) -> list[dict[str, Any]]:
    """Benchmark native and fused memory for one case."""
    try:
        if case.backend == "albumentations":
            native_thunk, fused_thunk = _albu_thunks(case.seq, case.device, case.batch)
        else:
            native_thunk, fused_thunk = _tensor_thunks(case.seq, case.backend, case.device, case.batch)
    except SkipCase as skip:
        reason = str(skip)
        return [case.record(mode=mode, status="skipped", skip_reason=reason) for mode in ("native", "fused")]
    return [
        _measure_mode(native_thunk, case, "native", warmup),
        _measure_mode(fused_thunk, case, "fused", warmup),
    ]


@dataclass
class BenchConfig:
    """Benchmark run configuration."""

    devices: list[Device]
    batch_sizes: tuple[int, ...]
    warmup: int
    quick: bool = False
    backends: tuple[str, ...] = ("kornia", "torchvision", "albumentations")
    sequences: list[Sequence] = field(default_factory=list)


def run_benchmark(cfg: BenchConfig) -> list[dict[str, Any]]:
    """Execute the device x batch x sequence x backend sweep."""
    results: list[dict[str, Any]] = []
    for device in cfg.devices:
        for batch in cfg.batch_sizes:
            for seq in cfg.sequences:
                for backend in cfg.backends:
                    print(f"  · {device.name}/b{batch}/{seq.label}/{backend}", flush=True)
                    results.extend(_run_case(CaseKey(seq, backend, device, batch), cfg.warmup))
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


def _pair_ratio(results: list[dict[str, Any]], key: tuple, metric: str) -> float | None:
    """Return the fused/native ratio for ``metric`` on a (seq, backend, device, batch) key."""
    seq, backend, device, batch = key
    lookup = {
        (r["sequence"], r["backend"], r["device"], r["batch"], r["mode"]): r for r in results if r.get("status") == "ok"
    }
    native = lookup.get((seq, backend, device, batch, "native"))
    fused = lookup.get((seq, backend, device, batch, "fused"))
    if not native or not fused or native.get(metric, 0) <= 0:
        return None
    return fused[metric] / native[metric]


_COLS: tuple[tuple[str, int], ...] = (
    ("sequence", 20),
    ("backend", 14),
    ("mode", 7),
    ("device", 7),
    ("batch", 5),
    ("peak MB", 9),
    ("allocs", 8),
    ("peak x", 7),
    ("alloc x", 8),
    ("notes", 34),
)


def _fmt_row(cells: list[str]) -> str:
    """Format one aligned table row from string cells."""
    return " ".join(cell.ljust(width) for cell, (_h, width) in zip(cells, _COLS, strict=False))


def _row_notes(record: dict[str, Any]) -> str:
    """Assemble the notes cell for one ok record (cross-checks + caveats)."""
    parts: list[str] = []
    if "tracemalloc_peak_mb" in record:
        parts.append(f"tm={record['tracemalloc_peak_mb']:.1f}")
    if "driver_mb" in record:
        parts.append(f"drv={record['driver_mb']:.0f}")
    if record.get("approx_allocs"):
        parts.append("allocs≈")
    return " ".join(parts)


def _ok_row(record: dict[str, Any], results: list[dict[str, Any]]) -> str:
    """Render one ``ok`` result as an aligned row, including fused/native ratios."""
    peak_x = alloc_x = ""
    if record["mode"] == "fused":
        key = (record["sequence"], record["backend"], record["device"], record["batch"])
        pr = _pair_ratio(results, key, "peak_mb")
        ar = _pair_ratio(results, key, "alloc_count")
        peak_x = f"{pr:.2f}x" if pr is not None else ""
        alloc_x = f"{ar:.2f}x" if ar is not None else ""
    return _fmt_row([
        record["sequence"],
        record["backend"],
        record["mode"],
        record["device"],
        str(record["batch"]),
        f"{record['peak_mb']:.1f}",
        str(record["alloc_count"]),
        peak_x,
        alloc_x,
        _row_notes(record),
    ])


def _text_table(results: list[dict[str, Any]]) -> str:
    """Render the results as an aligned fixed-width text table."""
    header = _fmt_row([h for h, _w in _COLS])
    rule = " ".join("-" * width for _h, width in _COLS)
    lines = [header, rule]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                _fmt_row([
                    r["sequence"],
                    r["backend"],
                    r["mode"],
                    r["device"],
                    str(r["batch"]),
                    "—",
                    "—",
                    "",
                    "",
                    f"skip: {r.get('skip_reason', '')}"[:34],
                ])
            )
            continue
        lines.append(_ok_row(r, results))
    return "\n".join(lines)


def _build_metadata(cfg: BenchConfig, source: str) -> dict[str, Any]:
    """Assemble the JSON metadata block."""
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "devices": {d.name: d.counter for d in cfg.devices},
        "image_shape": [3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "batch_sizes": list(cfg.batch_sizes),
        "warmup": cfg.warmup,
        "quick": cfg.quick,
        "sequence_source": source,
        "package_versions": {
            "fuse_augmentations": _pkg_version("fuse-augmentations"),
            "albumentations": _pkg_version("albumentations"),
            "kornia": _pkg_version("kornia"),
            "torchvision": _pkg_version("torchvision"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=None, help="Subset of devices (cpu cuda mps).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None, help="Batch sizes to sweep (default 1 8).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before measuring (default 3).")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: one geo + one mixed + crop probe.")
    parser.add_argument("--json", action="store_true", help="Also write a JSON dump to experiments/results/.")
    return parser.parse_args(argv)


def _select_sequences(sequences: list[Sequence], quick: bool) -> list[Sequence]:
    """Filter to the --quick subset when requested."""
    if not quick:
        return sequences
    return [s for s in sequences if s.label in _QUICK_LABELS]


def main(argv: list[str] | None = None) -> None:
    """Run the memory benchmark and print the aligned table (+ optional JSON)."""
    import json

    args = _parse_args(argv)
    torch.manual_seed(0)
    np.random.seed(0)

    sequences, source = _load_sequences()
    sequences = _select_sequences(sequences, args.quick)
    warmup = min(args.warmup, 2) if args.quick else args.warmup
    batches = tuple(args.batch_sizes) if args.batch_sizes else (_QUICK_BATCHES if args.quick else _FULL_BATCHES)

    cfg = BenchConfig(
        devices=_discover_devices(args.devices),
        batch_sizes=batches,
        warmup=warmup,
        quick=args.quick,
        sequences=sequences,
    )

    print(
        f"Torch {torch.__version__} | devices={[(d.name, d.counter) for d in cfg.devices]} "
        f"| batches={list(batches)} | warmup={warmup} "
        f"| sequences={len(sequences)} ({source})"
    )
    results = run_benchmark(cfg)

    print("\n" + _text_table(results))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skip = len(results) - n_ok
    print(f"\nResults: {n_ok} ok, {n_skip} skipped")
    print("Ratios are fused/native; <1.00x means fused uses less. 'allocs≈' = approximate count (MPS).")

    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"bench_memory_{_platform_slug()}.json"
        out_path.write_text(json.dumps({"metadata": _build_metadata(cfg, source), "results": results}, indent=2))
        print(f"JSON → {out_path}")


if __name__ == "__main__":
    main()
