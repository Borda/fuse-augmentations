#!/usr/bin/env python3
"""Measure fused-vs-native pixel parity per backend and operation, and check it against bounds.

``docs/research/quality-and-fidelity.md`` describes parity in prose: which surfaces are verified and
where the boundaries are. It carries no numbers, so a drift in resampling behaviour on any adapter
would not contradict anything written there. This script produces the missing numeric half and makes
it enforceable.

Each case runs a deterministic transform -- every parameter range collapsed to a single value, every
probability at 1.0 -- through the backend's own compose and through ``fuse_augmentations.Compose``,
then reports the maximum absolute per-pixel difference between the two renders. Determinism is what
makes the comparison meaningful: with no sampling left, any difference is resampling and composition
behaviour rather than two different random draws.

Usage
-----
Write the generated table (regenerates ``docs/research/parity-tolerances.md``)::

    python .github/scripts/measure_parity_tolerances.py --write-doc

Record the current measurements as the bounds to enforce::

    python .github/scripts/measure_parity_tolerances.py --write-bounds

Check the current measurements against the recorded bounds (used by CI)::

    python .github/scripts/measure_parity_tolerances.py --check

"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fuse_augmentations import Compose

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDS_PATH = REPO_ROOT / ".github" / "parity_baseline" / "parity_bounds.json"
DOC_PATH = REPO_ROOT / "docs" / "research" / "parity-tolerances.md"

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 48, 64

#: Absolute slack added to a recorded bound before a measurement counts as drift. Renderer versions
#: differ slightly across platforms, so an exact-equality gate would fail on the runner rather than
#: on a real change.
CHECK_SLACK = 5e-3


@dataclass(frozen=True, slots=True)
class ParityCase:
    """One backend/operation pair, with the native and fused renders to compare."""

    backend: str
    operation: str
    native: Callable[[torch.Tensor], torch.Tensor]
    fused: Callable[[torch.Tensor], torch.Tensor]
    #: How many resampling passes the native side performs. A multi-op geometric chain resamples
    #: once per op natively and exactly once when fused, so the difference is the fusion itself
    #: rather than a defect -- the number still has to stay stable, which is what the gate checks.
    native_passes: int = 1

    @property
    def key(self) -> str:
        """Return the stable identifier used in the bounds file."""
        return f"{self.backend}/{self.operation}"


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 48, 64)`` float32 batch with high-frequency structure.

    Smooth gradients hide resampling differences; the checkerboard term is what makes an off-by-one grid or a different
    interpolation kernel visible in the maximum absolute error.

    """
    generator = torch.Generator().manual_seed(0)
    base = torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH, generator=generator)
    rows = torch.arange(HEIGHT).reshape(1, 1, HEIGHT, 1)
    columns = torch.arange(WIDTH).reshape(1, 1, 1, WIDTH)
    return (base + ((rows + columns) % 2).float()).clamp(0.0, 1.0)


def _albumentations_cases() -> list[ParityCase]:
    """Return the Albumentations cases, or an empty list when the backend is absent."""
    try:
        import albumentations as albu
    except ImportError:
        return []

    def native(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = albu.Compose(transforms)

        def run(image: torch.Tensor) -> torch.Tensor:
            arrays = [pipeline(image=sample.permute(1, 2, 0).numpy())["image"] for sample in image]
            return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)

        return run

    def fused(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = Compose(transforms)
        return lambda image: torch.as_tensor(pipeline(image))

    specs: list[tuple[str, Callable[[], list[object]]]] = [
        ("hflip", lambda: [albu.HorizontalFlip(p=1.0)]),
        ("vflip", lambda: [albu.VerticalFlip(p=1.0)]),
        ("rotate", lambda: [albu.Affine(rotate=(15.0, 15.0), p=1.0)]),
        ("scale", lambda: [albu.Affine(scale=(1.15, 1.15), p=1.0)]),
        ("translate", lambda: [albu.Affine(translate_px=(3, 3), p=1.0)]),
        (
            "affine_chain",
            lambda: [
                albu.Affine(rotate=(15.0, 15.0), p=1.0),
                albu.Affine(translate_px=(3, 3), p=1.0),
            ],
        ),
        (
            "brightness_contrast",
            lambda: [
                albu.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.1),
                    contrast_limit=(0.1, 0.1),
                    p=1.0,
                )
            ],
        ),
    ]
    return [
        ParityCase(
            backend="albumentations",
            operation=name,
            native=native(build()),
            fused=fused(build()),
            native_passes=len(build()),
        )
        for name, build in specs
    ]


def _kornia_cases() -> list[ParityCase]:
    """Return the Kornia cases, or an empty list when the backend is absent."""
    try:
        import kornia.augmentation as kornia_aug
    except ImportError:
        return []

    def native(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = kornia_aug.AugmentationSequential(*transforms)
        return lambda image: torch.as_tensor(pipeline(image))

    def fused(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = Compose(transforms)
        return lambda image: torch.as_tensor(pipeline(image))

    specs: list[tuple[str, Callable[[], list[object]]]] = [
        ("hflip", lambda: [kornia_aug.RandomHorizontalFlip(p=1.0)]),
        ("vflip", lambda: [kornia_aug.RandomVerticalFlip(p=1.0)]),
        ("rotate", lambda: [kornia_aug.RandomRotation(degrees=(15.0, 15.0), p=1.0)]),
        (
            "affine_chain",
            lambda: [
                kornia_aug.RandomRotation(degrees=(15.0, 15.0), p=1.0),
                kornia_aug.RandomHorizontalFlip(p=1.0),
            ],
        ),
    ]
    return [
        ParityCase(
            backend="kornia",
            operation=name,
            native=native(build()),
            fused=fused(build()),
            native_passes=len(build()),
        )
        for name, build in specs
    ]


def _torchvision_cases() -> list[ParityCase]:
    """Return the TorchVision v2 cases, or an empty list when the backend is absent."""
    try:
        import torchvision.transforms.v2 as tv2
    except ImportError:
        return []

    def native(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = tv2.Compose(transforms)
        return lambda image: torch.as_tensor(pipeline(image))

    def fused(transforms: list[object]) -> Callable[[torch.Tensor], torch.Tensor]:
        pipeline = Compose(transforms)
        return lambda image: torch.as_tensor(pipeline(image))

    specs: list[tuple[str, Callable[[], list[object]]]] = [
        ("hflip", lambda: [tv2.RandomHorizontalFlip(p=1.0)]),
        ("vflip", lambda: [tv2.RandomVerticalFlip(p=1.0)]),
        ("rotate", lambda: [tv2.RandomRotation(degrees=(15.0, 15.0))]),
        (
            "affine_chain",
            lambda: [
                tv2.RandomRotation(degrees=(15.0, 15.0)),
                tv2.RandomHorizontalFlip(p=1.0),
            ],
        ),
    ]
    return [
        ParityCase(
            backend="torchvision",
            operation=name,
            native=native(build()),
            fused=fused(build()),
            native_passes=len(build()),
        )
        for name, build in specs
    ]


def _measure(case: ParityCase, image: torch.Tensor) -> float:
    """Return the maximum absolute per-pixel difference between the native and fused renders."""
    torch.manual_seed(0)
    np.random.seed(0)
    native_out = case.native(image.clone()).float()
    torch.manual_seed(0)
    np.random.seed(0)
    fused_out = case.fused(image.clone()).float()
    if native_out.shape != fused_out.shape:
        msg = f"{case.key}: shape mismatch {tuple(native_out.shape)} vs {tuple(fused_out.shape)}"
        raise RuntimeError(msg)
    return round(float((native_out - fused_out).abs().max()), 6)


def _cases() -> list[ParityCase]:
    """Return every case whose backend is installed, refusing to run with none of them."""
    cases = _albumentations_cases() + _kornia_cases() + _torchvision_cases()
    if not cases:
        msg = "no backends installed; install at least one of albumentations, kornia, torchvision"
        raise RuntimeError(msg)
    return cases


def _collect() -> dict[str, float]:
    """Run every available case and return the measured tolerance per case key."""
    image = _image()
    return {case.key: _measure(case, image) for case in _cases()}


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a padded markdown table.

    Columns are padded to their widest cell so the emitted file already matches what the repository's markdown formatter
    produces; otherwise the format-on-commit hook would rewrite this generated file and the CI staleness check would
    fail on formatting rather than on content.

    """
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    rule_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in rows]
    return [header_line, rule_line, *body]


def _render_doc(measured: dict[str, float]) -> str:
    """Render the generated markdown table for the measured tolerances.

    Paragraphs are emitted as single physical lines: the repository's markdown formatter unwraps
    them, and a hand-wrapped generated file would be rewritten on commit.

    """
    passes = {case.key: case.native_passes for case in _cases()}
    rows = [[*key.split("/", 1), str(passes.get(key, 1)), f"{measured[key]:.6f}"] for key in sorted(measured)]
    lines = [
        "<!-- GENERATED FILE -- do not edit by hand.",
        "     Regenerate with: python .github/scripts/measure_parity_tolerances.py --write-doc -->",
        "",
        "# Measured fused-vs-native tolerances",
        "",
        (
            "Maximum absolute per-pixel difference between each backend's own compose and"
            " `fuse_augmentations.Compose` running the same deterministic transform. Every parameter range"
            " is collapsed to a single value and every probability is 1.0, so no sampling remains and the"
            " number is resampling and composition behaviour rather than two different random draws."
        ),
        "",
        (
            "Images are float32 in `[0, 1]`, so a tolerance of `0.004` is roughly one step of an 8-bit"
            " level. These are measurements, not promises: they record what the current implementations do,"
            " and `.github/workflows/ci_parity-gate.yml` fails when one drifts past its recorded bound."
        ),
        "",
        (
            "A single-op row compares one warp against one warp, so its difference is small by construction"
            " and a large value there would be a defect. A chain row compares one fused warp against one"
            " native warp per operation: resampling twice is not the same as resampling once, so a visible"
            " difference there is the fusion working as designed on a deliberately high-frequency test"
            " image. Both kinds still have to stay stable, which is what the gate checks -- the number, not"
            " its size, is the signal."
        ),
        "",
        (
            "The prose companion to this table -- which surfaces are verified and where the boundaries are"
            " -- is `quality-and-fidelity.md`."
        ),
        "",
        *_table(["Backend", "Operation", "Native passes", "Max abs difference"], rows),
        "",
    ]
    return "\n".join(lines)


def _check(measured: dict[str, float]) -> int:
    """Compare measurements against the recorded bounds; return a process exit code."""
    if not BOUNDS_PATH.exists():
        print(f"no bounds file at {BOUNDS_PATH}; bootstrap it with --write-bounds")
        return 0

    bounds: dict[str, float] = json.loads(BOUNDS_PATH.read_text())["bounds"]
    failures: list[str] = []
    missing = sorted(set(measured) - set(bounds))
    for key in sorted(set(measured) & set(bounds)):
        limit = bounds[key] + CHECK_SLACK
        if measured[key] > limit:
            failures.append(f"  {key}: measured {measured[key]:.6f} > bound {bounds[key]:.6f} + slack")

    for key in sorted(set(measured) & set(bounds)):
        print(f"ok   {key}: {measured[key]:.6f} (bound {bounds[key]:.6f})")
    for key in missing:
        print(f"new  {key}: {measured[key]:.6f} (no recorded bound; run --write-bounds to record it)")

    if failures:
        print("\nparity drift detected:")
        print("\n".join(failures))
        return 1
    return 0


def main() -> int:
    """Parse arguments, measure, and write or check as requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-doc", action="store_true", help="regenerate the markdown table")
    parser.add_argument("--write-bounds", action="store_true", help="record current values as bounds")
    parser.add_argument("--check", action="store_true", help="fail when a measurement drifts past its bound")
    args = parser.parse_args()

    measured = _collect()

    if args.write_doc:
        DOC_PATH.write_text(_render_doc(measured))
        print(f"written: {DOC_PATH}")

    if args.write_bounds:
        BOUNDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BOUNDS_PATH.write_text(json.dumps({"slack": CHECK_SLACK, "bounds": measured}, indent=2, sort_keys=True) + "\n")
        print(f"written: {BOUNDS_PATH}")

    if args.check:
        return _check(measured)

    if not (args.write_doc or args.write_bounds):
        for key in sorted(measured):
            print(f"{key}: {measured[key]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
