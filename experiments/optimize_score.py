"""Fast composite benchmark score for the fuse-augmentations optimization campaign.

Measures the geometric mean of native/fused boost ratios for two representative
workloads across Kornia, TorchVision, and Albumentations:

- **Sequences** (b02_geom_3: Rotate+HFlip+Scale, b04_geom_5: 5-op geometric chain):
  these should be faster fused (boost > 1.0) because consecutive warps are
  collapsed into one.

- **Single ops** (a01_rotate, a04_scale): these should not regress relative to
  native because there is nothing to fuse; overhead alone determines the ratio.

For Kornia and TorchVision, native and fused both use BCHW float32 tensors.
For Albumentations, both native (``A.Compose``) and fused (``FuseCompose``) use
the Albumentations dict-input convention (``pipeline(image=hwc_uint8)``), which
avoids tensor round-trips and measures the true warp-fusion gain.

Score = geometric_mean(all 15 boost ratios).
Direction: higher is better.
Theoretical target = geometric_mean(nb_geom per case) = 75^(1/5) ≈ 2.37 for the
current 15 cases — recomputed automatically from ``nb_geom`` in ``_CASES`` /
``_ALBU_CASES`` whenever cases are added or changed.

Usage::

    uv run python scripts/optimize_score.py

Outputs two lines::

    real_score=X.XXXX
    theoretical_target=X.XXXX

"""

from __future__ import annotations

import copy
import statistics
import time

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
import torchvision.transforms.v2 as tv

from fuse_aug import Compose as FuseCompose

WARMUP: int = 10
REPS: int = 50
_IMG: torch.Tensor = torch.zeros(1, 3, 256, 256)
_IMG_NP: np.ndarray = np.zeros((256, 256, 3), dtype=np.uint8)


def _bench(fn: object) -> float:
    """Return mean ms per call for BCHW tensor input (warmup + REPS timed repetitions)."""
    for _ in range(WARMUP):
        fn(_IMG)  # type: ignore[operator]
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(_IMG)  # type: ignore[operator]
    return (time.perf_counter() - t0) / REPS * 1000.0


def _bench_albu(fn: object) -> float:
    """Return mean ms per call for Albumentations dict-input (warmup + REPS timed repetitions)."""
    for _ in range(WARMUP):
        fn(image=_IMG_NP)  # type: ignore[operator]
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(image=_IMG_NP)  # type: ignore[operator]
    return (time.perf_counter() - t0) / REPS * 1000.0


# ---------------------------------------------------------------------------
# Benchmark cases: (label, nb_geom, kornia_transforms, tv_transforms)
# nb_geom = number of geometric ops in the sequence; theoretical boost = nb_geom.
# Copied verbatim from bench_augmentation_pipelines.py so scores are comparable.
# ---------------------------------------------------------------------------
_CASES: list[tuple[str, int, list, list]] = [
    # b02_geom_3 — Rotate + HFlip + Scale
    (
        "b02_geom_3",
        3,
        [K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5), K.RandomAffine(0, scale=(0.8, 1.2))],
        [tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5), tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
    ),
    # b04_geom_5 — Rotate + HFlip + Shear + VFlip + Rotate
    (
        "b04_geom_5",
        5,
        [
            K.RandomRotation(10.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(0, shear=(-5.0, 5.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(5.0),
        ],
        [
            tv.RandomRotation(10),
            tv.RandomHorizontalFlip(0.5),
            tv.RandomAffine(degrees=0, shear=5),
            tv.RandomVerticalFlip(0.5),
            tv.RandomRotation(5),
        ],
    ),
    # b05_geom_5_warp — 5 pure warps: Rotate + Scale + Shear + Translate + Rotate
    # No flips: every native op requires a full warp, so fused should be ~3-4x faster.
    (
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
    ),
    # a01_rotate — single Rotate (no fusion gain, only overhead)
    (
        "a01_rotate",
        1,
        [K.RandomRotation(30.0)],
        [tv.RandomRotation(30)],
    ),
    # a04_scale — single Scale/Affine (no fusion gain, only overhead)
    (
        "a04_scale",
        1,
        [K.RandomAffine(0, scale=(0.8, 1.2))],
        [tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
    ),
]

# Albumentations cases — same four workloads benchmarked via the dict-input path.
# Transforms copied from bench_augmentation_pipelines.py for comparability.
_ALBU_CASES: list[tuple[str, int, list]] = [
    (
        "b02_geom_3",
        3,
        [A.Rotate(limit=30), A.HorizontalFlip(p=0.5), A.Affine(scale=(0.8, 1.2))],
    ),
    (
        "b04_geom_5",
        5,
        [
            A.Rotate(limit=10),
            A.HorizontalFlip(p=0.5),
            A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=5),
        ],
    ),
    (
        "b05_geom_5_warp",
        5,
        [
            A.Rotate(limit=30, p=1.0),
            A.Affine(scale=(0.8, 1.2), p=1.0),
            A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0),
            A.Rotate(limit=15, p=1.0),
        ],
    ),
    (
        "a01_rotate",
        1,
        [A.Rotate(limit=30, p=1.0)],
    ),
    (
        "a04_scale",
        1,
        [A.Affine(scale=(0.8, 1.2), p=1.0)],
    ),
]


def main() -> None:
    """Run all cases and print the composite score and theoretical target."""
    boosts: list[float] = []
    for _label, _nb_geom, k_tfms, tv_tfms in _CASES:
        for _backend, tfms, native_fn in [
            ("kornia", k_tfms, lambda t: K.AugmentationSequential(*t)),
            ("torchvision", tv_tfms, lambda t: tv.Compose(t)),
        ]:
            native = native_fn(copy.deepcopy(tfms))
            fused = FuseCompose(copy.deepcopy(tfms))
            # One warm-up forward to trigger JIT / lazy init before timing.
            native(_IMG)
            fused(_IMG)
            n_ms = _bench(native)
            f_ms = _bench(fused)
            boost = n_ms / f_ms if f_ms > 0 else 0.0
            boosts.append(boost)

    # Albumentations: compare A.Compose vs FuseCompose both via dict-input path.
    for _label, _nb_geom, albu_tfms in _ALBU_CASES:
        native = A.Compose(copy.deepcopy(albu_tfms))
        fused = FuseCompose(copy.deepcopy(albu_tfms))
        # Warm-up to trigger lazy init.
        native(image=_IMG_NP)
        fused(image=_IMG_NP)
        n_ms = _bench_albu(native)
        f_ms = _bench_albu(fused)
        boost = n_ms / f_ms if f_ms > 0 else 0.0
        boosts.append(boost)

    score = statistics.geometric_mean(boosts)

    # Theoretical ceiling: each case can achieve at most nb_geom x speedup.
    # geomean(nb_geom across all backends) = (prod nb_geom)^(1/N).
    all_nb_geom = (
        [nb for _, nb, _, _ in _CASES] * 2  # kornia + torchvision
        + [nb for _, nb, _ in _ALBU_CASES]
    )
    theoretical_target = statistics.geometric_mean(all_nb_geom)

    print(f"real_score={score:.4f}")
    print(f"theoretical_target={theoretical_target:.4f}")


if __name__ == "__main__":
    main()
