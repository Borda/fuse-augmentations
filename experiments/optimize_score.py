"""Fast composite benchmark score for the fuse-augmentations optimization campaign.

Measures the geometric mean of native/fused boost ratios across 45 cases:

- **a-group** (15 cases): single-op baselines x 3 backends (Kornia, TorchVision,
  Albumentations). Theoretical ceiling 1.0x per case -- any boost below 1.0 is
  pure passthrough overhead.

- **b-group** (15 cases): pure-geo chains of 2-5 ops x 3 backends. Fused collapses
  N warps into one ``grid_sample`` call; theoretical ceiling = ``nb_geom``x.

- **d-group AGGRESSIVE** (15 cases): interleaved geo+colour chains x 3 backends,
  measured under ``ReorderPolicy.AGGRESSIVE`` so all geo ops are grouped before
  fusion. Theoretical ceiling = ``nb_geom``x (bounded in practice by colour operation time).

Score = geometric_mean(all 45 boost ratios).
Direction: higher is better.
Theoretical target = geometric_mean(nb_geom per case across all 45 cases) ≈ 2.375.
The theoretical target is printed on every run; use it as the visual success criterion.

For Kornia and TorchVision, native and fused both use BCHW float32 tensors.
For Albumentations, both native (``A.Compose``) and fused (``FuseCompose``) use
the dict-input convention (``pipeline(image=hwc_uint8)``), matching the real
PyTorch training workflow and avoiding tensor round-trips.

Usage::

    uv run python experiments/optimize_score.py

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
from fuse_aug import ReorderPolicy

WARMUP_REPS: int = 10
NUM_REPEATS: int = 50
_IMAGE_TENSOR: torch.Tensor = torch.zeros(1, 3, 256, 256)
_IMAGE_NDARRAY: np.ndarray = np.zeros((256, 256, 3), dtype=np.uint8)


def _bench(func: object) -> float:
    """Return mean ms per call for BCHW tensor input (warmup + REPS timed repetitions)."""
    for _ in range(WARMUP_REPS):
        func(_IMAGE_TENSOR)  # type: ignore[operator]
    t_start = time.perf_counter()
    for _ in range(NUM_REPEATS):
        func(_IMAGE_TENSOR)  # type: ignore[operator]
    return (time.perf_counter() - t_start) / NUM_REPEATS * 1000.0


def _bench_albu(func: object) -> float:
    """Return mean ms per call for Albumentations dict-input (warmup + REPS timed repetitions)."""
    for _ in range(WARMUP_REPS):
        func(image=_IMAGE_NDARRAY)  # type: ignore[operator]
    t_start = time.perf_counter()
    for _ in range(NUM_REPEATS):
        func(image=_IMAGE_NDARRAY)  # type: ignore[operator]
    return (time.perf_counter() - t_start) / NUM_REPEATS * 1000.0


# ---------------------------------------------------------------------------
# a-group + b-group: Kornia and TorchVision.
# Format: (label, nb_geom, kornia_transforms, tv_transforms)
# Copied verbatim from bench_augmentation_pipelines.py for cross-script comparability.
# ---------------------------------------------------------------------------
_CASES: list[tuple[str, int, list, list]] = [
    # ── a: single-op baselines ──────────────────────────────────────────────
    ("a01_rotate", 1, [K.RandomRotation(30.0)], [tv.RandomRotation(30)]),
    ("a02_hflip", 1, [K.RandomHorizontalFlip(p=1.0)], [tv.RandomHorizontalFlip(1.0)]),
    ("a03_vflip", 1, [K.RandomVerticalFlip(p=1.0)], [tv.RandomVerticalFlip(1.0)]),
    (
        "a04_scale",
        1,
        [K.RandomAffine(0, scale=(0.8, 1.2))],
        [tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
    ),
    (
        "a05_shear",
        1,
        [K.RandomAffine(0, shear=(-10.0, 10.0))],
        [tv.RandomAffine(degrees=0, shear=10)],
    ),
    # ── b: geometric chains (2-5 ops) ───────────────────────────────────────
    (
        "b01_geom_2",
        2,
        [K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5)],
        [tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5)],
    ),
    (
        "b02_geom_3",
        3,
        [K.RandomRotation(30.0), K.RandomHorizontalFlip(p=0.5), K.RandomAffine(0, scale=(0.8, 1.2))],
        [tv.RandomRotation(30), tv.RandomHorizontalFlip(0.5), tv.RandomAffine(degrees=0, scale=(0.8, 1.2))],
    ),
    (
        "b03_geom_4",
        4,
        [
            K.RandomRotation(10.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(0, shear=(-5.0, 5.0)),
            K.RandomVerticalFlip(p=0.5),
        ],
        [
            tv.RandomRotation(10),
            tv.RandomHorizontalFlip(0.5),
            tv.RandomAffine(degrees=0, shear=5),
            tv.RandomVerticalFlip(0.5),
        ],
    ),
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
    # b05: 5 pure-warp ops (no flips, all prob=1.0) — fixed-cost architecture demo.
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
]

# ---------------------------------------------------------------------------
# a-group + b-group: Albumentations (dict-input path).
# Format: (label, nb_geom, albu_transforms)
# ---------------------------------------------------------------------------
_ALBU_CASES: list[tuple[str, int, list]] = [
    # ── a: single-op baselines ──────────────────────────────────────────────
    ("a01_rotate", 1, [A.Rotate(limit=30, p=1.0)]),
    ("a02_hflip", 1, [A.HorizontalFlip(p=1.0)]),
    ("a03_vflip", 1, [A.VerticalFlip(p=1.0)]),
    ("a04_scale", 1, [A.Affine(scale=(0.8, 1.2), p=1.0)]),
    ("a05_shear", 1, [A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0)]),
    # ── b: geometric chains ─────────────────────────────────────────────────
    ("b01_geom_2", 2, [A.Rotate(limit=30, p=1.0), A.HorizontalFlip(p=0.5)]),
    (
        "b02_geom_3",
        3,
        [A.Rotate(limit=30), A.HorizontalFlip(p=0.5), A.Affine(scale=(0.8, 1.2))],
    ),
    (
        "b03_geom_4",
        4,
        [
            A.Rotate(limit=10),
            A.HorizontalFlip(p=0.5),
            A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}),
            A.VerticalFlip(p=0.5),
        ],
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
]

# ---------------------------------------------------------------------------
# d-group interleaved pool — alternating geo/colour, 9 ops.
# Strict-prefix slices [:n] give d01 (n=5) … d05 (n=9).
# Identical to bench_augmentation_pipelines.py for cross-script comparability.
# ---------------------------------------------------------------------------
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

# d-group under AGGRESSIVE reordering — geo ops are grouped before fusion.
# Format: (label, nb_geom, albu_slice, kornia_slice, tv_slice)
# nb_geom = number of geo ops in the interleaved sequence.
_MIXED_AGR_CASES: list[tuple[str, int, list, list, list]] = [
    ("d01_mixed_g3c2__agr", 3, _D_POOL_ALB[:5], _D_POOL_K[:5], _D_POOL_TV[:5]),
    ("d02_mixed_g3c3__agr", 3, _D_POOL_ALB[:6], _D_POOL_K[:6], _D_POOL_TV[:6]),
    ("d03_mixed_g4c3__agr", 4, _D_POOL_ALB[:7], _D_POOL_K[:7], _D_POOL_TV[:7]),
    ("d04_mixed_g4c4__agr", 4, _D_POOL_ALB[:8], _D_POOL_K[:8], _D_POOL_TV[:8]),
    ("d05_mixed_g5c4__agr", 5, _D_POOL_ALB[:9], _D_POOL_K[:9], _D_POOL_TV[:9]),
]


def main() -> None:
    """Run all 45 cases and print the composite score and theoretical target."""
    # Fix random state for reproducibility across runs.
    torch.manual_seed(0)
    np.random.seed(0)

    boosts: list[float] = []

    # ── a-group + b-group: Kornia and TorchVision ────────────────────────────
    for _label, _num_geometric_ops, k_tfms, tv_tfms in _CASES:
        for _backend, tfms, native_fn in [
            ("kornia", k_tfms, lambda transform_list: K.AugmentationSequential(*transform_list)),
            ("torchvision", tv_tfms, lambda transform_list: tv.Compose(transform_list)),
        ]:
            native = native_fn(copy.deepcopy(tfms))
            fused = FuseCompose(copy.deepcopy(tfms))
            native(_IMAGE_TENSOR)
            fused(_IMAGE_TENSOR)
            native_ms = _bench(native)
            fused_ms = _bench(fused)
            boosts.append(native_ms / fused_ms if fused_ms > 0 else 0.0)

    # ── a-group + b-group: Albumentations (dict-input path) ─────────────────
    for _label, _num_geometric_ops, albu_tfms in _ALBU_CASES:
        native = A.Compose(copy.deepcopy(albu_tfms))
        fused = FuseCompose(copy.deepcopy(albu_tfms))
        native(image=_IMAGE_NDARRAY)
        fused(image=_IMAGE_NDARRAY)
        native_ms = _bench_albu(native)
        fused_ms = _bench_albu(fused)
        boosts.append(native_ms / fused_ms if fused_ms > 0 else 0.0)

    # ── d-group AGGRESSIVE: Kornia, TorchVision, Albumentations ─────────────
    for _label, _num_geometric_ops, alb_tfms, k_tfms, tv_tfms in _MIXED_AGR_CASES:
        # Kornia
        native_k = K.AugmentationSequential(*copy.deepcopy(k_tfms))
        fused_k = FuseCompose(copy.deepcopy(k_tfms), reorder=ReorderPolicy.AGGRESSIVE)
        native_k(_IMAGE_TENSOR)
        fused_k(_IMAGE_TENSOR)
        native_ms = _bench(native_k)
        fused_ms = _bench(fused_k)
        boosts.append(native_ms / fused_ms if fused_ms > 0 else 0.0)

        # TorchVision
        native_tv = tv.Compose(copy.deepcopy(tv_tfms))
        fused_tv = FuseCompose(copy.deepcopy(tv_tfms), reorder=ReorderPolicy.AGGRESSIVE)
        native_tv(_IMAGE_TENSOR)
        fused_tv(_IMAGE_TENSOR)
        native_ms = _bench(native_tv)
        fused_ms = _bench(fused_tv)
        boosts.append(native_ms / fused_ms if fused_ms > 0 else 0.0)

        # Albumentations
        native_alb = A.Compose(copy.deepcopy(alb_tfms))
        fused_alb = FuseCompose(copy.deepcopy(alb_tfms), reorder=ReorderPolicy.AGGRESSIVE)
        native_alb(image=_IMAGE_NDARRAY)
        fused_alb(image=_IMAGE_NDARRAY)
        native_ms = _bench_albu(native_alb)
        fused_ms = _bench_albu(fused_alb)
        boosts.append(native_ms / fused_ms if fused_ms > 0 else 0.0)

    score = statistics.geometric_mean(boosts)

    # Theoretical ceiling: geomean(num_geometric_ops per case across all backends).
    # Each sequence contributes its num_geometric_ops once per backend (3 backends total).
    all_num_geometric_ops = (
        [num_geometric_ops for _, num_geometric_ops, _, _ in _CASES] * 2  # kornia + torchvision
        + [num_geometric_ops for _, num_geometric_ops, _ in _ALBU_CASES]  # albumentations
        + [num_geometric_ops for _, num_geometric_ops, _, _, _ in _MIXED_AGR_CASES] * 3  # d-group: k + tv + alb
    )
    theoretical_target = statistics.geometric_mean(all_num_geometric_ops)

    print(f"real_score={score:.4f}")
    print(f"theoretical_target={theoretical_target:.4f}")


if __name__ == "__main__":
    main()
