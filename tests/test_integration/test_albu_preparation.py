"""Albumentations tensor preparation preserves seeded geometry and RNG streams."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
from fuse_augmentations.affine.segment import AlbuFusedAffineSegment

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu


def _seeded_transforms(factory: Callable[[], list[object]], *, same_on_batch: bool) -> list[object]:
    """Build transforms with reproducible backend-owned random streams."""
    transforms = factory()
    for index, transform in enumerate(transforms):
        transform.set_random_seed(40 + index)  # type: ignore[attr-defined]
        transform.same_on_batch = same_on_batch  # type: ignore[attr-defined]
    return transforms


def _run_seeded(
    factory: Callable[[], list[object]],
    image: torch.Tensor,
    *,
    same_on_batch: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...], tuple[float, ...]]:
    """Run one tensor pipeline and capture the next global and backend RNG values."""
    np.random.seed(19)
    transforms = _seeded_transforms(factory, same_on_batch=same_on_batch)
    output, matrix = Compose(transforms, execution="torch")(image.clone(), return_matrix=True)
    assert matrix is not None
    global_tail = tuple(float(value) for value in np.random.rand(4))
    backend_tail = tuple(float(transform.py_random.random()) for transform in transforms)  # type: ignore[attr-defined]
    return output, matrix, global_tail, backend_tail


def _legacy_adapter_matrix(transform: object) -> np.ndarray:
    """Sample one matrix through the tensor preparation path replaced by F10."""
    adapter = AlbumentationsAdapter()
    params = adapter.sample_params(transform, (1, 3, 32, 40), torch.device("cpu"))
    return adapter.build_matrix(transform, params, 32, 40)[0].double().cpu().numpy()


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
@pytest.mark.parametrize(
    ("kind", "factory"),
    [
        pytest.param("rotate", lambda: albu.Rotate(limit=(-11.0, 17.0), crop_border=False, p=1.0), id="rotate"),
        pytest.param("affine", lambda: albu.Affine(rotate=(-11.0, 17.0), scale=(0.7, 1.1), p=1.0), id="affine"),
        pytest.param(
            "perspective", lambda: albu.Perspective(scale=(0.04, 0.12), keep_size=True, p=1.0), id="perspective"
        ),
    ],
)
@pytest.mark.parametrize("same_on_batch", [False, True], ids=("per-sample", "shared"))
def test_native_preparation_matches_legacy_adapter_matrix_and_rng(
    kind: str,
    factory: Callable[[], object],
    same_on_batch: bool,
) -> None:
    """F10's direct sampler reproduces the replaced adapter matrix and RNG tails."""
    np.random.seed(19)
    legacy = _seeded_transforms(lambda: [factory()], same_on_batch=same_on_batch)[0]
    expected = _legacy_adapter_matrix(legacy)
    legacy_global_tail = tuple(float(value) for value in np.random.rand(4))
    legacy_backend_tail = float(legacy.py_random.random())  # type: ignore[attr-defined]

    np.random.seed(19)
    optimized = _seeded_transforms(lambda: [factory()], same_on_batch=same_on_batch)[0]
    adapter = AlbumentationsAdapter()
    tag = AlbuFusedAffineSegment._classify_transforms([optimized], adapter)[0]
    actual = AlbuFusedAffineSegment._sample_matrix_numpy(
        adapter,
        optimized,
        tag,
        channels=3,
        height=32,
        width=40,
        tensor_roundtrip=True,
    )
    optimized_global_tail = tuple(float(value) for value in np.random.rand(4))
    optimized_backend_tail = float(optimized.py_random.random())  # type: ignore[attr-defined]

    np.testing.assert_array_equal(actual, expected, strict=True, err_msg=kind)
    assert optimized_global_tail == legacy_global_tail
    assert optimized_backend_tail == legacy_backend_tail


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
@pytest.mark.parametrize(
    ("kind", "factory"),
    [
        pytest.param(
            "affine",
            lambda: [
                albu.Affine(rotate=(13.0, 13.0), scale=(0.8, 0.8), p=1.0),
                albu.Rotate(limit=(7.0, 7.0), crop_border=False, p=1.0),
            ],
            id="affine",
        ),
        pytest.param(
            "projective",
            lambda: [albu.Perspective(scale=(0.08, 0.08), keep_size=True, p=1.0)],
            id="projective",
        ),
    ],
)
@pytest.mark.parametrize("probability", [0.0, 1.0, 0.45], ids=("p0", "p1", "mixed"))
@pytest.mark.parametrize("same_on_batch", [False, True], ids=("per-sample", "shared"))
def test_tensor_native_preparation_preserves_seeded_results_and_rng(
    kind: str,
    factory: Callable[[], list[object]],
    probability: float,
    same_on_batch: bool,
) -> None:
    """Affine/projective preparation stays deterministic across probability modes."""

    def with_probability() -> list[object]:
        transforms = factory()
        for transform in transforms:
            transform.p = probability  # type: ignore[attr-defined]
        return transforms

    image = torch.rand(4, 3, 32, 40, generator=torch.Generator().manual_seed(9))
    first = _run_seeded(with_probability, image, same_on_batch=same_on_batch)
    second = _run_seeded(with_probability, image, same_on_batch=same_on_batch)

    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0, msg=kind)
    torch.testing.assert_close(first[1], second[1], rtol=0.0, atol=0.0, msg=kind)
    assert first[2:] == second[2:]
