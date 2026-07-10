"""Integration coverage for per-call matrices and plan caching."""

from __future__ import annotations

import concurrent.futures
import pickle

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations.affine.matrix import _singularity_threshold, inv3x3
from fuse_augmentations.affine.segment import _inv3x3_affine_np


def test_return_matrix_is_thread_local_for_different_image_shapes() -> None:
    """Concurrent calls return their own image-dtype pixel matrices."""
    pipe = Compose.from_params(rotation=(15.0, 15.0))

    def run(size: int) -> tuple[int, torch.Tensor]:
        image = torch.rand(1, 3, size, size)
        _output, matrix = pipe(image, return_matrix=True)
        assert matrix is not None
        return size, matrix

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, (8, 16)))

    matrices = dict(results)
    assert matrices[8].dtype == torch.float32
    assert matrices[16].dtype == torch.float32
    assert not torch.allclose(matrices[8], matrices[16])


def test_return_matrix_matches_compatibility_property() -> None:
    """The additive return value equals the matrix retained for compatibility."""
    pipe = Compose.from_params(rotation=(10.0, 10.0))
    _output, returned = pipe(torch.rand(1, 3, 8, 8), return_matrix=True)

    assert returned is not None
    assert pipe.transform_matrix is not None
    torch.testing.assert_close(returned, pipe.transform_matrix)


def test_singularity_threshold_is_shared_across_torch_and_numpy_paths() -> None:
    """Both inversion implementations reject determinants below the shared float32 threshold."""
    threshold = _singularity_threshold(torch.float32)
    matrix = torch.eye(3).unsqueeze(0)
    matrix[:, 0, 0] = threshold * 0.5
    with pytest.raises(ValueError, match="Near-singular"):
        inv3x3(matrix)

    numpy_matrix = matrix[0].numpy().astype("float64")
    with pytest.raises(ValueError, match="Singular affine"):
        _inv3x3_affine_np(numpy_matrix)


def test_fusion_plan_and_descriptors_are_cached_and_pickle_safe() -> None:
    """Plan properties return stable cached objects and survive serialization."""
    pipe = Compose.from_params(rotation=(0.0, 0.0), hflip_p=1.0)
    plan = pipe.fusion_plan
    descriptors = pipe.fusion_plan_descriptors
    assert pipe.fusion_plan is plan
    assert pipe.fusion_plan_descriptors is descriptors

    restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301
    assert restored.fusion_plan == plan
    assert restored.fusion_plan_descriptors == descriptors
