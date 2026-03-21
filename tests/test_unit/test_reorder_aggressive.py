"""Unit tests for ReorderPolicy.AGGRESSIVE (Phase B.3)."""
from __future__ import annotations

import pytest
import torch

from fuse_augmentations._types import ReorderPolicy, TransformCategory


class _StubGeo:
    _category = TransformCategory.GEOMETRIC_INTERP
    p = 1.0
    same_on_batch = False

    def __init__(self, name: str = "Geo") -> None:
        self.name = name


class _StubExact:
    _category = TransformCategory.GEOMETRIC_EXACT
    p = 1.0
    same_on_batch = False

    def __init__(self, name: str = "Exact") -> None:
        self.name = name


class _StubPointwise:
    _category = TransformCategory.POINTWISE
    p = 1.0
    same_on_batch = False

    def __init__(self, name: str = "PW") -> None:
        self.name = name


class _StubBarrier:
    _category = TransformCategory.SPATIAL_KERNEL
    p = 1.0
    same_on_batch = False

    def __init__(self, name: str = "Barrier") -> None:
        self.name = name


class _StubAdapter:
    def category(self, t):
        return getattr(t, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, t, shape, device):
        return {"_batch_size": torch.tensor([shape[0]])}

    def build_matrix(self, t, params, H, W):
        B = int(params["_batch_size"].item())
        return torch.eye(3).unsqueeze(0).expand(B, -1, -1)

    def call_nonfused(self, t, image, **kwargs):
        return image

    def exact_flip_dims(self, t):
        return [3]


class TestAggressiveNoLongerRaisesError:
    def test_fused_compose_accepts_aggressive_policy(self):
        from fuse_augmentations._compose import FusedCompose

        # Should NOT raise NotImplementedError anymore
        pipe = FusedCompose([], reorder=ReorderPolicy.AGGRESSIVE)
        assert pipe.reorder == ReorderPolicy.AGGRESSIVE

    def test_from_params_accepts_aggressive_policy(self):
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(
            rotation=(-30, 30), reorder=ReorderPolicy.AGGRESSIVE
        )
        assert pipe is not None


class TestReorderAggressiveFunction:
    def test_reorder_aggressive_importable(self):
        from fuse_augmentations.affine._segment import reorder_aggressive

        assert callable(reorder_aggressive)

    def test_aggressive_moves_pointwise_after_geometric(self):
        from fuse_augmentations.affine._segment import reorder_aggressive

        adapter = _StubAdapter()
        geo = _StubGeo()
        pw = _StubPointwise()
        result = reorder_aggressive([geo, pw], adapter)
        cats = [adapter.category(t) for t in result]
        assert cats == [TransformCategory.GEOMETRIC_INTERP, TransformCategory.POINTWISE]

    def test_aggressive_same_result_as_pointwise_for_standard_pipeline(self):
        """AGGRESSIVE and POINTWISE produce the same ordering for a standard pipeline."""
        from fuse_augmentations.affine._segment import reorder_aggressive, reorder_pointwise

        adapter = _StubAdapter()
        geo1, geo2 = _StubGeo("g1"), _StubGeo("g2")
        pw1, pw2 = _StubPointwise("p1"), _StubPointwise("p2")

        pipeline = [geo1, pw1, geo2, pw2]
        result_pointwise = reorder_pointwise(pipeline, adapter)
        result_aggressive = reorder_aggressive(pipeline, adapter)

        cats_pw = [adapter.category(t) for t in result_pointwise]
        cats_agg = [adapter.category(t) for t in result_aggressive]
        assert cats_pw == cats_agg

    def test_aggressive_never_crosses_spatial_kernel_barrier(self):
        from fuse_augmentations.affine._segment import reorder_aggressive

        adapter = _StubAdapter()
        pw = _StubPointwise()
        barrier = _StubBarrier()
        geo = _StubGeo()

        result = reorder_aggressive([pw, barrier, geo], adapter)
        # pw should stay before barrier -- NEVER crosses it
        positions = {id(pw): 0, id(barrier): 0, id(geo): 0}
        for i, t in enumerate(result):
            positions[id(t)] = i
        assert positions[id(pw)] < positions[id(barrier)]

    def test_aggressive_stable_idempotent(self):
        """Applying AGGRESSIVE twice produces the same result as once."""
        from fuse_augmentations.affine._segment import reorder_aggressive

        adapter = _StubAdapter()
        geo = _StubGeo()
        pw = _StubPointwise()

        once = reorder_aggressive([geo, pw], adapter)
        twice = reorder_aggressive(once, adapter)

        cats_once = [adapter.category(t) for t in once]
        cats_twice = [adapter.category(t) for t in twice]
        assert cats_once == cats_twice


class TestAggressivePipelineImage:
    """End-to-end: AGGRESSIVE policy produces correct pipeline structure."""

    kornia = pytest.importorskip("kornia")

    def test_aggressive_pipeline_runs_without_error(self):
        import kornia.augmentation as K

        from fuse_augmentations import Compose

        pipe = Compose(
            [K.RandomRotation(degrees=30, p=1.0), K.RandomHorizontalFlip(p=1.0)],
            reorder=ReorderPolicy.AGGRESSIVE,
        )
        x = torch.zeros(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == (2, 3, 32, 32)
