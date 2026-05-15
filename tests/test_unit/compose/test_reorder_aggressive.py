"""Unit tests for ReorderPolicy.AGGRESSIVE."""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose, FusedCompose
from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.affine.segment import reorder_aggressive, reorder_pointwise
from fuse_augmentations.types import ReorderPolicy, TransformCategory

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug


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
    def category(self, tfm):
        return getattr(tfm, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, tfm, shape, device):
        return {"_batch_size": torch.tensor([shape[0]])}

    def build_matrix(self, tfm, params, height, width):
        batch_size = int(params["_batch_size"].item())
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)

    def call_nonfused(self, tfm, image, **kwargs):
        return image

    def exact_flip_dims(self, tfm):
        return [3]


class TestAggressiveNoLongerRaisesError:
    """ReorderPolicy.AGGRESSIVE is now a fully accepted policy, not a placeholder."""

    def test_fused_compose_accepts_aggressive_policy(self):
        """FusedCompose constructor accepts AGGRESSIVE and stores it on the instance.

        Earlier versions raised NotImplementedError for AGGRESSIVE; this test guards against regressing to that
        placeholder behaviour.

        """
        # Should NOT raise NotImplementedError anymore
        pipe = FusedCompose([], reorder=ReorderPolicy.AGGRESSIVE)
        assert pipe.reorder == ReorderPolicy.AGGRESSIVE

    def test_from_params_accepts_aggressive_policy(self):
        """FusedCompose.from_params accepts AGGRESSIVE as the reorder kwarg."""
        pipe = FusedCompose.from_params(rotation=(-30, 30), reorder=ReorderPolicy.AGGRESSIVE)
        assert pipe is not None


class TestReorderAggressiveFunction:
    """Behavioural contract of the reorder_aggressive function on stub transforms."""

    def test_reorder_aggressive_importable(self):
        """reorder_aggressive is exported as a callable from affine._segment."""
        assert callable(reorder_aggressive)

    def test_aggressive_moves_pointwise_after_geometric(self):
        """[Geo, Pointwise] is reordered so the pointwise op runs after the geometric op."""
        adapter = _StubAdapter()
        geo = _StubGeo()
        pointwise = _StubPointwise()
        result = reorder_aggressive([geo, pointwise], adapter)
        cats = [adapter.category(transform) for transform in result]
        assert cats == [TransformCategory.GEOMETRIC_INTERP, TransformCategory.POINTWISE]

    def test_aggressive_same_result_as_pointwise_for_standard_pipeline(self):
        """AGGRESSIVE and POINTWISE produce the same ordering for a standard pipeline.

        For pipelines without projective ops or barriers between geometric stretches, the AGGRESSIVE policy degenerates
        to the POINTWISE policy — extra freedom is unused and both produce identical orderings.

        """
        adapter = _StubAdapter()
        geo1, geo2 = _StubGeo("g1"), _StubGeo("g2")
        pw1, pw2 = _StubPointwise("p1"), _StubPointwise("p2")

        pipeline = [geo1, pw1, geo2, pw2]
        result_pointwise = reorder_pointwise(pipeline, adapter)
        result_aggressive = reorder_aggressive(pipeline, adapter)

        cats_pw = [adapter.category(transform) for transform in result_pointwise]
        cats_agg = [adapter.category(transform) for transform in result_aggressive]
        assert cats_pw == cats_agg

    def test_aggressive_never_crosses_spatial_kernel_barrier(self):
        """Pointwise ops must not be moved across a SPATIAL_KERNEL barrier even under AGGRESSIVE.

        Spatial-kernel ops (e.g. blur) read neighbouring pixels, so a pointwise op appearing before the kernel cannot be
        deferred past it without changing semantics. AGGRESSIVE respects this invariant.

        """
        adapter = _StubAdapter()
        pointwise = _StubPointwise()
        barrier = _StubBarrier()
        geo = _StubGeo()

        result = reorder_aggressive([pointwise, barrier, geo], adapter)
        # pointwise should stay before barrier -- NEVER crosses it
        positions = {id(pointwise): 0, id(barrier): 0, id(geo): 0}
        for idx, transform in enumerate(result):
            positions[id(transform)] = idx
        assert positions[id(pointwise)] < positions[id(barrier)]

    def test_aggressive_stable_idempotent(self):
        """Applying AGGRESSIVE twice produces the same result as once.

        Idempotency guarantees that reorder is a fixed-point operation: pipelines
        already in canonical aggressive order do not get rewritten on re-application.

        """
        adapter = _StubAdapter()
        geo = _StubGeo()
        pointwise = _StubPointwise()

        once = reorder_aggressive([geo, pointwise], adapter)
        twice = reorder_aggressive(once, adapter)

        cats_once = [adapter.category(transform) for transform in once]
        cats_twice = [adapter.category(transform) for transform in twice]
        assert cats_once == cats_twice


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestAggressivePipelineImage:
    """End-to-end: AGGRESSIVE policy produces correct pipeline structure."""

    def test_aggressive_pipeline_runs_without_error(self):
        """Compose with AGGRESSIVE policy applied to a real kornia pipeline runs and preserves shape."""
        pipe = Compose(
            [
                kornia_aug.RandomRotation(degrees=30, p=1.0),
                kornia_aug.RandomHorizontalFlip(p=1.0),
            ],
            reorder=ReorderPolicy.AGGRESSIVE,
        )
        image = torch.zeros(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == (2, 3, 32, 32)
