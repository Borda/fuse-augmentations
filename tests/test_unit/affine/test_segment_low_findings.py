"""Regression tests for LOW-severity segment fixes (AFF-3, AFF-5, XB-4).

- AFF-3: the cv2/numpy fast paths must clone the composed matrix into ``last_matrix``
  so a matrix retained from call N is not mutated in place by call N+1.
- AFF-5: a directly instantiated ``ExactAffineSegment`` with a non-flip exact op must
  raise on box/keypoint auxiliary targets instead of passing them through untransformed.
- XB-4: a ``requires_grad`` input fed into a cv2 segment must run (detach before the raw
  ``.numpy()`` conversion) and return a detached output.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _CV2_AVAILABLE
from fuse_augmentations.affine.matrix import scale_matrix
from fuse_augmentations.affine.segment import ExactAffineSegment, FusedAffineSegment
from fuse_augmentations.types import TransformCategory


class _StubAdapter:
    """Minimal ``TransformAdapter`` for the fused/cv2 path -- no Kornia dependency."""

    def category(self, transform):
        """Return the transform's category attribute (default SPATIAL_KERNEL)."""
        return getattr(transform, "_category", TransformCategory.SPATIAL_KERNEL)

    def sample_params(self, transform, input_shape, device):
        """Return minimal canonical params carrying the batch size."""
        return {"_batch_size": torch.tensor([input_shape[0]])}

    def build_matrix(self, transform, params, height, width):
        """Delegate to ``transform.matrix_fn`` or return identity matrices."""
        batch_size = int(params["_batch_size"].item())
        if hasattr(transform, "matrix_fn"):
            return transform.matrix_fn(batch_size, height, width)
        return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).clone()

    def call_nonfused(self, transform, image, **kwargs):
        """Pass the image through unchanged (unused on the cv2 fast path)."""
        return image


class _StatefulScaleTransform:
    """Geometric transform whose scale (and thus matrix) changes on each build.

    Successive forward passes therefore compose to *different* matrices, so a
    ``last_matrix`` aliased to a reused buffer would visibly change after the next
    call -- exactly the AFF-3 regression this drives.

    """

    def __init__(self, scales: tuple[float, ...]) -> None:
        self.p = 1.0
        self._category = TransformCategory.GEOMETRIC_INTERP
        self._scales = scales
        self._calls = 0

    def matrix_fn(self, batch, height, width):
        """Return a (batch, 3, 3) scale matrix, cycling through ``self._scales``."""
        scale = self._scales[min(self._calls, len(self._scales) - 1)]
        self._calls += 1
        factor = torch.full((batch,), scale)
        return scale_matrix(factor, factor, height=height, width=width)


class _ExactStubAdapter:
    """Adapter driving ``ExactAffineSegment`` with a non-flip exact op."""

    def category(self, transform):
        """Return the transform's category attribute (default GEOMETRIC_EXACT)."""
        return getattr(transform, "_category", TransformCategory.GEOMETRIC_EXACT)

    def exact_flip_dims(self, transform):
        """Raise for a non-flip discrete op (rot90/transpose have no flip axes)."""
        raise NotImplementedError("non-flip exact op exposes no flip dims")

    def exact_apply(self, transform, image):
        """Apply the exact op as an identity for the stub (channels preserved)."""
        return image


class _ExactTransform:
    """A stub non-flip exact transform (e.g. rot90) that always applies."""

    def __init__(self) -> None:
        self.p = 1.0
        self._category = TransformCategory.GEOMETRIC_EXACT


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed -- fast path unavailable")
class TestLastMatrixStabilityCv2FastPath:
    """AFF-3: a retained ``last_matrix`` is stable across a following forward call."""

    def test_retained_matrix_unchanged_after_next_call(self):
        """The matrix returned by call N is not mutated in place by call N+1."""
        adapter = _StubAdapter()
        transforms = [_StatefulScaleTransform((0.5, 0.25)), _StatefulScaleTransform((1.0, 1.0))]
        seg = FusedAffineSegment(transforms, adapter)
        image = torch.rand(1, 3, 16, 16)  # B=1 CPU -> cv2 fast path

        seg(image)
        retained = seg.last_matrix
        snapshot = retained.clone()
        seg(image)  # call N+1 overwrites the reused cv2 buffer with a different matrix

        assert torch.equal(retained, snapshot)


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed -- fast path unavailable")
class TestRequiresGradThroughCv2Segment:
    """XB-4: a ``requires_grad`` input runs through the cv2 segment and detaches output."""

    def test_requires_grad_input_runs_and_returns_detached(self):
        """A grad-tracking input does not raise and yields a detached result."""
        adapter = _StubAdapter()
        transforms = [_StatefulScaleTransform((0.5,)), _StatefulScaleTransform((0.5,))]
        seg = FusedAffineSegment(transforms, adapter)
        image = torch.rand(1, 3, 16, 16, requires_grad=True)

        out = seg(image)

        assert out.requires_grad is False


class TestExactSegmentCoordAuxGuard:
    """AFF-5: non-flip exact op + coordinate aux must raise, mask-only must not."""

    @pytest.mark.parametrize(
        "coord_key",
        [
            pytest.param("bbox_xyxy", id="bbox_xyxy"),
            pytest.param("bbox_xywh", id="bbox_xywh"),
            pytest.param("keypoints", id="keypoints"),
        ],
    )
    def test_coord_aux_raises_on_non_flip_exact_op(self, coord_key):
        """Box/keypoint aux on a non-flip exact op raises instead of silent passthrough."""
        seg = ExactAffineSegment([_ExactTransform()], _ExactStubAdapter())
        image = torch.rand(1, 3, 8, 8)
        aux_targets = {coord_key: torch.zeros(1, 1, 4)}

        with pytest.raises(RuntimeError, match="non-flip exact op"):
            seg(image, aux_targets)

    def test_mask_only_aux_passes_through_non_flip_exact_op(self):
        """Mask-only aux on a non-flip exact op is handled without raising (contract case)."""
        seg = ExactAffineSegment([_ExactTransform()], _ExactStubAdapter())
        image = torch.rand(1, 3, 8, 8)
        aux_targets = {"mask": torch.zeros(1, 1, 8, 8)}

        _, out_aux = seg(image, aux_targets)

        assert "mask" in out_aux
