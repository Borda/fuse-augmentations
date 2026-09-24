"""Regression tests for LOW-severity segment fixes (AFF-3, K07, F03).

- AFF-3: the cv2/numpy fast paths must clone the composed matrix into ``last_matrix``
  so a matrix retained from call N is not mutated in place by call N+1.
- K07: a directly instantiated ``ExactAffineSegment`` routes non-flip D4 pixels,
  masks, and coordinate targets from one sampled matrix.
- F03: a ``requires_grad`` input must take the differentiable CPU path and propagate
  a finite, non-zero gradient to its source image.

"""

from __future__ import annotations

import math

import pytest
import torch

from fuse_augmentations._compat import _CV2_AVAILABLE
from fuse_augmentations.affine.matrix import rotation_matrix, scale_matrix
from fuse_augmentations.affine.segment import ExactAffineSegment, FusedAffineSegment
from fuse_augmentations.targets import transform_bbox_xyxy, transform_keypoints
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
    """Complete adapter stub for a sampled counter-clockwise quarter turn."""

    def category(self, transform):
        """Return the transform's category attribute (default GEOMETRIC_EXACT)."""
        return getattr(transform, "_category", TransformCategory.GEOMETRIC_EXACT)

    def exact_flip_dims(self, transform):
        """Raise for a non-flip discrete op (rot90/transpose have no flip axes)."""
        raise NotImplementedError("non-flip exact op exposes no flip dims")

    def sample_params(self, transform, input_shape, device):
        """Return one counter-clockwise quarter turn for every active sample."""
        return {"k90": torch.ones(input_shape[0], device=device, dtype=torch.int64)}

    def build_matrix(self, transform, params, height, width):
        """Return the centre-coordinate matrix for the sampled quarter turn."""
        angles = -params["k90"].to(dtype=torch.float32) * (math.pi / 2)
        return rotation_matrix(angles, height=height, width=width)

    def exact_apply(self, transform, image, *, params=None):
        """Apply the supplied sampled quarter turn losslessly to pixels and masks."""
        if params is None:
            raise AssertionError("non-flip exact application must consume the sampled parameters")
        k90 = params["k90"].to(device=image.device, dtype=torch.int64) % 4
        if k90.shape != (image.shape[0],):
            raise ValueError("k90 must contain one value per image")
        output = image.clone()
        for turns in range(1, 4):
            active = k90 == turns
            output[active] = torch.rot90(image[active], turns, dims=[2, 3])
        return output


class _ExactQuarterTurnTransform:
    """A non-flip exact transform (rot90) that always applies."""

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
    """F03: a ``requires_grad`` input uses a differentiable CPU segment path."""

    def test_requires_grad_input_propagates_finite_gradient(self):
        """A grad-tracking input retains a meaningful gradient through the CPU transform."""
        adapter = _StubAdapter()
        transforms = [_StatefulScaleTransform((0.5,)), _StatefulScaleTransform((0.5,))]
        seg = FusedAffineSegment(transforms, adapter)
        image = torch.rand(1, 3, 16, 16, requires_grad=True)

        out = seg(image)
        out.square().mean().backward()

        assert out.requires_grad
        assert image.grad is not None
        assert torch.isfinite(image.grad).all()
        assert image.grad.abs().sum() > 0


class TestExactSegmentNonFlipTargets:
    """K07: a sampled quarter turn shares its matrix with every target modality."""

    def test_nonflip_exact_routes_pixels_mask_and_coordinates_from_same_params(self):
        """A quarter turn maps image, mask, boxes, and keypoints by one supplied draw."""
        seg = ExactAffineSegment([_ExactQuarterTurnTransform()], _ExactStubAdapter())
        image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
        mask = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        boxes = torch.tensor([[[1.0, 2.0, 4.0, 6.0]]])
        keypoints = torch.tensor([[[1.0, 2.0], [6.0, 4.0]]])
        aux_targets = {"mask": mask, "bbox_xyxy": boxes, "keypoints": keypoints}
        expected_matrix = rotation_matrix(torch.tensor([-math.pi / 2]), height=8, width=8)

        out_image, out_aux = seg(image, aux_targets)

        torch.testing.assert_close(out_image, torch.rot90(image, 1, dims=[2, 3]))
        torch.testing.assert_close(out_aux["mask"], torch.rot90(mask, 1, dims=[2, 3]))
        torch.testing.assert_close(out_aux["bbox_xyxy"], transform_bbox_xyxy(boxes, expected_matrix))
        torch.testing.assert_close(out_aux["keypoints"], transform_keypoints(keypoints, expected_matrix))
        torch.testing.assert_close(seg.last_matrix, expected_matrix)
