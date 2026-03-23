"""Phase E demo tests — crystallise the API contract before implementation.

Each test demonstrates the intended feature behaviour and MUST FAIL against
the current codebase (the features do not exist yet).

E.1: from_params backend= kwarg + backend-native kwargs in TransformSpec.params
E.2: FusedColorSegment — build_segments folds POINTWISE_LINEAR ops into it
E.3: Adapter build_color_matrix — returns (B, 4, 4) homogeneous color matrices

Run to confirm all fail before implementing:
    pytest tests/test_unit/test_phase_e_demo.py -v
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# E.1 — from_params backend= kwarg
# ---------------------------------------------------------------------------


class TestE1FromParamsBackend:
    """from_params(specs=[...], backend=...) delegates to from_config semantics."""

    def test_from_params_specs_with_backend_runs(self):
        """from_params(specs=[...], backend='kornia') produces a working pipeline."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations._types import TransformSpec

        specs = [TransformSpec(op="hflip", params={}, p=0.5)]
        pipe = FusedCompose.from_params(specs=specs, backend="kornia")
        x = torch.zeros(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_from_params_specs_with_backend_same_segments_as_from_config(self):
        """from_params(specs=..., backend='kornia') produces the same segment structure as from_config."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations._types import TransformSpec

        specs = [
            TransformSpec(op="rotation", params={"degrees": (-30.0, 30.0)}, p=1.0),
            TransformSpec(op="hflip", params={}, p=0.5),
        ]
        pipe_params = FusedCompose.from_params(specs=specs, backend="kornia")
        pipe_config = FusedCompose.from_config(specs, backend="kornia")

        # Same number and kinds of segments
        desc_params = pipe_params.fusion_plan_descriptors
        desc_config = pipe_config.fusion_plan_descriptors
        assert len(desc_params) == len(desc_config)
        assert [d.kind for d in desc_params] == [d.kind for d in desc_config]

    def test_from_params_backend_native_kwarg_forwarded(self):
        """TransformSpec.params with a backend-native kwarg (same_on_batch) is forwarded without error."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations._types import TransformSpec

        specs = [
            TransformSpec(
                op="rotation",
                params={"degrees": (-15.0, 15.0), "same_on_batch": True},
                p=1.0,
            )
        ]
        pipe = FusedCompose.from_params(specs=specs, backend="kornia")
        x = torch.zeros(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])

    def test_from_params_unknown_backend_native_kwarg_raises(self):
        """TransformSpec.params with an unrecognised kwarg raises ValueError."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        from fuse_augmentations._compose import FusedCompose
        from fuse_augmentations._types import TransformSpec

        specs = [
            TransformSpec(
                op="rotation",
                params={"degrees": (-15.0, 15.0), "totally_unknown_kwarg": 42},
                p=1.0,
            )
        ]
        with pytest.raises((ValueError, TypeError)):
            FusedCompose.from_params(specs=specs, backend="kornia")

    def test_from_params_existing_call_sig_unchanged(self):
        """Existing from_params(rotation=...) call signature is not broken."""
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0), hflip_p=0.5)
        x = torch.zeros(2, 3, 32, 32)
        out = pipe(x)
        assert out.shape == torch.Size([2, 3, 32, 32])


# ---------------------------------------------------------------------------
# E.2 — FusedColorSegment
# ---------------------------------------------------------------------------


class TestE2FusedColorSegment:
    """build_segments folds consecutive POINTWISE_LINEAR ops into FusedColorSegment."""

    def test_fused_color_segment_importable(self):
        """FusedColorSegment can be imported from fuse_augmentations.affine._segment."""
        from fuse_augmentations.affine._segment import FusedColorSegment  # noqa: F401

    def test_build_segments_folds_pointwise_linear_run(self):
        """Two consecutive POINTWISE_LINEAR ops → single FusedColorSegment."""
        from fuse_augmentations.affine._segment import FusedColorSegment, build_segments
        from fuse_augmentations._types import TransformCategory

        class _PLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _PLAdapter:
            def category(self, t):
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, t, shape, device):
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_matrix(self, t, params, H, W):
                B = int(params["_batch_size"].item())
                return torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()

            def build_color_matrix(self, t, params):
                B = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

            def call_nonfused(self, t, image, **kw):
                return image

        t1, t2 = _PLTransform(), _PLTransform()
        adapter = _PLAdapter()
        segs = build_segments([t1, t2], adapter, "bilinear", "zeros")
        color_segs = [s for s in segs if isinstance(s, FusedColorSegment)]
        assert len(color_segs) == 1, f"Expected 1 FusedColorSegment, got segments: {segs}"

    def test_fused_color_segment_forward_returns_correct_shape(self):
        """FusedColorSegment.forward returns (B, C, H, W) matching input."""
        from fuse_augmentations.affine._segment import FusedColorSegment
        from fuse_augmentations._types import TransformCategory

        class _IdentityPLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityPLAdapter:
            def category(self, t):
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, t, shape, device):
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, t, params):
                B = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

            def call_nonfused(self, t, image, **kw):
                return image

        t = _IdentityPLTransform()
        seg = FusedColorSegment([t], _IdentityPLAdapter())
        x = torch.rand(2, 3, 16, 16)
        out = seg(x)
        assert out.shape == x.shape

    def test_fused_color_segment_identity_matrix_preserves_image(self):
        """Applying identity color matrices does not alter pixel values."""
        from fuse_augmentations.affine._segment import FusedColorSegment
        from fuse_augmentations._types import TransformCategory

        class _IdentityPLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityPLAdapter:
            def category(self, t):
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, t, shape, device):
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, t, params):
                B = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

            def call_nonfused(self, t, image, **kw):
                return image

        t = _IdentityPLTransform()
        seg = FusedColorSegment([t], _IdentityPLAdapter())
        x = torch.rand(2, 3, 16, 16)
        out = seg(x)
        torch.testing.assert_close(out, x)

    def test_build_segments_fallback_to_passthrough_when_adapter_raises(self):
        """If any adapter in a POINTWISE_LINEAR run raises NotImplementedError, fall back to passthrough."""
        from fuse_augmentations.affine._segment import FusedColorSegment, build_segments
        from fuse_augmentations._types import TransformCategory

        class _PLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _NoColorMatrixAdapter:
            def category(self, t):
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, t, shape, device):
                return {}

            def build_matrix(self, t, params, H, W):
                return torch.eye(3).unsqueeze(0)

            def build_color_matrix(self, t, params):
                raise NotImplementedError

            def call_nonfused(self, t, image, **kw):
                return image

        t1, t2 = _PLTransform(), _PLTransform()
        adapter = _NoColorMatrixAdapter()
        segs = build_segments([t1, t2], adapter, "bilinear", "zeros")
        color_segs = [s for s in segs if isinstance(s, FusedColorSegment)]
        # Fallback: no FusedColorSegment, transforms left as passthrough
        assert len(color_segs) == 0


# ---------------------------------------------------------------------------
# E.3 — Adapter build_color_matrix
# ---------------------------------------------------------------------------


class TestE3AdapterBuildColorMatrix:
    """Each adapter exposes build_color_matrix returning (B, 4, 4) tensors."""

    def test_kornia_adapter_has_build_color_matrix(self):
        """KorniaAdapter.build_color_matrix exists as a callable."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        assert callable(getattr(KorniaAdapter, "build_color_matrix", None))

    def test_kornia_random_brightness_returns_4x4(self):
        """KorniaAdapter.build_color_matrix for RandomBrightness returns (B, 4, 4)."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        import kornia.augmentation as K
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        B = 3
        transform = K.RandomBrightness(brightness=(0.5, 1.5), p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (B, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)
        assert mat.shape == (B, 4, 4), f"Expected ({B}, 4, 4), got {mat.shape}"

    def test_kornia_color_jitter_returns_4x4(self):
        """KorniaAdapter.build_color_matrix for ColorJitter returns (B, 4, 4)."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        import kornia.augmentation as K
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        B = 2
        transform = K.ColorJitter(brightness=0.2, contrast=0.2, p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (B, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)
        assert mat.shape == (B, 4, 4)

    def test_torchvision_adapter_has_build_color_matrix(self):
        """TorchVisionAdapter.build_color_matrix exists as a callable."""
        pytest.importorskip("torchvision")
        from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

        assert callable(getattr(TorchVisionAdapter, "build_color_matrix", None))

    def test_albumentations_adapter_has_build_color_matrix(self):
        """AlbumentationsAdapter.build_color_matrix exists as a callable."""
        pytest.importorskip("albumentations")
        from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter

        assert callable(getattr(AlbumentationsAdapter, "build_color_matrix", None))

    def test_build_color_matrix_last_row_is_homogeneous(self):
        """build_color_matrix bottom row is [0, 0, 0, 1] for all batch items."""
        kornia = pytest.importorskip("kornia")  # noqa: F841
        import kornia.augmentation as K
        from fuse_augmentations.adapters._kornia import KorniaAdapter

        B = 4
        transform = K.RandomBrightness(brightness=(0.8, 1.2), p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (B, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)

        expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for b in range(B):
            torch.testing.assert_close(mat[b, 3, :], expected_last_row)
