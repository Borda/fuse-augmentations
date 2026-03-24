"""Contract and regression tests for color fusion: from_params backend= kwarg,
FusedColorSegment, and adapter build_color_matrix implementations.

Covers:
- from_params(specs=..., backend=...) delegation to from_config semantics
- build_segments folding of POINTWISE_LINEAR ops into FusedColorSegment
- Adapter build_color_matrix returning (B, 4, 4) homogeneous color matrices
- FusedColorSegment forward edge cases (non-RGB, aux_targets)
- _try_build_color_matrix probe robustness
- scale_x/scale_y ValueError guard when backend= is set

Run to verify behaviour and guard against regressions:
    pytest tests/test_unit/test_color_fusion.py -v

"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# from_params backend= kwarg
# ---------------------------------------------------------------------------


class TestFromParamsBackend:
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
# FusedColorSegment — build_segments integration
# ---------------------------------------------------------------------------


class TestFusedColorSegment:
    """build_segments folds consecutive POINTWISE_LINEAR ops into FusedColorSegment."""

    def test_fused_color_segment_importable(self):
        """FusedColorSegment can be imported from fuse_augmentations.affine._segment."""
        from fuse_augmentations.affine._segment import FusedColorSegment  # noqa: F401

    def test_build_segments_folds_pointwise_linear_run(self):
        """Two consecutive POINTWISE_LINEAR ops → single FusedColorSegment."""
        from fuse_augmentations._types import TransformCategory
        from fuse_augmentations.affine._segment import FusedColorSegment, build_segments

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
        from fuse_augmentations._types import TransformCategory
        from fuse_augmentations.affine._segment import FusedColorSegment

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
        from fuse_augmentations._types import TransformCategory
        from fuse_augmentations.affine._segment import FusedColorSegment

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
        from fuse_augmentations._types import TransformCategory
        from fuse_augmentations.affine._segment import FusedColorSegment, build_segments

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

    def test_build_segments_kornia_color_jitter_sat_hue_passthrough(self):
        """Kornia ColorJitter with active saturation/hue must not be fused."""
        pytest.importorskip("kornia")
        import kornia.augmentation as K

        from fuse_augmentations.adapters._kornia import KorniaAdapter
        from fuse_augmentations.affine._segment import FusedColorSegment, build_segments

        t = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        segs = build_segments([t], KorniaAdapter(), "bilinear", "zeros")
        assert not any(isinstance(seg, FusedColorSegment) for seg in segs)
        assert segs == [t]


# ---------------------------------------------------------------------------
# Adapter build_color_matrix
# ---------------------------------------------------------------------------


class TestAdapterBuildColorMatrix:
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

    def test_kornia_color_jitter_with_sat_hue_raises_not_implemented(self):
        """ColorJitter with active saturation/hue is intentionally not fused."""
        pytest.importorskip("kornia")
        import kornia.augmentation as K

        from fuse_augmentations.adapters._kornia import KorniaAdapter

        B = 2
        transform = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (B, 3, 32, 32), torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="saturation/hue"):
            adapter.build_color_matrix(transform, params)

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


# ---------------------------------------------------------------------------
# FusedColorSegment forward edge cases
# ---------------------------------------------------------------------------


class TestFusedColorSegmentEdgeCases:
    """Edge-case coverage for FusedColorSegment forward path."""

    def _make_identity_seg(self):
        from fuse_augmentations._types import TransformCategory
        from fuse_augmentations.affine._segment import FusedColorSegment

        class _IdentityTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityAdapter:
            def category(self, t):
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, t, shape, device):
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, t, params):
                B = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

            def call_nonfused(self, t, image, **kw):
                return image

        t = _IdentityTransform()
        return FusedColorSegment([t], _IdentityAdapter())

    def test_forward_non_3channel_falls_back_to_passthrough(self):
        """FusedColorSegment falls back to sequential passthrough for non-RGB (C != 3) inputs."""
        seg = self._make_identity_seg()
        # 1-channel mask input
        x = torch.rand(2, 1, 16, 16)
        out = seg(x)
        assert out.shape == x.shape
        torch.testing.assert_close(out, x)

    def test_forward_with_aux_targets_returns_tuple(self):
        """FusedColorSegment returns (image, aux_targets) tuple when aux_targets is provided."""
        seg = self._make_identity_seg()
        x = torch.rand(2, 3, 16, 16)
        mask = torch.rand(2, 1, 16, 16)
        result = seg(x, aux_targets={"mask": mask})
        assert isinstance(result, tuple), "Expected (image, aux_targets) tuple"
        img_out, aux_out = result
        assert img_out.shape == x.shape
        assert "mask" in aux_out
        torch.testing.assert_close(aux_out["mask"], mask)

    def test_forward_aux_targets_none_returns_tensor(self):
        """FusedColorSegment returns bare Tensor when aux_targets is None."""
        seg = self._make_identity_seg()
        x = torch.rand(2, 3, 16, 16)
        out = seg(x, aux_targets=None)
        assert isinstance(out, torch.Tensor)

    def test_forward_non_3channel_with_aux_targets_returns_tuple(self):
        """Non-RGB fallback path also returns (image, aux_targets) when aux_targets is present."""
        seg = self._make_identity_seg()
        x = torch.rand(2, 1, 16, 16)  # 1-channel
        mask = torch.rand(2, 1, 16, 16)
        result = seg(x, aux_targets={"mask": mask})
        assert isinstance(result, tuple)
        img_out, _ = result
        assert img_out.shape == x.shape


# ---------------------------------------------------------------------------
# _try_build_color_matrix probe robustness
# ---------------------------------------------------------------------------


class TestTryBuildColorMatrixProbe:
    """_try_build_color_matrix correctly classifies adapter support."""

    def test_not_implemented_returns_false(self):
        """NotImplementedError from build_color_matrix → False (no support)."""
        from fuse_augmentations.affine._segment import _try_build_color_matrix

        class _NoSupportAdapter:
            def build_color_matrix(self, t, params):
                raise NotImplementedError

        assert _try_build_color_matrix(_NoSupportAdapter(), object()) is False

    def test_attribute_error_returns_false(self):
        """AttributeError (missing method) → False."""
        from fuse_augmentations.affine._segment import _try_build_color_matrix

        class _MissingMethodAdapter:
            pass

        assert _try_build_color_matrix(_MissingMethodAdapter(), object()) is False

    def test_key_error_returns_true(self):
        """KeyError (method exists but needs real params) → True."""
        from fuse_augmentations.affine._segment import _try_build_color_matrix

        class _NeedsParamsAdapter:
            def build_color_matrix(self, t, params):
                _ = params["brightness_factor"]  # KeyError when called with {}
                return torch.eye(4).unsqueeze(0)

        assert _try_build_color_matrix(_NeedsParamsAdapter(), object()) is True

    def test_runtime_error_returns_false(self):
        """RuntimeError from build_color_matrix → False (not classified as supported).

        A RuntimeError (e.g. GPU OOM, device mismatch) must NOT be silently treated as "method exists but needs real
        params" — it is classified as unsupported.

        """
        from fuse_augmentations.affine._segment import _try_build_color_matrix

        class _RuntimeErrorAdapter:
            def build_color_matrix(self, t, params):
                msg = "simulated GPU OOM or device mismatch"
                raise RuntimeError(msg)

        assert _try_build_color_matrix(_RuntimeErrorAdapter(), object()) is False


# ---------------------------------------------------------------------------
# scale_x/scale_y ValueError guard when backend= is set
# ---------------------------------------------------------------------------


class TestFromParamsScaleXYWithBackendRaises:
    """scale_x/scale_y kwargs raise ValueError when backend= is set."""

    def test_scale_x_with_backend_raises(self):
        """from_params(scale_x=..., backend='kornia') raises ValueError."""
        pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose

        with pytest.raises(ValueError, match="scale_x"):
            FusedCompose.from_params(scale_x=(0.8, 1.2), backend="kornia")

    def test_scale_y_with_backend_raises(self):
        """from_params(scale_y=..., backend='kornia') raises ValueError."""
        pytest.importorskip("kornia")
        from fuse_augmentations._compose import FusedCompose

        with pytest.raises(ValueError, match="scale_y"):
            FusedCompose.from_params(scale_y=(0.8, 1.2), backend="kornia")

    def test_scale_x_without_backend_works(self):
        """from_params(scale_x=...) without backend= is still valid."""
        from fuse_augmentations._compose import FusedCompose

        pipe = FusedCompose.from_params(scale_x=(0.8, 1.2))
        x = torch.zeros(2, 3, 32, 32)
        assert pipe(x).shape == x.shape
