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

from fuse_augmentations._compat import (
    _ALBUMENTATIONS_AVAILABLE,
    _KORNIA_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)
from fuse_augmentations._compose import FusedCompose
from fuse_augmentations._types import TransformCategory, TransformSpec
from fuse_augmentations.affine._segment import (
    FusedColorSegment,
    _try_build_color_matrix,
    build_segments,
)

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters._kornia import KorniaAdapter

if _TORCHVISION_AVAILABLE:
    from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

if _ALBUMENTATIONS_AVAILABLE:
    from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter


class TestFromParamsBackend:
    """from_params(specs=[...], backend=...) delegates to from_config semantics."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_from_params_specs_with_backend_runs(self):
        """from_params(specs=[...], backend='kornia') produces a working pipeline."""
        specs = [TransformSpec(operation="hflip", params={}, prob=0.5)]
        pipe = FusedCompose.from_params(specs=specs, backend="kornia")
        image = torch.zeros(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_from_params_specs_with_backend_same_segments_as_from_config(self):
        """from_params(specs=..., backend='kornia') produces the same segment structure as from_config."""
        specs = [
            TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)}, prob=1.0),
            TransformSpec(operation="hflip", params={}, prob=0.5),
        ]
        pipe_params = FusedCompose.from_params(specs=specs, backend="kornia")
        pipe_config = FusedCompose.from_config(specs, backend="kornia")

        desc_params = pipe_params.fusion_plan_descriptors
        desc_config = pipe_config.fusion_plan_descriptors
        assert len(desc_params) == len(desc_config)
        assert [desc.kind for desc in desc_params] == [desc.kind for desc in desc_config]

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_from_params_backend_native_kwarg_forwarded(self):
        """TransformSpec.params with a backend-native kwarg (same_on_batch) is forwarded without error."""
        specs = [
            TransformSpec(
                operation="rotation",
                params={"degrees": (-15.0, 15.0), "same_on_batch": True},
                prob=1.0,
            )
        ]
        pipe = FusedCompose.from_params(specs=specs, backend="kornia")
        image = torch.zeros(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_from_params_unknown_backend_native_kwarg_raises(self):
        """TransformSpec.params with an unrecognised kwarg raises ValueError."""
        specs = [
            TransformSpec(
                operation="rotation",
                params={"degrees": (-15.0, 15.0), "totally_unknown_kwarg": 42},
                prob=1.0,
            )
        ]
        with pytest.raises((ValueError, TypeError)):
            FusedCompose.from_params(specs=specs, backend="kornia")

    def test_from_params_existing_call_sig_unchanged(self):
        """Existing from_params(rotation=...) call signature is not broken."""
        pipe = FusedCompose.from_params(rotation=(-30.0, 30.0), hflip_p=0.5)
        image = torch.zeros(2, 3, 32, 32)
        out = pipe(image)
        assert out.shape == torch.Size([2, 3, 32, 32])


class TestFusedColorSegment:
    """build_segments folds consecutive POINTWISE_LINEAR ops into FusedColorSegment."""

    def test_fused_color_segment_importable(self):
        """FusedColorSegment can be imported from fuse_augmentations.affine._segment."""
        assert FusedColorSegment is not None

    def test_build_segments_folds_pointwise_linear_run(self):
        """Two consecutive POINTWISE_LINEAR ops → single FusedColorSegment."""

        class _PLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _PLAdapter:
            def category(self, tfm):
                """Return POINTWISE_LINEAR category."""
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, tfm, shape, device):
                """Return batch size param."""
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_matrix(self, tfm, params, height, width):
                """Return identity affine matrix."""
                batch_size = int(params["_batch_size"].item())
                return torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).clone()

            def build_color_matrix(self, tfm, params):
                """Return identity color matrix."""
                batch_size = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone()

            def call_nonfused(self, tfm, image, **kwargs):
                """Return image unchanged."""
                return image

        first_transform, second_transform = _PLTransform(), _PLTransform()
        adapter = _PLAdapter()
        segs = build_segments([first_transform, second_transform], adapter, "bilinear", "zeros")
        color_segs = [segment for segment in segs if isinstance(segment, FusedColorSegment)]
        assert len(color_segs) == 1, f"Expected 1 FusedColorSegment, got segments: {segs}"

    def test_fused_color_segment_forward_returns_correct_shape(self):
        """FusedColorSegment.forward returns (B, C, H, W) matching input."""

        class _IdentityPLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityPLAdapter:
            def category(self, tfm):
                """Return POINTWISE_LINEAR category."""
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, tfm, shape, device):
                """Return batch size param."""
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, tfm, params):
                """Return identity color matrix."""
                batch_size = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone()

            def call_nonfused(self, tfm, image, **kwargs):
                """Return image unchanged."""
                return image

        transform = _IdentityPLTransform()
        seg = FusedColorSegment([transform], _IdentityPLAdapter())
        image = torch.rand(2, 3, 16, 16)
        out = seg(image)
        assert out.shape == image.shape

    def test_fused_color_segment_identity_matrix_preserves_image(self):
        """Applying identity color matrices does not alter pixel values."""

        class _IdentityPLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityPLAdapter:
            def category(self, tfm):
                """Return POINTWISE_LINEAR category."""
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, tfm, shape, device):
                """Return batch size param."""
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, tfm, params):
                """Return identity color matrix."""
                batch_size = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone()

            def call_nonfused(self, tfm, image, **kwargs):
                """Return image unchanged."""
                return image

        transform = _IdentityPLTransform()
        seg = FusedColorSegment([transform], _IdentityPLAdapter())
        image = torch.rand(2, 3, 16, 16)
        out = seg(image)
        torch.testing.assert_close(out, image)

    def test_build_segments_fallback_to_passthrough_when_adapter_raises(self):
        """If any adapter in a POINTWISE_LINEAR run raises NotImplementedError, fall back to passthrough."""

        class _PLTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _NoColorMatrixAdapter:
            def category(self, tfm):
                """Return POINTWISE_LINEAR category."""
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, tfm, shape, device):
                """Return empty params."""
                return {}

            def build_matrix(self, tfm, params, height, width):
                """Return identity affine matrix."""
                return torch.eye(3).unsqueeze(0)

            def build_color_matrix(self, tfm, params):
                """Raise to indicate no color matrix support."""
                raise NotImplementedError

            def call_nonfused(self, tfm, image, **kwargs):
                """Return image unchanged."""
                return image

        first_transform, second_transform = _PLTransform(), _PLTransform()
        adapter = _NoColorMatrixAdapter()
        segs = build_segments([first_transform, second_transform], adapter, "bilinear", "zeros")
        color_segs = [segment for segment in segs if isinstance(segment, FusedColorSegment)]
        assert len(color_segs) == 0

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_build_segments_kornia_color_jitter_sat_hue_passthrough(self):
        """Kornia ColorJitter with active saturation/hue must not be fused."""
        transform = kornia_aug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        segs = build_segments([transform], KorniaAdapter(), "bilinear", "zeros")
        assert not any(isinstance(seg, FusedColorSegment) for seg in segs)
        assert segs == [transform]


class TestAdapterBuildColorMatrix:
    """Each adapter exposes build_color_matrix returning (B, 4, 4) tensors."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_adapter_has_build_color_matrix(self):
        """KorniaAdapter.build_color_matrix exists as a callable."""
        assert callable(getattr(KorniaAdapter, "build_color_matrix", None))

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_random_brightness_returns_4x4(self):
        """KorniaAdapter.build_color_matrix for RandomBrightness returns (B, 4, 4)."""
        batch = 3
        transform = kornia_aug.RandomBrightness(brightness=(0.5, 1.5), p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (batch, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)
        assert mat.shape == (batch, 4, 4), f"Expected ({batch}, 4, 4), got {mat.shape}"

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_color_jitter_returns_4x4(self):
        """KorniaAdapter.build_color_matrix for ColorJitter returns (B, 4, 4)."""
        batch = 2
        transform = kornia_aug.ColorJitter(brightness=0.2, contrast=0.2, p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (batch, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)
        assert mat.shape == (batch, 4, 4)

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_color_jitter_with_sat_hue_raises_not_implemented(self):
        """ColorJitter with active saturation/hue is intentionally not fused."""
        batch = 2
        transform = kornia_aug.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (batch, 3, 32, 32), torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="saturation/hue"):
            adapter.build_color_matrix(transform, params)

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_torchvision_adapter_has_build_color_matrix(self):
        """TorchVisionAdapter.build_color_matrix exists as a callable."""
        assert callable(getattr(TorchVisionAdapter, "build_color_matrix", None))

    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
    def test_albumentations_adapter_has_build_color_matrix(self):
        """AlbumentationsAdapter.build_color_matrix exists as a callable."""
        assert callable(getattr(AlbumentationsAdapter, "build_color_matrix", None))

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_build_color_matrix_last_row_is_homogeneous(self):
        """build_color_matrix bottom row is [0, 0, 0, 1] for all batch items."""
        batch = 4
        transform = kornia_aug.RandomBrightness(brightness=(0.8, 1.2), p=1.0)
        adapter = KorniaAdapter()
        params = adapter.sample_params(transform, (batch, 3, 32, 32), torch.device("cpu"))
        mat = adapter.build_color_matrix(transform, params)

        expected_last_row = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for batch_idx in range(batch):
            torch.testing.assert_close(mat[batch_idx, 3, :], expected_last_row)


class TestFusedColorSegmentEdgeCases:
    """Edge-case coverage for FusedColorSegment forward path."""

    def _make_identity_seg(self):
        """Build a FusedColorSegment that applies identity color matrices."""

        class _IdentityTransform:
            _category = TransformCategory.POINTWISE_LINEAR
            p = 1.0

        class _IdentityAdapter:
            def category(self, tfm):
                """Return POINTWISE_LINEAR category."""
                return TransformCategory.POINTWISE_LINEAR

            def sample_params(self, tfm, shape, device):
                """Return batch size param."""
                return {"_batch_size": torch.tensor([shape[0]])}

            def build_color_matrix(self, tfm, params):
                """Return identity color matrix."""
                batch_size = int(params["_batch_size"].item())
                return torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone()

            def call_nonfused(self, tfm, image, **kwargs):
                """Return image unchanged."""
                return image

        transform = _IdentityTransform()
        return FusedColorSegment([transform], _IdentityAdapter())

    def test_forward_non_3channel_falls_back_to_passthrough(self):
        """FusedColorSegment falls back to sequential passthrough for non-RGB (C != 3) inputs."""
        seg = self._make_identity_seg()
        image = torch.rand(2, 1, 16, 16)
        out = seg(image)
        assert out.shape == image.shape
        torch.testing.assert_close(out, image)

    def test_forward_with_aux_targets_returns_tuple(self):
        """FusedColorSegment returns (image, aux_targets) tuple when aux_targets is provided."""
        seg = self._make_identity_seg()
        image = torch.rand(2, 3, 16, 16)
        mask = torch.rand(2, 1, 16, 16)
        result = seg(image, aux_targets={"mask": mask})
        assert isinstance(result, tuple), "Expected (image, aux_targets) tuple"
        img_out, aux_out = result
        assert img_out.shape == image.shape
        assert "mask" in aux_out
        torch.testing.assert_close(aux_out["mask"], mask)

    def test_forward_aux_targets_none_returns_tensor(self):
        """FusedColorSegment returns bare Tensor when aux_targets is None."""
        seg = self._make_identity_seg()
        image = torch.rand(2, 3, 16, 16)
        out = seg(image, aux_targets=None)
        assert isinstance(out, torch.Tensor)

    def test_forward_non_3channel_with_aux_targets_returns_tuple(self):
        """Non-RGB fallback path also returns (image, aux_targets) when aux_targets is present."""
        seg = self._make_identity_seg()
        image = torch.rand(2, 1, 16, 16)
        mask = torch.rand(2, 1, 16, 16)
        result = seg(image, aux_targets={"mask": mask})
        assert isinstance(result, tuple)
        img_out, _ = result
        assert img_out.shape == image.shape


class TestTryBuildColorMatrixProbe:
    """_try_build_color_matrix correctly classifies adapter support."""

    def test_not_implemented_returns_false(self):
        """NotImplementedError from build_color_matrix → False (no support)."""

        class _NoSupportAdapter:
            def build_color_matrix(self, tfm, params):
                """Raise to signal no support."""
                raise NotImplementedError

        assert _try_build_color_matrix(_NoSupportAdapter(), object()) is False

    def test_attribute_error_returns_false(self):
        """AttributeError (missing method) → False."""

        class _MissingMethodAdapter:
            pass

        assert _try_build_color_matrix(_MissingMethodAdapter(), object()) is False

    def test_key_error_returns_true(self):
        """KeyError (method exists but needs real params) → True."""

        class _NeedsParamsAdapter:
            def build_color_matrix(self, tfm, params):
                """Access params to trigger KeyError when called with empty dict."""
                _ = params["brightness_factor"]
                return torch.eye(4).unsqueeze(0)

        assert _try_build_color_matrix(_NeedsParamsAdapter(), object()) is True

    def test_runtime_error_returns_false(self):
        """RuntimeError from build_color_matrix → False (not classified as supported).

        A RuntimeError (e.g. GPU OOM, device mismatch) must NOT be silently treated as "method exists but needs real
        params" — it is classified as unsupported.

        """

        class _RuntimeErrorAdapter:
            def build_color_matrix(self, tfm, params):
                """Raise RuntimeError to simulate GPU OOM or device mismatch."""
                msg = "simulated GPU OOM or device mismatch"
                raise RuntimeError(msg)

        assert _try_build_color_matrix(_RuntimeErrorAdapter(), object()) is False


class TestFromParamsScaleXYWithBackendRaises:
    """scale_x/scale_y kwargs raise ValueError when backend= is set."""

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_scale_x_with_backend_raises(self):
        """from_params(scale_x=..., backend='kornia') raises ValueError."""
        with pytest.raises(ValueError, match="scale_x"):
            FusedCompose.from_params(scale_x=(0.8, 1.2), backend="kornia")

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_scale_y_with_backend_raises(self):
        """from_params(scale_y=..., backend='kornia') raises ValueError."""
        with pytest.raises(ValueError, match="scale_y"):
            FusedCompose.from_params(scale_y=(0.8, 1.2), backend="kornia")

    def test_scale_x_without_backend_works(self):
        """from_params(scale_x=...) without backend= is still valid."""
        pipe = FusedCompose.from_params(scale_x=(0.8, 1.2))
        image = torch.zeros(2, 3, 32, 32)
        assert pipe(image).shape == image.shape
