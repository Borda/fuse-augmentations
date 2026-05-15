"""Integration tests for mixed-backend pipelines.

Requires both torchvision and kornia. Tests are skipped gracefully if either is not installed.

Mixed-backend pipelines allow transforms from different frameworks (e.g. TorchVision geometric + Kornia color) in a
single Compose call. Each transform is dispatched to its native adapter for parameter sampling and matrix construction.

"""

from __future__ import annotations

import pickle

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required"),
    pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required"),
]

HEIGHT, WIDTH, CHANNELS = 16, 16, 3


def _rand_image(batch_size: int = 2) -> torch.Tensor:
    return torch.rand(batch_size, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def img() -> torch.Tensor:
    return _rand_image()


class TestMixedBackendSmoke:
    def test_mixed_forward_does_not_raise(self, img):
        """A mixed TorchVision + Kornia pipeline does not raise on construction or forward."""
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3, p=1.0),
        ])
        out = pipe(img)
        assert out is not None


class TestMixedGeometricColor:
    def test_mixed_torchvision_geometric_kornia_color(self, img):
        """TorchVision geometric + Kornia color produces valid output with preserved shape."""
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3, p=1.0),
        ])
        out = pipe(img)
        assert out.shape == img.shape

    def test_mixed_output_shape_preserved(self):
        """Batch shape (B, C, H, W) is preserved through a mixed pipeline."""
        img = _rand_image()
        pipe = Compose([
            tv_trans.RandomRotation(degrees=15),
            kornia_aug.ColorJitter(brightness=0.1, p=1.0),
        ])
        out = pipe(img)
        assert out.shape == (2, CHANNELS, HEIGHT, WIDTH)


class TestMixedFusionPlan:
    def test_mixed_backend_fusion_plan(self):
        """Fusion plan shows geometric segment fused + color segment fused."""
        # brightness + contrast are the linearly-fusible subset; saturation/hue are not
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.ColorJitter(brightness=0.2, contrast=0.3, p=1.0),
        ])
        plan = pipe.fusion_plan
        # Geometric transform should be in a fused segment
        assert "fused" in plan or "exact" in plan, f"Expected fused/exact segment in plan: {plan}"
        # Color jitter (brightness+contrast only) should be a color-fused segment
        assert "color" in plan, f"Expected color segment in plan: {plan}"

    def test_mixed_backend_n_warps_saved(self):
        """Two TorchVision geometric transforms in a mixed pipeline save 1 warp."""
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            tv_trans.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            kornia_aug.ColorJitter(brightness=0.2, p=1.0),
        ])
        assert pipe.n_warps_saved >= 1, f"Expected at least 1 warp saved, got {pipe.n_warps_saved}"


class TestMixedGeometricSegments:
    def test_mixed_two_geometric_backends_separate_segments(self):
        """TorchVision rotation + Kornia rotation produce two separate fused segments.

        Different backends cannot share a single fused segment because each uses its own adapter for parameter sampling
        and matrix construction.

        """
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.RandomRotation(degrees=30, p=1.0),
        ])
        plan = pipe.fusion_plan
        # Should contain two separate fused segments (one per backend),
        # not a single fused segment combining both.
        fused_count = plan.count("fused(")
        assert fused_count == 2, (
            f"Expected 2 separate fused segments (one per backend), got {fused_count} in plan: {plan}"
        )

        # Each single-transform segment saves 0 warps on its own
        img = _rand_image()
        out = pipe(img)
        assert out.shape == img.shape


class TestMixedBackendDataKeys:
    def test_data_keys_mask_shape_preserved(self):
        """Mixed TV+Kornia pipeline with data_keys=["input", "mask"] preserves mask shape."""
        torch.manual_seed(0)
        img = torch.rand(2, 1, HEIGHT, WIDTH, dtype=torch.float32)
        mask = torch.rand(2, 1, HEIGHT, WIDTH, dtype=torch.float32)
        pipe = Compose(
            [
                tv_trans.RandomHorizontalFlip(p=1.0),
                kornia_aug.ColorJitter(brightness=0.2, p=1.0),
            ],
            data_keys=["input", "mask"],
        )
        _img_out, mask_out = pipe(img, mask)
        assert mask_out.shape == mask.shape, f"Expected mask shape {mask.shape}, got {mask_out.shape}"


class TestMixedBackendDuplicateTransform:
    def test_duplicate_object_emits_warning(self):
        """Same transform object at two positions in a mixed pipeline emits UserWarning."""
        shared_flip = tv_trans.RandomHorizontalFlip(p=1.0)
        with pytest.warns(UserWarning, match="(?i)same transform object"):
            Compose([shared_flip, kornia_aug.ColorJitter(brightness=0.2, p=1.0), shared_flip])


class TestMixedBackendSerialization:
    def test_pickle_roundtrip_preserves_passthrough_adapter_dispatch(self):
        """Pickle round-trip keeps mixed-backend passthrough dispatch bound to the right adapter."""
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3, p=1.0),
        ])
        loaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301

        img = _rand_image()
        torch.manual_seed(42)
        expected = pipe(img)
        torch.manual_seed(42)
        actual = loaded(img)

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    def test_pickle_roundtrip_tv_only_with_spatial_kernel_barrier(self):
        """Regression: pickle round-trip with a SPATIAL_KERNEL barrier dispatches correctly.

        Bug #2: id()-keyed adapter map broke after pickle because object ids change on deserialization. Index-based keys
        survive pickle.

        """
        pipe = Compose([
            tv_trans.RandomRotation(degrees=30),
            kornia_aug.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3, p=1.0),
            tv_trans.RandomHorizontalFlip(p=1.0),
        ])

        img = _rand_image()

        # Verify original works
        out_orig = pipe(img)
        assert out_orig.shape == img.shape

        # Pickle round-trip
        loaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301
        out_loaded = loaded(img)
        assert out_loaded.shape == img.shape
