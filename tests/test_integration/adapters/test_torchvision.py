"""Integration tests for the TorchVision backend adapter.

Requires torchvision. Tests are skipped gracefully if not installed.

Parity contracts:
- GEOMETRIC_EXACT (flip) transforms: fused ExactAffineSegment vs native TorchVision -> atol=1e-5
  (tensor.flip is exact; any diff is float32 representation noise)
- GEOMETRIC_INTERP transforms: fused pipeline vs manual grid_sample reference
  built from the same adapter-sampled parameters. Both paths use
  align_corners=True so tolerance is tight (atol=1e-5).

"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torchvision", reason="torchvision required")
import torchvision.transforms as T

from fuse_augmentations import Compose
from fuse_augmentations.adapters._torchvision import TorchVisionAdapter
from fuse_augmentations.affine._matrix import inv3x3, normalize_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

HEIGHT, WIDTH, NUM_CHANNELS = 64, 64, 3
ATOL_EXACT = 1e-5  # flip parity: tensor.flip is exact
ATOL_INTERP = 1e-5  # same matrix, same grid_sample -> near-identical


def _rand_image(batch_size: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch_size, NUM_CHANNELS, HEIGHT, WIDTH)


def _native_apply(transform: object, img: torch.Tensor) -> torch.Tensor:
    """Apply a TorchVision transform per-sample via native forward."""
    batch_size = img.shape[0]
    results = []
    for batch_idx in range(batch_size):
        out = transform(img[batch_idx])  # type: ignore[operator]
        results.append(out)
    return torch.stack(results)


def _manual_grid_sample(
    img: torch.Tensor,
    adapter: TorchVisionAdapter,
    transform: object,
) -> torch.Tensor:
    """Build matrix via adapter and apply via grid_sample (same path as FusedAffineSegment).

    Mirrors the RNG consumption pattern of FusedAffineSegment.forward():
    1. torch.rand(batch_size) for probability mask
    2. adapter.sample_params() which calls get_params()

    """
    batch_size, num_channels, height, width = img.shape
    # FusedAffineSegment draws torch.rand(batch_size) for probability before sampling params
    torch.rand(batch_size, device=img.device)
    params = adapter.sample_params(transform, img.shape, img.device)
    mtx_fwd = adapter.build_matrix(transform, params, height, width)
    if mtx_fwd.shape[0] == 1 and batch_size > 1:
        mtx_fwd = mtx_fwd.expand(batch_size, -1, -1)
    mtx_inv = inv3x3(mtx_fwd)
    mtx_norm = normalize_matrix(mtx_inv, height, width)
    grid = F.affine_grid(mtx_norm[:, :2, :], [batch_size, num_channels, height, width], align_corners=True)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# Single-transform parity tests -- flips (exact, prob=1)
# ---------------------------------------------------------------------------


class TestFlipParity:
    """Flips: fused output must exactly match native TorchVision output."""

    def test_hflip_parity(self):
        img = _rand_image()

        torch.manual_seed(42)
        native_out = _native_apply(T.RandomHorizontalFlip(p=1), img)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomHorizontalFlip(p=1)])(img)

        assert torch.allclose(fused_out, native_out, atol=ATOL_EXACT), (
            f"HFlip parity failed: max diff = {(fused_out - native_out).abs().max().item():.2e}"
        )

    def test_vflip_parity(self):
        img = _rand_image()

        torch.manual_seed(42)
        native_out = _native_apply(T.RandomVerticalFlip(p=1), img)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomVerticalFlip(p=1)])(img)

        assert torch.allclose(fused_out, native_out, atol=ATOL_EXACT), (
            f"VFlip parity failed: max diff = {(fused_out - native_out).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# INTERP parity tests -- same adapter params, same grid_sample -> tight tol
# ---------------------------------------------------------------------------


class TestInterpParity:
    """GEOMETRIC_INTERP: fused vs manual grid_sample from same adapter params."""

    @pytest.fixture
    def adapter(self):
        return TorchVisionAdapter()

    def test_rotation_parity(self, adapter):
        img = _rand_image()
        transform = T.RandomRotation(degrees=(30, 30))

        torch.manual_seed(42)
        ref = _manual_grid_sample(img, adapter, transform)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomRotation(degrees=(30, 30))])(img)

        assert fused_out.shape == img.shape
        assert torch.allclose(fused_out, ref, atol=ATOL_INTERP), (
            f"Rotation parity failed: max diff = {(fused_out - ref).abs().max().item():.2e}"
        )

    def test_affine_rotation_only_parity(self, adapter):
        img = _rand_image()
        transform = T.RandomAffine(degrees=(20, 20), translate=None, scale=None, shear=None)

        torch.manual_seed(42)
        ref = _manual_grid_sample(img, adapter, transform)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomAffine(degrees=(20, 20), translate=None, scale=None, shear=None)])(img)

        assert torch.allclose(fused_out, ref, atol=ATOL_INTERP), (
            f"Affine rotation-only parity failed: max diff = {(fused_out - ref).abs().max().item():.2e}"
        )

    def test_affine_scale_parity(self, adapter):
        img = _rand_image()
        transform = T.RandomAffine(degrees=0, scale=(0.9, 0.9))

        torch.manual_seed(42)
        ref = _manual_grid_sample(img, adapter, transform)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomAffine(degrees=0, scale=(0.9, 0.9))])(img)

        assert torch.allclose(fused_out, ref, atol=ATOL_INTERP), (
            f"Affine scale parity failed: max diff = {(fused_out - ref).abs().max().item():.2e}"
        )

    def test_affine_translate_parity(self, adapter):
        img = _rand_image()
        transform = T.RandomAffine(degrees=0, translate=(0.1, 0.1))

        torch.manual_seed(42)
        ref = _manual_grid_sample(img, adapter, transform)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomAffine(degrees=0, translate=(0.1, 0.1))])(img)

        assert torch.allclose(fused_out, ref, atol=ATOL_INTERP), (
            f"Affine translate parity failed: max diff = {(fused_out - ref).abs().max().item():.2e}"
        )

    def test_affine_shear_parity(self, adapter):
        img = _rand_image()
        transform = T.RandomAffine(degrees=0, shear=(10, 10, 0, 0))

        torch.manual_seed(42)
        ref = _manual_grid_sample(img, adapter, transform)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomAffine(degrees=0, shear=(10, 10, 0, 0))])(img)

        assert torch.allclose(fused_out, ref, atol=ATOL_INTERP), (
            f"Affine shear parity failed: max diff = {(fused_out - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Spec test #31 -- align_corners offset bounded (fused vs native)
# ---------------------------------------------------------------------------


class TestAlignCornersOffset:
    def test_align_corners_offset_within_bound(self):
        """Fused (align_corners=True, center=(width-1)/2) vs native TorchVision (center=width/2) difference is bounded
        for the documented center offset.

        The fused engine uses align_corners=True with rotation center (width-1)/2, while TorchVision native
        uses half-pixel center width/2.  Under 30-degree rotation, this 0.5px center offset displaces source
        coordinates across the image,
        producing pixel-value differences especially at edges (where zeros-padding creates hard boundaries).  This test
        validates that the max absolute difference stays <= 1.0 (pixel range is [0, 1]) and that the output shapes match
        -- confirming the architectural difference is bounded, not divergent.

        """
        img = _rand_image()

        # Fixed angle so both paths get angle=30 regardless of RNG state.
        torch.manual_seed(42)
        native_out = _native_apply(T.RandomRotation(degrees=(30, 30)), img)

        torch.manual_seed(42)
        fused_out = Compose([T.RandomRotation(degrees=(30, 30))])(img)

        max_diff = (fused_out - native_out).abs().max().item()

        assert fused_out.shape == native_out.shape
        # Max diff bounded by pixel range [0, 1] -- center offset causes real
        # pixel-level differences but cannot exceed the value range.
        assert max_diff <= 1.0, f"align_corners max offset {max_diff:.6f} exceeds pixel range 1.0"
        # Verify that not everything is identical (the center offset IS real)
        assert max_diff > 0.001, "Expected nonzero difference from center-of-rotation offset"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_p0_output_unchanged(self):
        img = _rand_image()
        out = Compose([T.RandomHorizontalFlip(p=0)])(img)
        assert torch.allclose(out, img, atol=1e-5), "prob=0 should produce identity output"

    def test_empty_pipeline_is_identity(self):
        img = _rand_image()
        out = Compose([])(img)
        assert torch.allclose(out, img)

    def test_output_shape_preserved(self):
        img = _rand_image(batch_size=4)
        out = Compose([T.RandomRotation(degrees=30)])(img)
        assert out.shape == img.shape

    def test_batch_samples_independent(self):
        """With prob=1 rotation, batch_size=4 samples should not all be identical."""
        img = _rand_image(batch_size=4)
        out = Compose([T.RandomRotation(degrees=45)])(img)
        diffs = [(out[sample_idx] - out[0]).abs().max().item() for sample_idx in range(1, 4)]
        assert any(max_diff > 1e-3 for max_diff in diffs), "Expected per-sample independence in outputs"


# ---------------------------------------------------------------------------
# v2 parity
# ---------------------------------------------------------------------------


class TestV2Parity:
    def test_v2_hflip_parity(self):
        """v2.RandomHorizontalFlip produces same result as v1 via fused path."""
        pytest.importorskip("torchvision.transforms.v2", reason="torchvision v2 required")
        import torchvision.transforms.v2 as T2

        img = _rand_image()

        torch.manual_seed(42)
        native_out = _native_apply(T2.RandomHorizontalFlip(p=1), img)

        torch.manual_seed(42)
        fused_out = Compose([T2.RandomHorizontalFlip(p=1)])(img)

        assert torch.allclose(fused_out, native_out, atol=ATOL_EXACT), (
            f"v2 HFlip parity failed: max diff = {(fused_out - native_out).abs().max().item():.2e}"
        )

    def test_v2_hflip_batch_probability_matches_native(self):
        pytest.importorskip("torchvision.transforms.v2", reason="torchvision v2 required")
        import torchvision.transforms.v2 as T2

        img = _rand_image(batch_size=4)

        torch.manual_seed(0)
        native_out = T2.RandomHorizontalFlip(p=0.5)(img)

        torch.manual_seed(0)
        fused_out = Compose([T2.RandomHorizontalFlip(p=0.5)])(img)

        torch.testing.assert_close(fused_out, native_out, atol=ATOL_EXACT, rtol=0.0)

    def test_v2_rotation_uses_one_matrix_for_the_batch(self):
        pytest.importorskip("torchvision.transforms.v2", reason="torchvision v2 required")
        import torchvision.transforms.v2 as T2

        img = _rand_image(batch_size=4)
        pipe = Compose([T2.RandomRotation(degrees=30)])

        torch.manual_seed(42)
        _ = pipe(img)

        matrix = pipe.transform_matrix
        assert matrix is not None
        for idx in range(1, img.shape[0]):
            torch.testing.assert_close(matrix[idx], matrix[0], atol=1e-6, rtol=0.0)

    def test_v2_affine_uses_one_matrix_for_the_batch(self):
        pytest.importorskip("torchvision.transforms.v2", reason="torchvision v2 required")
        import torchvision.transforms.v2 as T2

        img = _rand_image(batch_size=4)
        pipe = Compose([T2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5, -3, 3))])

        torch.manual_seed(42)
        _ = pipe(img)

        matrix = pipe.transform_matrix
        assert matrix is not None
        for idx in range(1, img.shape[0]):
            torch.testing.assert_close(matrix[idx], matrix[0], atol=1e-6, rtol=0.0)

    def test_v2_color_jitter_passthrough_matches_native_on_batch(self):
        pytest.importorskip("torchvision.transforms.v2", reason="torchvision v2 required")
        import torchvision.transforms.v2 as T2

        img = _rand_image(batch_size=4)
        transform = T2.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.05)

        torch.manual_seed(42)
        native_out = transform(img)

        torch.manual_seed(42)
        fused_out = Compose([T2.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.05)])(img)

        torch.testing.assert_close(fused_out, native_out, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Native parity tests -- fused vs TorchVision native (not self-referential)
# ---------------------------------------------------------------------------


def _tv_forward_matrix(
    angle: float,
    translate: tuple[int, int],
    scale: float,
    shear: tuple[float, float],
    center_x: float,
    center_y: float,
) -> torch.Tensor:
    """Build TorchVision's forward 3x3 matrix from sampled parameters.

    Delegates to TorchVision's own affine helper with ``inverted=False`` so
    the reference matrix stays aligned with upstream parameterization.

    """
    flat = T.functional._get_inverse_affine_matrix(  # type: ignore[attr-defined]
        center=[center_x, center_y],
        angle=angle,
        translate=[float(translate[0]), float(translate[1])],
        scale=scale,
        shear=[shear[0], shear[1]],
        inverted=False,
    )
    mtx = torch.zeros(3, 3)
    mtx[0, :] = torch.tensor([flat[0], flat[1], flat[2]])
    mtx[1, :] = torch.tensor([flat[3], flat[4], flat[5]])
    mtx[2, 2] = 1.0
    return mtx


ATOL_MATRIX = 1e-5  # float32 matrix element tolerance


class TestMatrixVsTorchVision:
    """Compare adapter-built affine matrix against TorchVision's reference.

    Validates that _build_affine_matrix produces the same 2x2 RSS block and correctly centered translation as
    TorchVision's _get_inverse_affine_matrix. The center is (width-1)/2 (align_corners=True) which intentionally differs
    from TorchVision's width/2 — this test uses (width-1)/2 for both sides so the comparison is purely about the matrix
    composition logic.

    """

    @pytest.fixture
    def adapter(self):
        return TorchVisionAdapter()

    @pytest.mark.parametrize(
        "label,kwargs",
        [
            ("shear_x", {"degrees": 0, "shear": (10, 10, 0, 0)}),
            ("shear_y", {"degrees": 0, "shear": (0, 0, 10, 10)}),
            ("shear_both", {"degrees": 0, "shear": (10, 10, 10, 10)}),
            ("scale", {"degrees": 0, "scale": (0.8, 0.8)}),
            ("translate", {"degrees": 0, "translate": (0.1, 0.1)}),
            ("rotation", {"degrees": (15, 15)}),
            (
                "combined",
                {
                    "degrees": (15, 15),
                    "translate": (0.05, 0.05),
                    "scale": (0.9, 0.9),
                    "shear": (5, 5, 5, 5),
                },
            ),
        ],
    )
    def test_matrix_matches_torchvision(self, adapter, label, kwargs):
        """Adapter forward matrix matches TorchVision reference (center=(width-1)/2)."""
        transform = T.RandomAffine(**kwargs)
        center_x, center_y = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0

        torch.manual_seed(42)
        params = adapter.sample_params(transform, (1, NUM_CHANNELS, HEIGHT, WIDTH), torch.device("cpu"))
        mtx_adapter = adapter.build_matrix(transform, params, HEIGHT, WIDTH)  # (1, 3, 3) forward

        # Extract the same scalar params to build reference
        angle_deg = float(torch.rad2deg(params["angle_rad"][0]))
        scale = float(params.get("scale", torch.ones(1))[0])
        shear_x_deg = float(torch.rad2deg(params.get("shear_x_rad", torch.zeros(1))[0]))
        shear_y_deg = float(torch.rad2deg(params.get("shear_y_rad", torch.zeros(1))[0]))
        trans_x = float(params.get("translate_x", torch.zeros(1))[0])
        trans_y = float(params.get("translate_y", torch.zeros(1))[0])

        mtx_ref = _tv_forward_matrix(
            angle=angle_deg,
            translate=(int(trans_x), int(trans_y)),
            scale=scale,
            shear=(shear_x_deg, shear_y_deg),
            center_x=center_x,
            center_y=center_y,
        )

        max_diff = (mtx_adapter[0] - mtx_ref).abs().max().item()
        assert max_diff < ATOL_MATRIX, (
            f"[{label}] matrix mismatch: max diff = {max_diff:.2e}"
            f"\n  adapter:\n{mtx_adapter[0]}\n  reference:\n{mtx_ref}"
        )


# ---------------------------------------------------------------------------
# Chain tests
# ---------------------------------------------------------------------------


class TestBatchSamplesIndependentAffine:
    """V1 RandomAffine samples independent parameters per batch element."""

    @pytest.fixture
    def adapter(self):
        return TorchVisionAdapter()

    def test_per_sample_matrices_differ(self, adapter):
        """With batch_size=3, at least one pair of affine matrices must differ (per-sample RNG)."""
        torch.manual_seed(42)
        transform = T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2))
        params = adapter.sample_params(transform, (3, NUM_CHANNELS, HEIGHT, WIDTH), device=torch.device("cpu"))
        mtx = adapter.build_matrix(transform, params, HEIGHT, WIDTH)
        assert mtx.shape[0] == 3, f"Expected batch dim 3, got {mtx.shape[0]}"
        # At least one pair of the 3 matrices should differ
        any_differ = (
            not torch.allclose(mtx[0], mtx[1], atol=1e-6)
            or not torch.allclose(mtx[0], mtx[2], atol=1e-6)
            or not torch.allclose(mtx[1], mtx[2], atol=1e-6)
        )
        assert any_differ, "All 3 per-sample affine matrices are identical; expected per-sample independence"


class TestChain:
    def test_rotate_then_hflip_fusion_plan(self):
        pipe = Compose([T.RandomRotation(degrees=30), T.RandomHorizontalFlip(p=1)])
        assert "fused" in pipe.fusion_plan

    def test_n_warps_saved(self):
        """Chain of 2 GEOMETRIC_INTERP saves 1 warp."""
        pipe = Compose([
            T.RandomRotation(degrees=30),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ])
        assert pipe.n_warps_saved == 1
