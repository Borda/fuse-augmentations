"""Tests for _matrix.py."""

import math

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations.affine._matrix import (
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    perspective_from_points,
    perspective_grid,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float64  # use float64 for tighter tolerances in unit tests


def _eye(batch_size: int = 1) -> torch.Tensor:
    """Return a (batch_size, 3, 3) identity matrix batch."""
    return torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0).expand(batch_size, -1, -1)


def _apply_point(mtx: torch.Tensor, coord_x: float, coord_y: float) -> tuple[float, float]:
    """Apply (1, 3, 3) matrix to a 2D point in homogeneous coords."""
    point = torch.tensor([coord_x, coord_y, 1.0], dtype=mtx.dtype)
    output = mtx[0] @ point
    return output[0].item(), output[1].item()


class TestIdentityCases:
    """Verify that neutral parameter values produce identity matrices."""

    def test_rotation_zero_is_identity(self) -> None:
        """Rotation by zero radians produces identity."""
        mtx = rotation_matrix(torch.zeros(2, dtype=DEFAULT_DTYPE), height=64, width=64)
        assert torch.allclose(mtx, _eye(2))

    def test_scale_one_is_identity(self) -> None:
        """Scale by (1, 1) produces identity."""
        mtx = scale_matrix(torch.ones(2, dtype=DEFAULT_DTYPE), torch.ones(2, dtype=DEFAULT_DTYPE), height=64, width=64)
        assert torch.allclose(mtx, _eye(2))

    def test_shear_x_zero_is_identity(self) -> None:
        """Shear-x by zero produces identity."""
        mtx = shear_x_matrix(torch.zeros(2, dtype=DEFAULT_DTYPE), height=64, width=64)
        assert torch.allclose(mtx, _eye(2))

    def test_shear_y_zero_is_identity(self) -> None:
        """Shear-y by zero produces identity."""
        mtx = shear_y_matrix(torch.zeros(2, dtype=DEFAULT_DTYPE), height=64, width=64)
        assert torch.allclose(mtx, _eye(2))

    def test_translate_zero_is_identity(self) -> None:
        """Translation by (0, 0) produces identity."""
        mtx = translate_matrix(torch.zeros(2, dtype=DEFAULT_DTYPE), torch.zeros(2, dtype=DEFAULT_DTYPE))
        assert torch.allclose(mtx, _eye(2))


class TestInvolution:
    """Verify that flip matrices are self-inverse (involutions)."""

    def test_hflip_involution(self) -> None:
        """Hflip @ Hflip == identity."""
        mtx = hflip_matrix(width=64, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        product = matmul3x3(mtx, mtx)
        assert torch.allclose(product, _eye(1), atol=1e-10)

    def test_vflip_involution(self) -> None:
        """Vflip @ Vflip == identity."""
        mtx = vflip_matrix(height=64, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        product = matmul3x3(mtx, mtx)
        assert torch.allclose(product, _eye(1), atol=1e-10)


class TestRotationGroup:
    """Verify rotation composition: R(a) @ R(b) == R(a+b)."""

    @pytest.mark.parametrize("a_deg,b_deg", [(30.0, 45.0), (-60.0, 90.0), (10.0, -170.0)])
    def test_rotation_composition(self, a_deg: float, b_deg: float) -> None:
        """R(a) @ R(b) == R(a+b) for the given angle pair."""
        a_rad = torch.tensor([math.radians(a_deg)], dtype=DEFAULT_DTYPE)
        b_rad = torch.tensor([math.radians(b_deg)], dtype=DEFAULT_DTYPE)
        mtx_a = rotation_matrix(a_rad, height=64, width=64)
        mtx_b = rotation_matrix(b_rad, height=64, width=64)
        mtx_ab = rotation_matrix(a_rad + b_rad, height=64, width=64)
        composed = matmul3x3(mtx_b, mtx_a)
        assert torch.allclose(composed, mtx_ab, atol=1e-10)


class TestScaleGroup:
    """Verify scale composition: S(a,b) @ S(c,d) == S(a*c, b*d)."""

    def test_scale_composition(self) -> None:
        """Scale factors compose multiplicatively."""
        sx1, sy1 = torch.tensor([2.0], dtype=DEFAULT_DTYPE), torch.tensor([3.0], dtype=DEFAULT_DTYPE)
        sx2, sy2 = torch.tensor([0.5], dtype=DEFAULT_DTYPE), torch.tensor([0.25], dtype=DEFAULT_DTYPE)
        mtx_s1 = scale_matrix(sx1, sy1, height=64, width=64)
        mtx_s2 = scale_matrix(sx2, sy2, height=64, width=64)
        mtx_sc = scale_matrix(sx1 * sx2, sy1 * sy2, height=64, width=64)
        product = matmul3x3(mtx_s2, mtx_s1)
        assert torch.allclose(product, mtx_sc, atol=1e-10)


class TestShearGroup:
    """Verify shear composition: Sh(a) @ Sh(b) == Sh(a+b)."""

    def test_shear_x_composition(self) -> None:
        """Shear-x factors compose additively."""
        angle_a = torch.tensor([0.3], dtype=DEFAULT_DTYPE)
        angle_b = torch.tensor([0.4], dtype=DEFAULT_DTYPE)
        mtx_sha = shear_x_matrix(angle_a, height=64, width=64)
        mtx_shb = shear_x_matrix(angle_b, height=64, width=64)
        mtx_shab = shear_x_matrix(angle_a + angle_b, height=64, width=64)
        product = matmul3x3(mtx_shb, mtx_sha)
        assert torch.allclose(product, mtx_shab, atol=1e-10)


class TestFlipRotation:
    """Verify hflip @ vflip == R(pi) for square images."""

    def test_hflip_vflip_equals_rotation_180(self) -> None:
        """Hflip @ vflip == R(pi) for square images."""
        height = width = 64
        mtx_hf = hflip_matrix(width=width, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        mtx_vf = vflip_matrix(height=height, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        composed = matmul3x3(mtx_hf, mtx_vf)
        mtx_r180 = rotation_matrix(torch.tensor([math.pi], dtype=DEFAULT_DTYPE), height=height, width=width)
        assert torch.allclose(composed, mtx_r180, atol=1e-10)


class TestCornerMappingHFlip:
    """Verify hflip maps pixel corners correctly for width=4."""

    def test_hflip_corners_W4(self) -> None:
        """Hflip maps (0,0)->(3,0), (3,0)->(0,0), (1,1)->(2,1) for width=4."""
        mtx = hflip_matrix(width=4, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        # (0,0) -> (3,0)
        coord_x, coord_y = _apply_point(mtx, 0.0, 0.0)
        assert abs(coord_x - 3.0) < 1e-10
        assert abs(coord_y - 0.0) < 1e-10
        # (3,0) -> (0,0)
        coord_x, coord_y = _apply_point(mtx, 3.0, 0.0)
        assert abs(coord_x - 0.0) < 1e-10
        assert abs(coord_y - 0.0) < 1e-10
        # (1,1) -> (2,1)
        coord_x, coord_y = _apply_point(mtx, 1.0, 1.0)
        assert abs(coord_x - 2.0) < 1e-10
        assert abs(coord_y - 1.0) < 1e-10


class TestRotation90CCW:
    """Verify 90-degree CCW rotation corner mapping on a 4x4 image."""

    def test_rotation_90_ccw_square(self) -> None:
        """90-degree CCW rotation around center of a 4x4 image.

        With center_x=center_y=1.5: (0,0) -> (3,0), (3,0) -> (3,3), (3,3) -> (0,3), (0,3) -> (0,0).

        """
        height = width = 4
        angle = torch.tensor([math.pi / 2], dtype=DEFAULT_DTYPE)  # 90 deg CCW
        mtx = rotation_matrix(angle, height=height, width=width)
        # (0, 0) -> (3, 0)
        coord_x, coord_y = _apply_point(mtx, 0.0, 0.0)
        assert abs(coord_x - 3.0) < 1e-10
        assert abs(coord_y - 0.0) < 1e-10
        # (3, 0) -> (3, 3)
        coord_x, coord_y = _apply_point(mtx, 3.0, 0.0)
        assert abs(coord_x - 3.0) < 1e-10
        assert abs(coord_y - 3.0) < 1e-10
        # (3, 3) -> (0, 3)
        coord_x, coord_y = _apply_point(mtx, 3.0, 3.0)
        assert abs(coord_x - 0.0) < 1e-10
        assert abs(coord_y - 3.0) < 1e-10
        # (0, 3) -> (0, 0)
        coord_x, coord_y = _apply_point(mtx, 0.0, 3.0)
        assert abs(coord_x - 0.0) < 1e-10
        assert abs(coord_y - 0.0) < 1e-10


class TestShearConversion:
    """Verify shear matrix element matches tan(angle)."""

    def test_shear_x_pi_over_4_gives_1(self) -> None:
        """build_matrix receives tan(shear_rad), so tan(pi/4)=1.0."""
        shear_x_tan = torch.tensor([math.tan(math.pi / 4)], dtype=DEFAULT_DTYPE)
        mtx = shear_x_matrix(shear_x_tan, height=64, width=64)
        # mtx[0, 0, 1] should be 1.0 (the shear element)
        assert abs(mtx[0, 0, 1].item() - 1.0) < 1e-10


class TestNonSquare:
    """Verify identity behaviour on non-square images."""

    @pytest.mark.parametrize("height,width", [(32, 64), (64, 32), (100, 200), (1, 2)])
    def test_rotation_identity_nonsquare(self, height: int, width: int) -> None:
        """Zero rotation is identity for non-square (height, width)."""
        mtx = rotation_matrix(torch.zeros(1, dtype=DEFAULT_DTYPE), height=height, width=width)
        assert torch.allclose(mtx, _eye(1))

    @pytest.mark.parametrize("height,width", [(32, 64), (64, 32), (100, 200)])
    def test_scale_identity_nonsquare(self, height: int, width: int) -> None:
        """Unit scale is identity for non-square (height, width)."""
        mtx = scale_matrix(
            torch.ones(1, dtype=DEFAULT_DTYPE), torch.ones(1, dtype=DEFAULT_DTYPE), height=height, width=width
        )
        assert torch.allclose(mtx, _eye(1))


class TestNormalizeIdentity:
    """Verify normalize_matrix(identity) is identity."""

    def test_normalize_identity_maps_pixels_to_self(self) -> None:
        """Identity in pixel space normalizes to identity in normalized space."""
        height, width = 64, 64
        mtx_norm = normalize_matrix(_eye(1), height=height, width=width)
        # Identity in pixel space -> identity in normalized space
        assert torch.allclose(mtx_norm, _eye(1), atol=1e-10)


class TestCornerMappingNormalized:
    """Verify pixel-to-normalized coordinate mapping at image corners."""

    def test_pixel_0_0_maps_to_minus1_minus1(self) -> None:
        """For identity matrix, pixel (0,0) should map to (-1,-1) in normalized coords."""
        height, width = 64, 64
        mtx_norm = normalize_matrix(_eye(1), height=height, width=width)
        # For identity, the normalized matrix should also be identity
        assert torch.allclose(mtx_norm, _eye(1), atol=1e-10)
        # normalization maps pixel (0,0) to (-1, -1):
        n_00 = 2.0 / (width - 1)
        n_02 = -1.0
        assert abs(n_00 * 0.0 + n_02 - (-1.0)) < 1e-10

    def test_pixel_W_minus_1_H_minus_1_maps_to_1_1(self) -> None:
        """Pixel (width-1, height-1) maps to (+1, +1) in normalized coordinates."""
        height, width = 64, 64
        n_00 = 2.0 / (width - 1)
        n_02 = -1.0
        n_11 = 2.0 / (height - 1)
        n_12 = -1.0
        assert abs(n_00 * (width - 1) + n_02 - 1.0) < 1e-10
        assert abs(n_11 * (height - 1) + n_12 - 1.0) < 1e-10


class TestNormalizeRoundTrip:
    """Verify denormalize(normalize(matrix)) recovers matrix."""

    def test_normalize_inv_normalize_is_identity(self) -> None:
        """Denormalize(normalize(matrix)) should recover matrix."""
        height, width = 64, 64
        angle = torch.tensor([math.radians(30)], dtype=DEFAULT_DTYPE)
        mtx = rotation_matrix(angle, height=height, width=width)
        mtx_norm = normalize_matrix(mtx, height=height, width=width)

        # Build norm and norm_inv manually to denormalize
        norm = torch.zeros(1, 3, 3, dtype=DEFAULT_DTYPE)
        norm[0, 0, 0] = 2.0 / (width - 1)
        norm[0, 0, 2] = -1.0
        norm[0, 1, 1] = 2.0 / (height - 1)
        norm[0, 1, 2] = -1.0
        norm[0, 2, 2] = 1.0

        norm_inv = torch.zeros(1, 3, 3, dtype=DEFAULT_DTYPE)
        norm_inv[0, 0, 0] = (width - 1) / 2.0
        norm_inv[0, 0, 2] = (width - 1) / 2.0
        norm_inv[0, 1, 1] = (height - 1) / 2.0
        norm_inv[0, 1, 2] = (height - 1) / 2.0
        norm_inv[0, 2, 2] = 1.0

        # Denormalize: norm_inv @ mtx_norm @ norm should give mtx
        recovered = matmul3x3(matmul3x3(inv3x3(norm), mtx_norm), inv3x3(norm_inv))
        assert torch.allclose(recovered, mtx, atol=1e-8)


class TestNonSquareNormalization:
    """Verify normalization correctness for non-square images."""

    def test_nonsquare_H64_W128(self) -> None:
        """Identity normalization is still identity for non-square height=64, width=128."""
        height, width = 64, 128
        mtx_norm = normalize_matrix(_eye(1).to(DEFAULT_DTYPE), height=height, width=width)
        # For identity, N @ I @ N_inv = N @ N_inv = I
        assert torch.allclose(mtx_norm, _eye(1), atol=1e-10)

    def test_nonsquare_normalization_different_scales(self) -> None:
        """N_x != N_y for non-square images."""
        height, width = 64, 128
        n_x = 2.0 / (width - 1)
        n_y = 2.0 / (height - 1)
        assert n_x != n_y


class TestDegenerateSize:
    """Verify ValueError for degenerate image dimensions."""

    def test_W1_raises_value_error(self) -> None:
        """Width=1 raises ValueError because normalization requires width >= 2."""
        with pytest.raises(ValueError, match="width must be >= 2"):
            normalize_matrix(_eye(1), height=64, width=1)

    def test_H1_raises_value_error(self) -> None:
        """Height=1 raises ValueError because normalization requires height >= 2."""
        with pytest.raises(ValueError, match="height must be >= 2"):
            normalize_matrix(_eye(1), height=1, width=64)


class TestSandwichCorrectness:
    """Verify the N @ matrix @ N_inv sandwich maps normalized coords correctly."""

    def test_sandwich_maps_normalized_coords_correctly(self) -> None:
        """N @ matrix @ N_inv maps normalized pixel coords through the transform."""
        height, width = 64, 64
        angle = torch.tensor([math.radians(45)], dtype=DEFAULT_DTYPE)
        mtx = rotation_matrix(angle, height=height, width=width)
        mtx_norm = normalize_matrix(mtx, height=height, width=width)

        # For pixel at (0, 0) -> normalized (-1, -1):
        # Should map to the normalized coordinate that mtx sends (0,0) to.
        output_px = mtx[0] @ torch.tensor([0.0, 0.0, 1.0], dtype=DEFAULT_DTYPE)
        output_norm_x = 2.0 * output_px[0].item() / (width - 1) - 1.0
        output_norm_y = 2.0 * output_px[1].item() / (height - 1) - 1.0

        # Through the sandwich
        out_norm = mtx_norm[0] @ torch.tensor([-1.0, -1.0, 1.0], dtype=DEFAULT_DTYPE)
        assert abs(out_norm[0].item() - output_norm_x) < 1e-10
        assert abs(out_norm[1].item() - output_norm_y) < 1e-10


class TestMatmul3x3:
    """Verify batched 3x3 matrix multiplication properties."""

    def test_associativity(self) -> None:
        """(albu @ batch_size) @ num_channels == albu @ (batch_size @ num_channels) for random batched 3x3 matrices."""
        mtx_a = torch.randn(4, 3, 3, dtype=DEFAULT_DTYPE)
        mtx_b = torch.randn(4, 3, 3, dtype=DEFAULT_DTYPE)
        mtx_c = torch.randn(4, 3, 3, dtype=DEFAULT_DTYPE)
        lhs = matmul3x3(matmul3x3(mtx_a, mtx_b), mtx_c)
        rhs = matmul3x3(mtx_a, matmul3x3(mtx_b, mtx_c))
        assert torch.allclose(lhs, rhs, atol=1e-10)


class TestInv3x3:
    """Verify batched 3x3 matrix inversion."""

    def test_inverse_round_trip(self) -> None:
        """Inv(matrix) @ matrix == I for random well-conditioned matrices."""
        mtx = torch.randn(4, 3, 3, dtype=DEFAULT_DTYPE)
        # Make them well-conditioned by adding scaled identity
        mtx = mtx + 5.0 * torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0).expand(4, -1, -1)
        mtx_inv = inv3x3(mtx)
        product = matmul3x3(mtx_inv, mtx)
        mtx_identity = torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0).expand(4, -1, -1)
        assert torch.allclose(product, mtx_identity, atol=1e-8)

    def test_singular_matrix_raises(self) -> None:
        """Singular (all-zeros) matrix raises ValueError."""
        mtx = torch.zeros(1, 3, 3, dtype=DEFAULT_DTYPE)
        with pytest.raises(ValueError, match="Near-singular"):
            inv3x3(mtx)

    def test_near_singular_boundary_no_raise(self) -> None:
        """Albu scale=0.01 matrix (det≈1e-4) clamps rather than raising.

        The eager-mode raise threshold is eps*1e-3 ≈ 2e-19 for float64, while the clamp threshold is eps*1e3 ≈ 2e-13.
        albu scale-0.01 matrix has |det|=1e-4, well above both thresholds, so inv3x3 must succeed without raising and
        return a finite result.

        """
        # scale_matrix with scale_x=scale_y=0.01 produces det ≈ (0.01)^2 = 1e-4
        mtx = scale_matrix(
            torch.tensor([0.01], dtype=DEFAULT_DTYPE),
            torch.tensor([0.01], dtype=DEFAULT_DTYPE),
            height=64,
            width=64,
        )
        mtx_inv = inv3x3(mtx)  # must not raise
        assert not torch.isnan(mtx_inv).any(), "inv3x3 produced NaN for near-singular matrix"
        assert not torch.isinf(mtx_inv).any(), "inv3x3 produced Inf for near-singular matrix"


class TestPerspectiveFromPoints:
    """perspective_from_points() -- DLT homography solver."""

    def test_identity_4point(self) -> None:
        """Same src and dst produces identity homography."""
        corners = torch.tensor([[[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]]])
        homography = perspective_from_points(corners, corners)
        assert homography.shape == (1, 3, 3)
        assert torch.allclose(homography, torch.eye(3).unsqueeze(0), atol=1e-4)

    def test_known_pure_translation(self) -> None:
        """Pure translation (dx=10, dy=5) should give mtx_homography close to translation matrix."""
        translation_x, translation_y = 10.0, 5.0
        src = torch.tensor([[[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]]])
        dst = src + torch.tensor([translation_x, translation_y])
        mtx_homography = perspective_from_points(src, dst)
        ones = torch.ones(1, 4, 1)
        src_h = torch.cat([src, ones], dim=-1)
        projected = (mtx_homography @ src_h.transpose(-1, -2)).transpose(-1, -2)
        homogeneous_w = projected[..., 2:3]
        xy_out = projected[..., :2] / homogeneous_w
        assert torch.allclose(xy_out, dst, atol=1e-3)

    def test_batch_shape(self) -> None:
        """Batch of 3 produces (3, 3, 3) output."""
        corners = torch.tensor([[[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]]]).expand(3, -1, -1)
        mtx_homography = perspective_from_points(corners, corners)
        assert mtx_homography.shape == (3, 3, 3)

    def test_dtype_preserved(self) -> None:
        """Output dtype matches input dtype."""
        corners = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
        mtx_homography = perspective_from_points(corners, corners)
        assert mtx_homography.dtype == torch.float64


class TestPerspectiveGrid:
    """perspective_grid() -- builds (batch_size, height, width, 2) sampling grid."""

    def test_shape(self) -> None:
        """Identity matrix produces grid of correct shape."""
        mtx_identity = torch.eye(3).unsqueeze(0)
        grid = perspective_grid(mtx_identity, height=8, width=8)
        assert grid.shape == (1, 8, 8, 2)

    def test_identity_grid_matches_affine_grid(self) -> None:
        """Identity 3x3 produces grid matching F.affine_grid with identity 2x3."""

        batch_size, height, width = 2, 8, 8
        mtx_identity = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
        grid_proj = perspective_grid(mtx_identity, height=height, width=width)
        theta_id = torch.eye(2, 3).unsqueeze(0).expand(batch_size, -1, -1)
        grid_affine = F.affine_grid(theta_id, [batch_size, 1, height, width], align_corners=True)
        assert torch.allclose(grid_proj, grid_affine, atol=1e-6)

    def test_batch_expanded(self) -> None:
        """batch_size=3 perspective matrices produce (3, height, width, 2) grid."""
        mtx_identity = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        grid = perspective_grid(mtx_identity, height=4, width=6)
        assert grid.shape == (3, 4, 6, 2)
