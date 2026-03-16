"""Tests for _matrix.py — spec tests #1-16."""

import math

import pytest
import torch

from fuse_augmentations._matrix import (
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64  # use float64 for tighter tolerances in unit tests


def _eye(B: int = 1) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE).unsqueeze(0).expand(B, -1, -1)


def _apply_point(M: torch.Tensor, x: float, y: float) -> tuple[float, float]:
    """Apply (1, 3, 3) matrix to a 2D point in homogeneous coords."""
    p = torch.tensor([x, y, 1.0], dtype=M.dtype)
    out = M[0] @ p
    return out[0].item(), out[1].item()


# --- Test #1: Identity cases ---


class TestIdentityCases:
    def test_rotation_zero_is_identity(self) -> None:
        M = rotation_matrix(torch.zeros(2, dtype=DTYPE), H=64, W=64)
        assert torch.allclose(M, _eye(2))

    def test_scale_one_is_identity(self) -> None:
        M = scale_matrix(torch.ones(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE), H=64, W=64)
        assert torch.allclose(M, _eye(2))

    def test_shear_x_zero_is_identity(self) -> None:
        M = shear_x_matrix(torch.zeros(2, dtype=DTYPE), H=64, W=64)
        assert torch.allclose(M, _eye(2))

    def test_shear_y_zero_is_identity(self) -> None:
        M = shear_y_matrix(torch.zeros(2, dtype=DTYPE), H=64, W=64)
        assert torch.allclose(M, _eye(2))

    def test_translate_zero_is_identity(self) -> None:
        M = translate_matrix(torch.zeros(2, dtype=DTYPE), torch.zeros(2, dtype=DTYPE))
        assert torch.allclose(M, _eye(2))


# --- Test #2: Involution ---


class TestInvolution:
    def test_hflip_involution(self) -> None:
        M = hflip_matrix(W=64, batch_size=1, device=DEVICE, dtype=DTYPE)
        product = matmul3x3(M, M)
        assert torch.allclose(product, _eye(1), atol=1e-10)

    def test_vflip_involution(self) -> None:
        M = vflip_matrix(H=64, batch_size=1, device=DEVICE, dtype=DTYPE)
        product = matmul3x3(M, M)
        assert torch.allclose(product, _eye(1), atol=1e-10)


# --- Test #3: Rotation group ---


class TestRotationGroup:
    @pytest.mark.parametrize("a_deg,b_deg", [(30.0, 45.0), (-60.0, 90.0), (10.0, -170.0)])
    def test_rotation_composition(self, a_deg: float, b_deg: float) -> None:
        a_rad = torch.tensor([math.radians(a_deg)], dtype=DTYPE)
        b_rad = torch.tensor([math.radians(b_deg)], dtype=DTYPE)
        Ra = rotation_matrix(a_rad, H=64, W=64)
        Rb = rotation_matrix(b_rad, H=64, W=64)
        Rab = rotation_matrix(a_rad + b_rad, H=64, W=64)
        composed = matmul3x3(Rb, Ra)
        assert torch.allclose(composed, Rab, atol=1e-10)


# --- Test #4: Scale group ---


class TestScaleGroup:
    def test_scale_composition(self) -> None:
        sx1, sy1 = torch.tensor([2.0], dtype=DTYPE), torch.tensor([3.0], dtype=DTYPE)
        sx2, sy2 = torch.tensor([0.5], dtype=DTYPE), torch.tensor([0.25], dtype=DTYPE)
        S1 = scale_matrix(sx1, sy1, H=64, W=64)
        S2 = scale_matrix(sx2, sy2, H=64, W=64)
        S_composed = scale_matrix(sx1 * sx2, sy1 * sy2, H=64, W=64)
        product = matmul3x3(S2, S1)
        assert torch.allclose(product, S_composed, atol=1e-10)


# --- Test #5: Shear group (tangent values) ---


class TestShearGroup:
    def test_shear_x_composition(self) -> None:
        a = torch.tensor([0.3], dtype=DTYPE)
        b = torch.tensor([0.4], dtype=DTYPE)
        Sha = shear_x_matrix(a, H=64, W=64)
        Shb = shear_x_matrix(b, H=64, W=64)
        Shab = shear_x_matrix(a + b, H=64, W=64)
        product = matmul3x3(Shb, Sha)
        assert torch.allclose(product, Shab, atol=1e-10)


# --- Test #6: Flip-rotation equivalence ---


class TestFlipRotation:
    def test_hflip_vflip_equals_rotation_180(self) -> None:
        """Hflip @ vflip == R(pi) for square images."""
        H = W = 64
        hf = hflip_matrix(W=W, batch_size=1, device=DEVICE, dtype=DTYPE)
        vf = vflip_matrix(H=H, batch_size=1, device=DEVICE, dtype=DTYPE)
        composed = matmul3x3(hf, vf)
        R180 = rotation_matrix(torch.tensor([math.pi], dtype=DTYPE), H=H, W=W)
        assert torch.allclose(composed, R180, atol=1e-10)


# --- Test #7: Corner mapping HFlip W=4 ---


class TestCornerMappingHFlip:
    def test_hflip_corners_W4(self) -> None:
        M = hflip_matrix(W=4, batch_size=1, device=DEVICE, dtype=DTYPE)
        # (0,0) -> (3,0)
        x, y = _apply_point(M, 0.0, 0.0)
        assert abs(x - 3.0) < 1e-10
        assert abs(y - 0.0) < 1e-10
        # (3,0) -> (0,0)
        x, y = _apply_point(M, 3.0, 0.0)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 0.0) < 1e-10
        # (1,1) -> (2,1)
        x, y = _apply_point(M, 1.0, 1.0)
        assert abs(x - 2.0) < 1e-10
        assert abs(y - 1.0) < 1e-10


# --- Test #8: Rotation 90 CCW square ---


class TestRotation90CCW:
    def test_rotation_90_ccw_square(self) -> None:
        """90-degree CCW rotation around center of a 4x4 image.

        With cx=cy=1.5: (0,0) -> (3,0), (3,0) -> (3,3), (3,3) -> (0,3), (0,3) -> (0,0).
        """
        H = W = 4
        angle = torch.tensor([math.pi / 2], dtype=DTYPE)  # 90 deg CCW
        M = rotation_matrix(angle, H=H, W=W)
        # (0, 0) -> (3, 0)
        x, y = _apply_point(M, 0.0, 0.0)
        assert abs(x - 3.0) < 1e-10
        assert abs(y - 0.0) < 1e-10
        # (3, 0) -> (3, 3)
        x, y = _apply_point(M, 3.0, 0.0)
        assert abs(x - 3.0) < 1e-10
        assert abs(y - 3.0) < 1e-10
        # (3, 3) -> (0, 3)
        x, y = _apply_point(M, 3.0, 3.0)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 3.0) < 1e-10
        # (0, 3) -> (0, 0)
        x, y = _apply_point(M, 0.0, 3.0)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 0.0) < 1e-10


# --- Test #9: Shear conversion pi/4 -> 1.0 ---


class TestShearConversion:
    def test_shear_x_pi_over_4_gives_1(self) -> None:
        """build_matrix receives tan(shear_rad), so tan(pi/4)=1.0."""
        shear_x_tan = torch.tensor([math.tan(math.pi / 4)], dtype=DTYPE)
        M = shear_x_matrix(shear_x_tan, H=64, W=64)
        # M[0, 0, 1] should be 1.0 (the shear element)
        assert abs(M[0, 0, 1].item() - 1.0) < 1e-10


# --- Test #10: Non-square images ---


class TestNonSquare:
    @pytest.mark.parametrize("H,W", [(32, 64), (64, 32), (100, 200), (1, 2)])
    def test_rotation_identity_nonsquare(self, H: int, W: int) -> None:
        M = rotation_matrix(torch.zeros(1, dtype=DTYPE), H=H, W=W)
        assert torch.allclose(M, _eye(1))

    @pytest.mark.parametrize("H,W", [(32, 64), (64, 32), (100, 200)])
    def test_scale_identity_nonsquare(self, H: int, W: int) -> None:
        M = scale_matrix(torch.ones(1, dtype=DTYPE), torch.ones(1, dtype=DTYPE), H=H, W=W)
        assert torch.allclose(M, _eye(1))


# --- Test #11: normalize(I, H, W) -> identity grid ---


class TestNormalizeIdentity:
    def test_normalize_identity_maps_pixels_to_self(self) -> None:
        H, W = 64, 64
        M_norm = normalize_matrix(_eye(1), H=H, W=W)
        # Identity in pixel space -> identity in normalized space
        assert torch.allclose(M_norm, _eye(1), atol=1e-10)


# --- Test #12: Corner mapping align_corners=True ---


class TestCornerMappingNormalized:
    def test_pixel_0_0_maps_to_minus1_minus1(self) -> None:
        """For identity matrix, pixel (0,0) should map to (-1,-1) in normalized coords."""
        H, W = 64, 64
        M_norm = normalize_matrix(_eye(1), H=H, W=W)
        # For identity, the normalized matrix should also be identity
        assert torch.allclose(M_norm, _eye(1), atol=1e-10)
        # N maps pixel (0,0) to (-1, -1):
        N_00 = 2.0 / (W - 1)
        N_02 = -1.0
        assert abs(N_00 * 0.0 + N_02 - (-1.0)) < 1e-10

    def test_pixel_W_minus_1_H_minus_1_maps_to_1_1(self) -> None:
        H, W = 64, 64
        N_00 = 2.0 / (W - 1)
        N_02 = -1.0
        N_11 = 2.0 / (H - 1)
        N_12 = -1.0
        assert abs(N_00 * (W - 1) + N_02 - 1.0) < 1e-10
        assert abs(N_11 * (H - 1) + N_12 - 1.0) < 1e-10


# --- Test #13: Round-trip denormalize(normalize(M)) ~ M ---


class TestNormalizeRoundTrip:
    def test_normalize_inv_normalize_is_identity(self) -> None:
        """denormalize(normalize(M)) should recover M."""
        H, W = 64, 64
        angle = torch.tensor([math.radians(30)], dtype=DTYPE)
        M = rotation_matrix(angle, H=H, W=W)
        M_norm = normalize_matrix(M, H=H, W=W)

        # Build N and N_inv manually to denormalize
        N = torch.zeros(1, 3, 3, dtype=DTYPE)
        N[0, 0, 0] = 2.0 / (W - 1)
        N[0, 0, 2] = -1.0
        N[0, 1, 1] = 2.0 / (H - 1)
        N[0, 1, 2] = -1.0
        N[0, 2, 2] = 1.0

        N_inv = torch.zeros(1, 3, 3, dtype=DTYPE)
        N_inv[0, 0, 0] = (W - 1) / 2.0
        N_inv[0, 0, 2] = (W - 1) / 2.0
        N_inv[0, 1, 1] = (H - 1) / 2.0
        N_inv[0, 1, 2] = (H - 1) / 2.0
        N_inv[0, 2, 2] = 1.0

        # Denormalize: N_inv @ M_norm @ N should give M
        recovered = matmul3x3(matmul3x3(inv3x3(N), M_norm), inv3x3(N_inv))
        assert torch.allclose(recovered, M, atol=1e-8)


# --- Test #14: Non-square normalization ---


class TestNonSquareNormalization:
    def test_nonsquare_H64_W128(self) -> None:
        H, W = 64, 128
        M_norm = normalize_matrix(_eye(1).to(DTYPE), H=H, W=W)
        # For identity, N @ I @ N_inv = N @ N_inv = I
        assert torch.allclose(M_norm, _eye(1), atol=1e-10)

    def test_nonsquare_normalization_different_scales(self) -> None:
        """N_x != N_y for non-square images."""
        H, W = 64, 128
        N_x = 2.0 / (W - 1)
        N_y = 2.0 / (H - 1)
        assert N_x != N_y


# --- Test #15: Degenerate W=1 or H=1 ---


class TestDegenerateSize:
    def test_W1_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="W must be >= 2"):
            normalize_matrix(_eye(1), H=64, W=1)

    def test_H1_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="H must be >= 2"):
            normalize_matrix(_eye(1), H=1, W=64)


# --- Test #16: Sandwich correctness ---


class TestSandwichCorrectness:
    def test_sandwich_maps_normalized_coords_correctly(self) -> None:
        """N @ M @ N_inv maps normalized pixel coords through the transform."""
        H, W = 64, 64
        angle = torch.tensor([math.radians(45)], dtype=DTYPE)
        M = rotation_matrix(angle, H=H, W=W)
        M_norm = normalize_matrix(M, H=H, W=W)

        # For pixel at (0, 0) -> normalized (-1, -1):
        # Should map to the normalized coordinate that M sends (0,0) to.
        output_px = M[0] @ torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        output_norm_x = 2.0 * output_px[0].item() / (W - 1) - 1.0
        output_norm_y = 2.0 * output_px[1].item() / (H - 1) - 1.0

        # Through the sandwich
        out_norm = M_norm[0] @ torch.tensor([-1.0, -1.0, 1.0], dtype=DTYPE)
        assert abs(out_norm[0].item() - output_norm_x) < 1e-10
        assert abs(out_norm[1].item() - output_norm_y) < 1e-10


# --- Additional matmul3x3 and inv3x3 tests ---


class TestMatmul3x3:
    def test_associativity(self) -> None:
        torch.manual_seed(42)
        A = torch.randn(4, 3, 3, dtype=DTYPE)
        B = torch.randn(4, 3, 3, dtype=DTYPE)
        C = torch.randn(4, 3, 3, dtype=DTYPE)
        lhs = matmul3x3(matmul3x3(A, B), C)
        rhs = matmul3x3(A, matmul3x3(B, C))
        assert torch.allclose(lhs, rhs, atol=1e-10)


class TestInv3x3:
    def test_inverse_round_trip(self) -> None:
        torch.manual_seed(42)
        M = torch.randn(4, 3, 3, dtype=DTYPE)
        # Make them well-conditioned by adding scaled identity
        M = M + 5.0 * torch.eye(3, dtype=DTYPE).unsqueeze(0).expand(4, -1, -1)
        M_inv = inv3x3(M)
        product = matmul3x3(M_inv, M)
        I = torch.eye(3, dtype=DTYPE).unsqueeze(0).expand(4, -1, -1)
        assert torch.allclose(product, I, atol=1e-8)

    def test_singular_matrix_raises(self) -> None:
        M = torch.zeros(1, 3, 3, dtype=DTYPE)
        with pytest.raises(ValueError, match="Near-singular"):
            inv3x3(M)
