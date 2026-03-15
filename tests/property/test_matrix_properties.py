"""Property-based tests for matrix primitives — spec tests #61-66."""

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from fuse_augmentations._matrix import (
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    rotation_matrix,
    scale_matrix,
    vflip_matrix,
)

DTYPE = torch.float64
DEVICE = torch.device("cpu")


# --- Test #61: Involution ---


@given(W=st.integers(min_value=2, max_value=4096))
@settings(max_examples=50)
def test_hflip_involution(W: int) -> None:
    M = hflip_matrix(W=W, batch_size=1, device=DEVICE, dtype=DTYPE)
    product = matmul3x3(M, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-10)


@given(H=st.integers(min_value=2, max_value=4096))
@settings(max_examples=50)
def test_vflip_involution(H: int) -> None:
    M = vflip_matrix(H=H, batch_size=1, device=DEVICE, dtype=DTYPE)
    product = matmul3x3(M, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-10)


# --- Test #62: Rotation group ---


@given(
    a_deg=st.floats(min_value=-180, max_value=180),
    b_deg=st.floats(min_value=-180, max_value=180),
    H=st.integers(min_value=2, max_value=512),
    W=st.integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_rotation_group(a_deg: float, b_deg: float, H: int, W: int) -> None:
    a_rad = torch.tensor([math.radians(a_deg)], dtype=DTYPE)
    b_rad = torch.tensor([math.radians(b_deg)], dtype=DTYPE)
    Ra = rotation_matrix(a_rad, H=H, W=W)
    Rb = rotation_matrix(b_rad, H=H, W=W)
    Rab = rotation_matrix(a_rad + b_rad, H=H, W=W)
    composed = matmul3x3(Rb, Ra)
    assert torch.allclose(composed, Rab, atol=1e-6)


# --- Test #63: Scale group ---


@given(
    a=st.floats(min_value=0.01, max_value=100.0),
    b=st.floats(min_value=0.01, max_value=100.0),
    c=st.floats(min_value=0.01, max_value=100.0),
    d=st.floats(min_value=0.01, max_value=100.0),
    H=st.integers(min_value=2, max_value=512),
    W=st.integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_scale_group(a: float, b: float, c: float, d: float, H: int, W: int) -> None:
    S1 = scale_matrix(torch.tensor([a], dtype=DTYPE), torch.tensor([b], dtype=DTYPE), H=H, W=W)
    S2 = scale_matrix(torch.tensor([c], dtype=DTYPE), torch.tensor([d], dtype=DTYPE), H=H, W=W)
    S_composed = scale_matrix(torch.tensor([a * c], dtype=DTYPE), torch.tensor([b * d], dtype=DTYPE), H=H, W=W)
    product = matmul3x3(S2, S1)
    assert torch.allclose(product, S_composed, atol=1e-6)


# --- Test #64: Normalize round-trip ---


@given(
    H=st.integers(min_value=2, max_value=512),
    W=st.integers(min_value=2, max_value=512),
    angle_deg=st.floats(min_value=-180, max_value=180),
)
@settings(max_examples=50)
def test_normalize_round_trip(H: int, W: int, angle_deg: float) -> None:
    angle_rad = torch.tensor([math.radians(angle_deg)], dtype=DTYPE)
    M = rotation_matrix(angle_rad, H=H, W=W)
    M_inv = inv3x3(M)
    M_norm = normalize_matrix(M_inv, H=H, W=W)

    # Denormalize: N_inv @ M_norm @ N
    N = torch.zeros(1, 3, 3, dtype=DTYPE)
    N[0, 0, 0] = 2.0 / (W - 1)
    N[0, 0, 2] = -1.0
    N[0, 1, 1] = 2.0 / (H - 1)
    N[0, 1, 2] = -1.0
    N[0, 2, 2] = 1.0

    N_inv = inv3x3(N)
    recovered = matmul3x3(matmul3x3(N_inv, M_norm), N)
    assert torch.allclose(recovered, M_inv, atol=1e-6)


# --- Test #65: Determinant product ---


@given(
    n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50)
def test_determinant_product(n: int, seed: int) -> None:
    torch.manual_seed(seed)
    matrices = [torch.randn(1, 3, 3, dtype=DTYPE) + 3.0 * torch.eye(3, dtype=DTYPE).unsqueeze(0) for _ in range(n)]

    # Compose
    acc = matrices[0]
    for i in range(1, n):
        acc = matmul3x3(matrices[i], acc)

    det_product = torch.ones(1, dtype=DTYPE)
    for m in matrices:
        det_product *= torch.det(m[0])

    det_composed = torch.det(acc[0])
    assert torch.allclose(det_composed, det_product, atol=1e-4)


# --- Test #66: Inverse round-trip ---


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=100)
def test_inverse_round_trip(seed: int) -> None:
    torch.manual_seed(seed)
    M = torch.randn(1, 3, 3, dtype=DTYPE) + 5.0 * torch.eye(3, dtype=DTYPE).unsqueeze(0)
    M_inv = inv3x3(M)
    product = matmul3x3(M_inv, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-8)
