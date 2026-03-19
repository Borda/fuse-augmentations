"""Property-based tests for matrix primitives using Hypothesis.

This module uses the Hypothesis library to verify algebraic invariants of the matrix primitives
in ``fuse_augmentations._matrix``. Unlike example-based unit tests that check specific
input/output pairs, property-based tests generate hundreds of random inputs per invariant and
verify the property holds for all of them. This catches edge cases (extreme angles, tiny/huge
scale factors, non-square dimensions) that hand-picked examples would miss.

Hypothesis manages its own RNG and shrinks failing inputs to minimal counterexamples, making
failures easy to reproduce and debug. Each test function declares the *property* (mathematical
invariant) that should hold, and Hypothesis searches for violations.

Invariants tested here:

- **Flip involution**: ``hflip @ hflip == I`` for any width; ``vflip @ vflip == I`` for any
  height. Flip matrices are self-inverse regardless of image dimensions.

- **Rotation group**: ``R(a) @ R(b) == R(a+b)`` for arbitrary angles and image sizes. Rotation
  matrices compose additively in angle, forming a group under multiplication.

- **Scale group**: ``S(a,b) @ S(c,d) == S(a*c, b*d)`` for positive scale factors. Scale
  matrices compose multiplicatively, forming a group under multiplication.

- **Normalize round-trip**: ``denormalize(normalize(M)) == M`` for arbitrary rotations and image
  sizes. The pixel-space-to-normalized-space sandwich transform is exactly invertible.

- **Determinant product rule**: ``det(A @ B @ ...) == det(A) * det(B) * ...`` for chains of
  well-conditioned matrices. This is a fundamental property of matrix determinants and validates
  that ``matmul3x3`` preserves it numerically.

- **Inverse round-trip**: ``inv(M) @ M == I`` for random well-conditioned matrices. The custom
  ``inv3x3`` implementation is numerically accurate.

Example: the rotation group test generates random ``(a_deg, b_deg, H, W)`` tuples and checks::

    Ra = rotation_matrix(radians(a_deg), H, W)
    Rb = rotation_matrix(radians(b_deg), H, W)
    Rab = rotation_matrix(radians(a_deg + b_deg), H, W)
    assert allclose(Rb @ Ra, Rab)

If this property fails for any generated input, Hypothesis shrinks the counterexample to the
smallest ``(a_deg, b_deg, H, W)`` that still triggers the failure, making the root cause obvious.

"""

import math

import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from fuse_augmentations.affine._matrix import (
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


@given(W=integers(min_value=2, max_value=4096))
@settings(max_examples=50)
def test_hflip_involution(W: int) -> None:
    """Hflip @ Hflip == identity for any width."""
    M = hflip_matrix(W=W, batch_size=1, device=DEVICE, dtype=DTYPE)
    product = matmul3x3(M, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-10)


@given(H=integers(min_value=2, max_value=4096))
@settings(max_examples=50)
def test_vflip_involution(H: int) -> None:
    """Vflip @ Vflip == identity for any height."""
    M = vflip_matrix(H=H, batch_size=1, device=DEVICE, dtype=DTYPE)
    product = matmul3x3(M, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-10)


@given(
    a_deg=floats(min_value=-180, max_value=180),
    b_deg=floats(min_value=-180, max_value=180),
    H=integers(min_value=2, max_value=512),
    W=integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_rotation_group(a_deg: float, b_deg: float, H: int, W: int) -> None:
    """R(a) @ R(b) == R(a+b) for arbitrary angles and image sizes."""
    a_rad = torch.tensor([math.radians(a_deg)], dtype=DTYPE)
    b_rad = torch.tensor([math.radians(b_deg)], dtype=DTYPE)
    Ra = rotation_matrix(a_rad, H=H, W=W)
    Rb = rotation_matrix(b_rad, H=H, W=W)
    Rab = rotation_matrix(a_rad + b_rad, H=H, W=W)
    composed = matmul3x3(Rb, Ra)
    assert torch.allclose(composed, Rab, atol=1e-6)


@given(
    a=floats(min_value=0.01, max_value=100.0),
    b=floats(min_value=0.01, max_value=100.0),
    c=floats(min_value=0.01, max_value=100.0),
    d=floats(min_value=0.01, max_value=100.0),
    H=integers(min_value=2, max_value=512),
    W=integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_scale_group(a: float, b: float, c: float, d: float, H: int, W: int) -> None:
    """S(a,b) @ S(c,d) == S(a*c, b*d) for arbitrary scale factors."""
    S1 = scale_matrix(torch.tensor([a], dtype=DTYPE), torch.tensor([b], dtype=DTYPE), H=H, W=W)
    S2 = scale_matrix(torch.tensor([c], dtype=DTYPE), torch.tensor([d], dtype=DTYPE), H=H, W=W)
    S_composed = scale_matrix(torch.tensor([a * c], dtype=DTYPE), torch.tensor([b * d], dtype=DTYPE), H=H, W=W)
    product = matmul3x3(S2, S1)
    assert torch.allclose(product, S_composed, atol=1e-6)


@given(
    H=integers(min_value=2, max_value=512),
    W=integers(min_value=2, max_value=512),
    angle_deg=floats(min_value=-180, max_value=180),
)
@settings(max_examples=50)
def test_normalize_round_trip(H: int, W: int, angle_deg: float) -> None:
    """Denormalize(normalize(M)) recovers M for arbitrary rotations and sizes."""
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


@given(
    n=integers(min_value=2, max_value=10),
    seed=integers(min_value=0, max_value=10000),
)
@settings(max_examples=50)
def test_determinant_product(n: int, seed: int) -> None:
    """Det(A @ B @ ...) == det(A) * det(B) * ...

    for random well-conditioned matrices.

    """
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


@given(seed=integers(min_value=0, max_value=10000))
@settings(max_examples=100)
def test_inverse_round_trip(seed: int) -> None:
    """Inv(M) @ M == I for random well-conditioned matrices."""
    torch.manual_seed(seed)
    M = torch.randn(1, 3, 3, dtype=DTYPE) + 5.0 * torch.eye(3, dtype=DTYPE).unsqueeze(0)
    M_inv = inv3x3(M)
    product = matmul3x3(M_inv, M)
    I = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(product, I, atol=1e-8)
