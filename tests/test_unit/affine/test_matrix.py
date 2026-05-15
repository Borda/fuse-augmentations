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

- **Normalize round-trip**: ``denormalize(normalize(matrix)) == matrix`` for arbitrary rotations and image
  sizes. The pixel-space-to-normalized-space sandwich transform is exactly invertible.

- **Determinant product rule**: ``det(albu @ batch_size @ ...) == det(albu) * det(batch_size) * ...`` for chains of
  well-conditioned matrices. This is a fundamental property of matrix determinants and validates
  that ``matmul3x3`` preserves it numerically.

- **Inverse round-trip**: ``inv(matrix) @ matrix == I`` for random well-conditioned matrices. The custom
  ``inv3x3`` implementation is numerically accurate.

- **Perspective from points**: ``height(src) ≈ dst`` for both pure translation and genuine
  per-corner projective distortions. The DLT homography correctly maps source corners to
  destination corners under perspective division.

Example: the rotation group test generates random ``(a_deg, b_deg, height, width)`` tuples and checks::

    Ra = rotation_matrix(radians(a_deg), height, width)
    Rb = rotation_matrix(radians(b_deg), height, width)
    Rab = rotation_matrix(radians(a_deg + b_deg), height, width)
    assert allclose(Rb @ Ra, Rab)

If this property fails for any generated input, Hypothesis shrinks the counterexample to the
smallest ``(a_deg, b_deg, height, width)`` that still triggers the failure, making the root cause obvious.

"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, lists, tuples

from fuse_augmentations.affine._matrix import (
    hflip_matrix,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    perspective_from_points,
    rotation_matrix,
    scale_matrix,
    vflip_matrix,
)

DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cpu")


@pytest.mark.parametrize(
    "matrix_fn, dim_kwarg",
    [
        pytest.param(hflip_matrix, "width", id="hflip"),
        pytest.param(vflip_matrix, "height", id="vflip"),
    ],
)
@given(dim=integers(min_value=2, max_value=4096))
@settings(max_examples=50)
def test_flip_involution(matrix_fn, dim_kwarg, dim: int) -> None:
    """Flip matrix is its own inverse for any image dimension."""
    mtx = matrix_fn(**{dim_kwarg: dim}, batch_size=1, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    product = matmul3x3(mtx, mtx)
    mtx_identity = torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0)
    assert torch.allclose(product, mtx_identity, atol=1e-10)


@given(
    a_deg=floats(min_value=-180, max_value=180),
    b_deg=floats(min_value=-180, max_value=180),
    height=integers(min_value=2, max_value=512),
    width=integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_rotation_group(a_deg: float, b_deg: float, height: int, width: int) -> None:
    """R(a) @ R(b) == R(a+b) for arbitrary angles and image sizes."""
    a_rad = torch.tensor([math.radians(a_deg)], dtype=DEFAULT_DTYPE)
    b_rad = torch.tensor([math.radians(b_deg)], dtype=DEFAULT_DTYPE)
    mtx_ra = rotation_matrix(a_rad, height=height, width=width)
    mtx_rb = rotation_matrix(b_rad, height=height, width=width)
    mtx_rab = rotation_matrix(a_rad + b_rad, height=height, width=width)
    mtx_composed = matmul3x3(mtx_rb, mtx_ra)
    assert torch.allclose(mtx_composed, mtx_rab, atol=1e-6)


@given(
    scale_a=floats(min_value=0.01, max_value=100.0),
    scale_b=floats(min_value=0.01, max_value=100.0),
    scale_c=floats(min_value=0.01, max_value=100.0),
    scale_d=floats(min_value=0.01, max_value=100.0),
    height=integers(min_value=2, max_value=512),
    width=integers(min_value=2, max_value=512),
)
@settings(max_examples=100)
def test_scale_group(scale_a: float, scale_b: float, scale_c: float, scale_d: float, height: int, width: int) -> None:
    """S(a,b) @ S(c,d) == S(a*c, b*d) for arbitrary scale factors."""
    mtx_s1 = scale_matrix(
        torch.tensor([scale_a], dtype=DEFAULT_DTYPE),
        torch.tensor([scale_b], dtype=DEFAULT_DTYPE),
        height=height,
        width=width,
    )
    mtx_s2 = scale_matrix(
        torch.tensor([scale_c], dtype=DEFAULT_DTYPE),
        torch.tensor([scale_d], dtype=DEFAULT_DTYPE),
        height=height,
        width=width,
    )
    mtx_s_composed = scale_matrix(
        torch.tensor([scale_a * scale_c], dtype=DEFAULT_DTYPE),
        torch.tensor([scale_b * scale_d], dtype=DEFAULT_DTYPE),
        height=height,
        width=width,
    )
    product = matmul3x3(mtx_s2, mtx_s1)
    assert torch.allclose(product, mtx_s_composed, atol=1e-6)


@given(
    height=integers(min_value=2, max_value=512),
    width=integers(min_value=2, max_value=512),
    angle_deg=floats(min_value=-180, max_value=180),
)
@settings(max_examples=50)
def test_normalize_round_trip(height: int, width: int, angle_deg: float) -> None:
    """Denormalize(normalize(matrix)) recovers matrix for arbitrary rotations and sizes."""
    angle_rad = torch.tensor([math.radians(angle_deg)], dtype=DEFAULT_DTYPE)
    mtx = rotation_matrix(angle_rad, height=height, width=width)
    mtx_inv = inv3x3(mtx)
    mtx_norm = normalize_matrix(mtx_inv, height=height, width=width)

    # Denormalize: N_inv @ M_norm @ N
    mtx_n = torch.zeros(1, 3, 3, dtype=DEFAULT_DTYPE)
    mtx_n[0, 0, 0] = 2.0 / (width - 1)
    mtx_n[0, 0, 2] = -1.0
    mtx_n[0, 1, 1] = 2.0 / (height - 1)
    mtx_n[0, 1, 2] = -1.0
    mtx_n[0, 2, 2] = 1.0

    mtx_n_inv = inv3x3(mtx_n)
    recovered = matmul3x3(matmul3x3(mtx_n_inv, mtx_norm), mtx_n)
    assert torch.allclose(recovered, mtx_inv, atol=1e-6)


@given(
    num_matrices=integers(min_value=2, max_value=10),
    seed=integers(min_value=0, max_value=10000),
)
@settings(max_examples=50)
def test_determinant_product(num_matrices: int, seed: int) -> None:
    """Determinant of a matrix chain equals the product of individual determinants for random well-conditioned inputs.

    This is a fundamental property of matrix determinants and validates that matmul3x3 preserves determinants
    numerically across chains of arbitrary length.

    """
    torch.manual_seed(seed)
    matrices = [
        torch.randn(1, 3, 3, dtype=DEFAULT_DTYPE) + 3.0 * torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0)
        for _ in range(num_matrices)
    ]

    # Compose
    mtx_acc = matrices[0]
    for idx in range(1, num_matrices):
        mtx_acc = matmul3x3(matrices[idx], mtx_acc)

    det_product = torch.ones(1, dtype=DEFAULT_DTYPE)
    for mtx in matrices:
        det_product *= torch.det(mtx[0])

    det_composed = torch.det(mtx_acc[0])
    assert torch.allclose(det_composed, det_product, atol=1e-4)


@given(seed=integers(min_value=0, max_value=10000))
@settings(max_examples=100)
def test_inverse_round_trip(seed: int) -> None:
    """Inv(matrix) @ matrix == I for random well-conditioned matrices."""
    torch.manual_seed(seed)
    mtx = torch.randn(1, 3, 3, dtype=DEFAULT_DTYPE) + 5.0 * torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0)
    mtx_inv = inv3x3(mtx)
    product = matmul3x3(mtx_inv, mtx)
    mtx_identity = torch.eye(3, dtype=DEFAULT_DTYPE).unsqueeze(0)
    assert torch.allclose(product, mtx_identity, atol=1e-8)


@given(
    delta_x=floats(min_value=-20, max_value=20, allow_nan=False),
    delta_y=floats(min_value=-20, max_value=20, allow_nan=False),
    width=integers(min_value=10, max_value=128),
    height=integers(min_value=10, max_value=128),
)
@settings(max_examples=100)
def test_perspective_from_points_maps_src_to_dst(delta_x: float, delta_y: float, height: int, width: int) -> None:
    """Height(src) ~ dst: computed homography maps source corners to destination corners (translation)."""
    src = torch.tensor(
        [[[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]], dtype=DEFAULT_DTYPE
    )
    dst = src + torch.tensor([delta_x, delta_y], dtype=DEFAULT_DTYPE)
    mtx_homography = perspective_from_points(src, dst)
    ones = torch.ones(1, 4, 1, dtype=DEFAULT_DTYPE)
    src_h = torch.cat([src, ones], dim=-1)
    projected = (mtx_homography @ src_h.transpose(-1, -2)).transpose(-1, -2)
    homogeneous_w = projected[..., 2:3]
    xy_out = projected[..., :2] / homogeneous_w
    assert torch.allclose(xy_out, dst, atol=1e-3), (
        f"Homography didn't map src->dst correctly for delta_x={delta_x}, delta_y={delta_y}"
    )


_corner_offset = tuples(
    floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False),
    floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False),
)


@given(
    offsets=lists(_corner_offset, min_size=4, max_size=4),
    width=integers(min_value=20, max_value=128),
    height=integers(min_value=20, max_value=128),
)
@settings(max_examples=100)
def test_perspective_from_points_projective_distortion(
    offsets: list[tuple[float, float]],
    height: int,
    width: int,
) -> None:
    """Height(src) ~ dst: DLT handles genuine per-corner projective distortions.

    Unlike the translation test (which only exercises the affine columns of height), this test applies an independent
    (dx, dy) offset to each corner, forcing the DLT to use the non-linear terms and exercise the full 8 degrees of
    freedom of the homography.

    """
    src = torch.tensor(
        [[[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]], dtype=DEFAULT_DTYPE
    )
    dst = src.clone()
    for idx, (delta_x, delta_y) in enumerate(offsets):
        dst[0, idx, 0] += delta_x
        dst[0, idx, 1] += delta_y

    mtx_homography = perspective_from_points(src, dst)
    ones = torch.ones(1, 4, 1, dtype=DEFAULT_DTYPE)
    src_h = torch.cat([src, ones], dim=-1)
    projected = (mtx_homography @ src_h.transpose(-1, -2)).transpose(-1, -2)
    homogeneous_w = projected[..., 2:3]
    xy_out = projected[..., :2] / homogeneous_w
    assert torch.allclose(xy_out, dst, atol=1e-3), f"Homography didn't map src->dst correctly for offsets={offsets}"
