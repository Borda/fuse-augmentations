"""Differential test guarding float64 matrix-chain accumulation (blueprint P2).

The fused affine engine accumulates a multi-transform ``(B, 3, 3)`` chain in
``float64`` for chains longer than one op, then casts back to the image dtype at
the ``grid_sample`` boundary (``affine/segment.py`` — ``_matrix_compose_dtype`` /
``_COMPOSE_DTYPE``). ``float64`` accumulation removes the chain-length-dependent
drift that ``float32`` matmul introduces; a single transform stays in the image
dtype because there is no chain to accumulate.

The existing precision suite does not guard this choice: reverting the
accumulation dtype to ``float32`` does not trip any precision assertion, because
the final ``float32`` cast at the ``grid_sample`` boundary and the interpolation
noise floor wash out the sub-``float32`` accumulation difference in the rendered
image. This module isolates the accumulation dtype at the matmul-chain level,
where the difference survives, and asserts the ``float64`` path is measurably
closer to a ``float64`` ground-truth composition than the ``float32`` path.

Design
    Build a 6-op ill-conditioned geometric chain (alternating large shears and
    small isotropic scales — a near-singular composition where ``float32``
    matmul drift is amplified) using the package's own ``float32`` per-op matrix
    builders, exactly as the backend adapters produce them. Accumulate the chain
    in the dtype ``_matrix_compose_dtype`` selects, and compare the result
    against a ``float64`` ground truth composed from the same ``float32``-built
    per-op matrices. The ``float64`` accumulation must beat a plain ``float32``
    accumulation by a clear Frobenius and determinant margin.

Determinism
    The chain uses fixed degenerate parameters (no RNG); the ``float32`` per-op
    matrices are byte-reproducible, so the accumulation error is deterministic
    across platforms. Not skipped on any device — the accumulation runs on CPU
    tensors independent of the image device (MPS ``float64`` fallback is out of
    scope: it is a device-compatibility branch of ``_matrix_compose_dtype``, not
    the accumulation-precision claim under test here).

Acceptance
    Forcing ``_matrix_compose_dtype`` to return ``float32`` for chains longer
    than one op collapses the ``float64`` path onto the ``float32`` path and
    trips ``test_selector_accumulation_beats_float32`` — see
    ``test_forced_float32_selector_trips_the_guard``.

"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch

from fuse_augmentations.affine import segment as _segment
from fuse_augmentations.affine.matrix import (
    matmul3x3,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
)

pytestmark = pytest.mark.integration


# Ill-conditioned chain geometry: alternating large shears and small isotropic
# scales drive the composed matrix near-singular (|det| ~ 3e-5), amplifying the
# float32 matmul drift well above the float32 epsilon floor.
_CHAIN_HEIGHT = 48
_CHAIN_WIDTH = 48
_SHEAR_A_DEG = 40.0
_SHEAR_B_DEG = 35.0
_SCALE_A = 0.06
_SCALE_B = 0.09
_ROT_A_DEG = 37.0
_ROT_B_DEG = 23.0

# The float64 accumulation must be at least this many times closer to the float64
# ground truth (Frobenius) than a plain float32 accumulation. Observed ratio on
# this chain is effectively infinite (float64 error is 0.0 to machine epsilon);
# 10x is a conservative cross-platform floor.
_MIN_FROBENIUS_RATIO = 10.0

# The float32 accumulation must drift from the ground truth by at least this
# Frobenius norm — proves the chain is genuinely ill-conditioned enough that the
# guard is not vacuous (a well-conditioned chain would leave both paths equal).
_MIN_FLOAT32_DRIFT = 1e-7


def _ill_conditioned_ops_float32() -> list[torch.Tensor]:
    """Six per-op ``(1, 3, 3)`` float32 matrices for the ill-conditioned chain.

    Built with the package's own float32 matrix constructors, matching how the
    backend adapters produce per-op matrices before accumulation.

    Returns:
        Per-op forward matrices in composition order (first applied first).

    """
    float32 = torch.float32
    shear_a = shear_x_matrix(
        torch.tan(torch.tensor([math.radians(_SHEAR_A_DEG)], dtype=float32)), _CHAIN_HEIGHT, _CHAIN_WIDTH
    )
    scale_a = scale_matrix(
        torch.tensor([_SCALE_A], dtype=float32), torch.tensor([_SCALE_A], dtype=float32), _CHAIN_HEIGHT, _CHAIN_WIDTH
    )
    rot_a = rotation_matrix(torch.tensor([math.radians(_ROT_A_DEG)], dtype=float32), _CHAIN_HEIGHT, _CHAIN_WIDTH)
    shear_b = shear_x_matrix(
        torch.tan(torch.tensor([math.radians(_SHEAR_B_DEG)], dtype=float32)), _CHAIN_HEIGHT, _CHAIN_WIDTH
    )
    scale_b = scale_matrix(
        torch.tensor([_SCALE_B], dtype=float32), torch.tensor([_SCALE_B], dtype=float32), _CHAIN_HEIGHT, _CHAIN_WIDTH
    )
    rot_b = rotation_matrix(torch.tensor([math.radians(_ROT_B_DEG)], dtype=float32), _CHAIN_HEIGHT, _CHAIN_WIDTH)
    return [shear_a, scale_a, rot_a, shear_b, scale_b, rot_b]


def _accumulate(ops: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """Compose per-op matrices in ``dtype`` (mirrors the segment loop), as float64.

    Args:
        ops: Per-op ``(1, 3, 3)`` matrices in composition order.
        dtype: Accumulation dtype (each op is cast to it before ``matmul3x3``).

    Returns:
        The composed ``(3, 3)`` matrix, upcast to float64 for error metrics.

    """
    acc = torch.eye(3, dtype=dtype).unsqueeze(0)
    for op in ops:
        acc = matmul3x3(op.to(dtype), acc)
    return acc[0].double()


def _ground_truth(ops: list[torch.Tensor]) -> torch.Tensor:
    """Compose the same float32-built ops in float64 (the ideal accumulation)."""
    return _accumulate([op.double() for op in ops], torch.float64)


def _accumulate_via_selector(ops: list[torch.Tensor]) -> torch.Tensor:
    """Compose using the dtype ``_matrix_compose_dtype`` picks for this chain.

    Drives the real production selector so a regression that forces the selector back to float32 for chains longer than
    one op is caught here.

    """
    dtype = _segment._matrix_compose_dtype(torch.float32, torch.device("cpu"), len(ops))
    return _accumulate(ops, dtype)


class TestLongChainAccumulationPrecision:
    """Float64 chain accumulation beats float32 on an ill-conditioned chain."""

    @pytest.fixture(autouse=True)
    def restore_compose_dtype(self) -> Iterator[None]:
        """Snapshot ``_COMPOSE_DTYPE`` on setup and restore it on teardown.

        This is a leak-containment guard only: it does NOT force a value, so a
        legitimate source flip of the ``_COMPOSE_DTYPE`` constant (segment.py) is
        still visible to these tests (setup captures the flipped value, teardown
        restores the same flipped value, and the guards see it) — that visibility
        is exactly what the WP-15 acceptance revert relies on. The snapshot exists
        only so that if any test in THIS class mutated the constant, teardown
        leaves the module global as it found it, keeping later tests clean. The
        production default is pinned separately by ``test_compose_dtype_constant_is_float64``,
        which deliberately runs outside this fixture.

        """
        original = _segment._COMPOSE_DTYPE
        try:
            yield
        finally:
            _segment._COMPOSE_DTYPE = original

    def test_chain_is_ill_conditioned(self) -> None:
        """The composed chain is near-singular, so float32 drift is amplified."""
        composed = _ground_truth(_ill_conditioned_ops_float32())
        determinant = torch.linalg.det(composed).abs().item()
        assert determinant < 1e-3, f"chain not ill-conditioned enough (|det|={determinant:.2e})"

    def test_float32_accumulation_visibly_drifts(self) -> None:
        """Plain float32 accumulation drifts from the float64 ground truth."""
        ops = _ill_conditioned_ops_float32()
        drift = torch.linalg.norm(_accumulate(ops, torch.float32) - _ground_truth(ops)).item()
        assert drift >= _MIN_FLOAT32_DRIFT, f"chain drift too small to guard (frobenius={drift:.2e})"

    def test_selector_accumulation_beats_float32(self) -> None:
        """The selector-chosen dtype composes far closer to ground truth than float32."""
        ops = _ill_conditioned_ops_float32()
        ground_truth = _ground_truth(ops)
        error_selector = torch.linalg.norm(_accumulate_via_selector(ops) - ground_truth).item()
        error_float32 = torch.linalg.norm(_accumulate(ops, torch.float32) - ground_truth).item()
        assert error_selector <= error_float32 / _MIN_FROBENIUS_RATIO, (
            f"selector accumulation not measurably closer than float32 "
            f"(selector frobenius={error_selector:.2e}, float32 frobenius={error_float32:.2e})"
        )

    def test_selector_determinant_matches_ground_truth(self) -> None:
        """The selector path preserves the determinant far better than float32."""
        ops = _ill_conditioned_ops_float32()
        det_truth = torch.linalg.det(_ground_truth(ops)).item()
        det_selector = torch.linalg.det(_accumulate_via_selector(ops)).item()
        det_float32 = torch.linalg.det(_accumulate(ops, torch.float32)).item()
        error_selector = abs(det_selector - det_truth)
        error_float32 = abs(det_float32 - det_truth)
        assert error_selector <= error_float32, (
            f"selector determinant error not below float32 (selector={error_selector:.2e}, float32={error_float32:.2e})"
        )


def test_forced_float32_selector_trips_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing the selector to float32 for chains>1 collapses the guard margin.

    Encodes the WP-15 acceptance check as an executable test: with
    ``_matrix_compose_dtype`` patched to always return float32, the
    selector-vs-float32 Frobenius margin that ``test_selector_accumulation_beats_float32``
    relies on no longer holds.

    """
    monkeypatch.setattr(_segment, "_matrix_compose_dtype", lambda *args, **kwargs: torch.float32)
    ops = _ill_conditioned_ops_float32()
    ground_truth = _ground_truth(ops)
    error_selector = torch.linalg.norm(_accumulate_via_selector(ops) - ground_truth).item()
    error_float32 = torch.linalg.norm(_accumulate(ops, torch.float32) - ground_truth).item()
    assert error_selector > error_float32 / _MIN_FROBENIUS_RATIO, (
        "forced-float32 selector unexpectedly still satisfies the float64 guard margin"
    )


def test_compose_dtype_constant_is_float64() -> None:
    """Pin the production accumulation default to float64 (the P2 constant).

    Fixture-free by design: this is a module-level function, so the class-scoped
    ``restore_compose_dtype`` snapshot fixture never runs for it and never
    overwrites the live constant. It reads ``_COMPOSE_DTYPE`` and the multi-op
    ``_matrix_compose_dtype`` return directly, so flipping the constant at its
    source definition (segment.py) to float32 trips this test — closing the gap
    where a class fixture that forced float64 would have masked that exact revert.

    """
    assert _segment._COMPOSE_DTYPE is torch.float64, (
        f"production accumulation constant regressed to {_segment._COMPOSE_DTYPE}"
    )
    selector_dtype = _segment._matrix_compose_dtype(torch.float32, torch.device("cpu"), 6)
    assert selector_dtype is torch.float64, (
        f"selector returned {selector_dtype} for a 6-op float32 chain; expected float64"
    )
