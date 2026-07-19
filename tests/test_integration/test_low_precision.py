"""GPU low-precision pipeline coverage with full-precision matrix assertions.

The MPS suite compares a deterministic geometric-plus-color pipeline to both its
MPS float32 execution and a CPU float32 reference. On this machine (PyTorch
2.10.0, MPS, 2026-07-19), the smooth 128-pixel fixture measured:

- bfloat16: 41.71 dB against MPS float32 and 24.21 dB against CPU float32;
- float16: 60.30 dB against MPS float32 and 24.53 dB against CPU float32.

The MPS-vs-CPU values include the MPS float32 sampler's own 24.53 dB parity,
so the low-precision gates compare to the MPS float32 path separately. The
committed floors retain at least 3 dB margin from the measured low-precision
values and do not claim float32-exact output.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations.types import PipelineDtypeStr

_MIN_MPS_PSNR_DB: dict[PipelineDtypeStr, float] = {
    "bfloat16": 38.0,
    "float16": 55.0,
}
_MIN_CPU_REFERENCE_PSNR_DB = 20.0


def _representative_pipeline(*, pipeline_dtype: PipelineDtypeStr | None = None) -> Compose:
    """Build a deterministic fused geometric and color pipeline for precision checks."""
    return Compose.from_params(
        rotation=(17.0, 17.0),
        scale=(1.13, 1.13),
        brightness=0.1,
        contrast=0.1,
        pipeline_dtype=pipeline_dtype,
    )


def _smooth_image() -> torch.Tensor:
    """Create a smooth RGB image so precision measures sampling rather than aliasing."""
    coords = torch.linspace(0.0, 1.0, 128)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    channels = torch.stack([
        (grid_x + 0.15 * torch.sin(2.0 * torch.pi * grid_y)).clamp(0.0, 1.0),
        (grid_y + 0.15 * torch.cos(2.0 * torch.pi * grid_x)).clamp(0.0, 1.0),
        0.5 * (grid_x + grid_y),
    ])
    return channels.unsqueeze(0).repeat(2, 1, 1, 1)


def _psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the float64 peak signal-to-noise ratio for images in ``[0, 1]``."""
    mse = torch.mean((actual.to(torch.float64) - expected.to(torch.float64)) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0, dtype=torch.float64) / mse))


@pytest.mark.parametrize("pipeline_dtype", ["bfloat16", "float16"])
def test_pipeline_dtype_is_cpu_noop_and_keeps_matrix_float32(pipeline_dtype: PipelineDtypeStr) -> None:
    """CPU ignores the opt-in dtype while retaining the float32 public matrix."""
    image = torch.rand(2, 3, 32, 32)
    reference = _representative_pipeline()
    candidate = _representative_pipeline(pipeline_dtype=pipeline_dtype)

    torch.manual_seed(7)
    reference_out, reference_matrix = reference(image, return_matrix=True)
    torch.manual_seed(7)
    candidate_out, candidate_matrix = candidate(image, return_matrix=True)

    assert isinstance(reference_out, torch.Tensor)
    assert isinstance(candidate_out, torch.Tensor)
    assert reference_matrix is not None
    assert candidate_matrix is not None
    assert candidate_out.dtype is torch.float32
    assert candidate_matrix.dtype is torch.float32
    torch.testing.assert_close(candidate_out, reference_out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate_matrix, reference_matrix, rtol=0.0, atol=0.0)


def test_default_pipeline_dtype_is_bit_identical() -> None:
    """Omitting pipeline_dtype preserves the existing float32 pipeline exactly."""
    image = _smooth_image()
    first = _representative_pipeline()
    second = _representative_pipeline()

    torch.manual_seed(11)
    first_out, first_matrix = first(image, return_matrix=True)
    torch.manual_seed(11)
    second_out, second_matrix = second(image, return_matrix=True)

    assert isinstance(first_out, torch.Tensor)
    assert isinstance(second_out, torch.Tensor)
    assert first_matrix is not None
    assert second_matrix is not None
    assert torch.equal(first_out, second_out)
    assert torch.equal(first_matrix, second_matrix)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required for GPU low-precision coverage")
@pytest.mark.parametrize("pipeline_dtype", ["bfloat16", "float16"])
def test_mps_pipeline_dtype_meets_documented_psnr_and_keeps_matrix_float32(
    pipeline_dtype: PipelineDtypeStr,
) -> None:
    """MPS low precision stays within the documented MPS and CPU-reference PSNR floors."""
    image = _smooth_image()
    cpu_reference = _representative_pipeline()
    mps_float32 = _representative_pipeline()
    mps_low_precision = _representative_pipeline(pipeline_dtype=pipeline_dtype)

    torch.manual_seed(7)
    cpu_output, _ = cpu_reference(image, return_matrix=True)
    torch.manual_seed(7)
    mps_output, mps_matrix = mps_float32(image.to("mps"), return_matrix=True)
    torch.manual_seed(7)
    low_output, low_matrix = mps_low_precision(image.to("mps"), return_matrix=True)

    assert isinstance(cpu_output, torch.Tensor)
    assert isinstance(mps_output, torch.Tensor)
    assert isinstance(low_output, torch.Tensor)
    assert mps_matrix is not None
    assert low_matrix is not None
    assert low_output.dtype is torch.float32
    assert low_matrix.dtype is torch.float32
    assert torch.equal(low_matrix, mps_matrix)

    mps_psnr = _psnr(low_output.cpu(), mps_output.cpu())
    cpu_reference_psnr = _psnr(low_output.cpu(), cpu_output)
    print(
        f"pipeline_dtype={pipeline_dtype} MPS-float32 PSNR={mps_psnr:.2f} dB "
        f"CPU-float32 PSNR={cpu_reference_psnr:.2f} dB"
    )
    assert mps_psnr >= _MIN_MPS_PSNR_DB[pipeline_dtype]
    assert cpu_reference_psnr >= _MIN_CPU_REFERENCE_PSNR_DB
