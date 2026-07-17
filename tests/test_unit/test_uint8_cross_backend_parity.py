"""Cross-backend parity test parametrized over the pipeline input's origin dtype ({float32, uint8}).

Every other cross-backend parity fixture in this suite constructs a float32 ``[0, 1]`` tensor directly via
``torch.rand`` (see ``test_integration/test_cross_backend.py`` and ``test_integration/test_mixed_backend.py``).
Real pipelines commonly feed ``uint8`` images instead; this file closes that gap by also sourcing the input from a
uint8 NumPy array via the public ``NumpyToTorchConverter``, so both origins reach both backends as tensors.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose, NumpyToTorchConverter
from fuse_augmentations._compat import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as tv_trans

pytestmark = pytest.mark.skipif(
    not (_KORNIA_AVAILABLE and _TORCHVISION_AVAILABLE), reason="kornia and torchvision required"
)

HEIGHT, WIDTH, CHANNELS, BATCH_SIZE = 16, 16, 3, 2


def _uint8_image() -> np.ndarray:
    """Return a deterministic ``(batch_size, height, width, channels)`` uint8 gradient, values 0-255."""
    gradient = np.linspace(0, 255, HEIGHT * WIDTH * CHANNELS, dtype=np.float64)
    single = gradient.reshape(1, HEIGHT, WIDTH, CHANNELS)
    return np.tile(single, (BATCH_SIZE, 1, 1, 1)).astype(np.uint8)


def _float32_image() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param("float32", id="float32"),
        pytest.param("uint8", id="uint8"),
    ],
)
def test_hflip_bit_exact_across_backends_by_input_origin(origin: str) -> None:
    """Kornia and TorchVision agree bit-exactly on ``HorizontalFlip(p=1.0)``, for both input origins.

    A single-transform ``HorizontalFlip(p=1.0)`` pipeline routes to ``ExactAffineSegment``, which flips the tensor
    directly (``image.flip(dims=[3])``) instead of going through ``grid_sample`` -- so the op is lossless and both
    backends apply the identical operation to the identical input tensor. Tolerance is 0 (``torch.equal``), not a
    loosened numeric tolerance, by design: the uint8-origin input is converted to float32 once, upfront, via the
    public ``NumpyToTorchConverter`` (documented uint8 -> float32 via division by 255), and both backends then
    receive that same already-converted tensor -- a lossless flip cannot introduce backend-specific rounding on top
    of it. An interpolating (grid_sample-based) op would need a non-zero, justified tolerance instead; that is a
    separate concern already covered by this suite's existing rotation/affine parity tests.

    """
    img = NumpyToTorchConverter().convert(_uint8_image()) if origin == "uint8" else _float32_image()

    pipe_kornia = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
    pipe_torchvision = Compose([tv_trans.RandomHorizontalFlip(p=1.0)])

    out_kornia = pipe_kornia(img.clone())
    out_torchvision = pipe_torchvision(img.clone())

    assert torch.equal(out_kornia, out_torchvision)
