"""Concrete BackendConverter implementations for cross-backend output.

Provides converters between PyTorch tensors and NumPy arrays, preserving the
``(B, C, H, W)`` pipeline invariant on the torch side and converting to/from
``(B, H, W, C)`` HWC layout on the NumPy side.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


class NumpyToTorchConverter:
    """Convert NumPy ``(H, W, C)`` or ``(B, H, W, C)`` arrays to ``(B, C, H, W)`` torch tensors.

    All pipeline outputs are ``(B, C, H, W)`` ``torch.Tensor`` regardless of backend.
    ``uint8`` inputs are normalised to ``float32`` in ``[0, 1]``; ``float32`` inputs
    are passed through unchanged.

    """

    @property
    def target_backend(self) -> str:
        """Target backend identifier."""
        return "torch"

    def convert(self, array: np.ndarray) -> torch.Tensor:
        """Convert a NumPy array (HWC or BHWC) to a torch tensor (BCHW).

        Args:
            array: NumPy array of shape ``(H, W, C)`` or ``(B, H, W, C)``.

        Returns:
            ``torch.Tensor`` of shape ``(1, C, H, W)`` for 3-D input or
            ``(B, C, H, W)`` for 4-D input.

        """
        tensor = torch.from_numpy(array)
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        # (H, W, C) -> (1, C, H, W) or (B, H, W, C) -> (B, C, H, W)
        return tensor.permute(2, 0, 1).unsqueeze(0) if tensor.ndim == 3 else tensor.permute(0, 3, 1, 2)


class TorchToNumpyConverter:
    """Convert ``(B, C, H, W)`` torch tensors to NumPy ``(H, W, C)`` or ``(B, H, W, C)`` arrays.

    Single-image batches ``(1, C, H, W)`` are squeezed to ``(H, W, C)`` for
    convenience. Multi-image batches produce ``(B, H, W, C)``.

    """

    @property
    def target_backend(self) -> str:
        """Target backend identifier."""
        return "numpy"

    def convert(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor (BCHW) to a NumPy array (HWC or BHWC).

        Args:
            tensor: ``torch.Tensor`` of shape ``(B, C, H, W)``.

        Returns:
            NumPy ``ndarray`` of shape ``(H, W, C)`` when ``B == 1``, or
            ``(B, H, W, C)`` otherwise. Dtype is preserved (typically ``float32``).

        """
        import numpy as np_mod

        arr = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        if arr.shape[0] == 1:
            # (1, H, W, C) -> (H, W, C)
            arr = np_mod.squeeze(arr, axis=0)
        return arr
