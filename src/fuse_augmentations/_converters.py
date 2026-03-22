"""Concrete BackendConverter implementations for cross-backend output.

Provides converters between PyTorch tensors and NumPy arrays, preserving the
``(B, C, H, W)`` pipeline invariant on the torch side and converting to/from
``(B, H, W, C)`` HWC layout on the NumPy side.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    def convert(self, array: NDArray[Any]) -> torch.Tensor:
        """Convert a NumPy array (HW/HWC/BHWC) to a float32 torch tensor (BCHW).

        Args:
            array: NumPy array in channel-last layout:
                ``(H, W)``, ``(H, W, C)``, or ``(B, H, W, C)``.

        Returns:
            ``torch.float32`` tensor of shape ``(1, 1, H, W)`` for 2-D input,
            ``(1, C, H, W)`` for 3-D input, or ``(B, C, H, W)`` for 4-D input.

        Raises:
            ValueError: If ``array`` is not 2-D/3-D/4-D, or if a 3-D input is
                ambiguous and does not look like ``(H, W, C)``.

        """
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            # (H, W) -> (1, H, W, 1)
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.ndim == 3:
            # Treat 3-D arrays as single-image HWC by default so arbitrary
            # channel counts round-trip cleanly. Batched grayscale must be
            # passed explicitly as (B, H, W, 1) to avoid ambiguity.
            if tensor.shape[-1] == 0:
                msg = (
                    f"Ambiguous 3-D array shape {tuple(tensor.shape)}; expected channel-last (H, W, C) "
                    "with a non-empty channel axis. For batched grayscale, pass (B, H, W, 1)."
                )
                raise ValueError(msg)
            # (H, W, C) -> (1, H, W, C)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            msg = f"Expected 2-D/3-D/4-D numpy array, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        elif tensor.dtype != torch.float32:
            # Keep the pipeline invariant: image tensors are float32.
            tensor = tensor.to(torch.float32)

        # (B, H, W, C) -> (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)


class TorchToNumpyConverter:
    """Convert ``(B, C, H, W)`` torch tensors to NumPy ``(H, W, C)`` or ``(B, H, W, C)`` arrays.

    Single-image batches ``(1, C, H, W)`` are squeezed to ``(H, W, C)`` for
    convenience. Multi-image batches produce ``(B, H, W, C)``.

    """

    @property
    def target_backend(self) -> str:
        """Target backend identifier."""
        return "numpy"

    def convert(self, tensor: torch.Tensor) -> NDArray[Any]:
        """Convert a torch tensor (BCHW) to a NumPy array (HWC or BHWC).

        Args:
            tensor: ``torch.Tensor`` of shape ``(B, C, H, W)``.

        Returns:
            NumPy ``ndarray`` of shape ``(H, W, C)`` when ``B == 1``, or
            ``(B, H, W, C)`` otherwise. Dtype is preserved (typically ``float32``).

        Raises:
            ValueError: If ``tensor`` is not a 4-D ``(B, C, H, W)`` tensor.

        """
        import numpy as np_mod

        if tensor.ndim != 4:
            msg = f"Expected 4-D tensor (B, C, H, W), got shape {tuple(tensor.shape)}"
            raise ValueError(msg)

        arr = tensor.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
        if arr.shape[0] == 1:
            # (1, H, W, C) -> (H, W, C)
            arr = np_mod.squeeze(arr, axis=0)
        return arr
