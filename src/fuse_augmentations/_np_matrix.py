"""NumPy matrix primitives for the Albumentations (cv2) backend.

Mirrors the pixel-space convention of ``_matrix.py`` for single-sample
(non-batched) operations. Used by ``AlbumentationsAdapter`` to build
flip matrices when the transform returns an empty parameter dict.

All matrices are forward pixel-space transforms (src→dst) using the same
``align_corners=True`` convention as ``_matrix.py``.

Example:
    >>> from fuse_augmentations._np_matrix import hflip_matrix_np, vflip_matrix_np
    >>> M = hflip_matrix_np(W=4)
    >>> M[0, 0]
    -1.0
    >>> M[0, 2]
    3.0

"""

from __future__ import annotations

import numpy as np


def hflip_matrix_np(W: int) -> np.ndarray:  # noqa: N803
    """Build a (3, 3) pixel-space forward horizontal flip matrix.

    Maps ``x' = W - 1 - x``, ``y' = y``.

    Args:
        W: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward horizontal flip matrix.

    Example:
        >>> M = hflip_matrix_np(W=4)
        >>> list(M[0])
        [-1.0, 0.0, 3.0]
        >>> list(M[1])
        [0.0, 1.0, 0.0]

    """
    return np.array(
        [
            [-1.0, 0.0, float(W - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def vflip_matrix_np(H: int) -> np.ndarray:  # noqa: N803
    """Build a (3, 3) pixel-space forward vertical flip matrix.

    Maps ``x' = x``, ``y' = H - 1 - y``.

    Args:
        H: Image height in pixels.

    Returns:
        ``(3, 3)`` float64 forward vertical flip matrix.

    Example:
        >>> M = vflip_matrix_np(H=4)
        >>> list(M[1])
        [0.0, -1.0, 3.0]
        >>> list(M[0])
        [1.0, 0.0, 0.0]

    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, float(H - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
