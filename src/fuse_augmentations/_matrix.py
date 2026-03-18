"""Matrix primitives for 2D affine transform composition.

All functions operate on batched ``(B, 3, 3)`` homogeneous matrices in
pixel coordinates with ``align_corners=True`` convention. Center of image
is ``cx = (W-1)/2``, ``cy = (H-1)/2``.

Example:
    >>> import torch
    >>> from fuse_augmentations._matrix import rotation_matrix, matmul3x3, inv3x3
    >>> M = rotation_matrix(torch.zeros(2), H=64, W=64)
    >>> M.shape
    torch.Size([2, 3, 3])

"""

from __future__ import annotations

import torch


def rotation_matrix(angle_rad: torch.Tensor, H: int, W: int) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space forward rotation matrix around image center.

    Args:
        angle_rad: Rotation angle in radians, shape ``(B,)``.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        ``(B, 3, 3)`` forward rotation matrix in pixel coordinates.

    Example:
        >>> import torch
        >>> M = rotation_matrix(torch.zeros(2), H=64, W=64)
        >>> torch.allclose(M, torch.eye(3).unsqueeze(0).expand(2, -1, -1))
        True

    """
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    B = angle_rad.shape[0]  # noqa: N806
    zeros = torch.zeros(B, device=angle_rad.device, dtype=angle_rad.dtype)
    ones = torch.ones(B, device=angle_rad.device, dtype=angle_rad.dtype)
    row0 = torch.stack([cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a], dim=-1)
    row1 = torch.stack([sin_a, cos_a, cy * (1 - cos_a) - cx * sin_a], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def scale_matrix(sx: torch.Tensor, sy: torch.Tensor, H: int, W: int) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space forward scale matrix around image center.

    Args:
        sx: X scale factor, shape ``(B,)``. 1.0 = identity.
        sy: Y scale factor, shape ``(B,)``. 1.0 = identity.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        ``(B, 3, 3)`` forward scale matrix.

    Example:
        >>> import torch
        >>> M = scale_matrix(torch.ones(1), torch.ones(1), H=64, W=64)
        >>> torch.allclose(M, torch.eye(3).unsqueeze(0))
        True

    """
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    B = sx.shape[0]  # noqa: N806
    zeros = torch.zeros(B, device=sx.device, dtype=sx.dtype)
    ones = torch.ones(B, device=sx.device, dtype=sx.dtype)
    row0 = torch.stack([sx, zeros, cx * (1 - sx)], dim=-1)
    row1 = torch.stack([zeros, sy, cy * (1 - sy)], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def shear_x_matrix(shear_x_tan: torch.Tensor, H: int, W: int) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space forward x-shear matrix around image center.

    ``shear_x_tan`` is ``tan(shear_x_rad)`` - already converted from degrees
    by the adapter.

    Args:
        shear_x_tan: Tangent of x-shear angle, shape ``(B,)``.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        ``(B, 3, 3)`` forward x-shear matrix.

    Example:
        >>> import torch
        >>> M = shear_x_matrix(torch.zeros(1), H=64, W=64)
        >>> torch.allclose(M, torch.eye(3).unsqueeze(0))
        True

    """
    cy = (H - 1) / 2.0
    B = shear_x_tan.shape[0]  # noqa: N806
    zeros = torch.zeros(B, device=shear_x_tan.device, dtype=shear_x_tan.dtype)
    ones = torch.ones(B, device=shear_x_tan.device, dtype=shear_x_tan.dtype)
    row0 = torch.stack([ones, shear_x_tan, -cy * shear_x_tan], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def shear_y_matrix(shear_y_tan: torch.Tensor, H: int, W: int) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space forward y-shear matrix around image center.

    ``shear_y_tan`` is ``tan(shear_y_rad)`` - already converted from degrees
    by the adapter.

    Args:
        shear_y_tan: Tangent of y-shear angle, shape ``(B,)``.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        ``(B, 3, 3)`` forward y-shear matrix.

    Example:
        >>> import torch
        >>> M = shear_y_matrix(torch.zeros(1), H=64, W=64)
        >>> torch.allclose(M, torch.eye(3).unsqueeze(0))
        True

    """
    cx = (W - 1) / 2.0
    B = shear_y_tan.shape[0]  # noqa: N806
    zeros = torch.zeros(B, device=shear_y_tan.device, dtype=shear_y_tan.dtype)
    ones = torch.ones(B, device=shear_y_tan.device, dtype=shear_y_tan.dtype)
    row0 = torch.stack([ones, zeros, zeros], dim=-1)
    row1 = torch.stack([shear_y_tan, ones, -cx * shear_y_tan], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def translate_matrix(tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) pixel-space translation matrix.

    Args:
        tx: X translation in pixels, shape ``(B,)``.
        ty: Y translation in pixels, shape ``(B,)``.

    Returns:
        ``(B, 3, 3)`` translation matrix.

    Example:
        >>> import torch
        >>> M = translate_matrix(torch.zeros(1), torch.zeros(1))
        >>> torch.allclose(M, torch.eye(3).unsqueeze(0))
        True

    """
    B = tx.shape[0]  # noqa: N806
    zeros = torch.zeros(B, device=tx.device, dtype=tx.dtype)
    ones = torch.ones(B, device=tx.device, dtype=tx.dtype)
    row0 = torch.stack([ones, zeros, tx], dim=-1)
    row1 = torch.stack([zeros, ones, ty], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def hflip_matrix(W: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space horizontal flip matrix.

    Args:
        W: Image width in pixels.
        batch_size: Batch size B.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(B, 3, 3)`` horizontal flip matrix.

    Example:
        >>> import torch
        >>> M = hflip_matrix(W=4, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        >>> M[0, 0, 0].item()
        -1.0
        >>> M[0, 0, 2].item()
        3.0

    """
    M = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)  # noqa: N806
    M[:, 0, 0] = -1.0
    M[:, 0, 2] = float(W - 1)
    M[:, 1, 1] = 1.0
    M[:, 2, 2] = 1.0
    return M


def vflip_matrix(H: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:  # noqa: N803
    """Build (B, 3, 3) pixel-space vertical flip matrix.

    Args:
        H: Image height in pixels.
        batch_size: Batch size B.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(B, 3, 3)`` vertical flip matrix.

    Example:
        >>> import torch
        >>> M = vflip_matrix(H=4, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        >>> M[0, 1, 1].item()
        -1.0
        >>> M[0, 1, 2].item()
        3.0

    """
    M = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)  # noqa: N806
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = -1.0
    M[:, 1, 2] = float(H - 1)
    M[:, 2, 2] = 1.0
    return M


def matmul3x3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """Batch 3x3 matrix multiply via element-wise ops - torch.compile-fusible.

    Uses ``unbind`` on rows/cols to avoid ``torch.bmm`` kernel-launch overhead
    on small matrices.

    Args:
        A: ``(B, 3, 3)`` left matrix.
        B: ``(B, 3, 3)`` right matrix.

    Returns:
        ``(B, 3, 3)`` product ``A @ B``.

    Example:
        >>> import torch
        >>> I = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        >>> R = matmul3x3(I, I)
        >>> torch.allclose(R, I)
        True

    """
    a = A.unbind(dim=-2)  # tuple of 3 rows, each (B, 3)
    b = B.unbind(dim=-1)  # tuple of 3 cols, each (B, 3)
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            elem = a[i][:, 0] * b[j][:, 0] + a[i][:, 1] * b[j][:, 1] + a[i][:, 2] * b[j][:, 2]
            row.append(elem.unsqueeze(-1))
        rows.append(torch.cat(row, dim=-1).unsqueeze(-2))
    return torch.cat(rows, dim=-2)


def inv3x3(M: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """Batch analytic 3x3 matrix inverse via Cramer's rule - torch.compile-fusible.

    Uses element-wise operations only (no ``torch.linalg.inv``). Includes a
    compile-safe singularity guard via determinant clamping and an eager-mode
    check that raises ``ValueError`` for near-singular matrices.

    Args:
        M: ``(B, 3, 3)`` batch of matrices.

    Returns:
        ``(B, 3, 3)`` batch of inverse matrices.

    Raises:
        ValueError: If any matrix in the batch has a near-zero determinant
            (eager mode only).

    Example:
        >>> import torch
        >>> I = torch.eye(3).unsqueeze(0)
        >>> torch.allclose(inv3x3(I), I)
        True

    """
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    eps = torch.finfo(M.dtype).eps * 1e3

    # Eager-mode check: raise for user-facing errors before compiled scope.
    # The raise threshold (eps * 1e-3) is lower than the clamp threshold (eps)
    # so that borderline cases like scale=0.01 are silently clamped rather than
    # raising, while truly degenerate matrices (det ≈ 0) still produce an error.
    if not _is_compiling() and (det.abs() < eps * 1e-3).any():
        msg = (
            f"Near-singular matrix detected (min |det|={det.abs().min().item():.2e}). "
            "Check for extreme scale values or degenerate transforms."
        )
        raise ValueError(msg)

    # Compile-safe singularity guard: clamp |det| to eps, preserve sign
    det = det.sign() * det.abs().clamp(min=eps)

    inv_det = 1.0 / det
    adj = torch.stack(
        [
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    result: torch.Tensor = adj * inv_det[:, None, None]
    return result


def _is_compiling() -> bool:
    """Return whether code is running under Torch compilation across versions."""
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())

    return False


def normalize_matrix(M: torch.Tensor, H: int, W: int) -> torch.Tensor:  # noqa: N803
    """Apply normalization sandwich ``N @ M @ N_inv`` for affine_grid input.

    Converts a pixel-space matrix to normalized ``[-1, 1]`` space
    with ``align_corners=True``. The matrix ``M`` is the composed forward
    transform (src->dst). ``affine_grid`` interprets the normalized theta
    as the mapping that generates sampling coordinates for ``grid_sample``.

    The normalization matrix is::

        N = [[2/(W-1), 0, -1], [0, 2/(H-1), -1], [0, 0, 1]]

    Args:
        M: ``(B, 3, 3)`` pixel-space forward matrix.
        H: Image height in pixels. Must be >= 2.
        W: Image width in pixels. Must be >= 2.

    Returns:
        ``(B, 3, 3)`` normalized matrix suitable for ``affine_grid``.

    Raises:
        ValueError: If ``W == 1`` or ``H == 1`` (division by zero).

    Example:
        >>> import torch
        >>> M_norm = normalize_matrix(torch.eye(3).unsqueeze(0), H=64, W=64)
        >>> M_norm.shape
        torch.Size([1, 3, 3])

    """
    if W == 1:
        msg = "W must be >= 2 for normalization (W=1 causes division by zero)"
        raise ValueError(msg)
    if H == 1:
        msg = "H must be >= 2 for normalization (H=1 causes division by zero)"
        raise ValueError(msg)

    B = M.shape[0]  # noqa: N806
    device = M.device
    dtype = M.dtype

    # N: pixel -> normalized [-1, 1]
    N = torch.zeros(B, 3, 3, device=device, dtype=dtype)  # noqa: N806
    N[:, 0, 0] = 2.0 / (W - 1)
    N[:, 0, 2] = -1.0
    N[:, 1, 1] = 2.0 / (H - 1)
    N[:, 1, 2] = -1.0
    N[:, 2, 2] = 1.0

    # N_inv: normalized -> pixel
    N_inv = torch.zeros(B, 3, 3, device=device, dtype=dtype)  # noqa: N806
    N_inv[:, 0, 0] = (W - 1) / 2.0
    N_inv[:, 0, 2] = (W - 1) / 2.0
    N_inv[:, 1, 1] = (H - 1) / 2.0
    N_inv[:, 1, 2] = (H - 1) / 2.0
    N_inv[:, 2, 2] = 1.0

    # Sandwich: N @ M @ N_inv
    return matmul3x3(matmul3x3(N, M), N_inv)
