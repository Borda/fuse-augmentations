"""Matrix primitives for 2D affine transform composition.

All functions operate on batched ``(B, 3, 3)`` homogeneous matrices in
pixel coordinates with ``align_corners=True`` convention. Center of image
is ``cx = (W-1)/2``, ``cy = (H-1)/2``.

Example:
    >>> import torch
    >>> from fuse_augmentations.affine.matrix import rotation_matrix, matmul3x3, inv3x3
    >>> matrix = rotation_matrix(torch.zeros(2), height=64, width=64)
    >>> matrix.shape
    torch.Size([2, 3, 3])

"""

from __future__ import annotations

import torch

_SINGULARITY_EPS_SCALE = 1.0


def _singularity_threshold(dtype: torch.dtype) -> float:
    """Return the shared near-singular determinant threshold for ``dtype``."""
    return float(torch.finfo(dtype).eps) * _SINGULARITY_EPS_SCALE


def rotation_matrix(angle_rad: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space forward rotation matrix around image center.

    Args:
        angle_rad: Rotation angle in radians, shape ``(batch_size,)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(batch_size, 3, 3)`` forward rotation matrix in pixel coordinates.

    Example:
        >>> import torch
        >>> matrix = rotation_matrix(torch.zeros(2), height=64, width=64)
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0).expand(2, -1, -1))
        True

    """
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    batch_size = angle_rad.shape[0]
    zeros = torch.zeros(batch_size, device=angle_rad.device, dtype=angle_rad.dtype)
    ones = torch.ones(batch_size, device=angle_rad.device, dtype=angle_rad.dtype)
    row0 = torch.stack([cos_a, -sin_a, center_x * (1 - cos_a) + center_y * sin_a], dim=-1)
    row1 = torch.stack([sin_a, cos_a, center_y * (1 - cos_a) - center_x * sin_a], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def scale_matrix(scale_x: torch.Tensor, scale_y: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space forward scale matrix around image center.

    Args:
        scale_x: X scale factor, shape ``(batch_size,)``. 1.0 = identity.
        scale_y: Y scale factor, shape ``(batch_size,)``. 1.0 = identity.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(batch_size, 3, 3)`` forward scale matrix.

    Example:
        >>> import torch
        >>> matrix = scale_matrix(torch.ones(1), torch.ones(1), height=64, width=64)
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0))
        True

    """
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    batch_size = scale_x.shape[0]
    zeros = torch.zeros(batch_size, device=scale_x.device, dtype=scale_x.dtype)
    ones = torch.ones(batch_size, device=scale_x.device, dtype=scale_x.dtype)
    row0 = torch.stack([scale_x, zeros, center_x * (1 - scale_x)], dim=-1)
    row1 = torch.stack([zeros, scale_y, center_y * (1 - scale_y)], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def shear_x_matrix(shear_x_tan: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space forward x-shear matrix around image center.

    ``shear_x_tan`` is ``tan(shear_x_rad)`` - already converted from degrees
    by the adapter.

    Args:
        shear_x_tan: Tangent of x-shear angle, shape ``(batch_size,)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(batch_size, 3, 3)`` forward x-shear matrix.

    Example:
        >>> import torch
        >>> matrix = shear_x_matrix(torch.zeros(1), height=64, width=64)
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0))
        True

    """
    center_y = (height - 1) / 2.0
    batch_size = shear_x_tan.shape[0]
    zeros = torch.zeros(batch_size, device=shear_x_tan.device, dtype=shear_x_tan.dtype)
    ones = torch.ones(batch_size, device=shear_x_tan.device, dtype=shear_x_tan.dtype)
    row0 = torch.stack([ones, shear_x_tan, -center_y * shear_x_tan], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def shear_y_matrix(shear_y_tan: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space forward y-shear matrix around image center.

    ``shear_y_tan`` is ``tan(shear_y_rad)`` - already converted from degrees
    by the adapter.

    Args:
        shear_y_tan: Tangent of y-shear angle, shape ``(batch_size,)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(batch_size, 3, 3)`` forward y-shear matrix.

    Example:
        >>> import torch
        >>> matrix = shear_y_matrix(torch.zeros(1), height=64, width=64)
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0))
        True

    """
    center_x = (width - 1) / 2.0
    batch_size = shear_y_tan.shape[0]
    zeros = torch.zeros(batch_size, device=shear_y_tan.device, dtype=shear_y_tan.dtype)
    ones = torch.ones(batch_size, device=shear_y_tan.device, dtype=shear_y_tan.dtype)
    row0 = torch.stack([ones, zeros, zeros], dim=-1)
    row1 = torch.stack([shear_y_tan, ones, -center_x * shear_y_tan], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def translate_matrix(translation_x: torch.Tensor, translation_y: torch.Tensor) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space translation matrix.

    Args:
        translation_x: X translation in pixels, shape ``(batch_size,)``.
        translation_y: Y translation in pixels, shape ``(batch_size,)``.

    Returns:
        ``(batch_size, 3, 3)`` translation matrix.

    Example:
        >>> import torch
        >>> matrix = translate_matrix(torch.zeros(1), torch.zeros(1))
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0))
        True

    """
    batch_size = translation_x.shape[0]
    zeros = torch.zeros(batch_size, device=translation_x.device, dtype=translation_x.dtype)
    ones = torch.ones(batch_size, device=translation_x.device, dtype=translation_x.dtype)
    row0 = torch.stack([ones, zeros, translation_x], dim=-1)
    row1 = torch.stack([zeros, ones, translation_y], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def hflip_matrix(width: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space horizontal flip matrix.

    Args:
        width: Image width in pixels.
        batch_size: Batch size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(batch_size, 3, 3)`` horizontal flip matrix.

    Example:
        >>> import torch
        >>> matrix = hflip_matrix(width=4, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        >>> matrix[0, 0, 0].item()
        -1.0
        >>> matrix[0, 0, 2].item()
        3.0

    """
    matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = -1.0
    matrix[:, 0, 2] = float(width - 1)
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 1.0
    return matrix


def vflip_matrix(height: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build (batch_size, 3, 3) pixel-space vertical flip matrix.

    Args:
        height: Image height in pixels.
        batch_size: Batch size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(batch_size, 3, 3)`` vertical flip matrix.

    Example:
        >>> import torch
        >>> matrix = vflip_matrix(height=4, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        >>> matrix[0, 1, 1].item()
        -1.0
        >>> matrix[0, 1, 2].item()
        3.0

    """
    matrix = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = -1.0
    matrix[:, 1, 2] = float(height - 1)
    matrix[:, 2, 2] = 1.0
    return matrix


# Canonical D4 (dihedral group of the square) forward pixel matrices, keyed by op
# name. Each returns the (3, 3) forward pixel matrix that maps an INPUT pixel
# coordinate to its OUTPUT position in the [0, W-1] x [0, H-1], align_corners=True
# convention. Feeding ``inv3x3`` of one of these to ``affine_grid`` reproduces the
# corresponding ``torch.flip`` / ``torch.rot90`` result bit-for-bit (verified against
# grid_sample). ``rot90``/``rot270``/``transpose``/``anti_transpose`` swap the two
# spatial axes, so they are shape-preserving only when ``height == width``.
_D4_OPS: tuple[str, ...] = (
    "identity",
    "hflip",
    "vflip",
    "rot180",
    "rot90",
    "rot270",
    "transpose",
    "anti_transpose",
)
# Ops whose linear part swaps the x and y axes (valid only on square images).
_D4_AXIS_SWAP: frozenset[str] = frozenset({"rot90", "rot270", "transpose", "anti_transpose"})
# Scale-free absolute tolerance on the pre-round residual (pixel units). Composing an
# exactly-90-degree kornia rotation with a flip leaves a residual that scales with the
# translation column (~4e-8 at 16 px up to ~9e-5 at 2048 px), whereas a sub-degree-off
# rotation (89.99 degrees) leaves >=1.3e-3 at every size — the two stay ~2000x apart.
# A FIXED 1e-4 sits between them at all realistic sizes: it accepts the true quarter-turn
# (<=9e-5 through 2048 px) and rejects the near-quarter-turn. Deliberately NOT scaled by
# the matrix magnitude — a magnitude-scaled tolerance widens the angular window with size
# and would silently snap near-90-degree rotations to an exact 90.
_D4_RESIDUAL_EPS: float = 1e-4


def _d4_forward_matrix(name: str, height: int, width: int) -> torch.Tensor:
    """Build the canonical ``(3, 3)`` float64 forward pixel matrix for a D4 op.

    Args:
        name: One of :data:`_D4_OPS`.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward pixel matrix.

    Raises:
        ValueError: If ``name`` is not a recognised D4 op.

    """
    w1 = float(width - 1)
    h1 = float(height - 1)
    if name == "identity":
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elif name == "hflip":
        rows = [[-1.0, 0.0, w1], [0.0, 1.0, 0.0]]
    elif name == "vflip":
        rows = [[1.0, 0.0, 0.0], [0.0, -1.0, h1]]
    elif name == "rot180":
        rows = [[-1.0, 0.0, w1], [0.0, -1.0, h1]]
    elif name == "rot90":  # torch.rot90(k=1, dims=[2, 3])
        rows = [[0.0, 1.0, 0.0], [-1.0, 0.0, w1]]
    elif name == "rot270":  # torch.rot90(k=3, dims=[2, 3])
        rows = [[0.0, -1.0, h1], [1.0, 0.0, 0.0]]
    elif name == "transpose":  # swap x and y
        rows = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    elif name == "anti_transpose":
        rows = [[0.0, -1.0, h1], [-1.0, 0.0, w1]]
    else:
        msg = f"Unknown D4 op name: {name!r}"
        raise ValueError(msg)
    matrix = torch.zeros(3, 3, dtype=torch.float64)
    matrix[:2, :] = torch.tensor(rows, dtype=torch.float64)
    matrix[2, 2] = 1.0
    return matrix


def classify_d4_batch(matrix: torch.Tensor, height: int, width: int) -> str | None:
    """Classify a ``(B, 3, 3)`` forward matrix as a shared D4 op via exact integers.

    Rounds the matrix to the nearest integer and *verifies* every entry is within a
    tight, scale-free absolute epsilon (``_D4_RESIDUAL_EPS``) of that integer, then
    checks equality against the eight canonical D4 forward matrices for the given
    ``(height, width)``. Returns the op name only when EVERY batch element is the same
    D4 element; otherwise ``None`` so the caller stays on the interpolating grid path.

    The epsilon is a fixed ``1e-4`` in pixel units (``_D4_RESIDUAL_EPS``), NOT a fraction
    of the matrix magnitude. A true quarter-turn (composed with flips) deviates from
    integer only by the ``sin(pi/2)`` float error propagated through the translation
    column (<=~9e-5 up to 2048 px), which passes; a near-quarter-turn such as an
    89.99-degree rotation leaves a residual of >=1.3e-3 at every size, which is rejected —
    so a sub-degree rotation is never silently snapped to an exact 90 degrees. Because the
    epsilon is scale-free, the angular acceptance window shrinks as the image grows,
    rather than widening (which a magnitude-scaled tolerance would do). A matrix whose net
    translation differs from the border-preserving canonical form is likewise rejected.

    Axis-swapping ops (``rot90``/``rot270``/``transpose``/``anti_transpose``) are
    reported only for square images (``height == width``); on non-square images they
    would change output dimensions and cannot replace a same-size warp.

    Args:
        matrix: ``(B, 3, 3)`` forward pixel matrix (input -> output), any dtype.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        A member of :data:`_D4_OPS` when the whole batch shares one D4 element,
        else ``None``.

    Example:
        >>> import torch
        >>> from fuse_augmentations.affine.matrix import hflip_matrix, classify_d4_batch
        >>> m = hflip_matrix(width=8, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> classify_d4_batch(m, height=8, width=8)
        'hflip'

    """
    # Work in float32 on the source device: MPS has no float64, and D4 entries are
    # small integers (linear part in {0, +/-1}, translation up to max(H, W)-1) so
    # float32 resolves them exactly at realistic image sizes.
    mat = matrix.detach().to(dtype=torch.float32)
    rounded = torch.round(mat)
    # Scale-free absolute residual gate (see _D4_RESIDUAL_EPS): accepts a true
    # quarter-turn, rejects a sub-degree-off rotation, at every realistic image size.
    if not bool((mat - rounded).abs().max().item() <= _D4_RESIDUAL_EPS):
        return None
    matched: str | None = None
    for name in _D4_OPS:
        if name in _D4_AXIS_SWAP and height != width:
            continue
        canonical = _d4_forward_matrix(name, height, width).to(dtype=torch.float32, device=mat.device)
        if bool(torch.equal(rounded, canonical.unsqueeze(0).expand_as(rounded))):
            matched = name
            break
    return matched


def apply_d4_image(image: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a D4 op to an image (or image-like) tensor via ``flip`` / ``rot90``.

    Uses only lossless index operations (``tensor.flip``, ``torch.rot90``,
    ``transpose``) so no interpolation error is introduced. The op is applied to the
    trailing two dims ``(..., H, W)``, so masks (single-channel) and multi-channel
    images share the same code path.

    Args:
        image: ``(..., H, W)`` tensor.
        name: One of :data:`_D4_OPS`.

    Returns:
        The transformed tensor.

    Raises:
        ValueError: If ``name`` is not a recognised D4 op.

    Example:
        >>> import torch
        >>> from fuse_augmentations.affine.matrix import apply_d4_image
        >>> apply_d4_image(torch.zeros(1, 1, 4, 4), "rot90").shape
        torch.Size([1, 1, 4, 4])

    """
    if name == "identity":
        return image
    if name == "hflip":
        return image.flip(dims=[-1])
    if name == "vflip":
        return image.flip(dims=[-2])
    if name == "rot180":
        return torch.rot90(image, k=2, dims=[-2, -1])
    if name == "rot90":
        return torch.rot90(image, k=1, dims=[-2, -1])
    if name == "rot270":
        return torch.rot90(image, k=3, dims=[-2, -1])
    if name == "transpose":
        return image.transpose(-2, -1)
    if name == "anti_transpose":
        return torch.rot90(image, k=1, dims=[-2, -1]).flip(dims=[-1])
    msg = f"Unknown D4 op name: {name!r}"
    raise ValueError(msg)


def matmul3x3(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
    """Batch 3x3 matrix multiply.

    Delegates to ``torch.bmm``, which is both faster on eager CPU and fully
    supported by ``torch.compile``'s inductor backend.

    Args:
        matrix_a: ``(B, 3, 3)`` left matrix.
        matrix_b: ``(B, 3, 3)`` right matrix.

    Returns:
        ``(B, 3, 3)`` product ``A @ B``.

    Example:
        >>> import torch
        >>> mtx_identity = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        >>> mtx_result = matmul3x3(mtx_identity, mtx_identity)
        >>> torch.allclose(mtx_result, mtx_identity)
        True

    """
    return torch.bmm(matrix_a, matrix_b)


def inv3x3(matrix: torch.Tensor, *, compiling: bool | None = None) -> torch.Tensor:
    """Batch 3x3 matrix inverse.

    Uses ``torch.linalg.inv`` on the eager CPU path (single LAPACK call, ~6x
    faster than Cramer's rule) and element-wise Cramer's rule both under
    ``torch.compile`` (kernel fusion) and on non-CPU eager tensors — the CPU
    raising guard reads ``(det.abs() < eps).any()``, which forces a
    device-to-host sync per call and would drain the CUDA/MPS stream on every
    warp; the Cramer branch clamps near-singular determinants instead, keeping
    the whole call asynchronous. Includes an eager CPU check that raises
    ``ValueError`` for near-singular matrices.

    Args:
        matrix: ``(batch_size, 3, 3)`` batch of matrices.
        compiling: Which branch to take. ``None`` (default) detects the current
            ``torch.compile`` tracing state via :func:`_is_compiling` — correct for
            ordinary eager callers, but ambient detection is fragile when this
            function is entered from a resumed dynamo frame (observed on torch
            2.2 as a spurious graph break). Callers that KNOW they are inside a
            compiled region (the module-level warp cores compiled by
            :func:`fuse_augmentations.affine.segment._compiled_warp_fn`) pass
            ``True`` explicitly so branch selection is a trace-time constant
            rather than a runtime probe.

    Returns:
        ``(batch_size, 3, 3)`` batch of inverse matrices.

    Raises:
        ValueError: If any matrix in the batch has a near-zero determinant
            (eager CPU mode only; non-CPU and compiled paths clamp instead).
            Note this aborts the WHOLE batch when a single
            sample draws a degenerate transform — keep scale ranges bounded so
            per-axis scale stays well above ``sqrt(finfo.eps)`` (≈3.5e-4 for
            float32) to avoid rare training crashes.

    Example:
        >>> import torch
        >>> mtx_identity = torch.eye(3).unsqueeze(0)
        >>> torch.allclose(inv3x3(mtx_identity), mtx_identity)
        True

    """
    # Single singularity threshold shared by ALL branches so eager and compiled
    # execution agree on which matrices count as near-singular. The eager CPU path
    # raises; the compile and non-CPU eager paths clamp at the SAME threshold
    # (data-dependent raises are not compile-safe, and on CUDA/MPS the raising
    # guard's `.any()` would host-sync every warp), so outputs only differ below
    # eps, where the CPU path would have raised anyway.
    eps = _singularity_threshold(matrix.dtype)

    is_compiling = _is_compiling() if compiling is None else compiling
    if not is_compiling and matrix.device.type == "cpu":
        # Fast path: single LAPACK call, ~6x faster than Cramer's rule
        det = torch.linalg.det(matrix)
        if (det.abs() < eps).any():
            msg = (
                f"Near-singular matrix detected (min |det|={det.abs().min().item():.2e}). "
                "Check for extreme scale values or degenerate transforms."
            )
            raise ValueError(msg)
        return torch.linalg.inv(matrix)  # type: ignore[no-any-return]

    # Compile path and non-CPU eager path: element-wise Cramer's rule — kernel
    # fusion under compile, sync-free clamping on accelerators
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
    # Compile-safe singularity guard: clamp |det| to eps, preserving a non-zero sign.
    safe_det = torch.where(det < 0, -det.new_tensor(eps), det.new_tensor(eps))
    det = torch.where(det.abs() < eps, safe_det, det)
    inv_det = 1.0 / det
    adj = torch.stack(
        [
            m11 * m22 - m12 * m21,
            m02 * m21 - m01 * m22,
            m01 * m12 - m02 * m11,
            m12 * m20 - m10 * m22,
            m00 * m22 - m02 * m20,
            m02 * m10 - m00 * m12,
            m10 * m21 - m11 * m20,
            m01 * m20 - m00 * m21,
            m00 * m11 - m01 * m10,
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


def normalize_matrix(matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Apply normalization sandwich ``N @ matrix @ N_inv`` for affine_grid input.

    Converts a pixel-space matrix to normalized ``[-1, 1]`` space with
    ``align_corners=True``. The sandwich is direction-agnostic — it re-expresses
    whatever pixel-space map is passed in normalized coordinates. Callers in this
    library pass the INVERSE (dst->src / sampling) matrix, because ``affine_grid``
    interprets the normalized theta as the map that generates sampling coordinates
    for ``grid_sample`` (output position -> input position). Passing the forward
    (src->dst) matrix here produces an inverted warp.

    The normalization matrix is::

        mtx_normalize = [[2/(width-1), 0, -1], [0, 2/(height-1), -1], [0, 0, 1]]

    Args:
        matrix: ``(batch_size, 3, 3)`` pixel-space sampling (dst->src) matrix, i.e.
            the inverse of the composed forward transform.
        height: Image height in pixels. Must be >= 2.
        width: Image width in pixels. Must be >= 2.

    Returns:
        ``(batch_size, 3, 3)`` normalized matrix suitable for ``affine_grid``.

    Raises:
        ValueError: If ``width == 1`` or ``height == 1`` (division by zero).

    Example:
        >>> import torch
        >>> mtx_norm = normalize_matrix(torch.eye(3).unsqueeze(0), height=64, width=64)
        >>> mtx_norm.shape
        torch.Size([1, 3, 3])

    """
    if width == 1:
        msg = "width must be >= 2 for normalization (width=1 causes division by zero)"
        raise ValueError(msg)
    if height == 1:
        msg = "height must be >= 2 for normalization (height=1 causes division by zero)"
        raise ValueError(msg)

    half_width: float = (width - 1) / 2.0  # N_inv[0,0] = N_inv[0,2]
    half_height: float = (height - 1) / 2.0  # N_inv[1,1] = N_inv[1,2]
    scale_x_norm: float = 2.0 / (width - 1)  # N[0,0]
    scale_y_norm: float = 2.0 / (height - 1)  # N[1,1]

    # Closed-form analytic expansion of N @ matrix @ N_inv.
    # Avoids allocating two (B,3,3) intermediate matrices and two bmm calls.
    #
    # Step 1 — tmp = N @ matrix, where N=[[scale_x_norm,0,-1],[0,scale_y_norm,-1],[0,0,1]]:
    #   tmp[b, 0, j] = scale_x_norm*matrix[b,0,j] - matrix[b,2,j]
    #   tmp[b, 1, j] = scale_y_norm*matrix[b,1,j] - matrix[b,2,j]
    #   tmp[b, 2, j] = matrix[b,2,j]
    #
    # Step 2 — result = tmp @ N_inv, where N_inv=[[half_width,0,half_width],[0,half_height,half_height],[0,0,1]]:
    #   result[b,i,0] = tmp[b,i,0]*half_width
    #   result[b,i,1] = tmp[b,i,1]*half_height
    #   result[b,i,2] = tmp[b,i,0]*half_width + tmp[b,i,1]*half_height + tmp[b,i,2]
    result = torch.empty_like(matrix)

    # Row 0: tmp[0,j] = scale_x_norm*matrix[0,j] - matrix[2,j]
    t00 = scale_x_norm * matrix[:, 0, 0] - matrix[:, 2, 0]
    t01 = scale_x_norm * matrix[:, 0, 1] - matrix[:, 2, 1]
    t02 = scale_x_norm * matrix[:, 0, 2] - matrix[:, 2, 2]
    result[:, 0, 0] = t00 * half_width
    result[:, 0, 1] = t01 * half_height
    result[:, 0, 2] = t00 * half_width + t01 * half_height + t02

    # Row 1: tmp[1,j] = scale_y_norm*matrix[1,j] - matrix[2,j]
    t10 = scale_y_norm * matrix[:, 1, 0] - matrix[:, 2, 0]
    t11 = scale_y_norm * matrix[:, 1, 1] - matrix[:, 2, 1]
    t12 = scale_y_norm * matrix[:, 1, 2] - matrix[:, 2, 2]
    result[:, 1, 0] = t10 * half_width
    result[:, 1, 1] = t11 * half_height
    result[:, 1, 2] = t10 * half_width + t11 * half_height + t12

    # Row 2: tmp[2,j] = matrix[2,j] (N[2,:] = [0,0,1])
    result[:, 2, 0] = matrix[:, 2, 0] * half_width
    result[:, 2, 1] = matrix[:, 2, 1] * half_height
    result[:, 2, 2] = matrix[:, 2, 0] * half_width + matrix[:, 2, 1] * half_height + matrix[:, 2, 2]

    return result


def crop_resize_matrix(
    top: torch.Tensor,
    left: torch.Tensor,
    crop_h: torch.Tensor,
    crop_w: torch.Tensor,
    target_h: torch.Tensor,
    target_w: torch.Tensor,
) -> torch.Tensor:
    """Build ``(B, 3, 3)`` pixel-space forward matrix for a crop-then-resize operation.

    Maps input pixel ``(x_in, y_in)`` to output pixel ``(x_out, y_out)`` via::

        x_out = (x_in - left) * (target_w - 1) / (crop_w - 1)
        y_out = (y_in - top)  * (target_h - 1) / (crop_h - 1)

    All inputs are per-sample ``(batch_size,)`` tensors.  ``target_h``/``target_w`` are
    typically constant across the batch (fixed output size) but are accepted as
    tensors for API uniformity.

    Args:
        top: Crop top coordinate in pixels, shape ``(batch_size,)``.
        left: Crop left coordinate in pixels, shape ``(batch_size,)``.
        crop_h: Crop height in pixels, shape ``(batch_size,)``.
        crop_w: Crop width in pixels, shape ``(batch_size,)``.
        target_h: Output height in pixels, shape ``(batch_size,)``.
        target_w: Output width in pixels, shape ``(batch_size,)``.

    Returns:
        ``(batch_size, 3, 3)`` forward affine matrix in pixel coordinates.

    Raises:
        ValueError: If any ``crop_h``, ``crop_w``, ``target_h``, or
            ``target_w`` element is ``<= 1`` (align-corners endpoint mapping
            would become singular).

    Example:
        >>> import torch
        >>> top = torch.zeros(1)
        >>> matrix = crop_resize_matrix(top, top, torch.full((1,), 32.0), torch.full((1,), 32.0),
        ...                        torch.full((1,), 32.0), torch.full((1,), 32.0))
        >>> torch.allclose(matrix, torch.eye(3).unsqueeze(0))
        True

    """
    if bool((crop_h <= 1).any() or (crop_w <= 1).any() or (target_h <= 1).any() or (target_w <= 1).any()):
        msg = "crop_resize_matrix requires crop and target sizes > 1 for align_corners endpoint mapping"
        raise ValueError(msg)

    scale_x = (target_w - 1.0) / (crop_w - 1.0)
    scale_y = (target_h - 1.0) / (crop_h - 1.0)
    translation_x = -left * scale_x
    translation_y = -top * scale_y
    zeros = torch.zeros_like(top)
    ones = torch.ones_like(top)
    row0 = torch.stack([scale_x, zeros, translation_x], dim=-1)
    row1 = torch.stack([zeros, scale_y, translation_y], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def estimate_scale(matrix: torch.Tensor) -> tuple[float, float]:
    """Estimate the per-axis output-vs-input scale of a forward pixel matrix.

    The two singular values of the ``2x2`` linear part of a forward affine
    matrix are the axis-aligned scale factors the warp applies (independent of
    any rotation or shear): a value ``< 1`` means that direction is *shrunk*
    (downscaled, aliasing risk), ``> 1`` means it is magnified. This is used at
    warp time to decide whether antialiasing is needed and, if so, how strong.
    The worst-axis (minimum) singular value governs the aliasing decision, so
    both are returned with the smaller one first.

    When the batch composes to more than one matrix, the smallest scale across
    the batch is used per axis, so a batch is antialiased whenever *any* sample
    downscales past the threshold (uniform output, no per-sample branching).

    Args:
        matrix: ``(batch_size, 3, 3)`` forward pixel-space matrix (input pixel ->
            output pixel). Any floating dtype and device.

    Returns:
        A ``(scale_min, scale_max)`` tuple of Python floats: the smaller and
        larger per-axis scale factors, with ``scale_min <= scale_max``.

    Example:
        >>> import torch
        >>> half = torch.eye(3).unsqueeze(0)
        >>> half[:, 0, 0] = 0.25  # shrink x to a quarter
        >>> half[:, 1, 1] = 0.5   # shrink y to a half
        >>> lo, hi = estimate_scale(half)
        >>> round(lo, 4), round(hi, 4)
        (0.25, 0.5)

    """
    linear = matrix[:, :2, :2].to(dtype=torch.float32)
    svals = torch.linalg.svdvals(linear)  # (batch_size, 2), descending
    scale_min = float(svals[:, 1].min().item())
    scale_max = float(svals[:, 0].min().item())
    return scale_min, scale_max


def normalize_matrix_io(
    matrix: torch.Tensor,
    height_in: int,
    width_in: int,
    height_out: int,
    width_out: int,
) -> torch.Tensor:
    """Normalize a pixel-space inverse matrix for ``affine_grid`` when input and output sizes differ.

    Applies the normalization sandwich ``N_in @ matrix @ N_out_inv`` where:

    - ``mtx_normalize_in`` maps input pixel coords → input normalized ``[-1, 1]`` (uses ``height_in``, ``width_in``).
    - ``mtx_normalize_out_inv`` maps output normalized ``[-1, 1]`` → output pixel coords
      (uses ``height_out``, ``width_out``).

    Use this instead of :func:`normalize_matrix` when the segment output size differs from the input
    size (e.g. for :class:`~fuse_augmentations.affine.segment.CropResizeSegment`).

    Args:
        matrix: ``(batch_size, 3, 3)`` pixel-space *inverse* matrix (output pixel → input pixel).
        height_in: Input image height in pixels.  Must be >= 2.
        width_in: Input image width in pixels.  Must be >= 2.
        height_out: Output image height in pixels.  Must be >= 2.
        width_out: Output image width in pixels.  Must be >= 2.

    Returns:
        ``(batch_size, 3, 3)`` normalized matrix suitable for ``F.affine_grid`` with output size
        ``[batch_size, channels, height_out, width_out]``.

    Raises:
        ValueError: If any dimension is < 2.

    Example:
        >>> import torch
        >>> mtx_inv = torch.eye(3).unsqueeze(0)
        >>> mtx_norm = normalize_matrix_io(mtx_inv, height_in=64, width_in=64, height_out=32, width_out=32)
        >>> mtx_norm.shape
        torch.Size([1, 3, 3])

    """
    for name, val in (
        ("height_in", height_in),
        ("width_in", width_in),
        ("height_out", height_out),
        ("width_out", width_out),
    ):
        if val < 2:
            msg = f"{name} must be >= 2 for normalization ({name}={val} causes division by zero)"
            raise ValueError(msg)

    batch_size = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype

    # N_in: input pixel → input normalized [-1, 1]
    matrix_n = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    matrix_n[:, 0, 0] = 2.0 / (width_in - 1)
    matrix_n[:, 0, 2] = -1.0
    matrix_n[:, 1, 1] = 2.0 / (height_in - 1)
    matrix_n[:, 1, 2] = -1.0
    matrix_n[:, 2, 2] = 1.0

    # N_out_inv: output normalized → output pixel
    matrix_n_inv = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    matrix_n_inv[:, 0, 0] = (width_out - 1) / 2.0
    matrix_n_inv[:, 0, 2] = (width_out - 1) / 2.0
    matrix_n_inv[:, 1, 1] = (height_out - 1) / 2.0
    matrix_n_inv[:, 1, 2] = (height_out - 1) / 2.0
    matrix_n_inv[:, 2, 2] = 1.0

    # Sandwich: N_in @ matrix @ N_out_inv
    return matmul3x3(matmul3x3(matrix_n, matrix), matrix_n_inv)


def perspective_from_points(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Build ``(batch_size, 3, 3)`` homography from 4 point correspondences using DLT.

    Solves the 8-unknown homography system for the finite transform ``mtx_homography`` such
    that ``dst ~ mtx_homography @ src`` in homogeneous coordinates, with ``mtx_homography[..., 2, 2]``
    fixed to ``1``.

    Args:
        src: Source points, shape ``(batch_size, 4, 2)``, ``[x, y]`` order.
        dst: Destination points, shape ``(batch_size, 4, 2)``, ``[x, y]`` order.

    Returns:
        ``(batch_size, 3, 3)`` forward homography matrices normalised so that
        ``H[..., 2, 2] = 1``.

    Example:
        >>> import torch
        >>> corners = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
        >>> mtx_homography = perspective_from_points(corners, corners)
        >>> torch.allclose(mtx_homography, torch.eye(3).unsqueeze(0), atol=1e-5)
        True

    """
    batch_size, num_points, _ = src.shape  # num_points=4
    if num_points != 4:
        msg = f"perspective_from_points expects exactly 4 point pairs, got {num_points}"
        raise ValueError(msg)

    src_x, src_y = src[..., 0], src[..., 1]  # (B, 4)
    dst_x, dst_y = dst[..., 0], dst[..., 1]
    zeros = torch.zeros_like(src_x)
    ones = torch.ones_like(src_x)
    # Solve A @ h = b for h = [h11, h12, h13, h21, h22, h23, h31, h32].
    row_x = torch.stack(
        [src_x, src_y, ones, zeros, zeros, zeros, -(dst_x * src_x), -(dst_x * src_y)], dim=-1
    )  # (B,4,8)
    row_y = torch.stack(
        [zeros, zeros, zeros, src_x, src_y, ones, -(dst_y * src_x), -(dst_y * src_y)], dim=-1
    )  # (B,4,8)
    matrix_a = torch.stack([row_x, row_y], dim=2).reshape(batch_size, 2 * num_points, 8)
    matrix_b = torch.stack([dst_x, dst_y], dim=2).reshape(batch_size, 2 * num_points, 1)
    homography_params = torch.linalg.solve(matrix_a, matrix_b).squeeze(-1)

    matrix_h = torch.empty((batch_size, 3, 3), device=src.device, dtype=homography_params.dtype)
    matrix_h[..., 0, 0] = homography_params[..., 0]
    matrix_h[..., 0, 1] = homography_params[..., 1]
    matrix_h[..., 0, 2] = homography_params[..., 2]
    matrix_h[..., 1, 0] = homography_params[..., 3]
    matrix_h[..., 1, 1] = homography_params[..., 4]
    matrix_h[..., 1, 2] = homography_params[..., 5]
    matrix_h[..., 2, 0] = homography_params[..., 6]
    matrix_h[..., 2, 1] = homography_params[..., 7]
    matrix_h[..., 2, 2] = 1.0
    return matrix_h.to(dtype=src.dtype)


def perspective_grid(matrix_inv_norm: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build ``(batch_size, height, width, 2)`` sampling grid from a normalised 3x3 perspective matrix.

    Unlike ``F.affine_grid`` (which uses a ``(batch_size, 2, 3)`` theta), this handles
    the full ``3x3`` case required by projective/perspective transforms by
    applying perspective division: ``coord_x' = transformed_x/homogeneous_w``,
    ``coord_y' = transformed_y/homogeneous_w``.

    No Python branching is used — the function is ``torch.compile``-friendly.

    Args:
        matrix_inv_norm: ``(batch_size, 3, 3)`` normalised inverse matrix in ``[-1, 1]`` space.
        height: Image height.
        width: Image width.

    Returns:
        ``(batch_size, height, width, 2)`` sampling grid for ``F.grid_sample`` with
        ``align_corners=True``.

    Example:
        >>> import torch
        >>> grid = perspective_grid(torch.eye(3).unsqueeze(0), height=4, width=4)
        >>> grid.shape
        torch.Size([1, 4, 4, 2])

    """
    batch_size = matrix_inv_norm.shape[0]
    device = matrix_inv_norm.device
    dtype = matrix_inv_norm.dtype

    linspace_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)  # (W,)
    linspace_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)  # (H,)
    grid_y, grid_x = torch.meshgrid(linspace_y, linspace_x, indexing="ij")  # both (H, W)
    ones = torch.ones(height, width, device=device, dtype=dtype)
    coords = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, height * width)  # (3, H*W)
    # Batched matrix multiply: (B, 3, 3) @ (3, H*W) -> (B, 3, H*W)
    coords_b = coords.unsqueeze(0).expand(batch_size, -1, -1)
    transformed = torch.bmm(matrix_inv_norm, coords_b)  # (B, 3, H*W)
    homogeneous_w = transformed[:, 2:3, :]  # (B, 1, H*W)
    # Clamp homogeneous_w away from zero (preserving sign) to avoid Inf/NaN in perspective division.
    eps = torch.finfo(dtype).eps
    tw_clamped = torch.where(
        homogeneous_w >= 0, torch.clamp(homogeneous_w, min=eps), torch.clamp(homogeneous_w, max=-eps)
    )
    normalized_coords = transformed[:, :2, :] / tw_clamped  # (B, 2, H*W)
    return normalized_coords.permute(0, 2, 1).reshape(batch_size, height, width, 2)
