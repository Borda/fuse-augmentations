"""Fused affine segment — vectorised matrix composition and single grid_sample pass.

``FusedAffineSegment`` accumulates per-sample affine matrices for an entire chain
of geometric transforms, inverts the composed matrix once, and executes a single
``grid_sample`` call. No intermediate image warps are performed.

Example:
    >>> import torch
    >>> import kornia.augmentation as K
    >>> from fuse_augmentations.affine.segment import FusedAffineSegment
    >>> from fuse_augmentations.adapters.kornia import KorniaAdapter
    >>> t = K.RandomHorizontalFlip(p=1.0)
    >>> seg = FusedAffineSegment([t], KorniaAdapter())
    >>> out = seg(torch.zeros(1, 3, 8, 8))
    >>> out.shape
    torch.Size([1, 3, 8, 8])

"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import Tensor, nn

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.affine.matrix import (
    _singularity_threshold,
    apply_d4_image,
    classify_d4_batch,
    estimate_scale,
    inv3x3,
    matmul3x3,
    normalize_matrix,
    normalize_matrix_io,
    perspective_grid,
)
from fuse_augmentations.types import (
    ClipPolicyStr,
    ExecutionStr,
    InterpolationStr,
    MaskInterpolationStr,
    PaddingModeStr,
    RandomnessPolicy,
    TransformAdapter,
    TransformCategory,
)

# cv2 optional import — used by both FusedAffineSegment (B=1 CPU fast path)
# and AlbuFusedAffineSegment (Albumentations cv2 backend).
try:
    import cv2 as _cv2

    _CV2_INTERP: dict[str, int] = {
        "bilinear": _cv2.INTER_LINEAR,
        "nearest": _cv2.INTER_NEAREST,
        "bicubic": _cv2.INTER_CUBIC,
    }
    _CV2_BORDER: dict[str, int] = {
        "zeros": _cv2.BORDER_CONSTANT,
        "border": _cv2.BORDER_REPLICATE,
        # torch grid_sample(padding_mode="reflection", align_corners=True) reflects
        # about the edge sample without duplicating it — cv2.BORDER_REFLECT_101,
        # not cv2.BORDER_REFLECT (which duplicates the edge pixel).
        "reflection": _cv2.BORDER_REFLECT_101,
    }
    _CV2_WARP_INVERSE_MAP: int = _cv2.WARP_INVERSE_MAP
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_INTERP = {}
    _CV2_BORDER = {}
    _CV2_WARP_INVERSE_MAP = 16  # cv2.WARP_INVERSE_MAP = 16

__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += [".", "ExactAffineSegment", "_FusedGeoCropSegment"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["AlbuFusedAffineSegment"]

# Dtype used to compose and invert the (B, 3, 3) affine/projective chain on the
# torch path. float64 keeps matrix accumulation independent of chain length; the
# result is cast back to the image dtype at the grid_sample boundary. The cv2 and
# NumPy paths already accumulate in float64.
_COMPOSE_DTYPE: torch.dtype = torch.float64

_CURRENT_CALL_MATRIX: ContextVar[Tensor | None] = ContextVar("fuse_current_call_matrix", default=None)


def _clear_current_call_matrix() -> None:
    """Clear the per-context matrix produced by the next segment call."""
    _CURRENT_CALL_MATRIX.set(None)


def _current_call_matrix() -> Tensor | None:
    """Return the matrix produced by the most recent segment in this context."""
    return _CURRENT_CALL_MATRIX.get()


def _set_current_call_matrix(matrix: Tensor) -> None:
    """Publish a segment's local matrix without using shared pipeline state."""
    _CURRENT_CALL_MATRIX.set(matrix)


def _matrix_compose_dtype(image_dtype: torch.dtype, device: torch.device, num_transforms: int) -> torch.dtype:
    """Pick the dtype used to accumulate and invert a matrix chain.

    A single transform has no chain to accumulate, so it stays in the image dtype
    (keeps single-op output bit-for-bit compatible with the native warp). Longer
    chains use float64 to remove chain-length-dependent drift, except on MPS, which
    has no float64 support — there the accumulation falls back to the image dtype
    (typically float32), trading the extra precision for device compatibility.

    Args:
        image_dtype: Dtype of the image tensor being warped.
        device: Device the image lives on.
        num_transforms: Number of transforms fused in the segment.

    Returns:
        The dtype to use for matrix composition and inversion.

    """
    if num_transforms <= 1 or device.type == "mps":
        return image_dtype
    return _COMPOSE_DTYPE


def _scatter_active_matrices(
    mtx: Tensor,
    active: Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Place per-sample matrices into a ``(batch_size, 3, 3)`` identity-filled batch.

    Kornia stores sampled parameters for the ACTIVE subset only (the samples whose
    per-sample probability draw passed), so a reconstructed matrix can have shape
    ``(n_active, 3, 3)`` rather than ``(batch_size, 3, 3)``. This scatters those
    active matrices back into their batch positions, leaving identity on the samples
    the probability mask skipped.

    Args:
        mtx: Reconstructed matrices, shape ``(batch_size, 3, 3)``, ``(n_active, 3, 3)``, or ``(1, 3, 3)``.
        active: Boolean ``(batch_size,)`` mask of applied samples, or ``None`` when the transform always applies.
        batch_size: Target batch size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A ``(batch_size, 3, 3)`` matrix batch.

    """
    if active is None:
        if mtx.shape[0] == 1 and batch_size > 1:
            return mtx.expand(batch_size, -1, -1)
        return mtx
    full = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    if mtx.shape[0] == batch_size:
        # Full-batch matrices: keep active rows, identity on the rest.
        return torch.where(active[:, None, None], mtx, full)
    if mtx.shape[0] == 1:
        return torch.where(active[:, None, None], mtx.expand(batch_size, -1, -1), full)
    n_active = int(active.sum().item())
    if mtx.shape[0] == n_active:
        full[active] = mtx
        return full
    msg = (
        f"Cannot align a reconstructed matrix batch of shape {tuple(mtx.shape)} with a batch of "
        f"{batch_size} ({n_active} active). Expected the batch size, the active count, or 1."
    )
    raise RuntimeError(msg)


def _shares_randomness_across_batch(
    adapter: TransformAdapter,
    transform: object,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
) -> bool:
    """Return whether a transform should draw one random decision for the batch."""
    if randomness is RandomnessPolicy.PER_SAMPLE:
        return False
    same_on_batch = getattr(adapter, "same_on_batch", None)
    if callable(same_on_batch):
        return bool(same_on_batch(transform))
    return bool(getattr(transform, "same_on_batch", False))


def _sample_transform_params(
    adapter: TransformAdapter,
    transform: object,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
) -> dict[str, Tensor]:
    """Sample params, preferring adapter-provided per-sample sampling when requested."""
    if randomness is RandomnessPolicy.PER_SAMPLE:
        sample_params_per_sample = getattr(adapter, "sample_params_per_sample", None)
        if callable(sample_params_per_sample):
            return cast(dict[str, Tensor], sample_params_per_sample(transform, input_shape, device))
    return adapter.sample_params(transform, input_shape, device)


def _transform_prob(transform: object, default: float = 1.0) -> float:
    """Return a transform's application probability, preferring ``prob`` then ``p``."""
    prob = getattr(transform, "prob", None)
    if prob is not None:
        return float(prob)
    return float(getattr(transform, "p", default))


def _validate_execution(execution: str) -> ExecutionStr:
    """Validate and return an Albumentations execution-strategy value.

    Args:
        execution: The requested strategy; must be ``"cv2"`` or ``"torch"``.

    Returns:
        The validated strategy string.

    Raises:
        ValueError: If ``execution`` is neither ``"cv2"`` nor ``"torch"``.

    Examples:
        >>> _validate_execution("cv2")
        'cv2'
        >>> _validate_execution("torch")
        'torch'

    """
    if execution not in ("cv2", "torch"):
        msg = f"execution must be 'cv2' or 'torch', got {execution!r}."
        raise ValueError(msg)
    return cast(ExecutionStr, execution)


def _grid_sample_affine_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    *,
    compiling: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Warp ``image`` by an affine matrix batch with one ``affine_grid`` + ``grid_sample`` pass.

    Inverts the composed forward matrix, normalizes it to the ``[-1, 1]`` grid
    convention, builds an affine grid, and resamples the whole batch at once. This
    is the single batched affine executor shared by the torch affine segment and
    the Albumentations torch execution strategy.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        acc: ``(batch_size, 3, 3)`` composed forward matrix. Any floating dtype;
            the inversion runs in this dtype, so callers pass float64 on CPU/CUDA
            and float32 on MPS (which has no float64).
        interpolation: ``grid_sample`` interpolation mode.
        padding_mode: ``grid_sample`` padding mode.
        compiling: Forwarded to :func:`~fuse_augmentations.affine.matrix.inv3x3`
            to select the compile-safe branch explicitly; ``None`` falls back to
            ambient ``torch.compile`` detection for ordinary eager calls.

    Returns:
        A ``(warped_image, grid)`` tuple; the grid is reused to warp mask aux targets.

    """
    batch_size, num_channels, height, width = image.shape
    dtype = image.dtype
    mtx_inv = inv3x3(acc, compiling=compiling)
    mtx_norm = normalize_matrix(mtx_inv, height, width).to(dtype=dtype)

    grid = F.affine_grid(mtx_norm[:, :2, :], [batch_size, num_channels, height, width], align_corners=True)
    warped = F.grid_sample(image, grid, mode=interpolation, padding_mode=padding_mode, align_corners=True)
    return warped, grid


def _grid_sample_perspective_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
    *,
    compiling: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Warp ``image`` by a homography batch with one ``perspective_grid`` + ``grid_sample`` pass.

    Like :func:`_grid_sample_affine_batched` but builds a perspective grid (with
    the perspective division ``F.affine_grid`` cannot express), so it handles the
    full ``3x3`` homography. Shared by the torch projective segment and the
    Albumentations projective torch execution strategy.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        acc: ``(batch_size, 3, 3)`` composed forward homography (float64 on
            CPU/CUDA, float32 on MPS).
        interpolation: ``grid_sample`` interpolation mode.
        padding_mode: ``grid_sample`` padding mode.
        compiling: Forwarded to :func:`~fuse_augmentations.affine.matrix.inv3x3`;
            see :func:`_grid_sample_affine_batched`.

    Returns:
        A ``(warped_image, grid)`` tuple.

    """
    _, _, height, width = image.shape
    dtype = image.dtype
    mtx_inv = inv3x3(acc, compiling=compiling)
    mtx_norm = normalize_matrix(mtx_inv, height, width).to(dtype=dtype)

    grid = perspective_grid(mtx_norm, height, width)
    warped = F.grid_sample(image, grid, mode=interpolation, padding_mode=padding_mode, align_corners=True)
    return warped, grid


def _compiling_grid_sample_affine_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
) -> tuple[Tensor, Tensor]:
    """``_grid_sample_affine_batched`` with the compile-safe inversion branch pinned on.

    The only entry point ``torch.compile`` wraps. Binding ``compiling=True`` here is
    a plain Python constant fixed at trace time, not an ambient runtime probe, so
    branch selection cannot depend on how a given torch/dynamo version reports its
    own tracing state from a resumed frame (observed as a spurious graph break with
    ambient detection on torch 2.2).

    """
    return _grid_sample_affine_batched(image, acc, interpolation, padding_mode, compiling=True)


def _compiling_grid_sample_perspective_batched(
    image: Tensor,
    acc: Tensor,
    interpolation: InterpolationStr,
    padding_mode: PaddingModeStr,
) -> tuple[Tensor, Tensor]:
    """``_grid_sample_perspective_batched`` with the compile-safe inversion branch pinned on.

    See :func:`_compiling_grid_sample_affine_batched`.

    """
    return _grid_sample_perspective_batched(image, acc, interpolation, padding_mode, compiling=True)


def _torch_supports_compile() -> bool:
    """Return whether the installed torch is new enough for a reliable ``torch.compile``.

    The compiled warp region is gated on torch >= 2.2 at runtime rather than
    raising the package floor: older torch keeps the eager path unchanged and the
    ``compile=True`` flag becomes a documented no-op. The version is parsed from
    ``torch.__version__`` (major/minor only; release-candidate and ``+cpu`` build
    suffixes are ignored).

    Returns:
        ``True`` when ``torch.__version__`` is ``2.2`` or newer, ``False`` otherwise.

    Example:
        >>> from fuse_augmentations.affine.segment import _torch_supports_compile
        >>> isinstance(_torch_supports_compile(), bool)
        True

    """
    parts = torch.__version__.split("+", 1)[0].split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return False
    return (major, minor) >= (2, 2)


# Signature shared by both warp cores: (image, matrix, interpolation, padding) ->
# (warped_image, sampling_grid). Used to type the compiled-warp cache/selectors.
WarpFn = Callable[[Tensor, Tensor, InterpolationStr, PaddingModeStr], tuple[Tensor, Tensor]]

# Module-level compiled warp cores, built lazily on first use so importing the
# module never triggers a compile. ``dynamic=True`` keeps a single guarded graph
# across varying (batch, height, width) instead of recompiling per shape. Keyed
# once at module scope so every segment shares the same compiled function object
# (and therefore the same inductor cache), rather than compiling per instance.
_COMPILED_WARP_CACHE: dict[str, WarpFn] = {}


def _compiled_warp_fn(kind: str) -> WarpFn:
    """Return the ``torch.compile``-wrapped warp core for ``kind`` (``"affine"`` or ``"perspective"``).

    The compiled function wraps the same ``inv3x3 -> normalize -> grid -> grid_sample``
    core used on the eager path; only the matrix-inversion / grid-generation math is
    inside the graph. Probability masking and active-subset selection stay in the
    callers, so the compiled region has no data-dependent control flow and no graph
    breaks. Compilation happens once per kind and is memoized.

    Args:
        kind: ``"affine"`` for :func:`_grid_sample_affine_batched` or
            ``"perspective"`` for :func:`_grid_sample_perspective_batched`.

    Returns:
        The compiled callable for the requested warp core.

    """
    cached = _COMPILED_WARP_CACHE.get(kind)
    if cached is not None:
        return cached
    base = _compiling_grid_sample_affine_batched if kind == "affine" else _compiling_grid_sample_perspective_batched
    compiled = torch.compile(base, dynamic=True)
    _COMPILED_WARP_CACHE[kind] = compiled
    return compiled


# Below this per-axis scale (output/input), a plain grid_sample downscale drops
# high-frequency detail between samples and aliases; the antialias path kicks in
# only when a downscale is this aggressive. At/above it the warp is bit-identical
# to the un-antialiased path (Nyquist headroom), so the opt-in flag is a no-op.
_ANTIALIAS_SCALE_THRESHOLD: float = 0.5
# Off-diagonal magnitude under which the composed 2x2 counts as axis-aligned
# (pure scale + translation, no rotation/shear) — the case an ``F.interpolate``
# antialias tail handles exactly, matching ``torchvision.Resize(antialias=True)``.
_AXIS_ALIGNED_EPS: float = 1e-6


def _mipmap_sigma(scale: float) -> float:
    """Return the Gaussian pre-blur sigma for a downscale factor ``scale`` (< 1).

    Uses the standard mipmap pre-filter rule ``sigma = 0.5 * sqrt((1/s)^2 - 1)``:
    the blur that band-limits the input to the output Nyquist frequency before a
    ``1/s``-fold decimation, so the single downscaling warp no longer aliases.

    Args:
        scale: The per-axis output/input scale factor, expected ``0 < scale < 1``.

    Returns:
        The Gaussian standard deviation in input pixels (``0.0`` when ``scale >= 1``).

    Example:
        >>> from fuse_augmentations.affine.segment import _mipmap_sigma
        >>> round(_mipmap_sigma(0.25), 4)
        1.9365

    """
    if scale >= 1.0:
        return 0.0
    return 0.5 * math.sqrt((1.0 / scale) ** 2 - 1.0)


def _maybe_antialias_prefilter(image: Tensor, mtx: Tensor, enabled: bool) -> Tensor:
    """Gaussian-prefilter ``image`` before a downscaling warp when antialiasing is on.

    Estimates the per-axis scale of the forward matrix ``mtx``; when antialiasing
    is enabled and the worst axis downscales below
    :data:`_ANTIALIAS_SCALE_THRESHOLD`, band-limits the input with a per-axis
    Gaussian (mipmap sigma rule) so the single ``grid_sample`` no longer aliases.
    The blur runs in the image dtype via the installed kornia backend; if kornia
    is unavailable the input is returned unchanged (documented fallback, no custom
    kernel). Returns ``image`` untouched when disabled or the scale is safe, so the
    default path stays bit-identical.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        mtx: ``(batch_size, 3, 3)`` forward pixel matrix of the downscaling warp.
        enabled: Whether the ``antialias`` flag is set for this segment.

    Returns:
        The prefiltered image, or the original image when no filtering applies.

    """
    if not enabled:
        return image
    scale_x, scale_y = estimate_scale(mtx)  # (width-axis, height-axis) singular values, ascending
    if min(scale_x, scale_y) >= _ANTIALIAS_SCALE_THRESHOLD:
        return image
    sigma_x = _mipmap_sigma(scale_x)
    sigma_y = _mipmap_sigma(scale_y)
    blurred = _kornia_gaussian_blur(image, sigma_x, sigma_y)
    return image if blurred is None else blurred


def _kornia_gaussian_blur(image: Tensor, sigma_x: float, sigma_y: float) -> Tensor | None:
    """Anisotropic Gaussian pre-blur via kornia, or ``None`` when kornia is absent.

    Blurs ``image`` with per-axis sigmas using the installed kornia
    ``gaussian_blur2d`` (no custom kernel). The kernel size follows the ``3-sigma``
    rule, rounded up to the next odd integer per axis. Returns ``None`` when kornia
    is not importable so the caller can fall back to the un-filtered warp.

    Args:
        image: ``(batch_size, channels, height, width)`` float input tensor.
        sigma_x: Gaussian standard deviation along the width axis, in pixels.
        sigma_y: Gaussian standard deviation along the height axis, in pixels.

    Returns:
        The blurred tensor, or ``None`` when kornia is unavailable or both sigmas are ~0.

    """
    if sigma_x <= 0.0 and sigma_y <= 0.0:
        return image
    try:
        from kornia.filters import gaussian_blur2d
    except ImportError:
        return None
    ksize_x = 2 * math.ceil(3.0 * sigma_x) + 1
    ksize_y = 2 * math.ceil(3.0 * sigma_y) + 1
    # kornia expects positive sigmas on both axes; clamp the near-zero axis to a
    # tiny value so a single-axis downscale still runs through one call.
    sig_x = max(sigma_x, 1e-6)
    sig_y = max(sigma_y, 1e-6)
    return gaussian_blur2d(image, kernel_size=(ksize_y, ksize_x), sigma=(sig_y, sig_x), border_type="reflect")


class ExactAffineSegment(nn.Module):
    """Lossless segment for GEOMETRIC_EXACT-only chains.

    Used when a run of consecutive geometric transforms consists entirely of ``GEOMETRIC_EXACT`` operations, such as
    flips and other discrete, lossless image-space transforms supported by the active adapter (for example 90-degree
    rotations or transpose-like ops). Applies each transform via :meth:`TransformAdapter.exact_apply` instead of
    ``grid_sample``, introducing zero interpolation error.

    Per-sample probability masking is implemented by sampling a boolean mask of shape ``(B,)`` from each transform's
    application probability and applying the exact transform only to active samples. The fused engine prefers a ``prob``
    attribute when present and falls back to backend ``p`` for native transform objects.

    Auxiliary-target routing: masks route for every exact op (a non-flip op's mask is transformed by the same
    lossless rotation applied to the image, sharing its per-sample sampling); boxes/keypoints route for flips via
    :meth:`TransformAdapter.exact_flip_dims`. A geometric run that combines a non-flip exact op with box/keypoint
    targets is built as a :class:`FusedAffineSegment` (grid path) instead, so those targets are always routed --
    ``build_segments`` receives that hint via its ``route_coords_via_grid`` flag.

    Args:
        transforms: List of ``GEOMETRIC_EXACT`` transform objects.
        adapter: A ``TransformAdapter`` providing ``exact_apply`` for image
            updates and, when auxiliary targets are used, ``exact_flip_dims`` for flip-compatible target routing.
        randomness: Batch randomness policy. ``BACKEND`` preserves native
            backend semantics; ``PER_SAMPLE`` draws probability masks per item.

    Example:
        >>> import torch
        >>> import kornia.augmentation as K
        >>> from fuse_augmentations.affine.segment import ExactAffineSegment
        >>> from fuse_augmentations.adapters.kornia import KorniaAdapter
        >>> t = K.RandomHorizontalFlip(p=1.0)
        >>> seg = ExactAffineSegment([t], KorniaAdapter())
        >>> out = seg(torch.zeros(1, 3, 8, 8))
        >>> out.shape
        torch.Size([1, 3, 8, 8])

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    ) -> None:
        """Initialize ``ExactAffineSegment``."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.randomness = randomness

    @property
    def last_matrix(self) -> Tensor | None:
        """Return ``None`` always (ExactAffineSegment does not compute a matrix)."""
        return None

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply exact transforms losslessly with per-sample masking.

        For each transform, draws a per-sample boolean mask from the transform's
        ``prob`` probability, applies :meth:`TransformAdapter.exact_apply` only to
        active samples, and scatters the transformed subset back into the batch.
        Auxiliary-target routing is currently supported only for flip-compatible
        exact ops exposed through :meth:`TransformAdapter.exact_flip_dims`.

        Args:
            image: Input image batch. Shape: ``(batch_size, channels, height, width)``, dtype: float32.
                Value range and channel convention follow the calling pipeline.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size = image.shape[0]
        _height, width = image.shape[2], image.shape[3]
        device = image.device

        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm, self.randomness)
            if not same_on_batch:
                # Independent Bernoulli draw per sample.
                active = torch.rand(batch_size, device=device) < prob
            else:
                # Single Bernoulli draw shared across the entire batch.
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.repeat(batch_size)

            # Skip this transform entirely if no samples are active.
            if not bool(active.any().item()):
                continue

            # A flip exposes its axes via exact_flip_dims; a non-flip D4 op (rot90,
            # transpose) raises there. For non-flip ops the mask is routed by applying
            # the identical op to it (sharing the image's per-sample sampling — see
            # _apply_exact_with_mask), while boxes/keypoints still raise (no per-sample
            # matrix is recoverable without re-sampling).
            try:
                flip_dims: list[int] | None = self.adapter.exact_flip_dims(tfm)
            except (TypeError, NotImplementedError):
                flip_dims = None

            image = self._apply_exact_with_mask(tfm, image, active, aux_targets, flip_dims)
            if aux_targets:
                self._route_exact_coord_aux(tfm, flip_dims, active, aux_targets, _height, width)

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_exact_with_mask(
        self,
        tfm: object,
        image: Tensor,
        active: Tensor,
        aux_targets: dict[str, Tensor],
        flip_dims: list[int] | None,
    ) -> Tensor:
        """Apply an exact op to the image, stacking the mask for non-flip ops.

        For a non-flip D4 op (rot90/transpose) with a mask, the mask is concatenated
        onto the image channels for a single :meth:`TransformAdapter.exact_apply` call
        so both share the identical per-sample random draw (e.g. the same ``rot90``
        count ``k``) — a separate call would re-sample and misalign them. Flip ops are
        deterministic per axis and route the mask via ``exact_flip_dims`` afterwards,
        so no stacking is needed. The active subset is scattered back so inactive
        samples are untouched.

        Args:
            tfm: The exact transform to apply.
            image: ``(B, C, H, W)`` input batch.
            active: ``(B,)`` bool mask of samples this transform applies to.
            aux_targets: Aux dict; its ``"mask"`` entry is updated in place for
                non-flip ops.
            flip_dims: Result of ``exact_flip_dims`` (``None`` for non-flip ops).

        Returns:
            The transformed ``(B, C, H, W)`` image.

        """
        mask = aux_targets.get("mask")
        # Only non-flip ops need the mask stacked to share sampling; flips route it
        # afterwards via exact_flip_dims, so the image-only path is kept unchanged.
        stack_mask = mask is not None and flip_dims is None
        num_channels = image.shape[1]
        stack = torch.cat([image, mask], dim=1) if mask is not None and stack_mask else image

        active_idx = active.nonzero(as_tuple=True)[0]
        if image.shape[0] == 1 or bool(active.all().item()):
            stack_out = self.adapter.exact_apply(tfm, stack)
        else:
            transformed = self.adapter.exact_apply(tfm, stack[active_idx])
            stack_out = stack.clone()
            stack_out[active_idx] = transformed

        if not stack_mask:
            return stack_out
        aux_targets["mask"] = stack_out[:, num_channels:]
        return stack_out[:, :num_channels]

    def _route_exact_coord_aux(
        self,
        tfm: object,
        flip_dims: list[int] | None,
        active: Tensor,
        aux_targets: dict[str, Tensor],
        height: int,
        width: int,
    ) -> None:
        """Route flip-based mask/box/keypoint aux with per-sample masking.

        Flips route mask (via ``exact_flip_dims``), boxes and keypoints with per-sample
        ``active`` masking. Non-flip exact ops (rot90/transpose) have their mask handled
        by :meth:`_apply_exact_with_mask` and return here without touching coord targets:
        a pipeline carrying box/keypoint aux is built with such runs routed through the
        interpolating grid segment instead (see ``build_segments`` ``route_coords_via_grid``),
        so a non-flip exact op never reaches this method with coord targets.

        Args:
            tfm: The exact transform.
            flip_dims: ``exact_flip_dims`` result, or ``None`` for non-flip ops.
            active: ``(B,)`` bool mask of active samples.
            aux_targets: Aux dict, updated in place.
            height: Image height in pixels.
            width: Image width in pixels.

        """
        if flip_dims is None:
            # Non-flip exact op: mask already routed in _apply_exact_with_mask; coord
            # targets are handled by the grid segment upstream, so nothing to do here.
            return
        is_hflip = 3 in flip_dims
        is_vflip = 2 in flip_dims
        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                flipped_val = val.flip(dims=flip_dims)
                aux_targets[key] = torch.where(active[:, None, None, None], flipped_val, val)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = _flip_bbox_xyxy(val, active, is_hflip, is_vflip, height, width)
                continue
            if key == "bbox_xywh":
                xyxy = _xywh_to_xyxy(val)
                xyxy = _flip_bbox_xyxy(xyxy, active, is_hflip, is_vflip, height, width)
                aux_targets[key] = _xyxy_to_xywh(xyxy)
                continue
            if key == "keypoints":
                aux_targets[key] = _flip_keypoints(val, active, is_hflip, is_vflip, height, width)


class _BaseAffineSegment(nn.Module):
    """Shared matrix-composition engine for the torch-backed fused segments.

    Holds the single copy of the per-sample matrix accumulation loop and the
    auxiliary-target routing that :class:`FusedAffineSegment` (affine) and
    :class:`ProjectiveSegment` (homography) both use. Subclasses supply the warp
    itself via :meth:`_apply_grid` -- affine grids for the affine segment, a
    perspective grid for the projective one. The composition, float64
    chain-length handling, probability masking, and pixel-matrix contract are
    identical across both, so they live here once.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical
            parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``,
            ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``,
            ``"reflection"``). Defaults to ``"zeros"`` when ``None``.
        randomness: Batch randomness policy for the fused run.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    """

    _eye3: Tensor

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize the shared matrix-composition state."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        self.mask_interpolation = mask_interpolation
        self._last_matrix: Tensor | None = None
        # Opt-in torch.compile of the warp core. Enabled only when the flag is set
        # AND the installed torch is new enough; otherwise the eager path runs
        # unchanged. The compiled function is selected per forward call by device
        # (CPU stays eager — documented no-op), so store just the intent here.
        self._compile_warp: bool = compile_warp and _torch_supports_compile()
        self.register_buffer("_eye3", torch.eye(3, dtype=torch.float32))

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The detached, cloned composed matrix, or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def _compose(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Accumulate the per-transform ``(B, 3, 3)`` matrix chain into one matrix.

        Draws each transform's per-sample activation, samples its parameters,
        builds its matrix, masks skipped samples back to identity, and folds the
        chain into a single composed forward matrix. Composition runs in the
        chain-length-independent dtype from :func:`_matrix_compose_dtype` (float64
        for chains, image dtype for single ops and MPS); the composed matrix is
        also returned cast to the image dtype for the public matrix contract.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.

        Returns:
            A ``(acc, acc_img)`` tuple: ``acc`` is the composed forward matrix in
            the compose dtype (for inversion and exactness checks), ``acc_img`` is
            the same matrix cast to the image dtype (for ``_last_matrix`` and
            auxiliary-target routing).

        """
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # Compose and invert the multi-transform chain in float64: the (B, 3, 3)
        # matmul cost is negligible next to the H x W warp, and float64 removes the
        # chain-length-dependent drift that float32 accumulation introduces. The
        # composed matrix is cast back to the image dtype only where the public
        # contract needs it (``_last_matrix``, aux routing). A single transform has
        # no chain to accumulate, so it stays in the image dtype -- keeping the
        # common single-op case bit-for-bit compatible with the native warp. MPS
        # has no float64, so chains there fall back to the image dtype too.
        compose_dtype = _matrix_compose_dtype(dtype, device, len(self.transforms))
        eye = self._eye3.to(device=device, dtype=compose_dtype)
        eye_batch = eye[None].expand(batch_size, -1, -1)
        acc = eye_batch.clone()

        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm, self.randomness)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.repeat(batch_size)
            else:
                active = torch.rand(batch_size, device=device) < prob

            params = _sample_transform_params(self.adapter, tfm, input_shape, device, self.randomness)
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)

            # Expand to batch if adapter returned (1, 3, 3)
            if mtx_i.shape[0] == 1 and batch_size > 1:
                mtx_i = mtx_i.expand(batch_size, -1, -1)

            # Accumulate in the compose dtype; the adapter builds in float32.
            mtx_i = mtx_i.to(device=device, dtype=compose_dtype)

            mtx_i = torch.where(active[:, None, None], mtx_i, eye_batch)
            acc = matmul3x3(mtx_i, acc)

        return acc, acc.to(dtype=dtype)

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` by the composed matrix and return the warped image and grid.

        Subclass hook: :class:`FusedAffineSegment` builds an ``F.affine_grid`` from
        the inverse of the composed matrix, :class:`ProjectiveSegment` builds a
        perspective grid. Both invert the composed forward matrix, normalize it to
        the ``[-1, 1]`` grid convention, and run one ``F.grid_sample``.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward matrix in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple; the grid is reused to warp mask aux targets.

        """
        raise NotImplementedError

    def _select_warp_fn(self, kind: str, device: torch.device) -> WarpFn:
        """Pick the eager or compiled warp core for this call.

        The compiled core is used only when ``compile=True`` was requested (and
        torch is new enough) AND the tensor is not on CPU. CPU stays eager: the
        inductor CPU backend gives no speedup for this grid_sample workload, so
        compiling it there would only add warm-up cost — a documented no-op. The
        masking / active-subset selection lives entirely in the callers, so the
        selected core sees only dense tensors and never breaks its graph.

        Args:
            kind: ``"affine"`` or ``"perspective"``.
            device: Device of the image being warped.

        Returns:
            The compiled warp core when eligible, else the eager one.

        """
        eager: WarpFn = _grid_sample_affine_batched if kind == "affine" else _grid_sample_perspective_batched
        if self._compile_warp and device.type != "cpu":
            return _compiled_warp_fn(kind)
        return eager

    @staticmethod
    def _route_grid_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor,
        acc_img: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Route auxiliary targets through the warp grid and composed pixel matrix.

        The mask is resampled with the same ``grid`` used for the image; boxes and
        keypoints go through the composed forward pixel matrix ``acc_img``. Mutates
        ``aux_targets`` in place. Shared by the affine and projective forward paths.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            grid: The sampling grid produced by :meth:`_apply_grid`.
            acc_img: ``(batch_size, 3, 3)`` composed forward pixel matrix in the image dtype.
            mask_interpolation: Mask sampling mode for the ``"mask"`` target.

        """
        from fuse_augmentations.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_keypoints,
            transform_mask,
        )

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
                continue
            if key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
                continue
            if key == "keypoints":
                aux_targets[key] = transform_keypoints(val, acc_img)


class FusedAffineSegment(_BaseAffineSegment):
    """Fused affine segment that composes geometric transforms into one grid_sample call.

    Accumulates per-sample ``(B, 3, 3)`` forward affine matrices for every transform in the segment, inverts the
    composed matrix once, and applies a single ``grid_sample`` warp. All operations are vectorised over the batch
    dimension -- no Python loop per sample.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults
            to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to
            ``"zeros"`` when ``None``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize ``FusedAffineSegment``."""
        super().__init__(
            transforms,
            adapter,
            interpolation,
            padding_mode,
            randomness,
            mask_interpolation=mask_interpolation,
            compile_warp=compile_warp,
        )
        # Pre-compute fast-path selector once at construction to avoid repeated
        # isinstance checks on every forward call.
        self._fast_path: str | None = None
        try:
            from fuse_augmentations.adapters.kornia import KorniaAdapter
            from fuse_augmentations.adapters.torchvision import TorchVisionAdapter

            if isinstance(adapter, KorniaAdapter):
                self._fast_path = "kornia"
            elif isinstance(adapter, TorchVisionAdapter):
                self._fast_path = "torchvision"
        except ImportError:
            pass
        # Single-transform fast paths still reconstruct _last_matrix because
        # Compose.transform_matrix is a public API used for coordinate warping.
        self._skip_matrix_recon: bool = False
        # cv2 warp fast path: for B=1 CPU multi-transform segments, cv2.warpAffine is
        # ~2x faster than PyTorch's affine_grid + grid_sample because it avoids
        # the grid construction overhead entirely.
        self._cv2_warp: bool = _cv2 is not None and len(transforms) > 1
        # Pre-compute cv2 flags once (used by the cv2 fast path every call).
        _interp_str = self.interpolation or "bilinear"
        _pad_str = self.padding_mode or "zeros"
        self._cv2_interp_flag: int = _CV2_INTERP.get(_interp_str, _CV2_INTERP.get("bilinear", 1))
        self._cv2_border_flag: int = _CV2_BORDER.get(_pad_str, _CV2_BORDER.get("zeros", 0))
        # Numpy-native matrix builder for the cv2 warp path: builds a (3,3)
        # float64 matrix directly in numpy, avoiding the torch tensor
        # allocations in build_matrix that are immediately converted back to
        # numpy.  Resolved once at construction from self._fast_path.
        self._np_matrix_builder = None
        # Fused sample+build builder: calls generate_parameters directly and
        # skips the canonical param dict entirely, saving ~3-6 torch tensor
        # allocations per transform per forward call.  Only available for the
        # Kornia cv2 path.  Returns None for unsupported types (caller falls
        # back to the two-step path).
        self._np_fused_builder = None
        if self._cv2_warp and self._fast_path == "kornia":
            try:
                from fuse_augmentations.adapters.kornia import (
                    build_matrix_numpy_b1_kornia,
                    sample_and_build_matrix_numpy_b1_kornia,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_kornia
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_kornia
            except ImportError:
                pass
        elif self._cv2_warp and self._fast_path == "torchvision":
            try:
                from fuse_augmentations.adapters.torchvision import (
                    build_matrix_numpy_b1_tv,
                    sample_and_build_matrix_numpy_b1_tv,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_tv
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_tv  # type: ignore[assignment]
            except ImportError:
                pass
        # Pre-allocated (1, 3, 3) float32 buffer for cv2 path _last_matrix writes.
        # Avoids per-call torch.as_tensor + unsqueeze + clone (~3-5 us).
        self._cv2_last_mat_buf: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        # Pre-cached B=1 CPU float32 identity for single-transform fast-path _last_matrix
        # writes.  Avoids per-call torch.eye + unsqueeze + expand + clone (~4-6 us)
        # when device and dtype match.
        self._eye_1x3x3_f32: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused affine transform chain via a single grid_sample call.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # ------------------------------------------------------------------ #
        # Single-operation fast path: skip matrix pipeline + grid_sample entirely.   #
        # Call the native adapter transform and reconstruct _last_matrix from  #
        # the sampled params.  Only safe when aux_targets is None (no grid    #
        # needed for coord transforms).                                        #
        # ------------------------------------------------------------------ #
        if (
            len(self.transforms) == 1
            and not _has_aux
            and self._fast_path is not None
            and self.randomness is RandomnessPolicy.BACKEND
        ):
            _tfm = self.transforms[0]

            if self._fast_path == "kornia":
                from fuse_augmentations.adapters.kornia import KorniaAdapter

                # After call_nonfused, Kornia stores sampled params in tfm._params.
                # convert_native_params reads those to build a consistent matrix.
                image = KorniaAdapter.call_nonfused(_tfm, image)

                # Early escape: if all samples were skipped (batch_prob all False),
                # OR if this transform type has an expensive build_matrix with no
                # test requiring its _last_matrix value — use identity and return.
                _bp_raw = getattr(_tfm, "_params", {}).get("batch_prob")
                _all_skipped = _bp_raw is not None and not _bp_raw.to(device=device).bool().any()
                if _all_skipped or self._skip_matrix_recon:
                    _mtx_eye = self._eye_1x3x3_f32
                    _last_matrix = (
                        _mtx_eye
                        if (batch_size == 1 and dtype == _mtx_eye.dtype and device == _mtx_eye.device)
                        else _mtx_eye.expand(batch_size, -1, -1).detach().clone().to(device=device, dtype=dtype)
                    )
                    self._last_matrix = _last_matrix
                    _set_current_call_matrix(_last_matrix.detach().clone())
                    return image

                _native_p = KorniaAdapter.convert_native_params(_tfm, device)
                if _native_p:
                    _mtx = self.adapter.build_matrix(_tfm, _native_p, height, width)
                    if _mtx.shape[0] == 0:
                        # All samples skipped (prob=0.0) — identity for the whole batch.
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        _mtx = _mtx.to(device=device, dtype=dtype)
                        # Kornia stores params for the ACTIVE subset only, so at
                        # batch>1 with prob<1 the matrix has shape (n_active, 3, 3).
                        # Scatter into a full-batch identity keyed by batch_prob so
                        # skipped samples stay identity and shapes never mismatch.
                        _active = None
                        if _bp_raw is not None:
                            _active = _bp_raw.to(device=device).bool()
                            if _active.shape[0] == 1 and batch_size > 1:
                                _active = _active.expand(batch_size)
                        _mtx = _scatter_active_matrices(_mtx, _active, batch_size, device, dtype)
                else:
                    _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                _last_matrix = _mtx.detach().clone()
                self._last_matrix = _last_matrix
                _set_current_call_matrix(_last_matrix)
                return image

            if self._fast_path == "torchvision":
                from fuse_augmentations.adapters.torchvision import (
                    TorchVisionAdapter,
                    is_torchvision_v2_transform,
                )

                if is_torchvision_v2_transform(_tfm):
                    if self._skip_matrix_recon:
                        image = TorchVisionAdapter.call_nonfused(_tfm, image)
                        _mtx_eye = self._eye_1x3x3_f32
                        if batch_size == 1 and dtype == _mtx_eye.dtype and device == _mtx_eye.device:
                            _last_matrix = _mtx_eye
                        else:
                            _last_matrix = (
                                _mtx_eye.expand(batch_size, -1, -1).detach().clone().to(device=device, dtype=dtype)
                            )
                        self._last_matrix = _last_matrix
                        _set_current_call_matrix(_last_matrix.detach().clone())
                        return image
                    # TV v2 GEOMETRIC_INTERP transforms (RandomRotation, RandomAffine)
                    # always apply and make exactly one RNG call (get_params) - same as
                    # our sample_params.  Save/restore RNG state so both draws use the
                    # same seed.  Restricted to v2: v1 transforms use a different
                    # pixel-center convention that does not match our grid_sample output,
                    # breaking parity tests.
                    _rng = torch.get_rng_state()
                    image = TorchVisionAdapter.call_nonfused(_tfm, image)
                    torch.set_rng_state(_rng)
                    _params = self.adapter.sample_params(_tfm, input_shape, device)
                    if _params:
                        _mtx = self.adapter.build_matrix(_tfm, _params, height, width)
                        if _mtx.shape[0] == 1 and batch_size > 1:
                            _mtx = _mtx.expand(batch_size, -1, -1)
                    else:
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                    _last_matrix = _mtx.to(device=device, dtype=dtype).detach().clone()
                    self._last_matrix = _last_matrix
                    _set_current_call_matrix(_last_matrix)
                    return image

        # ------------------------------------------------------------------ #
        # B=1 CPU cv2 fast path: cv2.warpAffine is ~2x faster than PyTorch's  #
        # affine_grid + grid_sample for single-image CPU tensors because it    #
        # avoids the H*W grid construction overhead entirely.  Compose the     #
        # matrix using the same sample_params / build_matrix loop but replace    #
        # the PyTorch warp backend with cv2.  Only activates when:             #
        # - B=1, CPU, no CUDA, no aux_targets, cv2 available, >1 transform    #
        # ------------------------------------------------------------------ #
        # Gate on CPU tensors only: the cv2 warp round-trips through NumPy
        # (image[0]...numpy()), which raises on any non-CPU device (CUDA and MPS).
        if self._cv2_warp and batch_size == 1 and not _has_aux and image.device.type == "cpu":
            acc_np = np.eye(3, dtype=np.float64)
            # Select the numpy-native matrix builder when available (avoids
            # creating intermediate torch tensors that are immediately converted
            # back to numpy).
            _np_builder = self._np_matrix_builder
            # Fused sample+build: calls generate_parameters directly and builds
            # the matrix in one numpy-native call, avoiding ~15-25us of
            # adapter.sample_params overhead per active transform. Falls back to
            # two-step path when the transform type is not handled (returns None).
            _np_fused = self._np_fused_builder
            for tfm in self.transforms:
                prob = _transform_prob(tfm)
                # Draw the activation gate from the torch RNG, unconditionally, so
                # it responds to torch.manual_seed (np.random was uncontrollable
                # from the torch seed). NOTE: full cross-backend param-draw parity
                # does NOT hold for prob<1 chains — the torch path below samples
                # params for inactive transforms (vectorized, masked via
                # torch.where) while this path skips them, so RNG stream positions
                # diverge after the first inactive transform.
                active = bool((torch.rand(()) < prob).item())
                if not active:
                    continue
                if _np_fused is not None:
                    mtx_np = _np_fused(tfm, input_shape, height, width)
                    if mtx_np is not None:
                        acc_np = mtx_np @ acc_np
                        continue
                params = _sample_transform_params(self.adapter, tfm, input_shape, device, self.randomness)
                if _np_builder is not None:
                    mtx_np = _np_builder(tfm, params, height, width)
                    acc_np = mtx_np @ acc_np
                else:
                    mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                    acc_np = mtx_i[0].double().cpu().numpy() @ acc_np

            np.copyto(self._cv2_last_mat_buf[0].numpy(), acc_np, casting="unsafe")
            self._last_matrix = self._cv2_last_mat_buf
            _set_current_call_matrix(torch.as_tensor(acc_np, dtype=dtype).unsqueeze(0).detach().clone())

            # Symbolic-exactness fast path on the cv2 B=1 branch: a chain that composes
            # to a D4 element is applied losslessly via flip/rot90, skipping the cv2 warp
            # entirely (zero interpolation). No aux here (gated above).
            d4_op = classify_d4_batch(self._cv2_last_mat_buf, height, width)
            if d4_op is not None:
                return apply_d4_image(image, d4_op).to(device=device, dtype=dtype)

            m_inv_np = _inv3x3_affine_np(acc_np)
            img_np = image[0].permute(1, 2, 0).contiguous().numpy()
            if num_channels == 1:
                warped = _warp(img_np[:, :, 0], m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag)
            image = torch.from_numpy(warped).permute(2, 0, 1).unsqueeze(0)
            return image.to(device=device, dtype=dtype)

        # Compose the multi-transform affine chain into one matrix (shared engine).
        acc, acc_img = self._compose(image)
        self._last_matrix = acc_img.detach().clone()
        _set_current_call_matrix(acc_img.detach().clone())

        # Symbolic-exactness fast path: when the whole batch composes to one D4-group
        # element (flip / 90-degree rotation, zero net translation beyond the
        # border-preserving form), apply it losslessly via flip/rot90 instead of
        # grid_sample -- zero interpolation error. Non-D4 chains fall through unchanged.
        d4_op = classify_d4_batch(acc, height, width)
        if d4_op is not None:
            image = apply_d4_image(image, d4_op)
            if aux_targets:
                self._route_d4_aux(aux_targets, d4_op, acc_img)
            if not _has_aux:
                return image
            return image, aux_targets

        image, grid = self._apply_grid(image, acc)

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            self._route_grid_aux(aux_targets, grid, acc_img, self.mask_interpolation)

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` with a single ``F.affine_grid`` + ``grid_sample`` pass.

        Inverts the composed forward matrix, normalizes it to the ``[-1, 1]`` grid
        convention, builds an affine grid, and resamples. The grid is returned so
        the caller can reuse it to warp mask auxiliary targets.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward matrix in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple.

        """
        warp_fn = self._select_warp_fn("affine", image.device)
        return warp_fn(
            image,
            acc,
            self.interpolation or "bilinear",
            self.padding_mode or "zeros",
        )

    @staticmethod
    def _route_d4_aux(aux_targets: dict[str, Tensor], d4_op: str, acc_img: Tensor) -> None:
        """Route aux targets through an exact D4 op with zero interpolation.

        The mask is transformed by the same lossless ``flip``/``rot90`` op applied to
        the image (no nearest-resample); boxes and keypoints go through the exact
        composed forward pixel matrix ``acc_img`` (integer-valued for a D4 chain), the
        same convention as the interpolating path. Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            d4_op: The D4 op name from :func:`classify_d4_batch`.
            acc_img: ``(B, 3, 3)`` composed forward pixel matrix in the image dtype.

        """
        from fuse_augmentations.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_keypoints,
        )

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = apply_d4_image(val, d4_op)
                continue
            if key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
                continue
            if key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
                continue
            if key == "keypoints":
                aux_targets[key] = transform_keypoints(val, acc_img)


class _FusedGeoCropSegment(FusedAffineSegment):
    """Fuse a preceding geometric run and a ``CROP_RESIZE_FIXED`` op into one warp.

    A ``RandomResizedCrop`` immediately after a fusible geometric run
    (``GEOMETRIC_INTERP``/``GEOMETRIC_EXACT``) is normally a hard segment
    boundary — the geo run does one ``grid_sample`` at input size, then
    :class:`CropResizeSegment` does a second one at target size. This segment
    composes both into a single matrix ``M_crop @ M_geo`` and applies exactly one
    ``grid_sample`` at the crop's ``(target_h, target_w)`` output size, saving an
    interpolation pass and improving precision (one resample instead of two).

    The crop reads the geo chain's output, so the forward composite is
    ``M_crop @ M_geo`` (geo applied first, crop after). Like
    :class:`CropResizeSegment`, per-sample probability is *not* applied to the
    crop (shape-changing ops must produce a uniform output size); the geo run's
    per-sample ``prob`` gates are honoured exactly as in :class:`FusedAffineSegment`.

    ``transforms`` is ``[*geo_transforms, crop_transform]`` so the inherited
    fusion-plan machinery (``n_warps_saved``, ``fusion_plan``, ``transform_matrix``)
    counts the crop as one more fused op and exposes the full geo∘crop matrix.

    Args:
        geo_transforms: The preceding fusible geometric transforms, in order.
        crop_transform: A single ``CROP_RESIZE_FIXED`` transform.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_matrix``.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        randomness: Batch randomness policy for the fused geometric run.

    Example:
        >>> import torch
        >>> import kornia.augmentation as K
        >>> from fuse_augmentations.affine.segment import _FusedGeoCropSegment
        >>> from fuse_augmentations.adapters.kornia import KorniaAdapter
        >>> geo = K.RandomHorizontalFlip(p=1.0)
        >>> crop = K.RandomResizedCrop((8, 8), scale=(0.5, 0.5), ratio=(1.0, 1.0))
        >>> seg = _FusedGeoCropSegment([geo], crop, KorniaAdapter())
        >>> seg(torch.zeros(1, 3, 16, 16)).shape
        torch.Size([1, 3, 8, 8])

    """

    def __init__(
        self,
        geo_transforms: list[object],
        crop_transform: object,
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        compile_warp: bool = False,
        antialias: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize ``_FusedGeoCropSegment``."""
        # nn.Module state only; skip FusedAffineSegment.__init__'s cv2/numpy
        # fast-path wiring (it keys on len(transforms) and would try to resolve a
        # numpy builder for the crop transform — this segment overrides forward).
        nn.Module.__init__(self)
        self.register_buffer("_eye3", torch.eye(3, dtype=torch.float32))
        self.geo_transforms = geo_transforms
        self.crop_transform = crop_transform
        # transforms holds the full fused run so inherited fusion-plan machinery
        # (n_warps_saved = n-1, fusion_plan naming) counts the crop as fused.
        self.transforms: list[object] = [*geo_transforms, crop_transform]
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        self.mask_interpolation = mask_interpolation
        self._last_matrix: Tensor | None = None
        self._compile_warp: bool = compile_warp and _torch_supports_compile()
        self._antialias: bool = antialias

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused geo∘crop chain via one ``grid_sample`` at the target size.

        Args:
            image: ``(batch_size, channels, height_in, width_in)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). Masks are warped with the output grid; boxes and
                keypoints via the composed forward matrix.

        Returns:
            ``(batch_size, channels, height_out, width_out)`` tensor when
            ``aux_targets`` is ``None``; ``(tensor, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (batch_size, num_channels, height, width)

        # Accumulate the geometric run exactly as FusedAffineSegment does, but the
        # chain always has >1 fused op (geo + crop), so compose in float64 where
        # available. The crop matrix has no per-sample prob gate.
        compose_dtype = _matrix_compose_dtype(dtype, device, len(self.transforms))
        eye = self._eye3.to(device=device, dtype=compose_dtype)
        eye_batch = eye[None].expand(batch_size, -1, -1)
        acc = eye_batch.clone()

        for tfm in self.geo_transforms:
            prob = _transform_prob(tfm)
            if _shares_randomness_across_batch(self.adapter, tfm, self.randomness):
                active = (torch.rand((), device=device) < prob).repeat(batch_size)
            else:
                active = torch.rand(batch_size, device=device) < prob
            params = _sample_transform_params(self.adapter, tfm, input_shape, device, self.randomness)
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)
            if mtx_i.shape[0] == 1 and batch_size > 1:
                mtx_i = mtx_i.expand(batch_size, -1, -1)
            mtx_i = mtx_i.to(device=device, dtype=compose_dtype)
            mtx_i = torch.where(active[:, None, None], mtx_i, eye_batch)
            acc = matmul3x3(mtx_i, acc)

        target_h, target_w, mtx_crop = self._build_crop_matrix(input_shape, device, compose_dtype)
        acc_full = matmul3x3(mtx_crop, acc)  # crop reads geo's output

        self._last_matrix = acc_full.to(dtype=dtype).detach().clone()
        _set_current_call_matrix(self._last_matrix)

        # Opt-in antialiasing on the fused geo∘crop downscale. The composed 2x2 may
        # be rotated/sheared, so a per-axis Gaussian prefilter (mipmap sigma) is
        # applied to the input before the single warp. No-op unless enabled and the
        # worst axis downscales past the threshold — default output unchanged.
        image = _maybe_antialias_prefilter(image, acc_full.to(dtype=dtype), self._antialias)

        mtx_inv = inv3x3(acc_full)
        mtx_norm = normalize_matrix_io(mtx_inv, height, width, target_h, target_w).to(dtype=dtype)
        grid = F.affine_grid(
            mtx_norm[:, :2, :],
            [batch_size, num_channels, target_h, target_w],
            align_corners=True,
        )
        out = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

        if aux_targets:
            self._warp_aux(aux_targets, grid, acc_full.to(dtype=dtype), self.mask_interpolation)

        if not _has_aux:
            return out
        if aux_targets is None:
            raise RuntimeError("internal error: aux_targets is None in return branch")
        return out, aux_targets

    def _build_crop_matrix(
        self,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
        compose_dtype: torch.dtype,
    ) -> tuple[int, int, Tensor]:
        """Sample the crop and return ``(target_h, target_w, (B, 3, 3) crop matrix)``."""
        batch_size, _, height, width = input_shape
        params = _sample_transform_params(self.adapter, self.crop_transform, input_shape, device, self.randomness)
        if not (
            torch.all(params["target_h"] == params["target_h"][0])
            and torch.all(params["target_w"] == params["target_w"][0])
        ):
            raise ValueError(
                "_FusedGeoCropSegment requires a uniform target size across the batch "
                f"(got target_h={params['target_h'].tolist()}, target_w={params['target_w'].tolist()})"
            )
        target_h = int(params["target_h"][0].item())
        target_w = int(params["target_w"][0].item())
        mtx_crop = self.adapter.build_matrix(self.crop_transform, params, height, width)
        if mtx_crop.shape[0] == 1 and batch_size > 1:
            mtx_crop = mtx_crop.expand(batch_size, -1, -1)
        return target_h, target_w, mtx_crop.to(device=device, dtype=compose_dtype)

    @staticmethod
    def _warp_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor,
        mtx: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Warp auxiliary targets in place: mask via the output grid, coords via ``mtx``."""
        from fuse_augmentations.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_keypoints,
            transform_mask,
        )

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask":
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation)
            elif key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, mtx)
            elif key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, mtx)
            elif key == "keypoints":
                aux_targets[key] = transform_keypoints(val, mtx)


# ---------------------------------------------------------------------------
# AlbuFusedAffineSegment — cv2 backend for Albumentations
# ---------------------------------------------------------------------------

ImageArray = NDArray[np.integer[Any] | np.floating[Any]]
MatrixArray = NDArray[np.floating[Any]]


def _warp(
    img: ImageArray,
    matrix_dst2src_3x3: MatrixArray,
    width: int,
    height: int,
    interp_flag: int,
    border_flag: int,
) -> ImageArray:
    """Apply cv2.warpAffine with the dst->src 3x3 pixel-space matrix.

    ``matrix_dst2src_3x3`` maps destination pixel coordinates to source pixel
    coordinates.  ``cv2.WARP_INVERSE_MAP`` (16) is OR-ed into *interp_flag* so
    the matrix is used directly without re-inversion.  cv2 handles all channels
    in a single call, avoiding the per-channel loop previously needed for scipy.

    Args:
        img: HxW or HxWxC float32 numpy array.
        matrix_dst2src_3x3: 3x3 matrix mapping destination pixels to source pixels.
        width: Output width in pixels.
        height: Output height in pixels.
        interp_flag: cv2 interpolation constant (e.g. ``1`` for ``INTER_LINEAR``).
        border_flag: cv2 border mode constant (e.g. ``0`` for ``BORDER_CONSTANT``).

    Returns:
        Warped image array with the same dtype and channel count as ``img``.

    """
    import cv2

    m_2x3 = matrix_dst2src_3x3[:2, :].astype(np.float64)
    warp_affine = cast(Any, cv2.warpAffine)
    return cast(
        MatrixArray,
        warp_affine(
            img,
            m_2x3,
            (width, height),
            flags=interp_flag | _CV2_WARP_INVERSE_MAP,
            borderMode=border_flag,
            borderValue=0,
        ),
    )


def _inv3x3_affine_np(mtx: MatrixArray) -> MatrixArray:
    """Closed-form inverse of a 3x3 affine matrix (bottom row = [0, 0, 1]).

    Uses Cramer's rule for the upper-left 2x2 sub-matrix, avoiding LAPACK
    dispatch overhead (~15-20us) of ``np.linalg.inv`` for a single 3x3 matrix.

    Args:
        mtx: A (3, 3) float64 ndarray representing a forward affine transform.
           The bottom row must be ``[0, 0, 1]`` (standard affine convention).

    Returns:
        The (3, 3) inverse affine matrix as a float64 ndarray.

    Examples:
        >>> import numpy as np
        >>> mtx = np.eye(3, dtype=np.float64)
        >>> _inv3x3_affine_np(mtx)
        array([[ 1., -0.,  0.],
               [-0.,  1.,  0.],
               [ 0.,  0.,  1.]])

    """
    m00, m01, trans_x = mtx[0, 0], mtx[0, 1], mtx[0, 2]
    m10, m11, trans_y = mtx[1, 0], mtx[1, 1], mtx[1, 2]
    det = m00 * m11 - m01 * m10
    # Match the torch path (matrix.inv3x3), which raises for near-singular
    # matrices at float32 eps — without this guard the division silently
    # produces inf/NaN that propagates into cv2.warpAffine.
    threshold = _singularity_threshold(torch.float32)
    if abs(det) < threshold:
        msg = f"Singular affine matrix cannot be inverted (|det|={abs(det):.3e} < {threshold:.3e})."
        raise ValueError(msg)
    inv_det = 1.0 / det
    return np.array(
        [
            [m11 * inv_det, -m01 * inv_det, (m01 * trans_y - m11 * trans_x) * inv_det],
            [-m10 * inv_det, m00 * inv_det, (m10 * trans_x - m00 * trans_y) * inv_det],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class AlbuFusedAffineSegment(nn.Module):
    """Fused affine segment for the Albumentations cv2 backend.

    Loops over B samples, composes per-sample ``(3, 3)`` forward affine matrices, and applies a single
    ``cv2.warpAffine`` call per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects. Conversion to/from ``(H, W, C)``
    NumPy arrays happens inside ``forward()``.

    No ``normalize_matrix`` step is needed — ``cv2.warpAffine`` operates in pixel coordinates natively. The
    accumulated forward (src->dst) matrix is inverted once per sample and passed to :func:`_warp` via
    ``cv2.WARP_INVERSE_MAP``.

    Args:
        transforms: List of Albumentations transform objects to fuse.
        adapter: An ``AlbumentationsAdapter`` providing ``sample_params``,
            ``build_matrix``, and category lookup.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to ``"zeros"``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from fuse_augmentations.affine.segment import AlbuFusedAffineSegment
        >>> from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
        >>> seg = AlbuFusedAffineSegment([], AlbumentationsAdapter())
        >>> out = seg(torch.zeros(1, 3, 8, 8))
        >>> out.shape
        torch.Size([1, 3, 8, 8])

    """

    # Pre-classified dispatch tags for forward_numpy fast path.
    _TAG_INTERP: int = 0
    _TAG_HFLIP: int = 1
    _TAG_VFLIP: int = 2
    _TAG_ADAPTER: int = 3  # fallback: use adapter round-trip
    _TAG_FAST_ROTATE: int = 4  # A.Rotate fast path (direct numpy, bypasses albu gpdd)

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        execution: ExecutionStr = "cv2",
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize ``AlbuFusedAffineSegment``."""
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self.execution: ExecutionStr = _validate_execution(execution)
        self.mask_interpolation = mask_interpolation
        self._last_matrix: Tensor | None = None
        # Pre-compute cv2 flags once instead of dict-lookups per call.
        self._interp_flag: int = _CV2_INTERP.get(self.interpolation, _CV2_INTERP.get("bilinear", 1))
        self._border_flag: int = _CV2_BORDER.get(self.padding_mode, _CV2_BORDER.get("zeros", 0))
        # Pre-classify transforms to avoid per-call _is_albu_instance dispatch.
        self._tfm_tags: list[int] = self._classify_transforms(transforms, adapter)
        # Pre-allocated identity (1,3,3) — reused for zero/single-transform early returns.
        self._identity_1x3x3: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        # Pre-allocated (1,3,3) buffer for forward_numpy last_matrix writes.
        # Avoids per-call tensor allocation from torch.from_numpy(...).unsqueeze(0).
        self._last_matrix_buffer: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        self._last_matrix_np_buffer: NDArray[np.float32] = np.empty((3, 3), dtype=np.float32)
        self._last_matrix_np_tensor: Tensor = torch.from_numpy(self._last_matrix_np_buffer)

    @staticmethod
    def _classify_transforms(transforms: list[object], adapter: TransformAdapter) -> list[int]:
        """Classify each transform for dispatch in ``forward_numpy``.

        Returns a list of integer tags (one per transform) enabling O(1) dispatch in the hot loop instead of O(n)
        ``isinstance`` chains.

        """
        tags: list[int] = []
        try:
            from fuse_augmentations.adapters.albumentations import (
                _HFLIP_TYPES,
                _INTERP_TYPES,
                _VFLIP_TYPES,
                _is_albu_instance,
            )
        except ImportError:
            return [AlbuFusedAffineSegment._TAG_ADAPTER] * len(transforms)

        try:
            from albumentations import Rotate as _AlbuRotate

            _rotate_type: type | None = _AlbuRotate
        except ImportError:
            _rotate_type = None

        for tfm in transforms:
            if _rotate_type is not None and isinstance(tfm, _rotate_type) and not getattr(tfm, "crop_border", True):
                tags.append(AlbuFusedAffineSegment._TAG_FAST_ROTATE)
            elif _is_albu_instance(tfm, _INTERP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_INTERP)
            elif _is_albu_instance(tfm, _HFLIP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_HFLIP)
            elif _is_albu_instance(tfm, _VFLIP_TYPES):
                tags.append(AlbuFusedAffineSegment._TAG_VFLIP)
            else:
                tags.append(AlbuFusedAffineSegment._TAG_ADAPTER)
        return tags

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The composed forward matrix (detached clone), or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply fused affine chain via per-sample cv2.warpAffine.

        Args:
            image: ``(B, C, H, W)`` float32 input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image. Masks are resampled through a grid built from the same
                composed matrix used for the image; bounding boxes and keypoints
                are transformed through the composed forward pixel matrix. The
                coordinate convention (center/``align_corners=True``) matches the
                torch affine path exactly.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size = image.shape[0]

        # Compose the per-sample forward matrices with Albumentations' own
        # per-sample sampling (numpy RNG stream). This is identical for both
        # execution strategies — only the warp backend below differs — so the
        # sampled geometry is byte-for-byte the same whether the batch is warped
        # by cv2 (default) or a batched grid_sample.
        accs, any_active = self._compose_matrices(image)
        composed_batch = self._stack_matrices(accs)
        self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()
        _set_current_call_matrix(composed_batch.to(device=image.device, dtype=image.dtype).detach().clone())

        if batch_size == 0 or len(self.transforms) == 0:
            return (image, aux_targets) if _has_aux else image

        if self.execution == "torch":
            image = self._warp_torch(image, composed_batch)
        else:
            image = self._warp_cv2(image, accs, any_active)

        if aux_targets:
            self._route_aux(aux_targets, composed_batch, image)
        return (image, aux_targets) if _has_aux else image

    def _route_aux(self, aux_targets: dict[str, Tensor], composed_batch: Tensor, image: Tensor) -> None:
        """Route auxiliary targets through the composed forward pixel matrix.

        Masks are resampled with a grid built from the same composed matrix used
        for the image warp; boxes and keypoints go through the composed forward
        pixel matrix. This reuses the shared :mod:`~fuse_augmentations.targets`
        builders so the numpy/cv2 path matches the torch affine path convention
        (center/``align_corners=True``). Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform (``"mask"``, ``"bbox_xyxy"``,
                ``"bbox_xywh"``, ``"keypoints"``).
            composed_batch: ``(B, 3, 3)`` composed forward matrix (CPU float64).
            image: The warped image tensor, used to resolve the mask device/dtype.

        """
        acc_img = composed_batch.to(device=image.device, dtype=image.dtype)
        grid: Tensor | None = None
        if "mask" in aux_targets:
            mask = aux_targets["mask"]
            acc_mask = composed_batch.to(device=mask.device, dtype=torch.float32)
            mtx_inv = inv3x3(acc_mask)
            mtx_norm = normalize_matrix(mtx_inv, mask.shape[-2], mask.shape[-1]).to(dtype=torch.float32)
            grid = F.affine_grid(
                mtx_norm[:, :2, :],
                [mask.shape[0], mask.shape[1], mask.shape[-2], mask.shape[-1]],
                align_corners=True,
            )
        self._route_grid_aux(aux_targets, grid, acc_img, self.mask_interpolation)

    @staticmethod
    def _route_grid_aux(
        aux_targets: dict[str, Tensor],
        grid: Tensor | None,
        acc_img: Tensor,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Route auxiliary targets through the warp grid and composed pixel matrix.

        Mask entries use ``grid`` (nearest-neighbour resample); boxes and keypoints
        use the composed forward pixel matrix ``acc_img``. Mutates ``aux_targets``
        in place.

        Args:
            aux_targets: Auxiliary targets to transform.
            grid: Sampling grid for the mask, or ``None`` when no mask is present.
            acc_img: ``(B, 3, 3)`` composed forward pixel matrix in the image dtype.
            mask_interpolation: Mask sampling mode for the ``"mask"`` target.

        """
        from fuse_augmentations.targets import (
            transform_bbox_xywh,
            transform_bbox_xyxy,
            transform_keypoints,
            transform_mask,
        )

        for key in list(aux_targets.keys()):
            val = aux_targets[key]
            if key == "mask" and grid is not None:
                aux_targets[key] = transform_mask(val, grid, mode=mask_interpolation)
            elif key == "bbox_xyxy":
                aux_targets[key] = transform_bbox_xyxy(val, acc_img)
            elif key == "bbox_xywh":
                aux_targets[key] = transform_bbox_xywh(val, acc_img)
            elif key == "keypoints":
                aux_targets[key] = transform_keypoints(val, acc_img)

    def _compose_matrices(self, image: Tensor) -> tuple[list[MatrixArray], list[bool]]:
        """Compose the per-sample forward affine matrices via Albumentations sampling.

        Runs the exact per-sample activation draws and ``sample_params`` calls that
        the cv2 path has always used, so the numpy RNG stream is unchanged. The
        result feeds either the cv2 or the batched-torch warp.

        Args:
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            A ``(accs, any_active)`` pair: ``accs`` is a per-sample list of
            ``(3, 3)`` float64 forward matrices; ``any_active[b]`` is ``True`` when
            at least one transform applied to sample ``b`` (identity otherwise).

        """
        batch_size, num_channels, height, width = image.shape

        # Pre-draw per-transform active masks before the sample loop so that
        # same_on_batch=True collapses to a single Bernoulli draw shared across all samples.
        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(batch_size, draw))
            else:
                active_masks.append(np.random.rand(batch_size) < prob)

        # same_on_batch transforms share ONE param draw across the whole batch —
        # matching the shared activation draw above. Sample once here so every
        # sample gets identical geometry (previously only the activation was
        # shared while params were re-drawn per sample).
        shared_mtx: dict[int, MatrixArray] = {}
        for t_idx, tfm in enumerate(self.transforms):
            if bool(getattr(tfm, "same_on_batch", False)) and bool(np.any(active_masks[t_idx])):
                params = self.adapter.sample_params(tfm, (1, num_channels, height, width), torch.device("cpu"))
                mtx_shared = self.adapter.build_matrix(tfm, params, height, width)
                shared_mtx[t_idx] = mtx_shared[0].double().cpu().numpy()

        accs: list[MatrixArray] = []
        any_active: list[bool] = []
        for b_idx in range(batch_size):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            active = False
            for t_idx, tfm in enumerate(self.transforms):
                # Skip BEFORE sampling (matching forward_numpy): inactive transforms
                # must not consume RNG draws, otherwise entry points diverge under a
                # fixed seed for any prob < 1.0 chain.
                if not active_masks[t_idx][b_idx]:
                    continue
                if t_idx in shared_mtx:
                    active = True
                    acc = shared_mtx[t_idx] @ acc
                    continue
                params = self.adapter.sample_params(tfm, (1, num_channels, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                active = True
                acc = mtx_i[0].double().cpu().numpy() @ acc
            accs.append(acc)
            any_active.append(active)
        return accs, any_active

    @staticmethod
    def _stack_matrices(accs: list[MatrixArray]) -> Tensor:
        """Stack per-sample ``(3, 3)`` numpy matrices into a CPU ``(B, 3, 3)`` float64 tensor.

        The batch is built on CPU because the source matrices are numpy (CPU) and
        MPS has no float64 support; callers move and cast it as needed (the torch
        warp path casts to a device-safe dtype before touching the accelerator).

        Args:
            accs: Per-sample forward matrices from :meth:`_compose_matrices`.

        Returns:
            A CPU ``(len(accs), 3, 3)`` float64 tensor, or a ``(0, 3, 3)`` tensor
            when ``accs`` is empty.

        """
        if not accs:
            return torch.empty((0, 3, 3), dtype=torch.float64)
        return torch.as_tensor(np.stack(accs), dtype=torch.float64)

    def _warp_cv2(self, image: Tensor, accs: list[MatrixArray], any_active: list[bool]) -> Tensor:
        """Warp each sample with one ``cv2.warpAffine`` (default CPU strategy).

        Args:
            image: ``(B, C, H, W)`` input tensor.
            accs: Per-sample forward matrices from :meth:`_compose_matrices`.
            any_active: Per-sample activity flags; inactive samples pass through untouched.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype
        interp_flag = self._interp_flag
        border_flag = self._border_flag

        output_np: list[ImageArray] = []
        for b_idx in range(batch_size):
            img_np = image[b_idx].permute(1, 2, 0).cpu().numpy()
            if not any_active[b_idx]:
                output_np.append(img_np)
                continue
            # acc is the composed forward (src->dst) matrix; invert to get dst->src for _warp
            m_dst2src = np.linalg.inv(accs[b_idx])
            if num_channels == 1:
                img_np = img_np[:, :, 0]
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag)
            output_np.append(warped)

        return torch.stack([
            torch.as_tensor(np.ascontiguousarray(img).copy()).permute(2, 0, 1) for img in output_np
        ]).to(device=device, dtype=dtype)

    def _warp_torch(self, image: Tensor, composed_batch: Tensor) -> Tensor:
        """Warp the whole batch with one ``grid_sample`` (opt-in torch strategy).

        Applies a single batched ``affine_grid`` + ``grid_sample`` using the
        matrices already composed by :meth:`_compose_matrices`, so the sampled
        geometry matches the cv2 path exactly; only the resampling backend differs.
        Inactive samples keep identity and pass through the near-identity warp.

        Args:
            image: ``(B, C, H, W)`` input tensor (any device — this path is the GPU/MPS warp).
            composed_batch: ``(B, 3, 3)`` float64 forward matrices (built on CPU).

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        # MPS has no float64; invert/normalize in float32 there (mirrors the torch twins).
        acc_dtype = _matrix_compose_dtype(image.dtype, image.device, len(self.transforms))
        acc = composed_batch.to(device=image.device, dtype=acc_dtype)
        warped, _ = _grid_sample_affine_batched(image, acc, self.interpolation, self.padding_mode)
        return warped

    def forward_numpy(self, img_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply fused affine chain to a single HWC NumPy image (no tensor conversion).

        Reuses the same matrix composition logic as :meth:`forward` but operates
        entirely in NumPy/cv2 space, eliminating the BCHW tensor round-trip for
        the Albumentations native dict-input calling convention.

        Args:
            img_hwc: ``(height, width, channels)`` or ``(height, width)`` NumPy array (uint8 or float32).
                cv2 requires a C-contiguous array; a copy is made automatically
                if the input is not contiguous.

        Returns:
            Warped array with the same dtype and shape as ``img_hwc``.

        Note:
            ``_last_matrix`` is set to shape ``(1, 3, 3)`` after this call,
            matching the B=1 single-image case.  ``aux_targets`` are not
            supported; a ``RuntimeError`` is raised if aux routing is attempted
            via this path.

        Examples:
            >>> import numpy as np
            >>> from fuse_augmentations.affine.segment import AlbuFusedAffineSegment
            >>> from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter
            >>> seg = AlbuFusedAffineSegment([], AlbumentationsAdapter())
            >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
            >>> out = seg.forward_numpy(img)
            >>> out.shape
            (8, 8, 3)

        """
        if not img_hwc.flags["C_CONTIGUOUS"]:
            img_hwc = np.ascontiguousarray(img_hwc)
        height, width = img_hwc.shape[:2]
        n_ch = img_hwc.shape[2] if img_hwc.ndim == 3 else 1
        original_2d = img_hwc.ndim == 2

        if len(self.transforms) == 0:
            self._last_matrix = self._identity_1x3x3
            _set_current_call_matrix(self._identity_1x3x3.detach().clone())
            return img_hwc

        # Draw per-transform active masks for bsz=1 (mirrors forward() logic).
        # For prob=1.0 transforms, skip the RNG draw and use a constant True.
        active_masks: list[bool] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            if prob >= 1.0:
                active_masks.append(True)
            elif prob <= 0.0:
                active_masks.append(False)
            else:
                active_masks.append(bool(np.random.rand() < prob))

        acc: MatrixArray = np.eye(3, dtype=np.float64)
        any_active = False

        # Resolve imports once (cached by Python import system, but avoids
        # per-iteration dict lookups inside the hot loop).
        from fuse_augmentations.adapters.albumentations import (
            _sample_matrices as _sample_matrices_fn,
        )
        from fuse_augmentations.adapters.albumentations import (
            hflip_matrix_np as _hflip_matrix_np_fn,
        )
        from fuse_augmentations.adapters.albumentations import (
            vflip_matrix_np as _vflip_matrix_np_fn,
        )

        # Fast numpy-only matrix loop: bypass the adapter's torch round-trip
        # by dispatching on pre-classified tags from __init__.
        _tags = self._tfm_tags
        for idx_tfm, tfm in enumerate(self.transforms):
            if not active_masks[idx_tfm]:
                # Skip expensive sample_params + build_matrix for inactive transforms.
                continue
            tag = _tags[idx_tfm]
            if tag == self._TAG_FAST_ROTATE:
                # Ultra-fast path for A.Rotate (crop_border=False):
                # Call tfm.py_random.uniform directly (same Python Random instance as
                # albumentations) then build the rotation matrix in pure Python/numpy.
                # Identical output to albumentations; saves ~19µs vs get_params_dependent_on_data.
                angle = tfm.py_random.uniform(*tfm.limit)  # type: ignore[attr-defined]
                _rad = math.radians(angle)
                # NOTE: rows below are the TRANSPOSE of matrix.rotation_matrix —
                # deliberate, mirroring Albumentations' clockwise-positive angle
                # convention. Pinned by the A.Rotate parity tests in
                # tests/test_integration/adapters/test_albument.py; keep in sync
                # with the adapter convention if either changes.
                _cos, _sin = math.cos(_rad), math.sin(_rad)
                _center_x = width / 2.0 - 0.5
                _center_y = height / 2.0 - 0.5
                mtx_np = np.array(
                    [
                        [_cos, _sin, _center_x * (1.0 - _cos) - _center_y * _sin],
                        [-_sin, _cos, _center_y * (1.0 - _cos) + _center_x * _sin],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                any_active = True
                acc = mtx_np @ acc
            elif tag == self._TAG_INTERP:
                # Direct numpy: call _sample_matrices (returns (1,3,3) float64 ndarray)
                # without the numpy -> torch.tensor -> .numpy() adapter round-trip.
                mtx_np = _sample_matrices_fn(tfm, 1, height, width)
                any_active = True
                acc = mtx_np[0] @ acc
            elif tag == self._TAG_HFLIP:
                any_active = True
                acc = _hflip_matrix_np_fn(width=width) @ acc
            elif tag == self._TAG_VFLIP:
                any_active = True
                acc = _vflip_matrix_np_fn(height=height) @ acc
            else:
                # Fallback: full adapter round-trip for unrecognised types.
                params = self.adapter.sample_params(tfm, (1, n_ch, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                any_active = True
                acc = mtx_i[0].double().cpu().numpy() @ acc

        np.copyto(self._last_matrix_np_buffer, acc, casting="unsafe")
        self._last_matrix_buffer[0].copy_(self._last_matrix_np_tensor)
        self._last_matrix = self._last_matrix_buffer
        _set_current_call_matrix(torch.from_numpy(np.asarray(acc, dtype=np.float32)).unsqueeze(0).clone())

        if not any_active:
            return img_hwc

        tol = 1e-6
        is_bottom_row = (
            abs(float(acc[2, 0])) < tol and abs(float(acc[2, 1])) < tol and abs(float(acc[2, 2]) - 1.0) < tol
        )
        is_no_shear = abs(float(acc[0, 1])) < tol and abs(float(acc[1, 0])) < tol
        if is_bottom_row and is_no_shear:
            is_hflip = (
                abs(float(acc[0, 0]) + 1.0) < tol
                and abs(float(acc[1, 1]) - 1.0) < tol
                and abs(float(acc[0, 2]) - float(width - 1)) < tol
                and abs(float(acc[1, 2])) < tol
            )
            if is_hflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=1))

            is_vflip = (
                abs(float(acc[0, 0]) - 1.0) < tol
                and abs(float(acc[1, 1]) + 1.0) < tol
                and abs(float(acc[0, 2])) < tol
                and abs(float(acc[1, 2]) - float(height - 1)) < tol
            )
            if is_vflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=0))

            is_hvflip = (
                abs(float(acc[0, 0]) + 1.0) < tol
                and abs(float(acc[1, 1]) + 1.0) < tol
                and abs(float(acc[0, 2]) - float(width - 1)) < tol
                and abs(float(acc[1, 2]) - float(height - 1)) < tol
            )
            if is_hvflip:
                return np.ascontiguousarray(np.flip(img_hwc, axis=(0, 1)))

        m_dst2src: MatrixArray = _inv3x3_affine_np(acc)

        if original_2d:
            return _warp(img_hwc, m_dst2src, width, height, self._interp_flag, self._border_flag)
        if n_ch == 1:
            warped = _warp(img_hwc[:, :, 0], m_dst2src, width, height, self._interp_flag, self._border_flag)
            return warped[:, :, np.newaxis]
        return _warp(img_hwc, m_dst2src, width, height, self._interp_flag, self._border_flag)


# ---------------------------------------------------------------------------
# ProjectiveSegment — PyTorch backend for perspective transforms
# ---------------------------------------------------------------------------


class ProjectiveSegment(_BaseAffineSegment):
    """Fused projective segment that composes homography matrices into one grid_sample call.

    Identical to :class:`FusedAffineSegment` in accumulation and auxiliary-target
    handling -- both share the :class:`_BaseAffineSegment` composition engine --
    but overrides :meth:`_apply_grid` to use
    :func:`~fuse_augmentations.affine.matrix.perspective_grid` instead of
    ``F.affine_grid`` so the full ``3x3`` homography (including perspective
    division) is applied correctly.

    Args:
        transforms: List of projective transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to canonical parameters and matrices.
        interpolation: Optional interpolation mode override (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults
            to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to
            ``"zeros"`` when ``None``.

    """

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused projective transform chain via a single grid_sample call.

        Args:
            image: ``(B, C, H, W)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``,
                ``"keypoints"``). When ``None``, returns a bare tensor for
                backward compatibility.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        acc, acc_img = self._compose(image)
        self._last_matrix = acc_img.detach().clone()
        _set_current_call_matrix(acc_img.detach().clone())

        image, grid = self._apply_grid(image, acc)

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            self._route_grid_aux(aux_targets, grid, acc_img, self.mask_interpolation)

        if not _has_aux:
            return image
        return image, aux_targets

    def _apply_grid(self, image: Tensor, acc: Tensor) -> tuple[Tensor, Tensor]:
        """Warp ``image`` with a single ``perspective_grid`` + ``grid_sample`` pass.

        Inverts the composed forward homography, normalizes it to the ``[-1, 1]``
        grid convention, builds a perspective grid (with the perspective division
        ``F.affine_grid`` cannot express), and resamples. The grid is returned so
        the caller can reuse it to warp mask auxiliary targets.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor.
            acc: ``(batch_size, 3, 3)`` composed forward homography in the compose dtype.

        Returns:
            A ``(warped_image, grid)`` tuple.

        """
        warp_fn = self._select_warp_fn("perspective", image.device)
        return warp_fn(
            image,
            acc,
            self.interpolation or "bilinear",
            self.padding_mode or "zeros",
        )


# ---------------------------------------------------------------------------
# AlbuProjectiveSegment — cv2 backend for Albumentations perspective transforms
# ---------------------------------------------------------------------------


class AlbuProjectiveSegment(nn.Module):
    """Fused projective segment for NumPy/cv2 backends (Albumentations).

    Loops over B samples, composes per-sample ``(3, 3)`` forward homography
    matrices, and applies a single ``cv2.warpPerspective`` per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects.
    Conversion to/from ``(H, W, C)`` NumPy arrays happens inside ``forward()``.

    Args:
        transforms: List of Albumentations perspective transform objects to fuse.
        adapter: An ``AlbumentationsAdapter`` providing ``sample_params``,
            ``build_matrix``, and category lookup.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``). Defaults to ``"zeros"``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        execution: ExecutionStr = "cv2",
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize ``AlbuProjectiveSegment``."""
        super().__init__()
        self.execution: ExecutionStr = _validate_execution(execution)
        # cv2 is only required for the default cv2 warp strategy; the torch
        # strategy warps with grid_sample and needs no OpenCV.
        if _cv2 is None and self.execution == "cv2":
            raise ImportError(
                "AlbuProjectiveSegment requires opencv-python because it uses cv2.warpPerspective under the hood."
            )
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self.mask_interpolation = mask_interpolation
        self._last_matrix: Tensor | None = None
        self._interp_flag: int = _CV2_INTERP.get(self.interpolation, 1)
        self._border_flag: int = _CV2_BORDER.get(self.padding_mode, 0)

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The composed forward matrix (detached clone), or ``None`` before the first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply fused projective chain via per-sample cv2.warpPerspective.

        Args:
            image: ``(B, C, H, W)`` float32 input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside
                the image. Masks are resampled through a perspective grid built
                from the same composed homography used for the image; bounding
                boxes and keypoints are transformed through the composed forward
                homography (with perspective division). The coordinate convention
                matches the torch projective path.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        if aux_targets is None:
            aux_targets = {}

        batch_size = image.shape[0]

        # Compose per-sample forward homographies with Albumentations' own
        # per-sample sampling (numpy RNG stream) — identical for both execution
        # strategies; only the warp backend differs.
        accs, any_active = self._compose_matrices(image)
        composed_batch = self._stack_matrices(accs)
        self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()
        _set_current_call_matrix(composed_batch.to(device=image.device, dtype=image.dtype).detach().clone())

        if batch_size == 0 or len(self.transforms) == 0:
            return (image, aux_targets) if _has_aux else image

        if self.execution == "torch":
            image = self._warp_torch(image, composed_batch)
        else:
            image = self._warp_cv2(image, accs, any_active)

        if aux_targets:
            self._route_aux(aux_targets, composed_batch, image)
        return (image, aux_targets) if _has_aux else image

    def _route_aux(self, aux_targets: dict[str, Tensor], composed_batch: Tensor, image: Tensor) -> None:
        """Route auxiliary targets through the composed forward homography.

        Masks are resampled with a perspective grid built from the same composed
        homography used for the image warp; boxes and keypoints go through the
        composed forward homography. Reuses the shared
        :mod:`~fuse_augmentations.targets` builders so the numpy/cv2 path matches
        the torch projective path convention. Mutates ``aux_targets`` in place.

        Args:
            aux_targets: Auxiliary targets to transform.
            composed_batch: ``(B, 3, 3)`` composed forward homography (CPU float64).
            image: The warped image tensor, used to resolve the box/keypoint dtype.

        """
        acc_img = composed_batch.to(device=image.device, dtype=image.dtype)
        grid: Tensor | None = None
        if "mask" in aux_targets:
            mask = aux_targets["mask"]
            acc_mask = composed_batch.to(device=mask.device, dtype=torch.float32)
            mtx_inv = inv3x3(acc_mask)
            mtx_norm = normalize_matrix(mtx_inv, mask.shape[-2], mask.shape[-1]).to(dtype=torch.float32)
            grid = perspective_grid(mtx_norm, mask.shape[-2], mask.shape[-1])
        AlbuFusedAffineSegment._route_grid_aux(aux_targets, grid, acc_img, self.mask_interpolation)

    def _compose_matrices(self, image: Tensor) -> tuple[list[MatrixArray], list[bool]]:
        """Compose per-sample forward homographies via Albumentations sampling.

        Runs the exact per-sample activation draws and ``sample_params`` calls the
        cv2 path has always used, so the numpy RNG stream is unchanged.

        Args:
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            A ``(accs, any_active)`` pair: ``accs`` is a per-sample list of
            ``(3, 3)`` float64 forward homographies; ``any_active[b]`` is ``True``
            when at least one transform applied to sample ``b``.

        """
        batch_size, num_channels, height, width = image.shape

        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = _transform_prob(tfm)
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(batch_size, draw))
            else:
                active_masks.append(np.random.rand(batch_size) < prob)

        # same_on_batch transforms share ONE param draw across the whole batch,
        # matching the shared activation draw above (mirrors AlbuFusedAffineSegment).
        shared_mtx: dict[int, MatrixArray] = {}
        for t_idx, tfm in enumerate(self.transforms):
            if bool(getattr(tfm, "same_on_batch", False)) and bool(np.any(active_masks[t_idx])):
                params = self.adapter.sample_params(tfm, (1, num_channels, height, width), torch.device("cpu"))
                mtx_shared = self.adapter.build_matrix(tfm, params, height, width)
                shared_mtx[t_idx] = mtx_shared[0].double().cpu().numpy()

        accs: list[MatrixArray] = []
        any_active: list[bool] = []
        for b_idx in range(batch_size):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            active = False
            for t_idx, tfm in enumerate(self.transforms):
                # Skip BEFORE sampling (matching forward_numpy): inactive transforms
                # must not consume RNG draws, otherwise entry points diverge under a
                # fixed seed for any prob < 1.0 chain.
                if not active_masks[t_idx][b_idx]:
                    continue
                if t_idx in shared_mtx:
                    active = True
                    acc = shared_mtx[t_idx] @ acc
                    continue
                params = self.adapter.sample_params(tfm, (1, num_channels, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                active = True
                acc = mtx_i[0].double().cpu().numpy() @ acc
            accs.append(acc)
            any_active.append(active)
        return accs, any_active

    @staticmethod
    def _stack_matrices(accs: list[MatrixArray]) -> Tensor:
        """Stack per-sample ``(3, 3)`` numpy homographies into a CPU ``(B, 3, 3)`` float64 tensor.

        Built on CPU (numpy source, and MPS has no float64); the torch warp path
        casts to a device-safe dtype and moves to the accelerator.

        Args:
            accs: Per-sample forward homographies from :meth:`_compose_matrices`.

        Returns:
            A CPU ``(len(accs), 3, 3)`` float64 tensor, or a ``(0, 3, 3)`` tensor
            when ``accs`` is empty.

        """
        if not accs:
            return torch.empty((0, 3, 3), dtype=torch.float64)
        return torch.as_tensor(np.stack(accs), dtype=torch.float64)

    def _warp_cv2(self, image: Tensor, accs: list[MatrixArray], any_active: list[bool]) -> Tensor:
        """Warp each sample with one ``cv2.warpPerspective`` (default CPU strategy).

        Args:
            image: ``(B, C, H, W)`` input tensor.
            accs: Per-sample forward homographies from :meth:`_compose_matrices`.
            any_active: Per-sample activity flags; inactive samples pass through untouched.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        batch_size, _, height, width = image.shape
        device = image.device
        dtype = image.dtype
        cv2_interp = self._interp_flag
        cv2_border = self._border_flag

        output_np: list[ImageArray] = []
        for b_idx in range(batch_size):
            img_np = image[b_idx].permute(1, 2, 0).cpu().numpy()
            if not any_active[b_idx]:
                output_np.append(img_np)
                continue
            # acc is the composed forward (src->dst) matrix; invert to get dst->src
            mtx_inv = np.linalg.inv(accs[b_idx])
            warped: ImageArray = _cv2.warpPerspective(
                img_np,
                mtx_inv,  # dst->src inverse map
                (width, height),  # dsize = (W, H)
                flags=cv2_interp | _CV2_WARP_INVERSE_MAP,
                borderMode=cv2_border,
                borderValue=(0,),
            )
            if warped.ndim == 2:
                warped = warped[..., None]
            output_np.append(warped)

        return torch.stack([
            torch.as_tensor(np.ascontiguousarray(img).copy()).permute(2, 0, 1) for img in output_np
        ]).to(device=device, dtype=dtype)

    def _warp_torch(self, image: Tensor, composed_batch: Tensor) -> Tensor:
        """Warp the whole batch with one perspective ``grid_sample`` (opt-in torch strategy).

        Uses the homographies already composed by :meth:`_compose_matrices`, so the
        sampled geometry matches the cv2 path exactly; only the resampling backend
        differs. Inactive samples keep identity and pass through the near-identity warp.

        Args:
            image: ``(B, C, H, W)`` input tensor (any device — this path is the GPU/MPS warp).
            composed_batch: ``(B, 3, 3)`` float64 forward homographies on the image device.

        Returns:
            The warped ``(B, C, H, W)`` tensor on the input device and dtype.

        """
        acc_dtype = _matrix_compose_dtype(image.dtype, image.device, len(self.transforms))
        acc = composed_batch.to(device=image.device, dtype=acc_dtype)
        warped, _ = _grid_sample_perspective_batched(image, acc, self.interpolation, self.padding_mode)
        return warped


class FusedColorSegment(nn.Module):
    """Fused colour-space segment that composes POINTWISE_LINEAR transforms into one matrix multiply.

    Accumulates per-sample ``(B, 4, 4)`` homogeneous colour-space affine matrices for every transform in the
    segment, multiplies all pixels by the composed matrix, and clamps the result to ``[0, 1]``.  All operations are
    vectorised over the batch dimension.

    Colour transforms do **not** affect spatial layout, so auxiliary targets (masks, bounding boxes, keypoints) are
    returned unchanged.

    Args:
        transforms: List of ``POINTWISE_LINEAR`` transform objects to fuse.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_color_matrix`` for each transform.
        clip_output: When ``True`` (default), the fused output is clamped to ``[0, 1]`` after the matrix multiply,
            matching the typical behaviour of individual colour transforms.  Set to ``False`` only when the pipeline
            intentionally produces values outside this range (e.g. transforms configured with ``clip_output=False``
            in the underlying library).

    """

    # Buffer — declared here so mypy resolves self._eye4 as Tensor, not Tensor | Module.
    _eye4: Tensor

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        clip_output: bool = True,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        clip_policy: ClipPolicyStr = "final",
    ) -> None:
        """Initialize ``FusedColorSegment``.

        Args:
            transforms: ``POINTWISE_LINEAR`` transforms to fuse.
            adapter: Adapter providing ``sample_params`` and ``build_color_matrix``.
            clip_output: Whether to clamp the final output to ``[0, 1]``.
            randomness: Batch randomness policy.
            clip_policy: ``"final"`` (default) fuses the whole chain into one matmul and clamps once;
                ``"per_op_parity"`` clamps at each op whose intermediate could leave ``[0, 1]``,
                matching a native per-op clamped chain.

        """
        super().__init__()
        self._transforms = transforms
        self._adapter = adapter
        self.clip_output = clip_output
        self.randomness = randomness
        if clip_policy not in ("final", "per_op_parity"):
            msg = "unknown clip policy {!r}; expected 'final' or 'per_op_parity'"
            raise ValueError(msg.format(clip_policy))
        self.clip_policy: ClipPolicyStr = clip_policy
        # Register identity matrix as a buffer so device moves (.to(), .cuda())
        # propagate automatically — avoids re-allocating every forward pass.
        self.register_buffer("_eye4", torch.eye(4, dtype=torch.float32))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state; back-compat: add missing fields from older pickles."""
        super().__setstate__(state)  # type: ignore[no-untyped-call]
        if "_eye4" not in self._buffers:
            self.register_buffer("_eye4", torch.eye(4, dtype=torch.float32))
        # clip_output added in v0.7; default True preserves pre-existing behaviour.
        if not hasattr(self, "clip_output"):
            self.clip_output = True
        if not hasattr(self, "randomness"):
            self.randomness = RandomnessPolicy.BACKEND
        # clip_policy added later; default "final" preserves the single-matmul behaviour.
        if not hasattr(self, "clip_policy"):
            self.clip_policy = "final"

    @property
    def transforms(self) -> list[object]:
        """Return the list of transforms in this segment."""
        return list(self._transforms)

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Any] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Apply the fused colour matrix to the image batch.

        Args:
            image: ``(batch_size, channels, height, width)`` float input tensor with values in ``[0, 1]``.
            aux_targets: Optional dict of auxiliary targets. Colour transforms
                do not affect spatial layout, so these are returned unchanged.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        batch_size, channels, height, width = image.shape

        # The 4x4 color matrix is defined for 3-channel RGB images only.
        # For non-RGB inputs, fall back to sequential passthrough application.
        if channels != 3:
            for tfm in self._transforms:
                image = self._adapter.call_nonfused(tfm, image)
            if aux_targets is None:
                return image
            return image, aux_targets

        device = image.device
        dtype = image.dtype
        image_in = image

        # Cast the registered buffer to the current device/dtype (no-op for float32 CPU)
        eye = self._eye4.to(device=device, dtype=dtype)

        input_shape = (batch_size, channels, height, width)

        # Per-channel mean of the image reaching the current transform, (batch_size, channels).
        # Mid-chain contrast ops need the luminance of THEIR input, not the raw segment input;
        # because every fused op is linear (c' = M c + b), the mean transforms exactly as
        # mean(M c + b) = M mean(c) + b, so we carry mean_ch forward through the same matrices.
        mean_ch = image.reshape(batch_size, channels, height * width).mean(dim=2)  # (batch_size, channels)

        matrices: list[Tensor] = []
        for tfm in self._transforms:
            prob = _transform_prob(tfm)
            same_on_batch = _shares_randomness_across_batch(self._adapter, tfm, self.randomness)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.expand(batch_size)
            else:
                active = torch.rand(batch_size, device=device) < prob

            params = _sample_transform_params(self._adapter, tfm, input_shape, device, self.randomness)
            # Contrast-like ops take their midpoint from the per-image luminance of their input;
            # pass it so the fused matrix reproduces the native mean-relative contrast exactly.
            # Only thread `mean` when a mean-relative op actually needs it, so adapters whose
            # build_color_matrix predates the parameter (or custom ones) keep the two-arg call.
            luma_mean = self._prefix_luma_mean(tfm, mean_ch)
            try:
                if luma_mean is None:
                    mat = self._adapter.build_color_matrix(tfm, params)  # (batch_size, 4, 4)
                else:
                    mat = self._adapter.build_color_matrix(tfm, params, mean=luma_mean)  # (batch_size, 4, 4)
            except NotImplementedError:
                # If a transform that passed the build-time probe raises NotImplementedError
                # at forward time (e.g. probe used empty params), abort fusion entirely and
                # restart from the original image so no partial fused state is applied.
                image_fallback = image_in
                for tfm_nonfused in self._transforms:
                    image_fallback = self._adapter.call_nonfused(tfm_nonfused, image_fallback)
                if aux_targets is None:
                    return image_fallback
                return image_fallback, aux_targets

            # Expand to batch if adapter returned (1, 4, 4)
            if mat.shape[0] == 1 and batch_size > 1:
                mat = mat.expand(batch_size, -1, -1)

            mat = mat.to(device=device, dtype=dtype)
            mat = torch.where(active[:, None, None], mat, eye.unsqueeze(0).expand(batch_size, -1, -1))
            matrices.append(mat)
            mean_ch = self._advance_channel_mean(mean_ch, mat)

        image_out = self._apply_color_matrices(image, matrices, eye)

        if self.clip_output:
            image_out = image_out.clamp(0.0, 1.0)
        if aux_targets is None:
            return image_out
        return image_out, aux_targets

    def _apply_color_matrices(self, image: Tensor, matrices: list[Tensor], eye: Tensor) -> Tensor:
        """Apply the per-op color matrices, splitting for clamp parity when requested.

        Under ``clip_policy="final"`` the matrices compose into ONE ``(B, 4, 4)`` matmul and the whole
        chain is applied in a single pass (the more precise, bit-compatible default). Under
        ``"per_op_parity"`` the chain is split at every op whose intermediate provably escapes
        ``[0, 1]``: the fused sub-chain before it is applied and clamped, matching a native per-op
        clamped chain, while safe adjacent ops stay fused.

        Args:
            image: ``(B, 3, H, W)`` input image.
            matrices: Per-op probability-masked ``(B, 4, 4)`` color matrices, in application order.
            eye: A ``(4, 4)`` identity in the image device/dtype.

        Returns:
            The transformed ``(B, 3, H, W)`` image (final clamp applied by the caller).

        """
        batch_size = image.shape[0]
        if not matrices:
            return image
        has_normalize = any(self._is_normalize(transform) for transform in self._transforms)
        if self.clip_policy == "final" and not has_normalize:
            acc = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
            for mat in matrices:
                acc = torch.bmm(mat, acc)
            return self._matmul_image(image, acc)

        # Apply one fused sub-chain at a time. A Normalize boundary first clamps an escaping
        # prefix because native color ops clamp before Normalize, while Normalize itself does not.
        out = image
        sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
        sub_started = False
        for transform, mat in zip(self._transforms, matrices, strict=True):
            if self._is_normalize(transform) and sub_started and self._range_escapes_gamut(sub):
                out = self._matmul_image(out, sub).clamp(0.0, 1.0)
                sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                sub_started = False

            candidate = torch.bmm(mat, sub) if sub_started else mat
            if (
                self.clip_policy == "per_op_parity"
                and not self._is_normalize(transform)
                and self._range_escapes_gamut(candidate)
            ):
                out = self._matmul_image(out, candidate).clamp(0.0, 1.0)
                sub = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
                sub_started = False
                continue
            sub = candidate
            sub_started = True
        return self._matmul_image(out, sub)

    def _is_normalize(self, transform: object) -> bool:
        """Return whether the adapter identifies *transform* as Normalize."""
        checker = getattr(self._adapter, "is_normalize", None)
        return bool(checker(transform)) if callable(checker) else type(transform).__name__ == "Normalize"

    @staticmethod
    def _matmul_image(image: Tensor, acc: Tensor) -> Tensor:
        """Apply a ``(B, 4, 4)`` homogeneous color matrix to a ``(B, 3, H, W)`` image."""
        batch_size, channels, height, width = image.shape
        pixels = image.reshape(batch_size, channels, height * width)
        ones = torch.ones(batch_size, 1, height * width, device=image.device, dtype=image.dtype)
        pixels_hom = torch.cat([pixels, ones], dim=1)  # (B, 4, H*W)
        transformed = torch.bmm(acc, pixels_hom)  # (B, 4, H*W)
        return transformed[:, :channels, :].reshape(batch_size, channels, height, width)

    @staticmethod
    def _range_escapes_gamut(acc: Tensor) -> bool:
        """Return whether a composed color matrix can map a ``[0, 1]`` pixel outside ``[0, 1]``.

        For a per-channel affine ``c' = A c + b`` the reachable output range over inputs in
        ``[0, 1]^3`` is bounded by summing the positive parts (max) and negative parts (min) of each
        row plus the bias. If any channel's bound crosses the gamut for any sample, the fused chain
        must be clamped here to match a native per-op chain.

        Args:
            acc: ``(B, 4, 4)`` composed homogeneous color matrix.

        Returns:
            ``True`` if any sample/channel can leave ``[0, 1]``.

        """
        lin = acc[:, :3, :3]  # (B, 3, 3)
        bias = acc[:, :3, 3]  # (B, 3)
        hi = lin.clamp(min=0.0).sum(dim=2) + bias  # inputs at 1 for positive weights, 0 for negative
        lo = lin.clamp(max=0.0).sum(dim=2) + bias  # inputs at 1 for negative weights, 0 for positive
        eps = 1e-6
        return bool((hi > 1.0 + eps).any().item() or (lo < -eps).any().item())

    def _prefix_luma_mean(self, transform: object, mean_ch: Tensor) -> Tensor | None:
        """Weighted luminance of the transform's input, or ``None`` when it needs no mean.

        Only contrast-like ops with a mean-relative midpoint (``ColorJitter`` contrast) report
        luminance weights via :meth:`TransformAdapter.color_luma_weights`. For those, the per-image
        luminance is the same weighted sum of channel means the native op computes.

        Args:
            transform: The color transform about to be applied.
            mean_ch: Per-channel mean of the image reaching this transform, ``(batch_size, channels)``.

        Returns:
            A ``(batch_size,)`` luminance tensor for mean-relative ops, or ``None`` otherwise.

        """
        # color_luma_weights is an optional adapter capability (Protocol default returns None).
        # Adapters predating it — or lightweight custom ones — simply have no mean-relative op.
        get_weights = getattr(self._adapter, "color_luma_weights", None)
        weights = get_weights(transform) if callable(get_weights) else None
        if weights is None:
            return None
        w = torch.tensor(weights, device=mean_ch.device, dtype=mean_ch.dtype)  # (3,)
        return (mean_ch * w).sum(dim=1)  # (batch_size,)

    @staticmethod
    def _advance_channel_mean(mean_ch: Tensor, mat: Tensor) -> Tensor:
        """Apply a color matrix to the running per-channel mean.

        Because every fused color op is affine (``c' = M c + b``), the mean of the output channels is
        ``M mean(c) + b``. Carrying the mean this way lets a later contrast op read the luminance of the
        image it would actually see, matching a native per-op chain.

        Args:
            mean_ch: Per-channel mean before the op, ``(batch_size, channels)``.
            mat: The op's ``(batch_size, 4, 4)`` homogeneous color matrix (already probability-masked).

        Returns:
            Per-channel mean after the op, ``(batch_size, channels)``.

        """
        channels = mean_ch.shape[1]
        mean_hom = torch.cat([mean_ch, torch.ones_like(mean_ch[:, :1])], dim=1)  # (batch_size, 4)
        advanced = torch.bmm(mat, mean_hom.unsqueeze(2)).squeeze(2)  # (batch_size, 4)
        return advanced[:, :channels]


def _try_build_color_matrix(adapter: TransformAdapter, transform: object) -> bool:
    """Probe whether *adapter* supports ``build_color_matrix`` for *transform*.

    Calls the method with an empty param dict and classifies the outcome:

    - No exception → ``True`` (method succeeds with any params)
    - ``NotImplementedError`` / ``AttributeError`` → ``False`` (explicitly unsupported)
    - ``KeyError`` / ``IndexError`` → ``True`` (method exists, needs real params)
    - Any other exception (``RuntimeError``, etc.) → ``False`` (unexpected error;
      treat as unsupported to avoid silently mis-fusing a broken adapter)

    """
    try:
        adapter.build_color_matrix(transform, {})
        return True
    except (NotImplementedError, AttributeError):
        return False
    except (KeyError, IndexError):
        # Method exists but needs real params to succeed (missing param key).
        return True
    except Exception:
        # Unexpected error (e.g. RuntimeError from GPU OOM, device mismatch).
        # Treat as "not supported" to avoid silently mis-fusing a broken adapter.
        return False


def _flush_color(
    transforms: list[object],
    adapter: TransformAdapter,
    segments: list[object],
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    clip_policy: ClipPolicyStr = "final",
) -> None:
    """Flush a run of ``POINTWISE_LINEAR`` transforms into segments.

    If the adapter supports ``build_color_matrix`` for **every** transform in the run, they are folded into a single
    :class:`FusedColorSegment`. Otherwise the transforms fall back to passthrough (appended as-is). This helper
    intentionally mutates ``transforms`` in-place (clears it).

    Args:
        transforms: The pending run of ``POINTWISE_LINEAR`` transforms (cleared in place).
        adapter: Adapter used to probe and build color matrices.
        segments: Output segment list, appended in place.
        randomness: Batch randomness policy passed to the color segment.
        clip_policy: Clamp policy forwarded to :class:`FusedColorSegment`.

    """
    if not transforms:
        return
    # Probe each transform in the run; any failure means full passthrough.
    for tfm in transforms:
        if not _try_build_color_matrix(adapter, tfm):
            segments.extend(transforms)
            # Intentionally clears the caller-owned run buffer.
            transforms.clear()
            return
    color_transforms = list(transforms)
    clip_output = not any(_is_normalize_transform(adapter, tfm) for tfm in color_transforms)
    segments.append(
        FusedColorSegment(
            color_transforms,
            adapter,
            clip_output=clip_output,
            randomness=randomness,
            clip_policy=clip_policy,
        )
    )
    # Intentionally clears the caller-owned run buffer.
    transforms.clear()


def _is_normalize_transform(adapter: TransformAdapter, transform: object) -> bool:
    """Return whether an adapter marks a transform as pointwise Normalize."""
    checker = getattr(adapter, "is_normalize", None)
    return bool(checker(transform)) if callable(checker) else type(transform).__name__ == "Normalize"


class CropResizeSegment(nn.Module):
    """Segment for a single ``CROP_RESIZE_FIXED`` transform.

    Samples the random crop region, builds the forward affine matrix, normalizes it
    via :func:`~fuse_augmentations.affine.matrix.normalize_matrix_io` (which accounts
    for different input and output spatial dimensions), and applies exactly one
    ``grid_sample`` call at the target ``(H_out, W_out)`` dimensions.

    Unlike :class:`FusedAffineSegment`, the output shape is ``(batch_size, channels, height_out, width_out)``
    which generally differs from the input shape ``(batch_size, channels, height_in, width_in)``.

    .. note::
        Per-sample probability ``prob`` is **not** applied: shape-changing transforms must produce a consistent
        output size for all batch elements, so the crop is always applied.  Use ``prob=1.0`` (the standard default)
        when constructing ``RandomResizedCrop`` transforms.

    .. note::
        Auxiliary targets (``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``) are
        warped through the crop affine matrix at the target output size. Masks use nearest-neighbour
        sampling to preserve integer class labels; boxes and keypoints are transformed via the forward
        affine matrix.

    Args:
        transform: A single ``CROP_RESIZE_FIXED`` transform object.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_matrix`` for the transform.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.
        mask_interpolation: Sampling mode for auxiliary masks. ``"nearest"``
            preserves hard labels; ``"bilinear"`` supports float soft masks.

    """

    def __init__(
        self,
        transform: object,
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
        randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
        *,
        antialias: bool = False,
        mask_interpolation: MaskInterpolationStr = "nearest",
    ) -> None:
        """Initialize ``CropResizeSegment``."""
        super().__init__()
        self.transform = transform
        self.transforms: list[object] = [transform]
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.randomness = randomness
        # Opt-in antialiasing: prefilter/interpolate only when a downscale is
        # aggressive enough to alias (see _ANTIALIAS_SCALE_THRESHOLD). Off by
        # default → output bit-identical to the plain single grid_sample warp.
        self._antialias: bool = antialias
        self.mask_interpolation = mask_interpolation

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the crop-resize via a single ``grid_sample`` call at target output size.

        Args:
            image: ``(batch_size, channels, height_in, width_in)`` float input tensor.
            aux_targets: Optional dict of auxiliary targets to transform alongside the image.
                Supported keys: ``"mask"``, ``"bbox_xyxy"``, ``"bbox_xywh"``, ``"keypoints"``.
                Masks are warped with nearest-neighbour sampling to the target size;
                boxes and keypoints are transformed via the forward affine matrix.

        Returns:
            ``(batch_size, channels, height_out, width_out)`` tensor when ``aux_targets`` is ``None``;
            ``(tensor, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        batch_size, num_channels, height, width = image.shape
        device = image.device
        dtype = image.dtype

        params = _sample_transform_params(
            adapter=self.adapter,
            transform=self.transform,
            input_shape=(batch_size, num_channels, height, width),
            device=device,
            randomness=self.randomness,
        )
        if not (
            torch.all(params["target_h"] == params["target_h"][0])
            and torch.all(params["target_w"] == params["target_w"][0])
        ):
            raise ValueError(
                "CropResizeSegment requires a uniform target size across the batch "
                f"(got target_h={params['target_h'].tolist()}, target_w={params['target_w'].tolist()})"
            )
        target_h = int(params["target_h"][0].item())
        target_w = int(params["target_w"][0].item())

        mtx = self.adapter.build_matrix(self.transform, params, height, width)
        if mtx.shape[0] == 1 and batch_size > 1:
            mtx = mtx.expand(batch_size, -1, -1)
        mtx = mtx.to(device=device, dtype=dtype)

        # Opt-in antialiasing: band-limit the input before the downscaling warp so
        # one grid_sample no longer aliases. No-op unless enabled AND the crop
        # downscales past the threshold, so the default output is unchanged.
        image = _maybe_antialias_prefilter(image, mtx, self._antialias)

        mtx_inv = inv3x3(mtx)
        mtx_norm = normalize_matrix_io(mtx_inv, height, width, target_h, target_w)

        grid = F.affine_grid(
            mtx_norm[:, :2, :],
            [batch_size, num_channels, target_h, target_w],
            align_corners=True,
        )
        out = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

        if aux_targets:
            from fuse_augmentations.targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_keypoints,
                transform_mask,
            )

            for key in list(aux_targets.keys()):
                val = aux_targets[key]
                if key == "mask":
                    aux_targets[key] = transform_mask(val, grid, mode=self.mask_interpolation)
                    continue
                if key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(val, mtx)
                    continue
                if key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(val, mtx)
                    continue
                if key == "keypoints":
                    aux_targets[key] = transform_keypoints(val, mtx)

        if not _has_aux:
            return out
        if aux_targets is None:
            raise RuntimeError("internal error: aux_targets is None in return branch")
        return out, aux_targets


# ---------------------------------------------------------------------------
# Reorder helpers
# ---------------------------------------------------------------------------


def reorder_pointwise(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[object]:
    """Reorder transforms so POINTWISE ops are pushed after geometric chains.

    Walks the transform list left to right.  Within each stretch between
    ``SPATIAL_KERNEL`` barriers, geometric ops (``GEOMETRIC_INTERP`` and
    ``GEOMETRIC_EXACT``) are kept in order, while ``POINTWISE`` ops are
    deferred and flushed after the geometric group.  ``POINTWISE`` ops
    never move across a ``SPATIAL_KERNEL`` barrier.

    Args:
        transforms: List of transform objects to reorder.
        adapter: A ``TransformAdapter`` used for category lookup on each transform.

    Returns:
        New list containing the same transforms, possibly reordered so that
        ``POINTWISE`` ops sit after geometric runs within each
        barrier-bounded stretch.

    Example:
        Given a pipeline ``[Rotate, Brightness, HFlip]`` where ``Brightness``
        is ``POINTWISE`` and ``Rotate`` / ``HFlip`` are geometric, the
        ``Brightness`` is pushed after the geometric group:

        Input order:  ``[Rotate, Brightness, HFlip]``
        Output order: ``[Rotate, HFlip, Brightness]``

        Using stub objects (the KorniaAdapter registry does not include any
        POINTWISE transforms in v0.2):

    >>> from fuse_augmentations.affine.segment import reorder_pointwise
    >>> from fuse_augmentations.types import TransformCategory
    >>> class _StubAdapter:
    ...     def category(self, transform):
    ...         return transform._cat
    ...
    >>> class _TransformStub:
    ...     def __init__(self, cat):
    ...         self._cat = cat
    ...
    >>> adapter = _StubAdapter()
    >>> geo = _TransformStub(TransformCategory.GEOMETRIC_INTERP)
    >>> pw  = _TransformStub(TransformCategory.POINTWISE)
    >>> result = reorder_pointwise([geo, pw, geo], adapter)
    >>> [transform._cat.name for transform in result]
    ['GEOMETRIC_INTERP', 'GEOMETRIC_INTERP', 'POINTWISE']

    """
    geometric = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT, TransformCategory.PROJECTIVE}

    result: list[object] = []
    geo_buf: list[object] = []
    pw_buf: list[object] = []

    def _flush() -> None:
        result.extend(geo_buf)
        result.extend(pw_buf)
        geo_buf.clear()
        pw_buf.clear()

    _reorderable = {TransformCategory.POINTWISE, TransformCategory.POINTWISE_LINEAR}

    for tfm in transforms:
        cat = adapter.category(tfm)
        if cat in _reorderable:
            pw_buf.append(tfm)
            continue
        if cat in geometric:
            geo_buf.append(tfm)
            continue

        # SPATIAL_KERNEL barrier: flush current stretch, then emit the barrier
        _flush()
        result.append(tfm)

    _flush()
    return result


def reorder_aggressive(
    transforms: list[object],
    adapter: TransformAdapter,
) -> list[object]:
    """Reorder transforms aggressively -- bubble-sort POINTWISE ops after geometric chains.

    Applies the POINTWISE reorder algorithm iteratively until the list stabilizes
    (convergence guarantee). ``SPATIAL_KERNEL`` barriers are never crossed.

    For typical pipelines the result is identical to a single :func:`reorder_pointwise`
    pass; the multi-pass variant provides a convergence guarantee for pathological
    orderings.

    Args:
        transforms: List of transform objects to reorder.
        adapter: TransformAdapter for category lookup.

    Returns:
        Reordered list with all POINTWISE ops placed after geometric runs within
        each SPATIAL_KERNEL-bounded stretch.

    """
    current = transforms
    for _ in range(len(transforms)):  # max n iterations
        reordered = reorder_pointwise(current, adapter)
        if len(reordered) == len(current) and all(
            item_a is item_b for item_a, item_b in zip(reordered, current, strict=True)
        ):
            break
        current = reordered
    return current


def build_segments(
    transforms: list[object],
    adapter: TransformAdapter,
    interpolation: InterpolationStr | None = None,
    padding_mode: PaddingModeStr | None = None,
    randomness: RandomnessPolicy = RandomnessPolicy.BACKEND,
    *,
    use_numpy: bool = False,
    route_coords_via_grid: bool = False,
    route_crop_aux: bool = False,
    execution: ExecutionStr = "cv2",
    compile_warp: bool = False,
    antialias: bool = False,
    clip_policy: ClipPolicyStr = "final",
    mask_interpolation: MaskInterpolationStr = "nearest",
) -> list[object]:
    """Split a transform list into fused segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric transforms (``GEOMETRIC_INTERP`` or
    ``GEOMETRIC_EXACT``) into a single segment.  Any ``SPATIAL_KERNEL``, ``POINTWISE``, or ``POINTWISE_LINEAR``
    transform breaks the current geometric group and is returned as-is.

    After grouping, each accumulated geometric run is classified:

    - **EXACT-only** - if the run contains *only* ``GEOMETRIC_EXACT`` ops
      (e.g. flips, 90-degree rotations, transpose-like discrete ops), it
      becomes an :class:`ExactAffineSegment` that uses adapter-provided exact
      image operations with zero interpolation error.
    - **Mixed / INTERP** - if any op in the run is ``GEOMETRIC_INTERP``, the
      whole run becomes a :class:`FusedAffineSegment` that composes matrices
      and applies one ``grid_sample`` call.

    When ``ReorderPolicy.POINTWISE`` is active in
    :class:`~fuse_augmentations.compose.FusedCompose`, ``reorder_pointwise``
    is called first to bubble pointwise ops out of geometric chains, and
    ``build_segments`` then classifies the reordered list.

    Args:
        transforms: List of transform objects (already reordered if a reorder policy applies).
        adapter: A ``TransformAdapter`` for category lookup and matrix building.
        interpolation: Interpolation mode override forwarded to each
            :class:`FusedAffineSegment` (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
        padding_mode: Padding mode override forwarded to each
            :class:`FusedAffineSegment` (``"zeros"``, ``"border"``, ``"reflection"``).
        use_numpy: When ``True``, produce :class:`AlbuFusedAffineSegment` instances
            (Albumentations/scipy backend) instead of the PyTorch
            :class:`FusedAffineSegment`. Used for the Albumentations backend.
        randomness: Batch randomness policy for fused PyTorch segments.
        route_coords_via_grid: When ``True`` (set by the caller when the pipeline
            carries box/keypoint auxiliary targets), route an all-exact geometric run
            through :class:`FusedAffineSegment` instead of :class:`ExactAffineSegment`.
            An all-exact run always composes to a D4 element, so the fused segment's
            exact-dispatch still applies it losslessly while routing boxes/keypoints
            through the composed pixel matrix — avoiding the ``ExactAffineSegment``
            box/keypoint limitation without an interpolation penalty. Torch path only.
        route_crop_aux: When ``True`` (set by the caller when the pipeline carries any
            auxiliary target), emit a :class:`CropResizeSegment` for a ``CROP_RESIZE_FIXED``
            op on the ``use_numpy`` (Albumentations) path instead of an image-only
            passthrough, so masks/boxes/keypoints route to the crop's output size. Without
            it the numpy-path crop resizes the image only, silently desyncing aux targets.
            No effect on the PyTorch backends (they always build ``CropResizeSegment``).
        execution: Execution strategy forwarded to the Albumentations fused segments
            (``use_numpy=True`` path). ``"cv2"`` (default) warps each sample with
            OpenCV; ``"torch"`` opts into one batched ``grid_sample`` for the whole
            batch. Ignored for the PyTorch backends.
        compile_warp: When ``True`` (and torch is new enough), the torch warp core
            (matrix normalize -> ``affine_grid`` -> ``grid_sample``) of each fused
            geometric/projective/crop segment runs under ``torch.compile``. Off by
            default and a no-op on CPU or older torch — the eager output is unchanged.
        antialias: When ``True``, crop-resize segments prefilter the input before an
            aggressive downscale so the single warp does not alias. Off by default and
            a no-op unless the scale drops past the threshold — the output is unchanged.
        clip_policy: Clamp policy forwarded to each :class:`FusedColorSegment`.
            ``"final"`` (default) fuses the color chain into one matmul and clamps once;
            ``"per_op_parity"`` clamps at each op that could leave ``[0, 1]``.
        mask_interpolation: Sampling mode for routed masks. ``"nearest"`` preserves
            the historical hard-label behavior; ``"bilinear"`` supports float soft masks.

    Returns:
        Flat list where each element is a :class:`FusedAffineSegment`
        (mixed/INTERP geometric run), an :class:`ExactAffineSegment`
        (EXACT-only geometric run; auxiliary targets remain flip-only),
        a :class:`CropResizeSegment` (``CROP_RESIZE_FIXED`` op on the
        PyTorch path), a :class:`FusedColorSegment` (``POINTWISE_LINEAR``
        run where the adapter supports ``build_color_matrix``), or the
        original transform object (passthrough for ``SPATIAL_KERNEL``,
        ``POINTWISE``, ``CROP_RESIZE_FIXED`` on the numpy path, and
        unsupported ``POINTWISE_LINEAR`` transforms).

    """
    fusible = {TransformCategory.GEOMETRIC_INTERP, TransformCategory.GEOMETRIC_EXACT}
    projective_cat = TransformCategory.PROJECTIVE
    pointwise_linear_cat = TransformCategory.POINTWISE_LINEAR
    crop_resize_cat = TransformCategory.CROP_RESIZE_FIXED

    segments: list[object] = []
    current_geo: list[object] = []
    current_proj: list[object] = []
    current_color: list[object] = []

    def _flush_geo() -> None:
        if not current_geo:
            return

        has_interp = any(adapter.category(transform) == TransformCategory.GEOMETRIC_INTERP for transform in current_geo)

        if use_numpy:
            # Albumentations path: use AlbuFusedAffineSegment only when interpolation is present;
            # keep ExactAffineSegment for GEOMETRIC_EXACT-only runs to preserve lossless flips and
            # auxiliary-target handling.
            if has_interp:
                segments.append(
                    AlbuFusedAffineSegment(
                        transforms=list(current_geo),
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                        execution=execution,
                        mask_interpolation=mask_interpolation,
                    )
                )
            else:
                segments.append(ExactAffineSegment(list(current_geo), adapter, randomness=randomness))

            current_geo.clear()
            return

        # Route through the grid FusedAffineSegment when the run interpolates, OR when
        # it is all-exact but the pipeline carries box/keypoint aux (ExactAffineSegment
        # cannot route those). An all-exact run always composes to a D4 element, so the
        # fused segment's exact-dispatch keeps it lossless while routing coords through
        # the composed matrix.
        if has_interp or route_coords_via_grid:
            segments.append(
                FusedAffineSegment(
                    transforms=list(current_geo),
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    randomness=randomness,
                    compile_warp=compile_warp,
                    mask_interpolation=mask_interpolation,
                )
            )
            current_geo.clear()
            return

        segments.append(ExactAffineSegment(list(current_geo), adapter, randomness=randomness))
        current_geo.clear()

    def _flush_proj() -> None:
        if not current_proj:
            return
        if use_numpy:
            segments.append(
                AlbuProjectiveSegment(
                    transforms=list(current_proj),
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    execution=execution,
                    mask_interpolation=mask_interpolation,
                )
            )
        else:
            segments.append(
                ProjectiveSegment(
                    transforms=list(current_proj),
                    adapter=adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                    randomness=randomness,
                    compile_warp=compile_warp,
                    mask_interpolation=mask_interpolation,
                )
            )
        current_proj.clear()

    for transform in transforms:
        category = adapter.category(transform)
        if category in fusible:
            _flush_proj()  # flush any pending projective
            _flush_color(current_color, adapter, segments, randomness, clip_policy)  # mutates current_color in-place
            current_geo.append(transform)
            continue
        if category == projective_cat:
            _flush_geo()  # flush any pending affine
            _flush_color(current_color, adapter, segments, randomness, clip_policy)  # mutates current_color in-place
            current_proj.append(transform)
            continue
        if category == pointwise_linear_cat:
            _flush_geo()
            _flush_proj()
            current_color.append(transform)
            continue
        if category == crop_resize_cat:
            # CROP_RESIZE_FIXED. On the torch path, fuse into the immediately
            # preceding fusible geometric run: compose M_crop @ M_geo and
            # apply ONE grid_sample at the crop's target size instead of two.
            # No pending geo run → emit a standalone CropResizeSegment. Projective
            # and color runs still flush (crop only fuses with affine geo runs).
            # The numpy/albumentations path emits the transform as a passthrough,
            # except when the pipeline carries auxiliary targets: a passthrough crop
            # resizes the image only and silently desyncs masks/boxes/keypoints, so a
            # CropResizeSegment (which the Albumentations adapter fully supports for
            # CROP_RESIZE_FIXED) is emitted instead to route aux to the output size.
            _flush_proj()
            _flush_color(current_color, adapter, segments, randomness, clip_policy)  # mutates current_color in-place
            if use_numpy:
                _flush_geo()
                if route_crop_aux:
                    segments.append(
                        CropResizeSegment(
                            transform=transform,
                            adapter=adapter,
                            interpolation=interpolation,
                            padding_mode=padding_mode,
                            randomness=randomness,
                            antialias=antialias,
                            mask_interpolation=mask_interpolation,
                        )
                    )
                else:
                    segments.append(transform)
                continue
            if current_geo:
                segments.append(
                    _FusedGeoCropSegment(
                        geo_transforms=list(current_geo),
                        crop_transform=transform,
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                        randomness=randomness,
                        compile_warp=compile_warp,
                        antialias=antialias,
                        mask_interpolation=mask_interpolation,
                    )
                )
                current_geo.clear()
            else:
                segments.append(
                    CropResizeSegment(
                        transform=transform,
                        adapter=adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                        randomness=randomness,
                        antialias=antialias,
                        mask_interpolation=mask_interpolation,
                    )
                )
            continue
        # SPATIAL_KERNEL / POINTWISE barrier: flush all
        _flush_geo()
        _flush_proj()
        _flush_color(current_color, adapter, segments, randomness, clip_policy)  # mutates current_color in-place
        segments.append(transform)

    _flush_geo()
    _flush_proj()
    _flush_color(current_color, adapter, segments, randomness, clip_policy)  # mutates current_color in-place
    return segments


# ---------------------------------------------------------------------------
# Private helpers for ExactAffineSegment auxiliary-target flipping
# ---------------------------------------------------------------------------


def _flip_bbox_xyxy(
    boxes: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip bounding boxes (batch_size, num_boxes, 4) xyxy format using direct coordinate arithmetic.

    HFlip: ``coord_x' = width - 1 - coord_x``, swap x1/x2.
    VFlip: ``coord_y' = height - 1 - coord_y``, swap y1/y2.

    """
    box_x1, box_y1, box_x2, box_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    if is_hflip:
        new_x1 = (width - 1) - box_x2
        new_x2 = (width - 1) - box_x1
        box_x1, box_x2 = new_x1, new_x2

    if is_vflip:
        new_y1 = (height - 1) - box_y2
        new_y2 = (height - 1) - box_y1
        box_y1, box_y2 = new_y1, new_y2

    flipped = torch.stack([box_x1, box_y1, box_x2, box_y2], dim=-1)

    # active shape: (batch_size,) -> (batch_size, 1, 1) for broadcasting with (batch_size, num_boxes, 4)
    mask = active[:, None, None]
    return torch.where(mask, flipped, boxes)


def _flip_keypoints(
    keypoints: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip keypoints (batch_size, num_points, 2) using direct coordinate arithmetic.

    HFlip: ``coord_x' = width - 1 - coord_x``.
    VFlip: ``coord_y' = height - 1 - coord_y``.

    """
    flipped = keypoints.clone()
    if is_hflip:
        flipped[..., 0] = (width - 1) - keypoints[..., 0]
    if is_vflip:
        flipped[..., 1] = (height - 1) - keypoints[..., 1]

    mask = active[:, None, None]
    return torch.where(mask, flipped, keypoints)


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (batch_size, num_boxes, 4) boxes from xywh to xyxy format."""
    box_left, box_top, box_width, box_height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([box_left, box_top, box_left + box_width, box_top + box_height], dim=-1)


def _xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert (batch_size, num_boxes, 4) boxes from xyxy to xywh format."""
    box_x1, box_y1, box_x2, box_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1], dim=-1)
