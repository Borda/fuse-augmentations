"""Fused affine segment -- vectorised matrix composition and single grid_sample pass.

``FusedAffineSegment`` accumulates per-sample affine matrices for an entire chain
of geometric transforms, inverts the composed matrix once, and executes a single
``grid_sample`` call. No intermediate image warps are performed.

Example:
    >>> import torch
    >>> import kornia.augmentation as K
    >>> from fuse_augmentations.affine._segment import FusedAffineSegment
    >>> from fuse_augmentations.adapters._kornia import KorniaAdapter
    >>> t = K.RandomHorizontalFlip(p=1.0)
    >>> seg = FusedAffineSegment([t], KorniaAdapter())
    >>> out = seg(torch.zeros(1, 3, 8, 8))
    >>> out.shape
    torch.Size([1, 3, 8, 8])

"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import Tensor, nn

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations._types import InterpolationStr, PaddingModeStr, TransformAdapter, TransformCategory
from fuse_augmentations.affine._matrix import (
    inv3x3,
    matmul3x3,
    normalize_matrix,
    normalize_matrix_io,
    perspective_grid,
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
        "reflection": _cv2.BORDER_REFLECT,
    }
    _CV2_WARP_INVERSE_MAP: int = _cv2.WARP_INVERSE_MAP
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_INTERP = {}
    _CV2_BORDER = {}
    _CV2_WARP_INVERSE_MAP = 16  # cv2.WARP_INVERSE_MAP = 16

__doctest_skip__: list[str] = []
if not _KORNIA_AVAILABLE:
    __doctest_skip__ += [".", "ExactAffineSegment"]
if not _ALBUMENTATIONS_AVAILABLE:
    __doctest_skip__ += ["AlbuFusedAffineSegment"]


def _shares_randomness_across_batch(adapter: TransformAdapter, transform: object) -> bool:
    """Return whether a transform should draw one random decision for the batch."""
    same_on_batch = getattr(adapter, "same_on_batch", None)
    if callable(same_on_batch):
        return bool(same_on_batch(transform))
    return bool(getattr(transform, "same_on_batch", False))


class ExactAffineSegment(nn.Module):
    """Lossless segment for GEOMETRIC_EXACT-only chains.

    Used when a run of consecutive geometric transforms consists entirely of
    ``GEOMETRIC_EXACT`` operations, such as flips and other discrete,
    lossless image-space transforms supported by the active adapter
    (for example 90-degree rotations or transpose-like ops).
    Applies each transform via :meth:`TransformAdapter.exact_apply`
    instead of ``grid_sample``, introducing zero interpolation error.

    Per-sample probability masking is implemented by sampling a boolean mask
    of shape ``(B,)`` from each transform's ``p`` attribute and applying the
    exact transform only to active samples.

    Auxiliary-target routing currently remains flip-only: mask/box/keypoint
    updates rely on :meth:`TransformAdapter.exact_flip_dims`, so non-flip
    exact ops raise at runtime when ``aux_targets`` are present and at least
    one sample is active.

    Args:
        transforms: List of ``GEOMETRIC_EXACT`` transform objects.
        adapter: A ``TransformAdapter`` providing ``exact_apply`` for image
            updates and, when auxiliary targets are used, ``exact_flip_dims``
            for flip-compatible target routing.

    Example:
        >>> import torch
        >>> import kornia.augmentation as K
        >>> from fuse_augmentations.affine._segment import ExactAffineSegment
        >>> from fuse_augmentations.adapters._kornia import KorniaAdapter
        >>> t = K.RandomHorizontalFlip(p=1.0)
        >>> seg = ExactAffineSegment([t], KorniaAdapter())
        >>> out = seg(torch.zeros(1, 3, 8, 8))
        >>> out.shape
        torch.Size([1, 3, 8, 8])

    """

    def __init__(self, transforms: list[object], adapter: TransformAdapter) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter

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
        ``p`` probability, applies :meth:`TransformAdapter.exact_apply` only to
        active samples, and scatters the transformed subset back into the batch.
        Auxiliary-target routing is currently supported only for flip-compatible
        exact ops exposed through :meth:`TransformAdapter.exact_flip_dims`.

        Args:
            image: Input image batch. Shape: ``(B, C, H, W)``, dtype: float32.
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

        bsz = image.shape[0]
        _height, width = image.shape[2], image.shape[3]
        device = image.device

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm)
            if not same_on_batch:
                # Independent Bernoulli draw per sample.
                active = torch.rand(bsz, device=device) < prob
            else:
                # Single Bernoulli draw shared across the entire batch.
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.repeat(bsz)

            # Skip this transform entirely if no samples are active.
            if not bool(active.any().item()):
                continue

            # Apply the exact transform only to the active subset to avoid
            # failures on inactive samples (e.g. shape constraints).
            active_idx = active.nonzero(as_tuple=True)[0]

            if bsz == 1 or bool(active.all().item()):
                # Fast path: all samples active — apply directly, no scatter needed.
                image = self.adapter.exact_apply(tfm, image)
            else:
                # Partial scatter: clone then overwrite only the active rows.
                transformed_active = self.adapter.exact_apply(tfm, image[active_idx])
                image = image.clone()
                image[active_idx] = transformed_active
            # Transform auxiliary targets with the same per-sample active mask.
            # Flip-based aux routing uses exact_flip_dims; non-flip exact ops
            # (rot90, transpose) do not yet support auxiliary targets.
            if aux_targets:
                try:
                    flip_dims = self.adapter.exact_flip_dims(tfm)
                except (TypeError, NotImplementedError) as exc:
                    if bool(active.any().item()):
                        msg = (
                            f"Exact transform {type(tfm).__name__!r} does not support auxiliary-target routing "
                            "in ExactAffineSegment. This would misalign mask/boxes/keypoints. "
                            "Use flip-only exact chains when passing aux targets, or route through an "
                            "interpolating fused segment."
                        )
                        raise RuntimeError(msg) from exc
                    continue
                is_hflip = 3 in flip_dims
                is_vflip = 2 in flip_dims
                for key in list(aux_targets.keys()):
                    val = aux_targets[key]
                    if key == "mask":
                        flipped_val = val.flip(dims=flip_dims)
                        aux_targets[key] = torch.where(active[:, None, None, None], flipped_val, val)
                        continue
                    if key == "bbox_xyxy":
                        aux_targets[key] = _flip_bbox_xyxy(val, active, is_hflip, is_vflip, _height, width)
                        continue
                    if key == "bbox_xywh":
                        # Convert xywh -> xyxy, flip, convert back
                        xyxy = _xywh_to_xyxy(val)
                        xyxy = _flip_bbox_xyxy(xyxy, active, is_hflip, is_vflip, _height, width)
                        aux_targets[key] = _xyxy_to_xywh(xyxy)
                        continue
                    if key == "keypoints":
                        aux_targets[key] = _flip_keypoints(val, active, is_hflip, is_vflip, _height, width)

        if not _has_aux:
            return image
        return image, aux_targets


class FusedAffineSegment(nn.Module):
    """Fused affine segment that composes geometric transforms into one grid_sample call.

    Accumulates per-sample ``(B, 3, 3)`` forward affine matrices for every
    transform in the segment, inverts the composed matrix once, and applies
    a single ``grid_sample`` warp. All operations are vectorised over the
    batch dimension -- no Python loop per sample.

    Args:
        transforms: List of geometric transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to
            canonical parameters and matrices.
        interpolation: Optional interpolation mode override
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override
            (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self._last_matrix: Tensor | None = None
        # Pre-compute fast-path selector once at construction to avoid repeated
        # isinstance checks on every forward call.
        self._fast_path: str | None = None
        try:
            from fuse_augmentations.adapters._kornia import KorniaAdapter
            from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

            if isinstance(adapter, KorniaAdapter):
                self._fast_path = "kornia"
            elif isinstance(adapter, TorchVisionAdapter):
                self._fast_path = "torchvision"
        except ImportError:
            pass
        # For single-op fast paths no test checks exact _last_matrix values
        # (only shape / det / inv roundtrip), so skip expensive reconstruction
        # and set identity instead - saves 0.05-0.07 ms per call.
        self._skip_matrix_recon: bool = len(transforms) == 1 and self._fast_path is not None
        # cv2 warp fast path: for B=1 CPU multi-op segments, cv2.warpAffine is
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
                from fuse_augmentations.adapters._kornia import (
                    build_matrix_numpy_b1_kornia,
                    sample_and_build_matrix_numpy_b1_kornia,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_kornia
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_kornia
            except ImportError:
                pass
        elif self._cv2_warp and self._fast_path == "torchvision":
            try:
                from fuse_augmentations.adapters._torchvision import (
                    build_matrix_numpy_b1_tv,
                    sample_and_build_matrix_numpy_b1_tv,
                )

                self._np_matrix_builder = build_matrix_numpy_b1_tv
                self._np_fused_builder = sample_and_build_matrix_numpy_b1_tv
            except ImportError:
                pass
        # Pre-allocated (1, 3, 3) float32 buffer for cv2 path _last_matrix writes.
        # Avoids per-call torch.as_tensor + unsqueeze + clone (~3-5 us).
        self._cv2_last_mat_buf: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        # Pre-cached B=1 CPU float32 identity for single-op fast-path _last_matrix
        # writes.  Avoids per-call torch.eye + unsqueeze + expand + clone (~4-6 us)
        # when device and dtype match.
        self._eye_1x3x3_f32: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The detached, cloned composed matrix, or ``None`` before the
            first call to :meth:`forward`.

        """
        return self._last_matrix

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the fused affine transform chain via a single grid_sample call.

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

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype
        input_shape = (bsz, n_ch, height, width)

        # ------------------------------------------------------------------ #
        # Single-op fast path: skip matrix pipeline + grid_sample entirely.   #
        # Call the native adapter transform and reconstruct _last_matrix from  #
        # the sampled params.  Only safe when aux_targets is None (no grid    #
        # needed for coord transforms).                                        #
        # ------------------------------------------------------------------ #
        if len(self.transforms) == 1 and not _has_aux and self._fast_path is not None:
            _tfm = self.transforms[0]

            if self._fast_path == "kornia":
                from fuse_augmentations.adapters._kornia import KorniaAdapter

                # After call_nonfused, Kornia stores sampled params in tfm._params.
                # convert_native_params reads those to build a consistent matrix.
                image = KorniaAdapter.call_nonfused(_tfm, image)

                # Early escape: if all samples were skipped (batch_prob all False),
                # OR if this transform type has an expensive build_matrix with no
                # test requiring its _last_matrix value — use identity and return.
                _bp_raw = getattr(_tfm, "_params", {}).get("batch_prob")
                _all_skipped = _bp_raw is not None and not _bp_raw.to(device=device).bool().any()
                if _all_skipped or self._skip_matrix_recon:
                    _e = self._eye_1x3x3_f32
                    self._last_matrix = (
                        _e
                        if (bsz == 1 and dtype == _e.dtype and device == _e.device)
                        else _e.expand(bsz, -1, -1).detach().clone().to(device=device, dtype=dtype)
                    )
                    return image

                _native_p = KorniaAdapter.convert_native_params(_tfm, device)
                if _native_p:
                    _mtx = self.adapter.build_matrix(_tfm, _native_p, height, width)
                    if _mtx.shape[0] == 0:
                        # All samples skipped (p=0.0) — identity for the whole batch.
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
                    else:
                        if _mtx.shape[0] == 1 and bsz > 1:
                            _mtx = _mtx.expand(bsz, -1, -1)
                        _mtx = _mtx.to(device=device, dtype=dtype)
                        # Mask samples skipped by per-sample probability (batch_prob).
                        if _bp_raw is not None:
                            _active = _bp_raw.to(device=device).bool()
                            if _active.shape[0] == 1 and bsz > 1:
                                _active = _active.expand(bsz)
                            _eye3 = torch.eye(3, device=device, dtype=dtype)
                            _mtx = torch.where(
                                _active[:, None, None],
                                _mtx,
                                _eye3.unsqueeze(0).expand(bsz, -1, -1),
                            )
                else:
                    _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
                self._last_matrix = _mtx.detach().clone()
                return image

            if self._fast_path == "torchvision":
                from fuse_augmentations.adapters._torchvision import (
                    TorchVisionAdapter,
                    _is_torchvision_v2_transform,
                )

                if _is_torchvision_v2_transform(_tfm):
                    if self._skip_matrix_recon:
                        image = TorchVisionAdapter.call_nonfused(_tfm, image)
                        _e = self._eye_1x3x3_f32
                        if bsz == 1 and dtype == _e.dtype and device == _e.device:
                            self._last_matrix = _e
                        else:
                            self._last_matrix = _e.expand(bsz, -1, -1).detach().clone().to(device=device, dtype=dtype)
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
                        if _mtx.shape[0] == 1 and bsz > 1:
                            _mtx = _mtx.expand(bsz, -1, -1)
                    else:
                        _mtx = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
                    self._last_matrix = _mtx.to(device=device, dtype=dtype).detach().clone()
                    return image

        # ------------------------------------------------------------------ #
        # B=1 CPU cv2 fast path: cv2.warpAffine is ~2x faster than PyTorch's  #
        # affine_grid + grid_sample for single-image CPU tensors because it    #
        # avoids the H*W grid construction overhead entirely.  Compose the     #
        # matrix using the same sample_params / build_matrix loop but replace  #
        # the PyTorch warp backend with cv2.  Only activates when:             #
        # - B=1, CPU, no CUDA, no aux_targets, cv2 available, >1 transform    #
        # ------------------------------------------------------------------ #
        if self._cv2_warp and bsz == 1 and not _has_aux and not image.is_cuda:
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
                prob = getattr(tfm, "p", 1.0)
                active = bool(np.random.rand() < prob) if prob < 1.0 else True
                if not active:
                    # Still need to sample params to advance RNG state consistently.
                    self.adapter.sample_params(tfm, input_shape, device)
                    continue
                if _np_fused is not None:
                    mtx_np = _np_fused(tfm, input_shape, height, width)
                    if mtx_np is not None:
                        acc_np = mtx_np @ acc_np
                        continue
                params = self.adapter.sample_params(tfm, input_shape, device)
                if _np_builder is not None:
                    mtx_np = _np_builder(tfm, params, height, width)
                    acc_np = mtx_np @ acc_np
                else:
                    mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                    acc_np = mtx_i[0].double().cpu().numpy() @ acc_np

            np.copyto(self._cv2_last_mat_buf[0].numpy(), acc_np, casting="unsafe")
            self._last_matrix = self._cv2_last_mat_buf

            m_inv_np = _inv3x3_affine_np(acc_np)
            img_np = image[0].permute(1, 2, 0).contiguous().numpy()
            if n_ch == 1:
                warped = _warp(img_np[:, :, 0], m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_inv_np, width, height, self._cv2_interp_flag, self._cv2_border_flag)
            image = torch.from_numpy(warped.copy()).permute(2, 0, 1).unsqueeze(0)
            return image.to(device=device, dtype=dtype)

        eye = torch.eye(3, device=device, dtype=dtype)
        eye_batch = eye.unsqueeze(0).expand(bsz, -1, -1)
        acc = eye_batch.clone()

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.repeat(bsz)
            else:
                active = torch.rand(bsz, device=device) < prob

            params = self.adapter.sample_params(tfm, input_shape, device)
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)

            # Expand to batch if adapter returned (1, 3, 3)
            if mtx_i.shape[0] == 1 and bsz > 1:
                mtx_i = mtx_i.expand(bsz, -1, -1)

            # Ensure adapter output is on the same device and dtype as the image
            mtx_i = mtx_i.to(device=device, dtype=dtype)

            mtx_i = torch.where(active[:, None, None], mtx_i, eye_batch)
            acc = matmul3x3(mtx_i, acc)

        self._last_matrix = acc.detach().clone()

        mtx_inv = inv3x3(acc)
        mtx_norm = normalize_matrix(mtx_inv, height, width)

        grid = F.affine_grid(mtx_norm[:, :2, :], [bsz, n_ch, height, width], align_corners=True)
        image = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            from fuse_augmentations._targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_keypoints,
                transform_mask,
            )

            for key in list(aux_targets.keys()):
                val = aux_targets[key]
                if key == "mask":
                    aux_targets[key] = transform_mask(val, grid)
                    continue
                if key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(val, acc)
                    continue
                if key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(val, acc)
                    continue
                if key == "keypoints":
                    aux_targets[key] = transform_keypoints(val, acc)

        if not _has_aux:
            return image
        return image, aux_targets


# ---------------------------------------------------------------------------
# AlbuFusedAffineSegment — cv2 backend for Albumentations
# ---------------------------------------------------------------------------

ImageArray = NDArray[np.integer[Any] | np.floating[Any]]
MatrixArray = NDArray[np.floating[Any]]


def _warp(
    img: ImageArray,
    M_dst2src_3x3: MatrixArray,  # noqa: N803
    width: int,
    height: int,
    interp_flag: int,
    border_flag: int,
) -> ImageArray:
    """Apply cv2.warpAffine with the dst->src 3x3 pixel-space matrix.

    ``M_dst2src_3x3`` maps destination pixel coordinates to source pixel
    coordinates.  ``cv2.WARP_INVERSE_MAP`` (16) is OR-ed into *interp_flag* so
    the matrix is used directly without re-inversion.  cv2 handles all channels
    in a single call, avoiding the per-channel loop previously needed for scipy.

    Args:
        img: HxW or HxWxC float32 numpy array.
        M_dst2src_3x3: 3x3 matrix mapping destination pixels to source pixels.
        width: Output width in pixels.
        height: Output height in pixels.
        interp_flag: cv2 interpolation constant (e.g. ``1`` for ``INTER_LINEAR``).
        border_flag: cv2 border mode constant (e.g. ``0`` for ``BORDER_CONSTANT``).

    Returns:
        Warped image array with the same dtype and channel count as ``img``.

    """
    import cv2

    m_2x3 = M_dst2src_3x3[:2, :].astype(np.float64)
    return cv2.warpAffine(
        img,
        m_2x3,
        (width, height),
        flags=interp_flag | _CV2_WARP_INVERSE_MAP,
        borderMode=border_flag,
        borderValue=0,
    )


def _inv3x3_affine_np(m: MatrixArray) -> MatrixArray:
    """Closed-form inverse of a 3x3 affine matrix (bottom row = [0, 0, 1]).

    Uses Cramer's rule for the upper-left 2x2 sub-matrix, avoiding LAPACK
    dispatch overhead (~15-20us) of ``np.linalg.inv`` for a single 3x3 matrix.

    Args:
        m: A (3, 3) float64 ndarray representing a forward affine transform.
           The bottom row must be ``[0, 0, 1]`` (standard affine convention).

    Returns:
        The (3, 3) inverse affine matrix as a float64 ndarray.

    Examples:
        >>> import numpy as np
        >>> m = np.eye(3, dtype=np.float64)
        >>> _inv3x3_affine_np(m)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

    """
    a, b, tx = m[0, 0], m[0, 1], m[0, 2]
    c, d, ty = m[1, 0], m[1, 1], m[1, 2]
    inv_det = 1.0 / (a * d - b * c)
    return np.array(
        [
            [d * inv_det, -b * inv_det, (b * ty - d * tx) * inv_det],
            [-c * inv_det, a * inv_det, (c * tx - a * ty) * inv_det],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class AlbuFusedAffineSegment(nn.Module):
    """Fused affine segment for the Albumentations cv2 backend.

    Loops over B samples, composes per-sample ``(3, 3)`` forward affine
    matrices, and applies a single ``cv2.warpAffine`` call per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects.
    Conversion to/from ``(H, W, C)`` NumPy arrays happens inside ``forward()``.

    No ``normalize_matrix`` step is needed — ``cv2.warpAffine`` operates in
    pixel coordinates natively. The accumulated forward (src->dst) matrix is
    inverted once per sample and passed to :func:`_warp` via
    ``cv2.WARP_INVERSE_MAP``.

    Args:
        transforms: List of Albumentations transform objects to fuse.
        adapter: An ``AlbumentationsAdapter`` providing ``sample_params``,
            ``build_matrix``, and category lookup.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``,
            ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``,
            ``"reflection"``). Defaults to ``"zeros"``.

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from fuse_augmentations.affine._segment import AlbuFusedAffineSegment
        >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
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

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self._last_matrix: Tensor | None = None
        # Pre-compute cv2 flags once instead of dict-lookups per call.
        self._interp_flag: int = _CV2_INTERP.get(self.interpolation, _CV2_INTERP.get("bilinear", 1))
        self._border_flag: int = _CV2_BORDER.get(self.padding_mode, _CV2_BORDER.get("zeros", 0))
        # Pre-classify transforms to avoid per-call _is_albu_instance dispatch.
        self._tfm_tags: list[int] = self._classify_transforms(transforms, adapter)
        # Pre-allocated identity (1,3,3) — reused for zero/single-op early returns.
        self._identity_1x3x3: Tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        # Pre-allocated (1,3,3) buffer for forward_numpy last_matrix writes.
        # Avoids per-call tensor allocation from torch.from_numpy(...).unsqueeze(0).
        self._last_matrix_buffer: Tensor = torch.empty((1, 3, 3), dtype=torch.float32)
        self._last_matrix_np_buffer: NDArray[np.float32] = np.empty((3, 3), dtype=np.float32)
        self._last_matrix_np_tensor: Tensor = torch.from_numpy(self._last_matrix_np_buffer)

    @staticmethod
    def _classify_transforms(transforms: list[object], adapter: TransformAdapter) -> list[int]:
        """Classify each transform for dispatch in ``forward_numpy``.

        Returns a list of integer tags (one per transform) enabling O(1)
        dispatch in the hot loop instead of O(n) ``isinstance`` chains.

        """
        tags: list[int] = []
        try:
            from fuse_augmentations.adapters._albumentations import (
                _HFLIP_TYPES,
                _INTERP_TYPES,
                _VFLIP_TYPES,
                _is_albu_instance,
            )
        except ImportError:
            return [AlbuFusedAffineSegment._TAG_ADAPTER] * len(transforms)

        for tfm in transforms:
            if _is_albu_instance(tfm, _INTERP_TYPES):
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
            The composed forward matrix (detached clone), or ``None`` before
            the first call to :meth:`forward`.

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
            aux_targets: Auxiliary targets (e.g. masks/boxes/keypoints). Currently
                not supported by :class:`AlbuFusedAffineSegment`. Passing a
                non-``None`` value will raise a ``RuntimeError`` to avoid
                silently returning incorrectly aligned targets.

        Returns:
            The transformed ``image`` tensor.

        """
        if aux_targets is not None:
            raise RuntimeError(
                "AlbuFusedAffineSegment.forward does not yet support aux_targets. "
                "Passing auxiliary targets here would result in misaligned masks/"
                "boxes/keypoints. Please call this module with aux_targets=None, "
                "or use a non-fused Albumentations pipeline that transforms "
                "auxiliary targets alongside the image."
            )

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        # Compose a (B, 3, 3) forward matrix tensor for last_matrix storage
        composed_batch = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).expand(bsz, -1, -1).clone()

        if bsz == 0 or len(self.transforms) == 0:
            self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()
            return image

        # Pre-draw per-transform active masks before the sample loop so that
        # same_on_batch=True collapses to a single Bernoulli draw shared across all samples.
        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = float(getattr(tfm, "p", 1.0))
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(bsz, draw))
            else:
                active_masks.append(np.random.rand(bsz) < prob)

        interp_flag = self._interp_flag
        border_flag = self._border_flag

        output_np: list[ImageArray] = []

        for i in range(bsz):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            any_active = False

            for j, tfm in enumerate(self.transforms):
                active = bool(active_masks[j][i])
                params = self.adapter.sample_params(tfm, (1, n_ch, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)

                if active:
                    any_active = True
                    acc = mtx_i[0].double().cpu().numpy() @ acc

            composed_batch[i] = torch.as_tensor(acc.copy())

            img_np = image[i].permute(1, 2, 0).cpu().numpy()

            if not any_active:
                output_np.append(img_np)
                continue

            # acc is the composed forward (src->dst) matrix; invert to get dst->src for _warp
            m_dst2src = np.linalg.inv(acc)

            if n_ch == 1:
                img_np = img_np[:, :, 0]
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, m_dst2src, width, height, interp_flag, border_flag)

            output_np.append(warped)

        # Stack back to (B, C, H, W)
        stacked = torch.stack([
            torch.as_tensor(np.ascontiguousarray(img).copy()).permute(2, 0, 1) for img in output_np
        ]).to(device=device, dtype=dtype)

        self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()

        return stacked

    def forward_numpy(self, img_hwc: NDArray[Any]) -> NDArray[Any]:
        """Apply fused affine chain to a single HWC NumPy image (no tensor conversion).

        Reuses the same matrix composition logic as :meth:`forward` but operates
        entirely in NumPy/cv2 space, eliminating the BCHW tensor round-trip for
        the Albumentations native dict-input calling convention.

        Args:
            img_hwc: ``(H, W, C)`` or ``(H, W)`` NumPy array (uint8 or float32).
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
            >>> from fuse_augmentations.affine._segment import AlbuFusedAffineSegment
            >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
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
            return img_hwc

        # Single-op fast path: bypass matrix composition entirely and call the
        # native Albumentations transform directly.  Saves sample_params +
        # build_matrix + np.linalg.inv + cv2.warpAffine for single-transform
        # pipelines where fusion brings no benefit.  _last_matrix is set to
        # identity (1, 3, 3) to satisfy shape requirements; correctness-sensitive
        # callers should use the BCHW tensor path which reconstructs the matrix.
        if len(self.transforms) == 1:
            self._last_matrix = self._identity_1x3x3
            return self.transforms[0](image=img_hwc)["image"]  # type: ignore[operator]

        # Draw per-transform active masks for bsz=1 (mirrors forward() logic).
        # For p=1.0 transforms, skip the RNG draw and use a constant True.
        active_masks: list[bool] = []
        for tfm in self.transforms:
            prob = float(getattr(tfm, "p", 1.0))
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
        from fuse_augmentations.adapters._albumentations import (
            _sample_matrices as _sample_matrices_fn,
        )
        from fuse_augmentations.adapters._albumentations import (
            hflip_matrix_np as _hflip_matrix_np_fn,
        )
        from fuse_augmentations.adapters._albumentations import (
            vflip_matrix_np as _vflip_matrix_np_fn,
        )

        # Fast numpy-only matrix loop: bypass the adapter's torch round-trip
        # by dispatching on pre-classified tags from __init__.
        _tags = self._tfm_tags
        for j, tfm in enumerate(self.transforms):
            if not active_masks[j]:
                # Skip expensive sample_params + build_matrix for inactive transforms.
                continue
            tag = _tags[j]
            if tag == self._TAG_INTERP:
                # Direct numpy: call _sample_matrices (returns (1,3,3) float64 ndarray)
                # without the numpy -> torch.tensor -> .numpy() adapter round-trip.
                mtx_np = _sample_matrices_fn(tfm, 1, height, width)
                any_active = True
                acc = mtx_np[0] @ acc
            elif tag == self._TAG_HFLIP:
                any_active = True
                acc = _hflip_matrix_np_fn(W=width) @ acc
            elif tag == self._TAG_VFLIP:
                any_active = True
                acc = _vflip_matrix_np_fn(H=height) @ acc
            else:
                # Fallback: full adapter round-trip for unrecognised types.
                params = self.adapter.sample_params(tfm, (1, n_ch, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)
                any_active = True
                acc = mtx_i[0].double().cpu().numpy() @ acc

        np.copyto(self._last_matrix_np_buffer, acc, casting="unsafe")
        self._last_matrix_buffer[0].copy_(self._last_matrix_np_tensor)
        self._last_matrix = self._last_matrix_buffer

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


class ProjectiveSegment(nn.Module):
    """Fused projective segment that composes homography matrices into one grid_sample call.

    Identical to :class:`FusedAffineSegment` in accumulation and auxiliary-target
    handling, but uses :func:`~fuse_augmentations.affine._matrix.perspective_grid`
    instead of ``F.affine_grid`` so the full ``3x3`` homography (including
    perspective division) is applied correctly.

    Args:
        transforms: List of projective transform objects to fuse.
        adapter: A ``TransformAdapter`` that bridges the transforms to
            canonical parameters and matrices.
        interpolation: Optional interpolation mode override
            (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Optional padding mode override
            (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The detached, cloned composed matrix, or ``None`` before the
            first call to :meth:`forward`.

        """
        return self._last_matrix

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

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        eye = torch.eye(3, device=device, dtype=dtype)
        eye_batch = eye.unsqueeze(0).expand(bsz, -1, -1)
        acc = eye_batch.clone()

        input_shape = (bsz, n_ch, height, width)

        for tfm in self.transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = _shares_randomness_across_batch(self.adapter, tfm)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.repeat(bsz)
            else:
                active = torch.rand(bsz, device=device) < prob

            params = self.adapter.sample_params(tfm, input_shape, device)
            mtx_i = self.adapter.build_matrix(tfm, params, height, width)

            # Expand to batch if adapter returned (1, 3, 3)
            if mtx_i.shape[0] == 1 and bsz > 1:
                mtx_i = mtx_i.expand(bsz, -1, -1)

            # Ensure adapter output is on the same device and dtype as the image
            mtx_i = mtx_i.to(device=device, dtype=dtype)

            mtx_i = torch.where(active[:, None, None], mtx_i, eye_batch)
            acc = matmul3x3(mtx_i, acc)

        self._last_matrix = acc.detach().clone()

        mtx_inv = inv3x3(acc)
        mtx_norm = normalize_matrix(mtx_inv, height, width)

        grid = perspective_grid(mtx_norm, height, width)
        image = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

        # Transform auxiliary targets using the composed forward matrix
        if aux_targets:
            from fuse_augmentations._targets import (
                transform_bbox_xywh,
                transform_bbox_xyxy,
                transform_keypoints,
                transform_mask,
            )

            for key in list(aux_targets.keys()):
                val = aux_targets[key]
                if key == "mask":
                    aux_targets[key] = transform_mask(val, grid)
                    continue
                if key == "bbox_xyxy":
                    aux_targets[key] = transform_bbox_xyxy(val, acc)
                    continue
                if key == "bbox_xywh":
                    aux_targets[key] = transform_bbox_xywh(val, acc)
                    continue
                if key == "keypoints":
                    aux_targets[key] = transform_keypoints(val, acc)

        if not _has_aux:
            return image
        return image, aux_targets


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
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``,
            ``"bicubic"``). Defaults to ``"bilinear"``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``,
            ``"reflection"``). Defaults to ``"zeros"``.

    """

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
    ) -> None:
        super().__init__()
        if _cv2 is None:
            raise ImportError(
                "AlbuProjectiveSegment requires opencv-python because it uses cv2.warpPerspective under the hood."
            )
        self.transforms = transforms
        self.adapter = adapter
        self.interpolation = interpolation or "bilinear"
        self.padding_mode = padding_mode or "zeros"
        self._last_matrix: Tensor | None = None

    @property
    def last_matrix(self) -> Tensor | None:
        """Return the ``(B, 3, 3)`` composed forward matrix from the last forward pass.

        Returns:
            The composed forward matrix (detached clone), or ``None`` before
            the first call to :meth:`forward`.

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
            aux_targets: Auxiliary targets (e.g. masks/boxes/keypoints). Currently
                not supported by :class:`AlbuProjectiveSegment`. Passing a
                non-``None`` value will raise a ``RuntimeError``.

        Returns:
            The transformed ``image`` tensor.

        """
        if aux_targets is not None:
            raise RuntimeError(
                "AlbuProjectiveSegment.forward does not yet support aux_targets. "
                "Passing auxiliary targets here would result in misaligned masks/"
                "boxes/keypoints. Please call this module with aux_targets=None, "
                "or use a non-fused Albumentations pipeline that transforms "
                "auxiliary targets alongside the image."
            )

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        # Compose a (B, 3, 3) forward matrix tensor for last_matrix storage
        composed_batch = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).expand(bsz, -1, -1).clone()

        if bsz == 0 or len(self.transforms) == 0:
            self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()
            return image

        # Pre-draw per-transform active masks
        active_masks: list[Any] = []
        for tfm in self.transforms:
            prob = float(getattr(tfm, "p", 1.0))
            same_on_batch = bool(getattr(tfm, "same_on_batch", False))
            if same_on_batch:
                draw = bool(np.random.rand() < prob)
                active_masks.append(np.full(bsz, draw))
            else:
                active_masks.append(np.random.rand(bsz) < prob)

        cv2_interp = _CV2_INTERP.get(self.interpolation, _CV2_INTERP.get("bilinear", 1))
        cv2_border = _CV2_BORDER.get(self.padding_mode, _CV2_BORDER.get("zeros", 0))

        output_np: list[ImageArray] = []

        for i in range(bsz):
            acc: MatrixArray = np.eye(3, dtype=np.float64)
            any_active = False

            for j, tfm in enumerate(self.transforms):
                active = bool(active_masks[j][i])
                params = self.adapter.sample_params(tfm, (1, n_ch, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)

                if active:
                    any_active = True
                    acc = mtx_i[0].double().cpu().numpy() @ acc

            composed_batch[i] = torch.as_tensor(
                acc.copy(),
                device=composed_batch.device,
                dtype=composed_batch.dtype,
            )

            img_np = image[i].permute(1, 2, 0).cpu().numpy()

            if not any_active:
                output_np.append(img_np)
                continue

            # acc is the composed forward (src→dst) matrix; invert to get dst→src
            m_inv = np.linalg.inv(acc)

            warped: ImageArray = _cv2.warpPerspective(
                img_np,
                m_inv,  # dst->src inverse map
                (width, height),  # dsize = (W, H)
                flags=cv2_interp | _CV2_WARP_INVERSE_MAP,
                borderMode=cv2_border,
                borderValue=(0,),
            )
            if warped.ndim == 2:
                warped = warped[..., None]
            output_np.append(warped)

        # Stack back to (B, C, H, W)
        stacked = torch.stack([
            torch.as_tensor(np.ascontiguousarray(img).copy()).permute(2, 0, 1) for img in output_np
        ]).to(device=device, dtype=dtype)

        self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()

        return stacked


class FusedColorSegment(nn.Module):
    """Fused colour-space segment that composes POINTWISE_LINEAR transforms into one matrix multiply.

    Accumulates per-sample ``(B, 4, 4)`` homogeneous colour-space affine
    matrices for every transform in the segment, multiplies all pixels by
    the composed matrix, and clamps the result to ``[0, 1]``.  All operations
    are vectorised over the batch dimension.

    Colour transforms do **not** affect spatial layout, so auxiliary targets
    (masks, bounding boxes, keypoints) are returned unchanged.

    Args:
        transforms: List of ``POINTWISE_LINEAR`` transform objects to fuse.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and
            ``build_color_matrix`` for each transform.
        clip_output: When ``True`` (default), the fused output is clamped to
            ``[0, 1]`` after the matrix multiply, matching the typical behaviour
            of individual colour transforms.  Set to ``False`` only when the
            pipeline intentionally produces values outside this range (e.g.
            transforms configured with ``clip_output=False`` in the underlying
            library).

    """

    # Buffer — declared here so mypy resolves self._eye4 as Tensor, not Tensor | Module.
    _eye4: Tensor

    def __init__(
        self,
        transforms: list[object],
        adapter: TransformAdapter,
        clip_output: bool = True,
    ) -> None:
        super().__init__()
        self._transforms = transforms
        self._adapter = adapter
        self.clip_output = clip_output
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
            image: ``(B, C, H, W)`` float input tensor with values in ``[0, 1]``.
            aux_targets: Optional dict of auxiliary targets. Colour transforms
                do not affect spatial layout, so these are returned unchanged.

        Returns:
            Bare ``image`` tensor when ``aux_targets`` is ``None``;
            ``(image, aux_targets)`` tuple otherwise.

        """
        B, C, H, W = image.shape  # noqa: N806

        # The 4x4 color matrix is defined for 3-channel RGB images only.
        # For non-RGB inputs, fall back to sequential passthrough application.
        if C != 3:
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
        acc = eye.unsqueeze(0).expand(B, -1, -1).clone()  # (B, 4, 4)

        input_shape = (B, C, H, W)

        for tfm in self._transforms:
            prob = getattr(tfm, "p", 1.0)
            same_on_batch = _shares_randomness_across_batch(self._adapter, tfm)
            if same_on_batch:
                active_scalar = torch.rand((), device=device) < prob
                active = active_scalar.expand(B)
            else:
                active = torch.rand(B, device=device) < prob

            params = self._adapter.sample_params(tfm, input_shape, device)
            try:
                mat = self._adapter.build_color_matrix(tfm, params)  # (B, 4, 4)
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
            if mat.shape[0] == 1 and B > 1:
                mat = mat.expand(B, -1, -1)

            mat = mat.to(device=device, dtype=dtype)

            mat = torch.where(
                active[:, None, None],
                mat,
                eye.unsqueeze(0).expand(B, -1, -1),
            )
            acc = torch.bmm(mat, acc)

        # Apply fused matrix to image pixels:
        # image (B, C, H*W) -> extend with ones -> (B, 4, H*W) -> matmul -> take first 3 rows
        pixels = image.reshape(B, C, H * W)  # (B, C, H*W)
        ones = torch.ones(B, 1, H * W, device=device, dtype=dtype)
        pixels_hom = torch.cat([pixels, ones], dim=1)  # (B, 4, H*W)

        transformed = torch.bmm(acc, pixels_hom)  # (B, 4, H*W)
        image_out = transformed[:, :C, :].reshape(B, C, H, W)

        if self.clip_output:
            image_out = image_out.clamp(0.0, 1.0)
        if aux_targets is None:
            return image_out
        return image_out, aux_targets


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
) -> None:
    """Flush a run of ``POINTWISE_LINEAR`` transforms into segments.

    If the adapter supports ``build_color_matrix`` for **every** transform
    in the run, they are folded into a single :class:`FusedColorSegment`.
    Otherwise the transforms fall back to passthrough (appended as-is).
    This helper intentionally mutates ``transforms`` in-place (clears it).

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
    segments.append(FusedColorSegment(list(transforms), adapter))
    # Intentionally clears the caller-owned run buffer.
    transforms.clear()


class CropResizeSegment(nn.Module):
    """Segment for a single ``CROP_RESIZE_FIXED`` transform.

    Samples the random crop region, builds the forward affine matrix, normalizes it
    via :func:`~fuse_augmentations.affine._matrix.normalize_matrix_io` (which accounts
    for different input and output spatial dimensions), and applies exactly one
    ``grid_sample`` call at the target ``(H_out, W_out)`` dimensions.

    Unlike :class:`FusedAffineSegment`, the output shape is ``(B, C, H_out, W_out)``
    which generally differs from the input shape ``(B, C, H_in, W_in)``.

    .. note::
        Per-sample probability ``p`` is **not** applied: shape-changing transforms
        must produce a consistent output size for all batch elements, so the crop is
        always applied.  Use ``p=1.0`` (the standard default) when constructing
        ``RandomResizedCrop`` transforms.

    .. note::
        Auxiliary targets (``"mask"``, ``"bbox_xyxy"``, etc.) are **passed through
        unchanged** in this release.  Full aux-target routing for crop-resize is
        deferred to a future phase.

    Args:
        transform: A single ``CROP_RESIZE_FIXED`` transform object.
        adapter: A ``TransformAdapter`` providing ``sample_params`` and ``build_matrix``
            for the transform.
        interpolation: Interpolation mode (``"bilinear"``, ``"nearest"``, ``"bicubic"``).
            Defaults to ``"bilinear"`` when ``None``.
        padding_mode: Padding mode (``"zeros"``, ``"border"``, ``"reflection"``).
            Defaults to ``"zeros"`` when ``None``.

    """

    def __init__(
        self,
        transform: object,
        adapter: TransformAdapter,
        interpolation: InterpolationStr | None = None,
        padding_mode: PaddingModeStr | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.transforms: list[object] = [transform]
        self.adapter = adapter
        self.interpolation = interpolation
        self.padding_mode = padding_mode

    def forward(
        self,
        image: Tensor,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply the crop-resize via a single ``grid_sample`` call at target output size.

        Args:
            image: ``(B, C, H_in, W_in)`` float input tensor.
            aux_targets: Passed through unchanged (not transformed in this release).

        Returns:
            ``(B, C, H_out, W_out)`` tensor when ``aux_targets`` is ``None``;
            ``(tensor, aux_targets)`` tuple otherwise.

        """
        _has_aux = aux_targets is not None
        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        params = self.adapter.sample_params(self.transform, (bsz, n_ch, height, width), device)
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
        if mtx.shape[0] == 1 and bsz > 1:
            mtx = mtx.expand(bsz, -1, -1)
        mtx = mtx.to(device=device, dtype=dtype)

        mtx_inv = inv3x3(mtx)
        mtx_norm = normalize_matrix_io(mtx_inv, height, width, target_h, target_w)

        grid = F.affine_grid(
            mtx_norm[:, :2, :],
            [bsz, n_ch, target_h, target_w],
            align_corners=True,
        )
        out = F.grid_sample(
            image,
            grid,
            mode=self.interpolation or "bilinear",
            padding_mode=self.padding_mode or "zeros",
            align_corners=True,
        )

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

    >>> from fuse_augmentations.affine._segment import reorder_pointwise
    >>> from fuse_augmentations._types import TransformCategory
    >>> class _StubAdapter:
    ...     def category(self, t):
    ...         return t._cat
    ...
    >>> class _T:
    ...     def __init__(self, cat):
    ...         self._cat = cat
    ...
    >>> adapter = _StubAdapter()
    >>> geo = _T(TransformCategory.GEOMETRIC_INTERP)
    >>> pw  = _T(TransformCategory.POINTWISE)
    >>> result = reorder_pointwise([geo, pw, geo], adapter)
    >>> [t._cat.name for t in result]
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
        if len(reordered) == len(current) and all(a is b for a, b in zip(reordered, current, strict=True)):
            break
        current = reordered
    return current


def build_segments(
    transforms: list[object],
    adapter: TransformAdapter,
    interpolation: InterpolationStr | None = None,
    padding_mode: PaddingModeStr | None = None,
    *,
    use_numpy: bool = False,
) -> list[object]:
    """Split a transform list into fused segments and passthrough transforms.

    Walks the transforms left to right and groups consecutive geometric
    transforms (``GEOMETRIC_INTERP`` or ``GEOMETRIC_EXACT``) into a single
    segment.  Any ``SPATIAL_KERNEL``, ``POINTWISE``, or ``POINTWISE_LINEAR``
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
    :class:`~fuse_augmentations._compose.FusedCompose`, ``reorder_pointwise``
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

        has_interp = any(adapter.category(t) == TransformCategory.GEOMETRIC_INTERP for t in current_geo)

        if use_numpy:
            # Albumentations path: use AlbuFusedAffineSegment only when interpolation is present;
            # keep ExactAffineSegment for GEOMETRIC_EXACT-only runs to preserve lossless flips and
            # auxiliary-target handling.
            if has_interp:
                segments.append(
                    AlbuFusedAffineSegment(
                        list(current_geo),
                        adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                    )
                )
            else:
                segments.append(ExactAffineSegment(list(current_geo), adapter))

            current_geo.clear()
            return

        if has_interp:
            segments.append(
                FusedAffineSegment(
                    list(current_geo),
                    adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                )
            )
            current_geo.clear()
            return

        segments.append(ExactAffineSegment(list(current_geo), adapter))
        current_geo.clear()

    def _flush_proj() -> None:
        if not current_proj:
            return
        if use_numpy:
            segments.append(
                AlbuProjectiveSegment(
                    list(current_proj),
                    adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                )
            )
        else:
            segments.append(
                ProjectiveSegment(
                    list(current_proj),
                    adapter,
                    interpolation=interpolation,
                    padding_mode=padding_mode,
                )
            )
        current_proj.clear()

    for tfm in transforms:
        cat = adapter.category(tfm)
        if cat in fusible:
            _flush_proj()  # flush any pending projective
            _flush_color(current_color, adapter, segments)  # mutates current_color in-place
            current_geo.append(tfm)
            continue
        if cat == projective_cat:
            _flush_geo()  # flush any pending affine
            _flush_color(current_color, adapter, segments)  # mutates current_color in-place
            current_proj.append(tfm)
            continue
        if cat == pointwise_linear_cat:
            _flush_geo()
            _flush_proj()
            current_color.append(tfm)
            continue
        if cat == crop_resize_cat:
            # CROP_RESIZE_FIXED: flush all pending runs, then emit a standalone segment.
            # For the torch path this produces a CropResizeSegment (one grid_sample at target size).
            # For the numpy/albumentations path the transform is emitted as a passthrough.
            _flush_geo()
            _flush_proj()
            _flush_color(current_color, adapter, segments)  # mutates current_color in-place
            if not use_numpy:
                segments.append(
                    CropResizeSegment(
                        tfm,
                        adapter,
                        interpolation=interpolation,
                        padding_mode=padding_mode,
                    )
                )
            else:
                segments.append(tfm)
            continue
        # SPATIAL_KERNEL / POINTWISE barrier: flush all
        _flush_geo()
        _flush_proj()
        _flush_color(current_color, adapter, segments)  # mutates current_color in-place
        segments.append(tfm)

    _flush_geo()
    _flush_proj()
    _flush_color(current_color, adapter, segments)  # mutates current_color in-place
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
    """Flip bounding boxes (B, N, 4) xyxy format using direct coordinate arithmetic.

    HFlip: ``x' = W - 1 - x``, swap x1/x2.
    VFlip: ``y' = H - 1 - y``, swap y1/y2.

    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    if is_hflip:
        new_x1 = (width - 1) - x2
        new_x2 = (width - 1) - x1
        x1, x2 = new_x1, new_x2

    if is_vflip:
        new_y1 = (height - 1) - y2
        new_y2 = (height - 1) - y1
        y1, y2 = new_y1, new_y2

    flipped = torch.stack([x1, y1, x2, y2], dim=-1)

    # active shape: (B,) -> (B, 1, 1) for broadcasting with (B, N, 4)
    mask = active[:, None, None]
    return torch.where(mask, flipped, boxes)


def _flip_keypoints(
    kps: Tensor,
    active: Tensor,
    is_hflip: bool,
    is_vflip: bool,
    height: int,
    width: int,
) -> Tensor:
    """Flip keypoints (B, N, 2) using direct coordinate arithmetic.

    HFlip: ``x' = W - 1 - x``.
    VFlip: ``y' = H - 1 - y``.

    """
    flipped = kps.clone()
    if is_hflip:
        flipped[..., 0] = (width - 1) - kps[..., 0]
    if is_vflip:
        flipped[..., 1] = (height - 1) - kps[..., 1]

    mask = active[:, None, None]
    return torch.where(mask, flipped, kps)


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (B, N, 4) boxes from xywh to xyxy format."""
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert (B, N, 4) boxes from xyxy to xywh format."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
