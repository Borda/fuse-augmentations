"""NumPy/cv2 fused affine segment for the Albumentations backend.

``NumpyFusedAffineSegment`` accumulates per-sample ``(3, 3)`` forward affine
matrices for a chain of Albumentations transforms, inverts the composed matrix
once per sample, and executes a single ``cv2.warpAffine`` call. This reduces
``n`` sequential warp calls to 1 per sample.

Unlike ``FusedAffineSegment`` (which is fully vectorised over the batch
dimension via ``F.affine_grid`` + ``F.grid_sample``), this class loops over B
samples — matching Albumentations' per-sample execution model.

Example:
    >>> import numpy as np
    >>> import torch
    >>> from fuse_augmentations._np_segment import NumpyFusedAffineSegment
    >>> from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
    >>> seg = NumpyFusedAffineSegment([], AlbumentationsAdapter())
    >>> out = seg(torch.zeros(1, 3, 8, 8))
    >>> out.shape
    torch.Size([1, 3, 8, 8])

"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from fuse_augmentations._types import TransformAdapter

# cv2 flag mapping — import lazily to keep cv2 optional at import time
_INTERP_FLAGS: dict[str, int] | None = None
_BORDER_FLAGS: dict[str, int] | None = None


def _get_flags() -> tuple[dict[str, int], dict[str, int]]:
    """Return cv2 interpolation and border mode flag dicts (lazy import)."""
    global _INTERP_FLAGS, _BORDER_FLAGS
    if _INTERP_FLAGS is None:
        import cv2

        _INTERP_FLAGS = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "bicubic": cv2.INTER_CUBIC,
        }
        _BORDER_FLAGS = {
            "zeros": cv2.BORDER_CONSTANT,
            "border": cv2.BORDER_REPLICATE,
            "reflection": cv2.BORDER_REFLECT_101,
        }
    return _INTERP_FLAGS, _BORDER_FLAGS  # type: ignore[return-value]


class NumpyFusedAffineSegment(nn.Module):
    """Fused affine segment for NumPy/cv2 backends (Albumentations).

    Loops over B samples, composes per-sample ``(3, 3)`` forward affine
    matrices, and applies a single ``cv2.warpAffine`` per sample.

    The input and output are ``(B, C, H, W)`` float32 ``torch.Tensor`` objects.
    Conversion to/from ``(H, W, C)`` NumPy arrays happens inside ``forward()``.

    No ``normalize_matrix`` step is needed — ``cv2.warpAffine`` operates
    in pixel coordinates natively. The forward 2x3 matrix is passed
    directly to ``cv2.warpAffine`` (which inverts it internally).

    Args:
        transforms: List of Albumentations transform objects to fuse.
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
        interpolation: str | None = None,
        padding_mode: str | None = None,
    ) -> None:
        super().__init__()
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
        """Apply fused affine chain via per-sample cv2.warpAffine.

        Args:
            image: ``(B, C, H, W)`` float32 input tensor.
            aux_targets: Auxiliary targets (e.g. masks/boxes/keypoints). Currently
                not supported by :class:`NumpyFusedAffineSegment`. Passing a
                non-``None`` value will raise a ``RuntimeError`` to avoid
                silently returning incorrectly aligned targets.

        Returns:
            The transformed ``image`` tensor.

        """
        if aux_targets is not None:
            raise RuntimeError(
                "NumpyFusedAffineSegment.forward does not yet support aux_targets. "
                "Passing auxiliary targets here would result in misaligned masks/"
                "boxes/keypoints. Please call this module with aux_targets=None, "
                "or use a non-fused Albumentations pipeline that transforms "
                "auxiliary targets alongside the image."
            )

        interp_flags, border_flags = _get_flags()
        interp_flag = interp_flags.get(self.interpolation, interp_flags["bilinear"])
        border_flag = border_flags.get(self.padding_mode, border_flags["zeros"])

        bsz, n_ch, height, width = image.shape
        device = image.device
        dtype = image.dtype

        # Compose a (B, 3, 3) forward matrix tensor for last_matrix storage
        composed_batch = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(bsz, -1, -1)
            .clone()
        )

        # Per-sample warp
        output_np: list[np.ndarray] = []

        for i in range(bsz):
            acc = np.eye(3, dtype=np.float64)
            any_active = False

            for tfm in self.transforms:
                prob = float(getattr(tfm, "p", 1.0))
                active = np.random.rand() < prob

                params = self.adapter.sample_params(tfm, (1, n_ch, height, width), torch.device("cpu"))
                mtx_i = self.adapter.build_matrix(tfm, params, height, width)

                if active:
                    any_active = True
                    acc = mtx_i[0].double().cpu().numpy() @ acc

            composed_batch[i] = torch.from_numpy(acc)

            if len(self.transforms) == 0 or not any_active:
                # Identity: no warp needed when no transforms or all inactive
                img_np = image[i].permute(1, 2, 0).cpu().numpy()
                output_np.append(img_np)
                continue

            # cv2.warpAffine without WARP_INVERSE_MAP treats M as forward (src→dst)
            # and internally inverts it to produce dst[p'] = src[M^{-1}*p'].
            # Albumentations stores forward pixel-space matrices, so we pass acc
            # directly — no explicit inversion needed.
            M_fwd_2x3 = acc[:2, :].astype(np.float32)  # noqa: N806

            img_np = image[i].permute(1, 2, 0).cpu().numpy()

            if n_ch == 1:
                img_np = img_np[:, :, 0]
                warped = _warp(img_np, M_fwd_2x3, width, height, interp_flag, border_flag)
                warped = warped[:, :, np.newaxis]
            else:
                warped = _warp(img_np, M_fwd_2x3, width, height, interp_flag, border_flag)

            output_np.append(warped)

        # Stack back to (B, C, H, W)
        stacked = torch.stack([torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1) for img in output_np]).to(
            device=device, dtype=dtype
        )

        # Store a float32 detached clone of the composed transform matrix on the input device
        self._last_matrix = composed_batch.to(dtype=torch.float32).clone().detach()

        return stacked


def _warp(
    img: np.ndarray,
    M_fwd_2x3: np.ndarray,  # noqa: N803
    width: int,
    height: int,
    interp_flag: int,
    border_flag: int,
) -> np.ndarray:
    """Apply cv2.warpAffine with the forward 2x3 pixel-space matrix.

    cv2.warpAffine without WARP_INVERSE_MAP treats M as forward (src→dst)
    and inverts it internally, so dst[p'] = src[M^{-1}*p'].
    """
    import cv2

    result: np.ndarray = cv2.warpAffine(
        img,
        M_fwd_2x3,
        (width, height),
        flags=interp_flag,
        borderMode=border_flag,
    )
    return result
