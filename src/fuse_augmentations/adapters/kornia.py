"""Kornia backend adapter for the fused affine engine.

Bridges Kornia augmentation transforms to the canonical parameter
representation and matrix primitives used by ``FusedAffineSegment``.

Example:
    >>> from fuse_augmentations.adapters.kornia import KorniaAdapter
    >>> adapter = KorniaAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...KorniaAdapter...>

"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from fuse_augmentations.affine.matrix import (
    crop_resize_matrix,
    hflip_matrix,
    perspective_from_points,
    rotation_matrix,
    vflip_matrix,
)
from fuse_augmentations.types import SamplingSemantics, TransformCategory

# ---------------------------------------------------------------------------
# Transform registry - lazy import guards kornia as optional dependency
# ---------------------------------------------------------------------------

try:
    from kornia.augmentation import ColorJitter as _ColorJitter
    from kornia.augmentation import Normalize as _KorniaNormalize
    from kornia.augmentation import RandomAffine as _RandomAffine
    from kornia.augmentation import RandomBrightness as _RandomBrightness
    from kornia.augmentation import RandomContrast as _RandomContrast
    from kornia.augmentation import RandomHorizontalFlip as _RandomHorizontalFlip
    from kornia.augmentation import RandomPerspective as _RandomPerspective
    from kornia.augmentation import RandomResizedCrop as _RandomResizedCrop
    from kornia.augmentation import RandomRotation as _RandomRotation
    from kornia.augmentation import RandomRotation90 as _RandomRotation90
    from kornia.augmentation import RandomSaturation as _RandomSaturation
    from kornia.augmentation import RandomShear as _RandomShear
    from kornia.augmentation import RandomTranslate as _RandomTranslate
    from kornia.augmentation import RandomVerticalFlip as _RandomVerticalFlip
    from kornia.geometry.transform import get_affine_matrix2d as _get_affine_matrix2d

    TRANSFORM_REGISTRY: dict[type, TransformCategory] = {
        _RandomRotation: TransformCategory.GEOMETRIC_INTERP,
        _RandomAffine: TransformCategory.GEOMETRIC_INTERP,
        _RandomShear: TransformCategory.GEOMETRIC_INTERP,
        _RandomTranslate: TransformCategory.GEOMETRIC_INTERP,
        _RandomHorizontalFlip: TransformCategory.GEOMETRIC_EXACT,
        _RandomVerticalFlip: TransformCategory.GEOMETRIC_EXACT,
        _RandomRotation90: TransformCategory.GEOMETRIC_EXACT,
        _RandomPerspective: TransformCategory.PROJECTIVE,
        _RandomBrightness: TransformCategory.POINTWISE_LINEAR,
        _RandomContrast: TransformCategory.POINTWISE_LINEAR,
        _ColorJitter: TransformCategory.POINTWISE_LINEAR,
        _KorniaNormalize: TransformCategory.POINTWISE_LINEAR,
        # Saturation/hue ops are pixel-wise (non-linear in RGB) — reorderable
        # but not linearly composable into a FusedColorSegment matrix.
        _RandomSaturation: TransformCategory.POINTWISE,
        _RandomResizedCrop: TransformCategory.CROP_RESIZE_FIXED,
    }

    _COLOR_TYPES: frozenset[type] = frozenset({_RandomBrightness, _RandomContrast, _ColorJitter})
    _NORMALIZE_TYPES: frozenset[type] = frozenset({_KorniaNormalize})
except ImportError:
    TRANSFORM_REGISTRY = {}
    _KorniaNormalize: type = object  # type: ignore[no-redef]
    _COLOR_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]
    _NORMALIZE_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]


class KorniaAdapter:
    """Adapter between Kornia augmentation transforms and the fused affine engine.

    Implements the ``TransformAdapter`` protocol for the Kornia backend.
    Supports ``RandomRotation``, ``RandomAffine``, ``RandomShear``,
    ``RandomTranslate``, ``RandomHorizontalFlip``, ``RandomVerticalFlip``
    (affine path), and ``RandomPerspective`` (projective path).

    Example:
        >>> adapter = KorniaAdapter()
        >>> isinstance(adapter, KorniaAdapter)
        True

    """

    #: Canonical op names Kornia can build (mirrors ``resolver._kornia_registry``).
    capabilities: frozenset[str] = frozenset({
        "rotation",
        "affine",
        "shear",
        "translate",
        "hflip",
        "vflip",
        "scale",
        "perspective",
        "rotation90",
        "brightness",
        "contrast",
    })

    #: Kornia draws one parameter set per batch.
    sampling_semantics: SamplingSemantics = "per_batch"

    @staticmethod
    def category(transform: object) -> TransformCategory:
        """Return the TransformCategory of the given Kornia transform.

        Args:
            transform: A Kornia augmentation transform.

        Returns:
            The category for the transform. Unknown transforms default to
            ``SPATIAL_KERNEL`` with a ``UserWarning``.

        """
        cat = TRANSFORM_REGISTRY.get(type(transform))
        if cat is not None:
            return cat
        warnings.warn(
            f"Unknown Kornia transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
            UserWarning,
            stacklevel=2,
        )
        return TransformCategory.SPATIAL_KERNEL

    @staticmethod
    def sample_params(
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample random parameters for a batch of images.

        Calls Kornia's ``generate_parameters(input_shape)`` and converts
        to canonical units (radians, pixels, scale factors).

        Args:
            transform: A Kornia augmentation transform.
            input_shape: ``(batch_size, channels, height, width)`` shape tuple.
            device: Target device for parameter tensors.

        Returns:
            Dict of canonical parameter tensors. Empty for flip transforms.

        """
        ttype = type(transform)

        batch_size, _channels, _height, _width = input_shape

        # POINTWISE (non-linear color ops): no spatial matrix — return sentinel.
        if TRANSFORM_REGISTRY and ttype is _RandomSaturation:
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        # Flip transforms have no sampled params — prob-mask handles them.
        # Store batch metadata so build_matrix can construct the right shape.
        if TRANSFORM_REGISTRY and ttype in (
            _RandomHorizontalFlip,
            _RandomVerticalFlip,
        ):
            return {
                "_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64),
            }

        # Generate Kornia-native params
        params = transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]

        if TRANSFORM_REGISTRY and ttype is _RandomRotation90:
            return {
                "_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64),
                "k90": params["times"].to(device=device, dtype=torch.int64) % 4,
            }

        if TRANSFORM_REGISTRY and ttype is _RandomRotation:
            # Negate: Kornia's positive angle is CW; our rotation_matrix uses CCW convention.
            return {
                "angle_rad": -torch.deg2rad(params["degrees"].to(device=device)),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomAffine:
            result: dict[str, torch.Tensor] = {}

            # Rotation (degrees -> radians); negate to match Kornia's CW sign convention.
            if "angle" in params:
                result["angle_rad"] = -torch.deg2rad(params["angle"].to(device=device))

            # Translation - already in pixels (batch_size, 2); do NOT multiply by width/height
            if "translations" in params:
                translation_factors = params["translations"].to(device=device)
                result["translate_x"] = translation_factors[:, 0]
                result["translate_y"] = translation_factors[:, 1]

            # Scale (batch_size, 2) - factors
            if "scale" in params:
                scale_factors = params["scale"].to(device=device)
                result["scale_x"] = scale_factors[:, 0]
                result["scale_y"] = scale_factors[:, 1]

            # Shear - degrees -> radians; negate to match Kornia's sign convention.
            # Kornia emits separate "shear_x" / "shear_y" keys (shape (batch_size,)).
            if "shear_x" in params:
                result["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))

            return result

        if TRANSFORM_REGISTRY and ttype is _RandomShear:
            # RandomShear emits "shear_x" / "shear_y" keys in degrees (shape (batch_size,)).
            # Same conversion as the RandomAffine shear path.
            result_shear: dict[str, torch.Tensor] = {}
            if "shear_x" in params:
                result_shear["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result_shear["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))
            return result_shear

        if TRANSFORM_REGISTRY and ttype is _RandomTranslate:
            # RandomTranslate emits "translate_x" / "translate_y" in pixels (shape (batch_size,)).
            return {
                "translate_x": params["translate_x"].to(device=device),
                "translate_y": params["translate_y"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomPerspective:
            return {
                "start_points": params["start_points"].to(device=device),
                "end_points": params["end_points"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomResizedCrop:
            # src: (batch_size, 4, 2) corners (x, y) in order:
            #   [0]=top-left, [1]=top-right, [2]=bottom-right, [3]=bottom-left
            # output_size: (batch_size, 2) as (height, width)
            src_points = params["src"].to(device=device)  # (batch_size, 4, 2)
            output_size = params["output_size"].to(device=device)  # (batch_size, 2)
            left = src_points[:, 0, 0].float()
            top = src_points[:, 0, 1].float()
            right = src_points[:, 1, 0].float()
            bottom = src_points[:, 3, 1].float()
            return {
                "crop_top": top,
                "crop_left": left,
                "crop_h": (bottom - top + 1.0),
                "crop_w": (right - left + 1.0),
                "target_h": output_size[:, 0].float(),
                "target_w": output_size[:, 1].float(),
            }

        # --- Color transforms (POINTWISE_LINEAR) ---

        if TRANSFORM_REGISTRY and ttype is _RandomBrightness:
            # Kornia: c' = c + (brightness_factor - 1)  (additive brightness)
            return {
                "brightness_factor": params["brightness_factor"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomContrast:
            # Kornia: c' = contrast_factor * c  (multiplicative contrast)
            return {
                "contrast_factor": params["contrast_factor"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype is _ColorJitter:
            # ColorJitter applies brightness, contrast, saturation, hue in random order.
            # Brightness: c' = brightness_factor * c  (multiplicative)
            # Contrast: c' = contrast_factor * c + mean * (1 - contrast_factor) (data-dependent)
            # We extract all factors; build_color_matrix handles the linear approximation.
            out = {
                "brightness_factor": params["brightness_factor"].to(device=device),
                "contrast_factor": params["contrast_factor"].to(device=device),
                "order": params["order"].to(device=device),
            }
            if "saturation_factor" in params:
                out["saturation_factor"] = params["saturation_factor"].to(device=device)
            if "hue_factor" in params:
                out["hue_factor"] = params["hue_factor"].to(device=device)
            return out

        if TRANSFORM_REGISTRY and ttype is _KorniaNormalize:
            return {"_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64)}

        # Unknown - return empty
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build a (batch_size, 3, 3) pixel-space forward affine matrix from sampled params.

        Args:
            transform: A Kornia augmentation transform.
            params: Canonical-unit parameter dict from ``sample_params``.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            ``(batch_size, 3, 3)`` forward affine matrix in pixel coordinates.

        """
        ttype = type(transform)

        if TRANSFORM_REGISTRY and ttype is _RandomSaturation:
            # Non-linear color op: no spatial change → identity matrix.
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        if TRANSFORM_REGISTRY and ttype is _RandomHorizontalFlip:
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return hflip_matrix(width=width, batch_size=batch_size, device=device, dtype=torch.float32)

        if TRANSFORM_REGISTRY and ttype is _RandomVerticalFlip:
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device
            return vflip_matrix(height=height, batch_size=batch_size, device=device, dtype=torch.float32)

        if TRANSFORM_REGISTRY and ttype is _RandomRotation90:
            # Build per-sample 90-degree multiple rotation matrices.
            # Expect an integer parameter "k90" in {0, 1, 2, 3} per batch element.
            batch_size = int(params["_batch_size"].item())
            device = params["_batch_size"].device

            k_rotations = params.get("k90")
            if k_rotations is None:
                # Fallback: sample k uniformly if not provided in params.
                k_rotations = torch.randint(0, 4, (batch_size,), device=device)

            # Kornia's quarter-turn matrix uses the same CW-positive convention
            # as RandomRotation, while rotation_matrix is CCW-positive.
            angles_rad = -k_rotations.to(dtype=torch.float32) * (torch.pi / 2.0)
            return rotation_matrix(angles_rad, height=height, width=width)
        if TRANSFORM_REGISTRY and ttype is _RandomRotation:
            angle_rad = params["angle_rad"]
            return rotation_matrix(angle_rad, height=height, width=width)

        if TRANSFORM_REGISTRY and ttype is _RandomAffine:
            return KorniaAdapter._build_affine_matrix(params, height, width)

        if TRANSFORM_REGISTRY and ttype is _RandomShear:
            return KorniaAdapter._build_affine_matrix(params, height, width)

        if TRANSFORM_REGISTRY and ttype is _RandomTranslate:
            return KorniaAdapter._build_affine_matrix(params, height, width)

        if TRANSFORM_REGISTRY and ttype is _RandomPerspective:
            return perspective_from_points(params["start_points"], params["end_points"])

        if TRANSFORM_REGISTRY and ttype is _RandomResizedCrop:
            return crop_resize_matrix(
                top=params["crop_top"],
                left=params["crop_left"],
                crop_h=params["crop_h"],
                crop_w=params["crop_w"],
                target_h=params["target_h"],
                target_w=params["target_w"],
            )

        # Fallback: identity
        return torch.eye(3).unsqueeze(0)

    @staticmethod
    def _build_affine_matrix(
        params: dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build Kornia's native affine matrix from canonical adapter parameters.

        Canonical rotation and shear angles are negated when sampled so they
        match this package's CCW-positive convention. Kornia's native
        ``get_affine_matrix2d`` uses the opposite convention and composes the
        centered rotation-scale matrix before its coupled x/y shear matrix.
        Reconstructing that call avoids changing the order or units of combined
        ``RandomAffine`` parameters.

        Args:
            params: Canonical-unit parameter dict.
            height: Image height.
            width: Image width.

        Returns:
            ``(batch_size, 3, 3)`` composed forward matrix.

        """
        # Determine batch size from any param
        batch_size = None
        device = torch.device("cpu")
        dtype = torch.float32
        for param_value in params.values():
            if isinstance(param_value, torch.Tensor):
                batch_size = param_value.shape[0]
                device = param_value.device
                dtype = param_value.dtype
                break
        if batch_size is None:
            return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

        zero = torch.zeros(batch_size, device=device, dtype=dtype)
        one = torch.ones(batch_size, device=device, dtype=dtype)
        translations = torch.stack((params.get("translate_x", zero), params.get("translate_y", zero)), dim=1)
        center = torch.tensor([(width - 1) * 0.5, (height - 1) * 0.5], device=device, dtype=dtype).expand(
            batch_size, -1
        )
        scale = torch.stack((params.get("scale_x", one), params.get("scale_y", one)), dim=1)

        # Undo canonical sign conversion before calling Kornia's native builder.
        angle_deg = -torch.rad2deg(params.get("angle_rad", zero))
        shear_x = -params.get("shear_x_rad", zero)
        shear_y = -params.get("shear_y_rad", zero)
        return _get_affine_matrix2d(translations, center, scale, angle_deg, shear_x, shear_y)

    @staticmethod
    def exact_flip_dims(transform: object) -> list[int]:
        """Return the spatial dims to flip for GEOMETRIC_EXACT transforms.

        Args:
            transform: A Kornia flip transform (``RandomHorizontalFlip`` or
                ``RandomVerticalFlip``).

        Returns:
            List containing the dimension index to flip: ``[3]`` for HFlip
            (width axis) or ``[2]`` for VFlip (height axis).

        Raises:
            TypeError: If the transform is not a recognised flip type.

        """
        ttype = type(transform)
        if TRANSFORM_REGISTRY and ttype is _RandomHorizontalFlip:
            return [3]
        if TRANSFORM_REGISTRY and ttype is _RandomVerticalFlip:
            return [2]
        raise TypeError(f"Cannot determine flip dims for {ttype.__name__!r}")

    @staticmethod
    def exact_apply(transform: object, image: torch.Tensor) -> torch.Tensor:
        """Apply a GEOMETRIC_EXACT transform losslessly.

        Dispatches flips via ``tensor.flip`` and 90-degree rotations via
        ``torch.rot90``. For ``RandomRotation90``, the rotation count ``k``
        is sampled from the transform's ``times`` range.

        Args:
            transform: A Kornia GEOMETRIC_EXACT transform.
            image: ``(batch_size, channels, height, width)`` input tensor.

        Returns:
            Transformed ``(batch_size, channels, height, width)`` tensor.

        Raises:
            RuntimeError: If ``RandomRotation90`` is used on non-square images
                with an odd rotation count (k=1 or k=3), since these change
                spatial dimensions.

        """
        ttype = type(transform)
        if TRANSFORM_REGISTRY and ttype is _RandomHorizontalFlip:
            return image.flip(dims=[3])
        if TRANSFORM_REGISTRY and ttype is _RandomVerticalFlip:
            return image.flip(dims=[2])
        if TRANSFORM_REGISTRY and ttype is _RandomRotation90:
            # Sample per-batch k values using Kornia's native sampler.
            params = transform.generate_parameters(  # type: ignore[attr-defined]
                torch.Size(image.shape),
            )
            k_values = params["times"].to(device=image.device).to(dtype=torch.int64) % 4
            if image.shape[2] != image.shape[3] and bool(((k_values == 1) | (k_values == 3)).any().item()):
                msg = (
                    "RandomRotation90 with k in {1, 3} changes spatial dimensions "
                    f"({image.shape[2]}x{image.shape[3]}). "
                    "ExactAffineSegment requires shape-preserving ops. "
                    "Use square images for exact 90-degree rotations."
                )
                raise RuntimeError(msg)
            if k_values.numel() == 0:
                return image
            # Fast path: shared k across batch.
            if bool((k_values == k_values[0]).all().item()):
                return torch.rot90(image, k=int(k_values[0].item()), dims=[2, 3])

            out = image.clone()
            for kval in (1, 2, 3):
                idx = torch.nonzero(k_values == kval, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                out[idx] = torch.rot90(image[idx], k=kval, dims=[2, 3])
            return out
        msg = f"Cannot apply exact op for {ttype.__name__!r}"
        raise TypeError(msg)

    @staticmethod
    def color_luma_weights(transform: object) -> tuple[float, float, float] | None:
        """Return Kornia's RGB luminance weights for mean-relative contrast (``ColorJitter`` only).

        Kornia ``ColorJitter`` contrast uses ``adjust_contrast_with_mean_subtraction``, whose
        midpoint is ``rgb_to_grayscale(image).mean()`` with float weights ``(0.299, 0.587, 0.114)``.
        ``RandomContrast`` is pure multiplicative (no mean) and ``RandomBrightness`` additive, so
        they report ``None``.

        Args:
            transform: A Kornia color augmentation transform.

        Returns:
            ``(0.299, 0.587, 0.114)`` for ``ColorJitter``; ``None`` otherwise.

        """
        if TRANSFORM_REGISTRY and type(transform) is _ColorJitter:
            return (0.299, 0.587, 0.114)
        return None

    @staticmethod
    def is_normalize(transform: object) -> bool:
        """Return whether *transform* is Kornia's pointwise Normalize transform."""
        return bool(TRANSFORM_REGISTRY) and type(transform) is _KorniaNormalize

    @staticmethod
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Maps the linear color transform ``c' = M @ c + b`` to the 4x4
        homogeneous form ``[[M, b], [0^T, 1]]``.

        Supported transforms:

        - ``RandomBrightness``: additive brightness ``c' = c + beta``
          where ``beta = brightness_factor - 1``.
          Matrix: ``M = I₃``, ``b = (beta, beta, beta)``.
        - ``RandomContrast``: multiplicative contrast ``c' = alpha * c``
          where ``alpha = contrast_factor``.
          Matrix: ``M = alpha * I₃``, ``b = 0``.
        - ``ColorJitter``: composes brightness (multiplicative) and contrast
          in the sampled ``order``. Contrast is mean-relative:
          ``c' = cf*c + (1-cf)*mid``, where ``mid`` is the per-image luminance
          from ``mean`` (matching native) or ``0.5`` when ``mean`` is ``None``.
          Brightness: ``M = brightness_factors * I₃``, ``b = 0``.
          Contrast: ``M = contrast_factors * I₃``, ``b = (1-contrast_factors) * mid``.

        Args:
            transform: A Kornia color augmentation transform.
            params: Parameter dict from ``sample_params()``.
            mean: Optional per-image luminance of the transform's input, shape ``(B,)``. Used only for
                ``ColorJitter`` contrast; ``None`` falls back to the fixed ``0.5`` midpoint.

        Returns:
            ``(B, 4, 4)`` homogeneous color-space affine matrix.

        Raises:
            NotImplementedError: If the transform type is not supported.

        """
        ttype = type(transform)

        if TRANSFORM_REGISTRY and ttype is _RandomBrightness:
            brightness_factors = params["brightness_factor"]  # (batch_size,)
            return _brightness_additive_matrix(brightness_factors)

        if TRANSFORM_REGISTRY and ttype is _RandomContrast:
            contrast_factors = params["contrast_factor"]  # (batch_size,)
            return _contrast_multiplicative_matrix(contrast_factors)

        if TRANSFORM_REGISTRY and ttype is _ColorJitter:
            # NOTE: The fused ColorJitter path currently only supports
            # brightness/contrast. Saturation and hue are treated as identity
            # in the underlying _color_jitter_matrix. To avoid silently
            # changing semantics when non-trivial saturation or hue are
            # requested, detect these cases and force a fallback.
            sat_cfg = getattr(transform, "saturation", 0.0)
            hue_cfg = getattr(transform, "hue", 0.0)
            has_sat_cfg = sat_cfg not in (0.0, (1.0, 1.0))
            has_hue_cfg = hue_cfg not in (0.0, (0.0, 0.0))
            if has_sat_cfg or has_hue_cfg:
                raise NotImplementedError(
                    "Fused ColorJitter does not support non-identity saturation/hue; fall back to non-fused execution."
                )
            sat = params.get("saturation_factor")
            # Identity saturation corresponds to a factor of 1.0.
            if sat is not None and not torch.allclose(sat, torch.ones_like(sat)):
                raise NotImplementedError(
                    "Fused ColorJitter does not support non-identity saturation; fall back to non-fused execution."
                )
            hue = params.get("hue_factor")
            # Identity hue corresponds to a shift of 0.0.
            if hue is not None and not torch.allclose(hue, torch.zeros_like(hue)):
                raise NotImplementedError(
                    "Fused ColorJitter does not support non-identity hue; fall back to non-fused execution."
                )
            return _color_jitter_matrix(params, mean)

        if TRANSFORM_REGISTRY and ttype is _KorniaNormalize:
            return _build_normalize_matrix(transform, params)

        msg = f"build_color_matrix not supported for {ttype.__name__!r}"
        raise NotImplementedError(msg)

    @staticmethod
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply a Kornia transform directly via its native forward method.

        Args:
            transform: A Kornia augmentation transform.
            image: ``(batch_size, channels, height, width)`` image tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Transformed image tensor.

        """
        return transform(image)  # type: ignore[operator, no-any-return]

    @staticmethod
    def convert_native_params(
        transform: object,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert Kornia's stored ``_params`` to the canonical param dict.

        After a native forward call, Kornia transforms store their sampled
        parameters in ``transform._params``. This method converts that dict
        to the same canonical format returned by :meth:`sample_params`,
        enabling post-hoc matrix reconstruction via :meth:`build_matrix`.

        Args:
            transform: A Kornia augmentation transform that has been called
                (i.e. ``transform._params`` is populated).
            device: Target device for parameter tensors.

        Returns:
            Canonical parameter dict suitable for :meth:`build_matrix`.

        """
        ttype = type(transform)
        params = transform._params  # type: ignore[attr-defined]

        if TRANSFORM_REGISTRY and ttype is _RandomRotation:
            return {
                "angle_rad": -torch.deg2rad(params["degrees"].to(device=device)),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomAffine:
            result: dict[str, torch.Tensor] = {}
            if "angle" in params:
                result["angle_rad"] = -torch.deg2rad(params["angle"].to(device=device))
            if "translations" in params:
                translation_factors = params["translations"].to(device=device)
                result["translate_x"] = translation_factors[:, 0]
                result["translate_y"] = translation_factors[:, 1]
            if "scale" in params:
                scale_factors = params["scale"].to(device=device)
                result["scale_x"] = scale_factors[:, 0]
                result["scale_y"] = scale_factors[:, 1]
            if "shear_x" in params:
                result["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))
            return result

        if TRANSFORM_REGISTRY and ttype is _RandomShear:
            result_shear: dict[str, torch.Tensor] = {}
            if "shear_x" in params:
                result_shear["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result_shear["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))
            return result_shear

        if TRANSFORM_REGISTRY and ttype is _RandomTranslate:
            return {
                "translate_x": params["translate_x"].to(device=device),
                "translate_y": params["translate_y"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype in (_RandomHorizontalFlip, _RandomVerticalFlip):
            batch_size = int(params["batch_prob"].shape[0])
            return {
                "_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomRotation90:
            batch_size = int(params["batch_prob"].shape[0])
            return {
                "_batch_size": torch.tensor([batch_size], device=device, dtype=torch.int64),
                "k90": params["times"].to(device=device, dtype=torch.int64) % 4,
            }

        # Unsupported transform type — warn loudly: the caller falls back to an
        # identity _last_matrix, which silently misreports geometry if this path
        # is ever reached by a genuinely geometric transform.
        warnings.warn(
            f"convert_native_params has no handler for {ttype.__name__}; caller will fall back "
            "to an identity transform matrix. If this transform is geometric, its effect will "
            "not be reflected in transform_matrix / aux-target warps.",
            UserWarning,
            stacklevel=2,
        )
        return {}


# ---------------------------------------------------------------------------
# NumPy-native matrix builder for B=1 CPU cv2 warp path
# ---------------------------------------------------------------------------


def _rotation_np_b1(angle_rad: float, center_x: float, center_y: float) -> NDArray[np.float64]:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, -sin_a, center_x * (1.0 - cos_a) + center_y * sin_a],
            [sin_a, cos_a, center_y * (1.0 - cos_a) - center_x * sin_a],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _affine_matrix_np_b1(
    params: dict[str, torch.Tensor],
    center_x: float,
    center_y: float,
) -> NDArray[np.float64]:
    """Reconstruct Kornia's native affine order from canonical B=1 parameters."""
    # RandomAffine's native builder receives the original CW-positive angle;
    # canonical params store its CCW-positive inverse.
    angle = -float(params.get("angle_rad", torch.zeros(1)).item())
    scale_x = float(params.get("scale_x", torch.ones(1)).item())
    scale_y = float(params.get("scale_y", torch.ones(1)).item())
    translate_x = float(params.get("translate_x", torch.zeros(1)).item())
    translate_y = float(params.get("translate_y", torch.zeros(1)).item())
    shear_x = -float(params.get("shear_x_rad", torch.zeros(1)).item())
    shear_y = -float(params.get("shear_y_rad", torch.zeros(1)).item())

    cosine, sine = np.cos(angle), np.sin(angle)
    rotation_scale = np.array(
        [[cosine * scale_x, -sine * scale_y, 0.0], [sine * scale_x, cosine * scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_scale[0, 2] = center_x - rotation_scale[0, :2] @ np.array([center_x, center_y]) + translate_x
    rotation_scale[1, 2] = center_y - rotation_scale[1, :2] @ np.array([center_x, center_y]) + translate_y

    shear_x_tan, shear_y_tan = np.tan(shear_x), np.tan(shear_y)
    shear = np.array(
        [
            [1.0, -shear_x_tan, shear_x_tan * center_y],
            [-shear_y_tan, 1.0 + shear_x_tan * shear_y_tan, shear_y_tan * (center_x - shear_x_tan * center_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation_scale @ shear


def build_matrix_numpy_b1_kornia(
    transform: object,
    params: dict[str, torch.Tensor],
    height: int,
    width: int,
) -> NDArray[np.float64]:
    """Return (3, 3) float64 pixel-space matrix for batch_size=1, bypassing torch tensor creation.

    Drop-in replacement for
    ``KorniaAdapter.build_matrix(transform, params, height, width)[0].double().cpu().numpy()``
    in the batch_size=1 CPU cv2 warp path.  Extracts scalar values from ``params`` via
    ``.item()`` and computes the matrix directly in NumPy, avoiding the 8-12
    intermediate ``(1,)`` / ``(1, 3, 3)`` torch tensor allocations produced by
    the torch-based construction path.

    Args:
        transform: A Kornia augmentation transform (type-dispatched).
        params: Canonical parameter dict from ``KorniaAdapter.sample_params``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward affine matrix in pixel coordinates.

    """
    center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5
    ttype = type(transform)

    if TRANSFORM_REGISTRY and ttype is _RandomRotation:
        return _rotation_np_b1(float(params["angle_rad"].item()), center_x, center_y)

    if TRANSFORM_REGISTRY and ttype is _RandomHorizontalFlip:
        return np.array([[-1.0, 0.0, float(width - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    if TRANSFORM_REGISTRY and ttype is _RandomVerticalFlip:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)

    if TRANSFORM_REGISTRY and ttype in (_RandomAffine, _RandomShear, _RandomTranslate):
        return _affine_matrix_np_b1(params, center_x, center_y)

    # Fallback: torch path for uncommon types (RandomRotation90, RandomPerspective,
    # RandomResizedCrop, color ops, etc.).
    return KorniaAdapter.build_matrix(transform, params, height, width)[0].double().cpu().numpy()


def sample_and_build_matrix_numpy_b1_kornia(
    transform: object,
    input_shape: tuple[int, int, int, int],
    height: int,
    width: int,
) -> NDArray[np.float64]:
    """Sample params and build (3, 3) float64 matrix for batch_size=1, entirely in numpy.

    Fuses :func:`KorniaAdapter.sample_params` + :func:`build_matrix_numpy_b1_kornia`
    into a single call that calls ``generate_parameters`` directly, extracts scalar
    floats without creating intermediate canonical torch tensors, and computes the
    affine matrix in NumPy.  Compared to the two-step path this avoids 3-6 small
    torch tensor allocations per transform per forward call (``-torch.deg2rad(...)``,
    splits for scale/translation, etc.).

    Only handles the types common in the benchmark hot path
    (``RandomRotation``, ``RandomAffine``, ``RandomShear``, ``RandomTranslate``,
    ``RandomHorizontalFlip``, ``RandomVerticalFlip``).  All other types fall back
    to the two-step path and return ``None`` to signal that the caller should fall
    back.

    Args:
        transform: A Kornia augmentation transform (type-dispatched).
        input_shape: ``(batch_size, channels, height, width)`` shape tuple passed to ``generate_parameters``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(3, 3)`` float64 forward affine matrix, or ``None`` if this type is not
        handled (caller must fall back to the two-step path).

    """
    center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5
    ttype = type(transform)

    if not TRANSFORM_REGISTRY:
        return None  # type: ignore[return-value]

    if ttype is _RandomHorizontalFlip:
        # No generate_parameters needed — matrix is constant.
        transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]
        return np.array([[-1.0, 0.0, float(width - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype is _RandomVerticalFlip:
        transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)

    if ttype is _RandomRotation:
        raw = transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]
        # Kornia degrees: positive = CW; negate for CCW convention.
        angle_rad = -float(raw["degrees"].item()) * (np.pi / 180.0)
        return _rotation_np_b1(angle_rad, center_x, center_y)

    if ttype in (_RandomAffine, _RandomShear, _RandomTranslate):
        raw = transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]
        params: dict[str, torch.Tensor] = {}
        if "angle" in raw:
            params["angle_rad"] = -torch.deg2rad(raw["angle"])
        if "scale" in raw:
            params["scale_x"] = raw["scale"][:, 0]
            params["scale_y"] = raw["scale"][:, 1]
        if "shear_x" in raw:
            params["shear_x_rad"] = -torch.deg2rad(raw["shear_x"])
        if "shear_y" in raw:
            params["shear_y_rad"] = -torch.deg2rad(raw["shear_y"])
        if "translations" in raw:
            params["translate_x"] = raw["translations"][:, 0]
            params["translate_y"] = raw["translations"][:, 1]
        elif "translate_x" in raw and "translate_y" in raw:
            params["translate_x"] = raw["translate_x"]
            params["translate_y"] = raw["translate_y"]

        return _affine_matrix_np_b1(params, center_x, center_y)

    # Unsupported type: signal caller to fall back to the two-step path.
    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Private helpers -- color matrix construction
# ---------------------------------------------------------------------------

_MIDPOINT = 0.5  # Fixed midpoint for contrast approximation in ColorJitter


def _make_eye4(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return (batch_size, 4, 4) identity matrices."""
    return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()


def _brightness_additive_matrix(brightness_factor: torch.Tensor) -> torch.Tensor:
    """Build (batch_size, 4, 4) for Kornia RandomBrightness: ``c' = c + (brightness_factor - 1)``.

    M = I₃, b = (beta, beta, beta) where beta = brightness_factor - 1.

    Args:
        brightness_factor: ``(batch_size,)`` brightness factor tensor.

    Returns:
        ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

    """
    batch_size = brightness_factor.shape[0]
    device = brightness_factor.device
    dtype = brightness_factor.dtype
    mat = _make_eye4(batch_size, device, dtype)
    beta = brightness_factor - 1.0  # (batch_size,)
    mat[:, 0, 3] = beta
    mat[:, 1, 3] = beta
    mat[:, 2, 3] = beta
    return mat


def _contrast_multiplicative_matrix(contrast_factor: torch.Tensor) -> torch.Tensor:
    """Build (batch_size, 4, 4) for Kornia RandomContrast: ``c' = contrast_factor * c``.

    M = contrast_factor * I₃, b = 0.

    Args:
        contrast_factor: ``(batch_size,)`` contrast factor tensor.

    Returns:
        ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

    """
    batch_size = contrast_factor.shape[0]
    device = contrast_factor.device
    dtype = contrast_factor.dtype
    mat = _make_eye4(batch_size, device, dtype)
    mat[:, 0, 0] = contrast_factor
    mat[:, 1, 1] = contrast_factor
    mat[:, 2, 2] = contrast_factor
    return mat


def _build_normalize_matrix(transform: object, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build Kornia Normalize's channel-wise affine color matrix."""
    batch_size = int(params["_batch_size"].item())
    flags = cast(Any, transform).flags
    mean = torch.as_tensor(flags["mean"], dtype=torch.float32).flatten()
    std = torch.as_tensor(flags["std"], dtype=torch.float32).flatten()
    if mean.numel() == 1:
        mean = mean.expand(3)
    if std.numel() == 1:
        std = std.expand(3)
    if mean.numel() != 3 or std.numel() != 3:
        raise NotImplementedError("Kornia Normalize fusion requires three RGB mean/std values")
    device = params["_batch_size"].device
    dtype = mean.dtype
    mat = _make_eye4(batch_size, device, dtype)
    alpha = std.to(device=device).reciprocal()
    beta = -mean.to(device=device) * alpha
    mat[:, :3, :3] = torch.diag(alpha).expand(batch_size, -1, -1)
    mat[:, :3, 3] = beta
    return mat


def _color_jitter_matrix(params: dict[str, torch.Tensor], mean: torch.Tensor | None = None) -> torch.Tensor:
    """Build (batch_size, 4, 4) for Kornia ColorJitter (brightness + contrast).

    ColorJitter applies sub-transforms in a random ``order``. This function composes brightness
    (multiplicative: ``brightness_factors * c``) and contrast (mean-relative:
    ``contrast_factors * c + (1-contrast_factors) * mid``) in the sampled order. Saturation and hue
    steps are treated as identity.

    Contrast's ``mid`` is the per-image luminance of the image the contrast step actually sees. When
    ``mean`` is provided it is the luminance of this ColorJitter's input; brightness applied earlier in
    the same ``order`` scales that luminance, so the running luminance is advanced through each intra-op
    step (matching a native per-op chain). When ``mean`` is ``None``, a fixed ``0.5`` midpoint is used.

    Args:
        params: Parameter dict containing ``brightness_factor``, ``contrast_factor``,
            and ``order`` tensors.
        mean: Optional per-image input luminance, shape ``(batch_size,)``. ``None`` uses ``0.5``.

    Returns:
        ``(batch_size, 4, 4)`` homogeneous color-space affine matrix.

    """
    brightness_factors = params["brightness_factor"]  # (batch_size,)
    contrast_factors = params["contrast_factor"]  # (batch_size,)
    order = params["order"]  # (num_ops,) -- typically (4,) for [b, c, s, h]
    batch_size = brightness_factors.shape[0]
    device = brightness_factors.device
    dtype = brightness_factors.dtype

    # Start with identity
    mtx_acc = _make_eye4(batch_size, device, dtype)
    # Running per-image luminance seen by the NEXT sub-op. Uniform-diagonal color ops act on the
    # luminance the same scalar way, so a scalar carry reproduces the native intra-op mean exactly.
    luma = mean if mean is not None else None
    if luma is not None and batch_size == 1 and luma.shape[0] > 1:
        brightness_factors = brightness_factors.expand(luma.shape[0])
        contrast_factors = contrast_factors.expand(luma.shape[0])
        batch_size = luma.shape[0]

    # Apply in sampled order; index 0=brightness, 1=contrast, 2=saturation, 3=hue
    for idx_t in order.tolist():
        idx = int(idx_t)
        if idx == 0:
            # Brightness (multiplicative in ColorJitter): c' = brightness_factors * c
            mtx_step = _make_eye4(batch_size, device, dtype)
            mtx_step[:, 0, 0] = brightness_factors
            mtx_step[:, 1, 1] = brightness_factors
            mtx_step[:, 2, 2] = brightness_factors
            if luma is not None:
                luma = luma * brightness_factors
        elif idx == 1:
            # Contrast (mean-relative): c' = contrast_factors * c + (1-contrast_factors)*mid
            mtx_step = _make_eye4(batch_size, device, dtype)
            mtx_step[:, 0, 0] = contrast_factors
            mtx_step[:, 1, 1] = contrast_factors
            mtx_step[:, 2, 2] = contrast_factors
            mid = luma if luma is not None else _MIDPOINT
            bias = (1.0 - contrast_factors) * mid
            mtx_step[:, 0, 3] = bias
            mtx_step[:, 1, 3] = bias
            mtx_step[:, 2, 3] = bias
            if luma is not None:
                luma = contrast_factors * luma + bias
        else:
            # Saturation (idx=2) and hue (idx=3) treated as identity
            continue
        # Compose: mtx_acc = mtx_step @ mtx_acc (mtx_step applied after current accumulation)
        mtx_acc = torch.bmm(mtx_step, mtx_acc)

    return mtx_acc
