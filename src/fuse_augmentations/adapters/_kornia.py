"""Kornia backend adapter for the fused affine engine.

Bridges Kornia augmentation transforms to the canonical parameter
representation and matrix primitives used by ``FusedAffineSegment``.

Example:
    >>> from fuse_augmentations.adapters._kornia import KorniaAdapter
    >>> adapter = KorniaAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...KorniaAdapter...>

"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import (
    crop_resize_matrix,
    hflip_matrix,
    matmul3x3,
    perspective_from_points,
    rotation_matrix,
    scale_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_matrix,
    vflip_matrix,
)

# ---------------------------------------------------------------------------
# Transform registry - lazy import guards kornia as optional dependency
# ---------------------------------------------------------------------------

try:
    from kornia.augmentation import ColorJitter as _ColorJitter
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
        # Saturation/hue ops are pixel-wise (non-linear in RGB) — reorderable
        # but not linearly composable into a FusedColorSegment matrix.
        _RandomSaturation: TransformCategory.POINTWISE,
        _RandomResizedCrop: TransformCategory.CROP_RESIZE_FIXED,
    }

    _COLOR_TYPES: frozenset[type] = frozenset({_RandomBrightness, _RandomContrast, _ColorJitter})
except ImportError:
    TRANSFORM_REGISTRY = {}
    _COLOR_TYPES: frozenset[type] = frozenset()  # type: ignore[no-redef]


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

            # Convert discrete k to angle in radians: 0, π/2, π, 3π/2
            angles_rad = k_rotations.to(dtype=torch.float32) * (torch.pi / 2.0)
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
        """Compose the full RandomAffine matrix: T @ Sh_y @ Sh_x @ S @ R.

        Composition order: rotation first, then scale, then x-shear,
        then y-shear, then translation. This matches Kornia's internal
        convention for RandomAffine.

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

        # Start with identity
        acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Rotation — skip when all angles are zero (e.g. RandomAffine(degrees=0))
        if "angle_rad" in params and not torch.all(params["angle_rad"] == 0):
            matrix_r = rotation_matrix(params["angle_rad"], height=height, width=width)
            acc = matmul3x3(matrix_r, acc)

        # Scale — skip when both factors are 1.0 (e.g. RandomAffine with no scale)
        if (
            "scale_x" in params
            and "scale_y" in params
            and not (torch.all(params["scale_x"] == 1) and torch.all(params["scale_y"] == 1))
        ):
            matrix_s = scale_matrix(params["scale_x"], params["scale_y"], height=height, width=width)
            acc = matmul3x3(matrix_s, acc)

        # X-Shear — skip when all values are zero (e.g. RandomAffine with no shear)
        if "shear_x_rad" in params and not torch.all(params["shear_x_rad"] == 0):
            # shear_x_rad already carries the Kornia→CCW negation from sample_params.
            # Parity verified by test_kornia_adapter.py::test_shear_sign_parity.
            shear_x_tan = torch.tan(params["shear_x_rad"])
            matrix_sh_x = shear_x_matrix(shear_x_tan, height=height, width=width)
            acc = matmul3x3(matrix_sh_x, acc)

        # Y-Shear — skip when all values are zero
        if "shear_y_rad" in params and not torch.all(params["shear_y_rad"] == 0):
            shear_y_tan = torch.tan(params["shear_y_rad"])
            matrix_sh_y = shear_y_matrix(shear_y_tan, height=height, width=width)
            acc = matmul3x3(matrix_sh_y, acc)

        # Translation — skip when all offsets are zero
        if (
            "translate_x" in params
            and "translate_y" in params
            and not (torch.all(params["translate_x"] == 0) and torch.all(params["translate_y"] == 0))
        ):
            matrix_t = translate_matrix(params["translate_x"], params["translate_y"])
            acc = matmul3x3(matrix_t, acc)

        return acc

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
    def build_color_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
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
          (approximated with midpoint 0.5) in the sampled ``order``.
          Brightness: ``M = brightness_factors * I₃``, ``b = 0``.
          Contrast: ``M = contrast_factors * I₃``, ``b = (1-contrast_factors) * 0.5``.

        Args:
            transform: A Kornia color augmentation transform.
            params: Parameter dict from ``sample_params()``.

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
            return _color_jitter_matrix(params)

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

        # Unsupported transform type — return empty
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
    """Compose rotation/scale/shear/translate from canonical params (batch_size=1, NumPy only)."""
    mtx_acc = np.eye(3, dtype=np.float64)

    if "angle_rad" in params:
        angle = float(params["angle_rad"].item())
        if angle != 0.0:
            mtx_acc = _rotation_np_b1(angle, center_x, center_y) @ mtx_acc

    if "scale_x" in params and "scale_y" in params:
        scale_x = float(params["scale_x"].item())
        scale_y = float(params["scale_y"].item())
        if scale_x != 1.0 or scale_y != 1.0:
            mtx_scale = np.array(
                [
                    [scale_x, 0.0, center_x * (1.0 - scale_x)],
                    [0.0, scale_y, center_y * (1.0 - scale_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            mtx_acc = mtx_scale @ mtx_acc

    if "shear_x_rad" in params:
        shear_x_rad = float(params["shear_x_rad"].item())
        if shear_x_rad != 0.0:
            shear_x_tan = np.tan(shear_x_rad)
            mtx_acc = (
                np.array(
                    [[1.0, shear_x_tan, -center_y * shear_x_tan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
                )
                @ mtx_acc
            )

    if "shear_y_rad" in params:
        shear_y_rad = float(params["shear_y_rad"].item())
        if shear_y_rad != 0.0:
            shear_y_tan = np.tan(shear_y_rad)
            mtx_acc = (
                np.array(
                    [[1.0, 0.0, 0.0], [shear_y_tan, 1.0, -center_x * shear_y_tan], [0.0, 0.0, 1.0]], dtype=np.float64
                )
                @ mtx_acc
            )

    if "translate_x" in params and "translate_y" in params:
        translation_x = float(params["translate_x"].item())
        translation_y = float(params["translate_y"].item())
        if translation_x != 0.0 or translation_y != 0.0:
            mtx_acc = (
                np.array([[1.0, 0.0, translation_x], [0.0, 1.0, translation_y], [0.0, 0.0, 1.0]], dtype=np.float64)
                @ mtx_acc
            )

    return mtx_acc


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
        mtx_acc = np.eye(3, dtype=np.float64)

        # Rotation
        if "angle" in raw:
            angle = -float(raw["angle"].item()) * (np.pi / 180.0)
            if angle != 0.0:
                mtx_acc = _rotation_np_b1(angle, center_x, center_y) @ mtx_acc

        # Scale
        if "scale" in raw:
            scale_raw = raw["scale"]
            scale_x = float(scale_raw[0, 0].item())
            scale_y = float(scale_raw[0, 1].item())
            if scale_x != 1.0 or scale_y != 1.0:
                mtx_acc = (
                    np.array(
                        [
                            [scale_x, 0.0, center_x * (1.0 - scale_x)],
                            [0.0, scale_y, center_y * (1.0 - scale_y)],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    )
                    @ mtx_acc
                )

        # X-Shear
        if "shear_x" in raw:
            shear_x_deg = float(raw["shear_x"].item())
            if shear_x_deg != 0.0:
                shear_x_tan = np.tan(-shear_x_deg * (np.pi / 180.0))
                mtx_acc = (
                    np.array(
                        [[1.0, shear_x_tan, -center_y * shear_x_tan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    @ mtx_acc
                )

        # Y-Shear
        if "shear_y" in raw:
            shear_y_deg = float(raw["shear_y"].item())
            if shear_y_deg != 0.0:
                shear_y_tan = np.tan(-shear_y_deg * (np.pi / 180.0))
                mtx_acc = (
                    np.array(
                        [[1.0, 0.0, 0.0], [shear_y_tan, 1.0, -center_x * shear_y_tan], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    @ mtx_acc
                )

        # Translation
        if "translations" in raw:
            translation_x = float(raw["translations"][0, 0].item())
            translation_y = float(raw["translations"][0, 1].item())
            if translation_x != 0.0 or translation_y != 0.0:
                mtx_acc = (
                    np.array([[1.0, 0.0, translation_x], [0.0, 1.0, translation_y], [0.0, 0.0, 1.0]], dtype=np.float64)
                    @ mtx_acc
                )
        elif "translate_x" in raw and "translate_y" in raw:
            translation_x = float(raw["translate_x"].item())
            translation_y = float(raw["translate_y"].item())
            if translation_x != 0.0 or translation_y != 0.0:
                mtx_acc = (
                    np.array([[1.0, 0.0, translation_x], [0.0, 1.0, translation_y], [0.0, 0.0, 1.0]], dtype=np.float64)
                    @ mtx_acc
                )

        return mtx_acc

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


def _color_jitter_matrix(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build (batch_size, 4, 4) for Kornia ColorJitter (brightness + contrast).

    ColorJitter applies sub-transforms in a random ``order``. This function
    composes brightness (multiplicative: ``brightness_factors * c``) and contrast (linear
    approximation around midpoint 0.5: ``contrast_factors * c + (1-contrast_factors) * 0.5``) in the
    sampled order. Saturation and hue steps are treated as identity.

    Args:
        params: Parameter dict containing ``brightness_factor``, ``contrast_factor``,
            and ``order`` tensors.

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

    # Apply in sampled order; index 0=brightness, 1=contrast, 2=saturation, 3=hue
    for idx_t in order.tolist():
        idx = int(idx_t)
        if idx == 0:
            # Brightness (multiplicative in ColorJitter): c' = brightness_factors * c
            mtx_step = _make_eye4(batch_size, device, dtype)
            mtx_step[:, 0, 0] = brightness_factors
            mtx_step[:, 1, 1] = brightness_factors
            mtx_step[:, 2, 2] = brightness_factors
        elif idx == 1:
            # Contrast (midpoint approximation): c' = contrast_factors * c + (1-contrast_factors)*midpoint
            mtx_step = _make_eye4(batch_size, device, dtype)
            mtx_step[:, 0, 0] = contrast_factors
            mtx_step[:, 1, 1] = contrast_factors
            mtx_step[:, 2, 2] = contrast_factors
            bias = (1.0 - contrast_factors) * _MIDPOINT
            mtx_step[:, 0, 3] = bias
            mtx_step[:, 1, 3] = bias
            mtx_step[:, 2, 3] = bias
        else:
            # Saturation (idx=2) and hue (idx=3) treated as identity
            continue
        # Compose: mtx_acc = mtx_step @ mtx_acc (mtx_step applied after current accumulation)
        mtx_acc = torch.bmm(mtx_step, mtx_acc)

    return mtx_acc
