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

import torch

from fuse_augmentations._types import TransformCategory
from fuse_augmentations.affine._matrix import (
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
    from kornia.augmentation import RandomAffine as _RandomAffine
    from kornia.augmentation import RandomHorizontalFlip as _RandomHorizontalFlip
    from kornia.augmentation import RandomPerspective as _RandomPerspective
    from kornia.augmentation import RandomRotation as _RandomRotation
    from kornia.augmentation import RandomRotation90 as _RandomRotation90
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
    }
except ImportError:
    TRANSFORM_REGISTRY = {}


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
        """Sample random parameters for a batch of B images.

        Calls Kornia's ``generate_parameters(input_shape)`` and converts
        to canonical units (radians, pixels, scale factors).

        Args:
            transform: A Kornia augmentation transform.
            input_shape: ``(B, C, H, W)`` shape tuple.
            device: Target device for parameter tensors.

        Returns:
            Dict of canonical parameter tensors. Empty for flip transforms.

        """
        ttype = type(transform)

        B, _C, _H, _W = input_shape  # noqa: N806

        # Exact discrete transforms have no sampled params — p-mask handles them.
        # Store batch metadata so build_matrix can construct the right shape.
        if TRANSFORM_REGISTRY and ttype in (
            _RandomHorizontalFlip,
            _RandomVerticalFlip,
            _RandomRotation90,
        ):
            return {
                "_batch_size": torch.tensor([B], device=device, dtype=torch.int64),
            }

        # Generate Kornia-native params
        params = transform.generate_parameters(torch.Size(input_shape))  # type: ignore[attr-defined]

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

            # Translation - already in pixels (B, 2); do NOT multiply by W/H
            if "translations" in params:
                trans = params["translations"].to(device=device)
                result["translate_x"] = trans[:, 0]
                result["translate_y"] = trans[:, 1]

            # Scale (B, 2) - factors
            if "scale" in params:
                sc = params["scale"].to(device=device)
                result["scale_x"] = sc[:, 0]
                result["scale_y"] = sc[:, 1]

            # Shear - degrees -> radians; negate to match Kornia's sign convention.
            # Kornia emits separate "shear_x" / "shear_y" keys (shape (B,)).
            if "shear_x" in params:
                result["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))

            return result

        if TRANSFORM_REGISTRY and ttype is _RandomShear:
            # RandomShear emits "shear_x" / "shear_y" keys in degrees (shape (B,)).
            # Same conversion as the RandomAffine shear path.
            result_shear: dict[str, torch.Tensor] = {}
            if "shear_x" in params:
                result_shear["shear_x_rad"] = -torch.deg2rad(params["shear_x"].to(device=device))
            if "shear_y" in params:
                result_shear["shear_y_rad"] = -torch.deg2rad(params["shear_y"].to(device=device))
            return result_shear

        if TRANSFORM_REGISTRY and ttype is _RandomTranslate:
            # RandomTranslate emits "translate_x" / "translate_y" in pixels (shape (B,)).
            return {
                "translate_x": params["translate_x"].to(device=device),
                "translate_y": params["translate_y"].to(device=device),
            }

        if TRANSFORM_REGISTRY and ttype is _RandomPerspective:
            return {
                "start_points": params["start_points"].to(device=device),
                "end_points": params["end_points"].to(device=device),
            }

        # Unknown - return empty
        return {}

    @staticmethod
    def build_matrix(
        transform: object,
        params: dict[str, torch.Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> torch.Tensor:
        """Build a (B, 3, 3) pixel-space forward affine matrix from sampled params.

        Args:
            transform: A Kornia augmentation transform.
            params: Canonical-unit parameter dict from ``sample_params``.
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            ``(B, 3, 3)`` forward affine matrix in pixel coordinates.

        """
        ttype = type(transform)

        if TRANSFORM_REGISTRY and ttype is _RandomHorizontalFlip:
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            return hflip_matrix(W=W, batch_size=B, device=device, dtype=torch.float32)

        if TRANSFORM_REGISTRY and ttype is _RandomVerticalFlip:
            B = int(params["_batch_size"].item())  # noqa: N806
            device = params["_batch_size"].device
            return vflip_matrix(H=H, batch_size=B, device=device, dtype=torch.float32)

        if TRANSFORM_REGISTRY and ttype is _RandomRotation90:
            # Discrete exact op handled by exact_apply; return identity
            B = int(params["_batch_size"].item())  # noqa: N806
            return torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()

        if TRANSFORM_REGISTRY and ttype is _RandomRotation:
            angle_rad = params["angle_rad"]
            return rotation_matrix(angle_rad, H=H, W=W)

        if TRANSFORM_REGISTRY and ttype is _RandomAffine:
            return KorniaAdapter._build_affine_matrix(params, H, W)

        if TRANSFORM_REGISTRY and ttype is _RandomShear:
            return KorniaAdapter._build_affine_matrix(params, H, W)

        if TRANSFORM_REGISTRY and ttype is _RandomTranslate:
            return KorniaAdapter._build_affine_matrix(params, H, W)

        if TRANSFORM_REGISTRY and ttype is _RandomPerspective:
            return perspective_from_points(params["start_points"], params["end_points"])

        # Fallback: identity
        return torch.eye(3).unsqueeze(0)

    @staticmethod
    def _build_affine_matrix(
        params: dict[str, torch.Tensor],
        H: int,  # noqa: N803
        W: int,  # noqa: N803
    ) -> torch.Tensor:
        """Compose the full RandomAffine matrix: T @ Sh_y @ Sh_x @ S @ R.

        Composition order: rotation first, then scale, then x-shear,
        then y-shear, then translation. This matches Kornia's internal
        convention for RandomAffine.

        Args:
            params: Canonical-unit parameter dict.
            H: Image height.
            W: Image width.

        Returns:
            ``(B, 3, 3)`` composed forward matrix.
        """
        # Determine batch size from any param
        B = None  # noqa: N806
        device = torch.device("cpu")
        dtype = torch.float32
        for v in params.values():
            if isinstance(v, torch.Tensor):
                B = v.shape[0]  # noqa: N806
                device = v.device
                dtype = v.dtype
                break
        if B is None:
            return torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

        # Start with identity
        acc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()

        # Rotation
        if "angle_rad" in params:
            R = rotation_matrix(params["angle_rad"], H=H, W=W)  # noqa: N806
            acc = matmul3x3(R, acc)

        # Scale
        if "scale_x" in params and "scale_y" in params:
            S = scale_matrix(params["scale_x"], params["scale_y"], H=H, W=W)  # noqa: N806
            acc = matmul3x3(S, acc)

        # X-Shear
        if "shear_x_rad" in params:
            # shear_x_rad already carries the Kornia→CCW negation from sample_params.
            # Parity verified by test_kornia_adapter.py::test_shear_sign_parity.
            shear_x_tan = torch.tan(params["shear_x_rad"])
            Sh_x = shear_x_matrix(shear_x_tan, H=H, W=W)  # noqa: N806
            acc = matmul3x3(Sh_x, acc)

        # Y-Shear
        if "shear_y_rad" in params:
            shear_y_tan = torch.tan(params["shear_y_rad"])
            Sh_y = shear_y_matrix(shear_y_tan, H=H, W=W)  # noqa: N806
            acc = matmul3x3(Sh_y, acc)

        # Translation
        if "translate_x" in params and "translate_y" in params:
            T = translate_matrix(params["translate_x"], params["translate_y"])  # noqa: N806
            acc = matmul3x3(T, acc)

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
            image: ``(B, C, H, W)`` input tensor.

        Returns:
            Transformed ``(B, C, H, W)`` tensor.

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
                    "Use square images or pair with a GEOMETRIC_INTERP transform "
                    "to route through FusedAffineSegment instead."
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
    def call_nonfused(
        transform: object,
        image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply a Kornia transform directly via its native forward method.

        Args:
            transform: A Kornia augmentation transform.
            image: ``(B, C, H, W)`` image tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Transformed image tensor.

        """
        return transform(image)  # type: ignore[operator, no-any-return]
