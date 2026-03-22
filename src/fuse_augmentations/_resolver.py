"""Backend resolver for canonical operation names.

Maps ``(op: str, backend: str) -> type`` so that
:meth:`FusedCompose.from_config
<fuse_augmentations._compose.FusedCompose.from_config>` can construct
backend-specific transforms from a declarative :class:`TransformSpec`.

Each backend adapter exposes wrapper classes (e.g. Kornia's
``_RandomRotation``, TorchVision's ``RandomRotation``). This module
builds a reverse lookup from canonical op names
(``"rotation"``, ``"hflip"``, etc.) to those wrapper classes,
importing each backend lazily to avoid hard dependencies.

Example:
    >>> from fuse_augmentations._resolver import SUPPORTED_OPS
    >>> "rotation" in SUPPORTED_OPS
    True

"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

SUPPORTED_OPS: frozenset[str] = frozenset({
    "rotation",
    "affine",
    "shear",
    "translate",
    "hflip",
    "vflip",
    "scale",
    "perspective",
    "rotation90",
})

SUPPORTED_BACKENDS: frozenset[str] = frozenset({
    "kornia",
    "torchvision",
    "albumentations",
})


@lru_cache(maxsize=1)
def _kornia_registry() -> dict[str, type]:
    """Build op -> class map for the Kornia backend (lazy import)."""
    from kornia.augmentation import RandomAffine as _RandomAffine
    from kornia.augmentation import RandomHorizontalFlip as _RandomHorizontalFlip
    from kornia.augmentation import RandomPerspective as _RandomPerspective
    from kornia.augmentation import RandomRotation as _RandomRotation
    from kornia.augmentation import RandomRotation90 as _RandomRotation90
    from kornia.augmentation import RandomShear as _RandomShear
    from kornia.augmentation import RandomTranslate as _RandomTranslate
    from kornia.augmentation import RandomVerticalFlip as _RandomVerticalFlip

    return {
        "rotation": _RandomRotation,
        "affine": _RandomAffine,
        "shear": _RandomShear,
        "translate": _RandomTranslate,
        "hflip": _RandomHorizontalFlip,
        "vflip": _RandomVerticalFlip,
        "scale": _RandomAffine,  # scale is a subset of affine
        "perspective": _RandomPerspective,
        "rotation90": _RandomRotation90,
    }


@lru_cache(maxsize=1)
def _torchvision_registry() -> dict[str, type]:
    """Build op -> class map for the TorchVision backend (lazy import)."""
    # Prefer v2 when available, fall back to v1
    try:
        from torchvision.transforms.v2 import (
            RandomAffine,
            RandomHorizontalFlip,
            RandomPerspective,
            RandomRotation,
            RandomVerticalFlip,
        )
    except ImportError:
        from torchvision.transforms import (
            RandomAffine,
            RandomHorizontalFlip,
            RandomPerspective,
            RandomRotation,
            RandomVerticalFlip,
        )

    return {
        "rotation": RandomRotation,
        "affine": RandomAffine,
        "hflip": RandomHorizontalFlip,
        "vflip": RandomVerticalFlip,
        "scale": RandomAffine,  # scale is a subset of affine
        "perspective": RandomPerspective,
    }


@lru_cache(maxsize=1)
def _albumentations_registry() -> dict[str, type]:
    """Build op -> class map for the Albumentations backend (lazy import)."""
    from albumentations import Affine as _Affine
    from albumentations import HorizontalFlip as _HorizontalFlip
    from albumentations import Perspective as _Perspective
    from albumentations import RandomRotate90 as _RandomRotate90
    from albumentations import Rotate as _Rotate
    from albumentations import VerticalFlip as _VerticalFlip

    return {
        "rotation": _Rotate,
        "affine": _Affine,
        "hflip": _HorizontalFlip,
        "vflip": _VerticalFlip,
        "scale": _Affine,  # scale is a subset of affine
        "perspective": _Perspective,
        "rotation90": _RandomRotate90,
    }


_BACKEND_REGISTRY_BUILDERS: dict[str, Callable[[], dict[str, type]]] = {
    "kornia": _kornia_registry,
    "torchvision": _torchvision_registry,
    "albumentations": _albumentations_registry,
}


def resolve_op(op: str, backend: str) -> type:
    """Resolve a canonical operation name to its backend transform class.

    Args:
        op: Canonical operation name (e.g. ``"rotation"``, ``"hflip"``).
            Must be one of :data:`SUPPORTED_OPS`.
        backend: Backend name (e.g. ``"kornia"``, ``"torchvision"``,
            ``"albumentations"``). Must be one of :data:`SUPPORTED_BACKENDS`.

    Returns:
        The backend-specific transform class for the given operation.

    Raises:
        ValueError: If ``op`` is not in :data:`SUPPORTED_OPS` or ``backend``
            is not in :data:`SUPPORTED_BACKENDS`, or the backend does not
            support the requested operation.

    Example:
        >>> resolve_op("hflip", "torchvision")  # doctest: +SKIP
        <class 'torchvision.transforms.v2.RandomHorizontalFlip'>

    """
    if backend not in SUPPORTED_BACKENDS:
        msg = f"unknown backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
        raise ValueError(msg)
    if op not in SUPPORTED_OPS:
        msg = f"unknown op {op!r}; supported: {sorted(SUPPORTED_OPS)}"
        raise ValueError(msg)

    builder = _BACKEND_REGISTRY_BUILDERS[backend]
    try:
        registry = builder()
    except (ImportError, ModuleNotFoundError) as exc:
        msg = (
            f"backend {backend!r} is not available because its optional dependency is "
            f"not installed. Install it with e.g. `pip install fuse-augmentations[{backend}]`."
        )
        raise ValueError(msg) from exc
    if op not in registry:
        msg = f"backend {backend!r} does not support op {op!r}; supported ops for {backend}: {sorted(registry)}"
        raise ValueError(msg)
    return registry[op]
