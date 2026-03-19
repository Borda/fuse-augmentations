"""Backend detection for augmentation transform pipelines.

Inspects transform module paths to determine which backend framework
(Kornia, Albumentations, TorchVision) is in use.

Example:
    >>> from fuse_augmentations._backend import detect_backend
    >>> detect_backend([])
    <Backend.UNKNOWN: 'unknown'>

"""

from __future__ import annotations

import warnings
from enum import Enum


class Backend(Enum):
    """Supported augmentation backend frameworks."""

    KORNIA = "kornia"
    ALBUMENTATIONS = "albumentations"
    TORCHVISION = "torchvision"
    UNKNOWN = "unknown"


_BACKEND_PREFIXES: dict[str, Backend] = {
    "kornia.": Backend.KORNIA,
    "albumentations.": Backend.ALBUMENTATIONS,
    "torchvision.": Backend.TORCHVISION,
}


def detect_backend(transforms: list[object]) -> Backend:
    """Detect the backend from a list of transforms by inspecting module paths.

    Args:
        transforms: List of transform objects.

    Returns:
        A ``Backend`` enum member.

    Raises:
        ValueError: If transforms come from more than one backend.

    Example:
        >>> detect_backend([])
        <Backend.UNKNOWN: 'unknown'>

    """
    backends: set[Backend] = set()

    for t in transforms:
        module = type(t).__module__ or ""
        backend = _match_backend(module)
        if backend is None:
            warnings.warn(
                f"Unrecognized transform {type(t).__name__!r}; treating as SPATIAL_KERNEL barrier.",
                UserWarning,
                stacklevel=2,
            )
        else:
            backends.add(backend)

    if len(backends) > 1:
        msg = "Mixed backends are not supported in v0.1-v0.4. All transforms must use the same backend."
        raise ValueError(msg)

    if len(backends) == 1:
        return backends.pop()
    return Backend.UNKNOWN


def detect_backends_per_transform(transforms: list[object]) -> list[Backend | None]:
    """Return a per-transform backend list without raising on mixed backends.

    Each entry is the ``Backend`` for the corresponding transform, or ``None``
    if the transform's module could not be matched to any known backend prefix.
    Unrecognised transforms emit a ``UserWarning``.

    Args:
        transforms: List of transform objects.

    Returns:
        List of ``Backend | None``, same length as *transforms*.

    Example:
        >>> detect_backends_per_transform([])
        []

    """
    result: list[Backend | None] = []
    for t in transforms:
        module = type(t).__module__ or ""
        backend = _match_backend(module)
        if backend is None:
            warnings.warn(
                f"Unrecognized transform {type(t).__name__!r}; treating as SPATIAL_KERNEL barrier.",
                UserWarning,
                stacklevel=2,
            )
        result.append(backend)
    return result


def _match_backend(module: str) -> Backend | None:
    """Match a module path to a known backend prefix.

    Args:
        module: The ``__module__`` attribute of a transform type.

    Returns:
        ``Backend`` enum member, or ``None`` if no prefix matches.

    """
    for prefix, backend in _BACKEND_PREFIXES.items():
        if module.startswith(prefix):
            return backend
    return None
