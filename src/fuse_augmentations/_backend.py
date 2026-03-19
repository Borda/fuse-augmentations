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
        msg = (
            "Mixed backends are not supported by detect_backend(); all transforms must "
            "use the same backend. For mixed-backend pipelines, use "
            "detect_backends_per_transform()."
        )
        raise ValueError(msg)

    if len(backends) == 1:
        return backends.pop()
    return Backend.UNKNOWN


def detect_backends_per_transform(transforms: list[object]) -> list[Backend | None]:
    """Return a per-transform backend list without raising on mixed backends.

    Each entry is the ``Backend`` for the corresponding transform, or ``None``
    if the transform's module could not be matched to any known backend prefix.
    Unrecognised transforms emit a ``UserWarning``.

    When a direct module-prefix match fails, the function falls back to
    checking the transform's MRO (method resolution order) for any ancestor
    class whose ``__module__`` matches a known backend prefix. This handles
    subclasses defined outside the backend package (e.g. a user-defined
    ``class MyRot(torchvision.transforms.RandomRotation)`` in ``__main__``).

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
            backend = _match_backend_from_mro(type(t))
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


def _match_backend_from_mro(cls: type) -> Backend | None:
    """Walk the MRO looking for an ancestor whose module matches a known backend.

    Skips ``object`` and the class itself (already checked by the caller via
    its direct ``__module__``).

    Args:
        cls: The type of the transform.

    Returns:
        ``Backend`` enum member from the first matching ancestor, or ``None``.

    """
    for ancestor in cls.__mro__[1:]:
        if ancestor is object:
            continue
        module = ancestor.__module__ or ""
        backend = _match_backend(module)
        if backend is not None:
            return backend
    return None
