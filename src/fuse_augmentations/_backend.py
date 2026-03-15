"""Backend detection for augmentation transform pipelines.

Inspects transform module paths to determine which backend framework
(Kornia, Albumentations, TorchVision) is in use.

Example:
    >>> from fuse_augmentations._backend import detect_backend
    >>> detect_backend([])
    'unknown'
"""

from __future__ import annotations

import warnings

_BACKEND_PREFIXES: dict[str, str] = {
    "kornia.": "kornia",
    "albumentations.": "albumentations",
    "torchvision.": "torchvision",
}


def detect_backend(transforms: list[object]) -> str:
    """Detect the backend from a list of transforms by inspecting module paths.

    Args:
        transforms: List of transform objects.

    Returns:
        One of ``"kornia"``, ``"albumentations"``, ``"torchvision"``, or ``"unknown"``.

    Raises:
        ValueError: If transforms come from more than one backend.

    Example:
        >>> detect_backend([])
        'unknown'
    """
    backends: set[str] = set()

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
        msg = "Mixed backends are not supported in v0.1-v0.5. All transforms must use the same backend."
        raise ValueError(msg)

    if len(backends) == 1:
        return backends.pop()
    return "unknown"


def _match_backend(module: str) -> str | None:
    """Match a module path to a known backend prefix.

    Args:
        module: The ``__module__`` attribute of a transform type.

    Returns:
        Backend name string, or ``None`` if no prefix matches.
    """
    for prefix, name in _BACKEND_PREFIXES.items():
        if module.startswith(prefix):
            return name
    return None
