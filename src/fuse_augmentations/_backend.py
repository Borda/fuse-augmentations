"""Backend detection for augmentation transform pipelines.

Inspects transform module paths to determine which backend framework (Kornia, Albumentations, TorchVision) is in use.

Detection is driven by a pluggable adapter registry. The three built-in backends self-register at import time via
:func:`register_adapter`. Third-party adapters may register through the ``fuse_augmentations.adapters`` entry-point
group; those are loaded **lazily** on the first detection miss (never at package import) so that ``import
fuse_augmentations`` neither executes third-party code nor pays their import cost.

Example:
    >>> from fuse_augmentations._backend import detect_backend
    >>> detect_backend([])
    <Backend.UNKNOWN: 'unknown'>

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuse_augmentations.types import TransformAdapter


class Backend(Enum):
    """Supported augmentation backend frameworks."""

    KORNIA = "kornia"
    ALBUMENTATIONS = "albumentations"
    TORCHVISION = "torchvision"
    UNKNOWN = "unknown"


#: Name of the entry-point group third-party packages use to register adapters.
ADAPTERS_ENTRY_POINT_GROUP = "fuse_augmentations.adapters"


@dataclass(frozen=True, slots=True)
class _Entry:
    """A registered adapter: its backend tag, module prefixes, and capabilities.

    Attributes:
        backend: The ``Backend`` this adapter detects (``Backend.UNKNOWN`` for third-party adapters that map to no
            built-in enum member).
        adapter: The adapter instance implementing ``TransformAdapter``.
        prefixes: Module-path prefixes (e.g. ``"kornia."``) that identify transforms handled by this adapter.
        capabilities: Canonical op names the adapter can build.
    """

    backend: Backend
    adapter: TransformAdapter
    prefixes: tuple[str, ...]
    capabilities: frozenset[str] = field(default_factory=frozenset)


#: name -> _Entry. Built-ins self-register at import; third-party adapters load lazily (see ``_load_entrypoints``).
_ADAPTER_REGISTRY: dict[str, _Entry] = {}

#: Guard so the entry-point group is scanned at most once (idempotent; a failed scan is not retried repeatedly).
_ENTRYPOINTS_LOADED = False


def adapter_capabilities(adapter: object) -> frozenset[str]:
    """Return an adapter's declared ``capabilities``, defaulting to empty.

    ``capabilities`` is an optional member of the :class:`~fuse_augmentations.types.TransformAdapter` protocol; adapters
    predating it need not define it. This getattr helper keeps the registry backwards-compatible.

    Args:
        adapter: An adapter instance (or class).

    Returns:
        The adapter's ``capabilities`` as a ``frozenset[str]``, or an empty frozenset when the member is absent.

    Example:
        >>> adapter_capabilities(object())
        frozenset()

    """
    return frozenset(getattr(adapter, "capabilities", frozenset()))


def register_adapter(
    name: str,
    adapter: TransformAdapter,
    module_prefixes: str | tuple[str, ...] | list[str],
    *,
    backend: Backend = Backend.UNKNOWN,
) -> None:
    """Register a backend adapter for detection and fusion.

    .. warning::

        **Experimental.** ``register_adapter`` and the ``fuse_augmentations.adapters`` entry-point group are a
        provisional third-party extension API. The signature may change until an external adapter validates it.

    Args:
        name: Unique registry key for the adapter (re-registering the same name overwrites the prior entry).
        adapter: An instance implementing the :class:`~fuse_augmentations.types.TransformAdapter` protocol.
        module_prefixes: One or more module-path prefixes (e.g. ``"mypkg.transforms."``) whose transforms this
            adapter handles. A trailing dot is recommended to avoid spurious prefix collisions.
        backend: The :class:`Backend` enum member this adapter maps to. Defaults to ``Backend.UNKNOWN`` for
            third-party backends without a built-in enum member.

    Example:
        >>> class _Dummy:
        ...     capabilities = frozenset({"rotation"})
        >>> register_adapter("dummy", _Dummy(), "dummypkg.")
        >>> "dummy" in _ADAPTER_REGISTRY
        True

    """
    prefixes = (module_prefixes,) if isinstance(module_prefixes, str) else tuple(module_prefixes)
    _ADAPTER_REGISTRY[name] = _Entry(
        backend=backend,
        adapter=adapter,
        prefixes=prefixes,
        capabilities=adapter_capabilities(adapter),
    )


def _load_entrypoints() -> None:
    """Lazily load third-party adapters from the entry-point group (idempotent, failure-isolated).

    Called on the first detection miss, never at package import (avoids executing third-party code and
    paying its import cost on ``import fuse_augmentations``). Each entry point is loaded in isolation; a failing
    ``load()`` or ``register()`` is warned and skipped so one broken plugin cannot break detection for the rest.

    """
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True

    from importlib import metadata

    try:
        entry_points = metadata.entry_points(group=ADAPTERS_ENTRY_POINT_GROUP)
    except Exception as exc:
        warnings.warn(f"Failed to query adapter entry points: {exc!r}", UserWarning, stacklevel=2)
        return

    for ep in entry_points:
        _load_one_entrypoint(ep)


def _load_one_entrypoint(ep: object) -> None:
    """Load and invoke a single adapter entry point in isolation.

    Failure isolation is per entry point: a broken ``load()`` or ``register()`` is warned and skipped so one bad
    plugin cannot break detection for the rest (kept a separate function to isolate the broad ``except`` from the
    discovery loop).

    Args:
        ep: An ``importlib.metadata.EntryPoint`` whose ``load()`` returns a zero-arg registration callable.

    """
    try:
        register = ep.load()  # type: ignore[attr-defined]
        register()
    except Exception as exc:
        name = getattr(ep, "name", ep)
        warnings.warn(f"Failed to load adapter entry point {name!r}: {exc!r}; skipping.", UserWarning, stacklevel=2)


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

    for transform in transforms:
        module = type(transform).__module__ or ""
        backend = _match_backend(module)
        if backend is None:
            warnings.warn(
                f"Unrecognized transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
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

    Note:
        This is a semi-public API accessible via ``fuse_augmentations._backend``.
        It is not part of the stable public surface and may change without notice.

    Args:
        transforms: List of transform objects.

    Returns:
        List of ``Backend | None``, same length as *transforms*.

    Example:
        >>> detect_backends_per_transform([])
        []

    """
    result: list[Backend | None] = []
    for transform in transforms:
        module = type(transform).__module__ or ""
        backend = _match_backend(module)
        if backend is None:
            backend = _match_backend_from_mro(type(transform))
        if backend is None:
            warnings.warn(
                f"Unrecognized transform {type(transform).__name__!r}; treating as SPATIAL_KERNEL barrier.",
                UserWarning,
                stacklevel=2,
            )
        result.append(backend)
    return result


def _lookup_prefix(module: str) -> Backend | None:
    """Return the backend whose registered prefix best (longest) matches *module*, or ``None``.

    Args:
        module: A transform type's ``__module__`` string.

    Returns:
        The ``Backend`` of the longest matching registered prefix, or ``None`` when nothing matches.

    """
    best_len = -1
    best: Backend | None = None
    for entry in _ADAPTER_REGISTRY.values():
        for prefix in entry.prefixes:
            if module.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best = entry.backend
    return best


def _match_backend(module: str) -> Backend | None:
    """Match a module path to a registered backend prefix.

    Consults the adapter registry (longest-prefix match wins). On a miss, third-party adapters are loaded lazily from
    the entry-point group and the lookup is retried once.

    Args:
        module: The ``__module__`` attribute of a transform type.

    Returns:
        ``Backend`` enum member, or ``None`` if no prefix matches.

    """
    backend = _lookup_prefix(module)
    if backend is not None:
        return backend
    if not _ENTRYPOINTS_LOADED:
        _load_entrypoints()
        return _lookup_prefix(module)
    return None


def _match_backend_from_mro(cls: type) -> Backend | None:
    """Walk the MRO looking for an ancestor whose module matches a known backend.

    Skips ``object`` and the class itself (already checked by the caller via its direct ``__module__``).

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


def _register_builtins() -> None:
    """Self-register the three built-in adapters (kornia, torchvision, albumentations) at import.

    Kept as a function (invoked at module bottom) so import order is explicit and the registry is populated exactly
    once. Adapter classes carry their own ``capabilities`` frozenset; ``register_adapter`` reads it via the getattr
    helper, so an adapter missing the member simply registers with empty capabilities.

    """
    from fuse_augmentations.adapters import AlbumentationsAdapter, KorniaAdapter, TorchVisionAdapter

    register_adapter("kornia", KorniaAdapter(), "kornia.", backend=Backend.KORNIA)
    register_adapter("albumentations", AlbumentationsAdapter(), "albumentations.", backend=Backend.ALBUMENTATIONS)
    register_adapter("torchvision", TorchVisionAdapter(), "torchvision.", backend=Backend.TORCHVISION)


_register_builtins()
