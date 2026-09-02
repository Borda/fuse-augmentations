"""Random-draw helpers that honour a caller-supplied :class:`torch.Generator`.

Every pipeline-owned draw (per-transform probability gates, direct parameter
sampling) routes through :func:`rand` or :func:`uniform` so a caller can own the
random stream instead of sharing the global one.

Two invariants hold across this module:

- ``generator=None`` reproduces the historical call *exactly* — same function,
  same device, no extra hop — so existing pipelines keep their bit-for-bit
  behaviour on every device.
- A generator whose device differs from the target device draws on the
  generator's device and copies the result across, so one CPU generator can seed
  a CUDA or MPS pipeline. ``torch`` itself rejects that combination.

"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["rand", "reject_backend_randomness", "uniform"]


def _draw_device(generator: torch.Generator, device: torch.device) -> torch.device:
    """Return the device to draw on for ``generator`` targeting ``device``.

    Args:
        generator: The caller-supplied generator.
        device: The device the drawn tensor is needed on.

    Returns:
        ``device`` when the generator can drive it directly, else the generator's own device.

    """
    return device if generator.device.type == device.type else generator.device


def rand(
    shape: tuple[int, ...] | int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw uniform ``[0, 1)`` samples, optionally from a caller-owned generator.

    Args:
        shape: Output shape; ``()`` for a scalar draw.
        device: Device the returned tensor must live on.
        generator: Caller-owned generator, or ``None`` for the global stream.

    Returns:
        A float tensor of ``shape`` on ``device``.

    Examples:
        ```pycon
        >>> import torch
        >>> from fuse_augmentations._random import rand
        >>> gen = torch.Generator().manual_seed(0)
        >>> first = rand(3, device=torch.device("cpu"), generator=gen)
        >>> gen.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> bool(torch.equal(first, rand(3, device=torch.device("cpu"), generator=gen)))
        True

        ```

    """
    if generator is None:
        return torch.rand(shape, device=device)
    draw_device = _draw_device(generator, device)
    drawn = torch.rand(shape, device=draw_device, generator=generator)
    return drawn if draw_device == device else drawn.to(device)


def uniform(
    shape: tuple[int, ...] | int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw uniform ``[low, high)`` samples, optionally from a caller-owned generator.

    Args:
        shape: Output shape; ``()`` for a scalar draw.
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        device: Device the returned tensor must live on.
        dtype: Output dtype, or ``None`` for the torch default.
        generator: Caller-owned generator, or ``None`` for the global stream.

    Returns:
        A tensor of ``shape`` on ``device`` filled with uniform samples.

    Examples:
        ```pycon
        >>> import torch
        >>> from fuse_augmentations._random import uniform
        >>> gen = torch.Generator().manual_seed(0)
        >>> drawn = uniform(4, -1.0, 1.0, device=torch.device("cpu"), generator=gen)
        >>> bool((drawn >= -1.0).all() and (drawn < 1.0).all())
        True

        ```

    """
    if generator is None:
        return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)
    draw_device = _draw_device(generator, device)
    drawn = torch.empty(shape, device=draw_device, dtype=dtype).uniform_(low, high, generator=generator)
    return drawn if draw_device == device else drawn.to(device)


def reject_backend_randomness(generator: torch.Generator | None, source: str) -> None:
    """Raise when a caller-owned generator would reach a backend-owned draw.

    Backend transforms sample through their own libraries (Kornia's samplers,
    Albumentations' ``py_random`` and ``np.random``, TorchVision's ``make_params``),
    none of which accept a ``torch.Generator``. Falling back to the global stream
    there would look reproducible without being reproducible, so the pipeline
    refuses the combination instead.

    Args:
        generator: The caller-owned generator, or ``None``.
        source: Human-readable description of the draw that cannot be seeded.

    Raises:
        ValueError: If ``generator`` is not ``None``.

    """
    if generator is None:
        return
    msg = (
        f"generator= cannot seed {source}: that draw belongs to the backend library and does not accept a "
        "torch.Generator. Use FusedCompose.from_params(...) (or backend='native') for a fully caller-seeded "
        "pipeline, or drop generator= and seed the backend's own global stream."
    )
    raise ValueError(msg)
