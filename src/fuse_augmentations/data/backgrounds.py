"""Canvas fillers the synthetic generator draws its objects on top of.

A background is a *type*, not a mode string. :class:`Background` is abstract and every concrete
subclass carries exactly the parameters its own mode needs, with defaults, and renders itself. That
removes the question a ``mode="noise"`` field would create — which parameters are legal together —
because a class that has no ``sigma`` cannot be given one, and the constructor says so rather than a
hand-written cross-field check. It is also the extension point: a third party's own
:class:`Background` works with no registration, matching the
:func:`~fuse_augmentations.data.writers.register_writer` culture already in the package.

Every background renders from a **side stream**, never from the generator's own placement stream, so
turning one on cannot move an object. :attr:`Background.consumes_randomness` is what
:class:`~fuse_augmentations.data.generator.SyntheticGenerator` asks before taking that side stream at
all, which is why a background that draws nothing is handed ``None`` and must not touch it.

Numpy only at module scope, plus Pillow inside the one method that needs a resampler: importing this
module is as cheap as importing :mod:`~fuse_augmentations.data.config`, which it deliberately does
not force to grow heavier.

Examples:
    ```pycon
    >>> import numpy as np
    >>> from fuse_augmentations.data.backgrounds import NoiseBackground, SolidBackground
    >>> SolidBackground((10, 20, 30)).render(None, 4).shape
    (4, 4, 3)
    >>> canvas = NoiseBackground(sigma=8.0).render(np.random.default_rng(0), 8)
    >>> canvas.dtype
    dtype('uint8')

    ```

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fuse_augmentations.data.config import ColorLike, Fill

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: The grey every mode falls back to, and the fill the generator drew before backgrounds were types.
#: Spelled once here so the default of five dataclasses cannot drift apart.
DEFAULT_BASE: tuple[int, int, int] = (128, 128, 128)

#: Highest sigma :class:`NoiseBackground` may use while the rasterized-ink oracle in
#: ``tests/test_unit/test_data/test_bbox_from_polygon.py`` stays sound. That oracle finds ink by
#: distance to pure red under a tolerance of 90, and Gaussian noise is unbounded, so its tail decides
#: this rather than its mean. Measured on a 192x192x3 canvas around ``(128, 128, 128)``: 0 false-ink
#: pixels at sigma 8, 16 and 32, then 36 at 48 and 248 at 64. A single stray pixel widens the
#: oracle's box, so the bound is the last value that produced none.
INK_SAFE_SIGMA = 32.0


def _rgb(color: ColorLike) -> NDArray[np.float32]:
    """Return a fill as a ``(3,)`` float array, validating the spelling on the way through."""
    return np.asarray(Fill.parse(color).rgb, dtype=np.float32)


def _to_canvas(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Round a float canvas to the writable, C-contiguous ``uint8`` array a renderer must return.

    Args:
        values: ``(img_size, img_size, 3)`` float values in any range.

    Returns:
        The same canvas clipped to ``[0, 255]`` and rounded once, as a fresh ``uint8`` array. One
        rounding rather than several is what keeps two implementations of the same ramp agreeing.

    """
    return np.ascontiguousarray(np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8))


def _require_stream(rng: np.random.Generator | None, mode: str) -> np.random.Generator:
    """Return the side stream a drawing background was promised, or say who broke the contract.

    Args:
        rng: The stream handed to :meth:`Background.render`.
        mode: The class name, so the message names the background that needs one.

    Returns:
        ``rng`` unchanged.

    Raises:
        TypeError: If ``rng`` is ``None``. Only a background whose
            :attr:`Background.consumes_randomness` is ``False`` is handed ``None``, so reaching here
            means the two disagree — a caller error, and one worth naming rather than letting it
            surface as an attribute error on ``None``.

    """
    if rng is None:
        raise TypeError(f"{mode} draws randomness but was handed no side stream; consumes_randomness says it needs one")
    return rng


class Background(ABC):
    """A canvas filler whose subclass *is* the mode, so there is no mode field.

    Implement :meth:`render` and, when the mode draws nothing, override :attr:`consumes_randomness` — the generator
    reads it to decide whether to take a side stream at all, and a background that claims to draw nothing is handed
    ``None`` in place of one.

    """

    @property
    def consumes_randomness(self) -> bool:
        """Return whether :meth:`render` draws from the generator it is handed.

        Defaults to ``True``, which is always safe: a side stream that is taken and not used costs one cheap child and
        moves nothing, while a background that draws from a stream it said it would not need would receive ``None`` and
        fail loudly rather than quietly.

        """
        return True

    @abstractmethod
    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return the canvas this background fills, as ``(img_size, img_size, 3)`` ``uint8``.

        Args:
            rng: The side stream to draw from, or ``None`` when :attr:`consumes_randomness` is
                ``False`` for this instance. It is never the generator's placement stream.
            img_size: Square canvas side length in pixels.

        Returns:
            A writable, C-contiguous RGB canvas.

        """


@dataclass(frozen=True)
class SolidBackground(Background):
    """One flat colour across the whole canvas — the behaviour that predates this module.

    Args:
        color: The fill, as a :class:`~fuse_augmentations.data.config.Color`, an ``(r, g, b)``
            triple, or a :class:`~fuse_augmentations.data.config.Fill`. Normalized to a ``Fill`` at
            construction, like every other fill in the package.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.backgrounds import SolidBackground
        >>> SolidBackground((20, 30, 40)).render(None, 2)[0, 0].tolist()
        [20, 30, 40]

        ```

    """

    color: ColorLike = DEFAULT_BASE

    def __post_init__(self) -> None:
        """Normalize the fill once, so every later read holds a :class:`Fill`."""
        object.__setattr__(self, "color", Fill.parse(self.color))

    @property
    def consumes_randomness(self) -> bool:
        """Return ``False``: a flat fill has nothing to draw."""
        return False

    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return a canvas of one repeated colour, ignoring ``rng`` entirely."""
        return np.ascontiguousarray(np.broadcast_to(_rgb(self.color).astype(np.uint8), (img_size, img_size, 3)))


@dataclass(frozen=True)
class GradientBackground(Background):
    """A linear or radial ramp between two stops.

    Breaks the "background is one value" assumption a colour-mode classifier can otherwise lean on:
    the same object fill now sits on a different local intensity depending on where it landed.

    Args:
        stops: The two ends of the ramp. For a linear ramp the first is at the low end of the
            direction axis; for a radial one it is at the canvas centre and the second at the
            corners.
        direction: Ramp angle in radians for a linear ramp, or ``None`` to sample one per image.
            Ignored entirely when ``radial`` is set, which has no direction to speak of.
        radial: Ramp outward from the centre rather than across the canvas.

    Raises:
        ValueError: If ``stops`` does not hold exactly two fills.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.backgrounds import GradientBackground
        >>> ramp = GradientBackground(direction=0.0).render(None, 4)
        >>> bool(ramp[0, 0, 0] < ramp[0, -1, 0])
        True

        ```

    """

    stops: tuple[ColorLike, ColorLike] = ((64, 64, 64), (192, 192, 192))
    direction: float | None = None
    radial: bool = False

    def __post_init__(self) -> None:
        """Normalize both stops and reject a ramp that does not have exactly two ends."""
        if len(tuple(self.stops)) != 2:
            raise ValueError(f"stops must hold exactly two fills, got {len(tuple(self.stops))}")
        object.__setattr__(self, "stops", tuple(Fill.parse(stop) for stop in self.stops))

    @property
    def consumes_randomness(self) -> bool:
        """Return whether the ramp angle still has to be sampled.

        Only a linear ramp with no fixed ``direction`` draws. A radial ramp is centred and has no angle at all, so it
        draws nothing whatever ``direction`` says — which is why the draw count is keyed on this property rather than on
        ``direction`` alone.

        """
        return self.direction is None and not self.radial

    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return the ramp, interpolating between the stops in float and rounding once."""
        first, second = (_rgb(stop) for stop in self.stops)
        return _to_canvas(first + self._ramp(rng, img_size)[..., None] * (second - first))

    def _ramp(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.float32]:
        """Return the scalar field the stops are interpolated over, normalized to ``[0, 1]``.

        The linear field is rescaled by its own extent rather than by a closed form, so the ramp spans both stops
        exactly at every angle instead of compressing toward the diagonals.

        """
        rows, columns = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
        if self.radial:
            centre = (img_size - 1) / 2.0
            distance = np.hypot(rows - centre, columns - centre)
            return np.asarray(distance / max(float(distance.max()), 1e-6), dtype=np.float32)
        if self.direction is not None:
            angle = float(self.direction)
        else:
            angle = float(_require_stream(rng, type(self).__name__).uniform(0.0, 2.0 * math.pi))
        field = math.cos(angle) * columns + math.sin(angle) * rows
        span = float(field.max() - field.min())
        return np.asarray((field - field.min()) / max(span, 1e-6), dtype=np.float32)


@dataclass(frozen=True)
class NoiseBackground(Background):
    """Per-pixel Gaussian noise around a base colour.

    The first knob that makes a small object genuinely hard: it removes the trivial edge detector a
    flat canvas hands a model for free.

    Args:
        base: The colour the noise is centred on.
        sigma: Per-channel standard deviation in 8-bit units. Keep it at or below
            :data:`INK_SAFE_SIGMA` for any run scored by a rasterized-ink oracle; see that constant
            for the measurement.

    Raises:
        ValueError: If ``sigma`` is negative.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.backgrounds import NoiseBackground
        >>> NoiseBackground(sigma=4.0).render(np.random.default_rng(0), 4).shape
        (4, 4, 3)

        ```

    """

    base: ColorLike = DEFAULT_BASE
    sigma: float = 16.0

    def __post_init__(self) -> None:
        """Normalize the base fill and reject a negative spread."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return the base colour plus one Gaussian field, added rather than multiplied."""
        stream = _require_stream(rng, type(self).__name__)
        noise = stream.standard_normal((img_size, img_size, 3)).astype(np.float32) * float(self.sigma)
        return _to_canvas(_rgb(self.base) + noise)


@dataclass(frozen=True)
class ImpulseNoiseBackground(Background):
    """Salt-and-pepper pixels scattered over a base colour.

    The same axis as :class:`NoiseBackground` but heavy-tailed, and the reason it is a separate
    difficulty step: impulse pixels survive a blur that erases a Gaussian field.

    Args:
        base: The colour the surviving pixels keep.
        amount: Fraction of pixels replaced, in ``[0, 1]``.
        salt_ratio: Of those, the fraction set white rather than black, in ``[0, 1]``.

    Raises:
        ValueError: If ``amount`` or ``salt_ratio`` falls outside ``[0, 1]``.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.backgrounds import ImpulseNoiseBackground
        >>> canvas = ImpulseNoiseBackground(amount=0.5).render(np.random.default_rng(0), 16)
        >>> bool((canvas == 255).any() and (canvas == 0).any())
        True

        ```

    """

    base: ColorLike = DEFAULT_BASE
    amount: float = 0.05
    salt_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Normalize the base fill and reject fractions outside the unit interval."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        for name, value in (("amount", self.amount), ("salt_ratio", self.salt_ratio)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")

    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return the base colour with a fraction of pixels replaced outright by black or white.

        Replacement, not addition: adding a fixed salt value to an arbitrary base would not produce endpoint pixels,
        which is the whole point of impulse noise and the reason it survives a blur.

        """
        stream = _require_stream(rng, type(self).__name__)
        canvas = np.broadcast_to(_rgb(self.base), (img_size, img_size, 3)).copy()
        selected = stream.random((img_size, img_size)) < float(self.amount)
        salt = stream.random((img_size, img_size)) < float(self.salt_ratio)
        canvas[selected & salt] = 255.0
        canvas[selected & ~salt] = 0.0
        return _to_canvas(canvas)


@dataclass(frozen=True)
class TextureBackground(Background):
    """Value noise at a chosen spatial frequency, optionally posterized.

    The first background that puts *structure* at object scale, so a false positive becomes possible
    rather than merely a matter of contrast.

    Args:
        base: The colour the field deviates around.
        amplitude: Peak deviation from ``base`` in 8-bit units.
        frequency: Lattice cells across the image at the first octave.
        octaves: Number of octaves summed, each at twice the frequency and half the weight.
        quantize: Number of levels to posterize the field to before it is mapped to colour, or
            ``None`` for a continuous field.

    Raises:
        ValueError: If ``amplitude`` is negative, ``frequency`` is not positive, ``octaves`` is below
            1, or ``quantize`` is given and below 2.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> from fuse_augmentations.data.backgrounds import TextureBackground
        >>> TextureBackground(octaves=2).render(np.random.default_rng(0), 32).shape
        (32, 32, 3)

        ```

    """

    base: ColorLike = DEFAULT_BASE
    amplitude: float = 48.0
    frequency: float = 8.0
    octaves: int = 3
    quantize: int | None = None

    def __post_init__(self) -> None:
        """Normalize the base fill and reject a lattice that cannot be built."""
        object.__setattr__(self, "base", Fill.parse(self.base))
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be at least 1, got {self.octaves}")
        if self.quantize is not None and self.quantize < 2:
            raise ValueError(f"quantize must be at least 2 levels when given, got {self.quantize}")

    def render(self, rng: np.random.Generator | None, img_size: int) -> NDArray[np.uint8]:
        """Return the base colour plus the summed octaves, added rather than multiplied.

        Additive on purpose: a multiplicative field would make the effective contrast depend on
        ``base``, so the same ``amplitude`` would mean different things on a dark and a light canvas.

        """
        field = self._field(_require_stream(rng, type(self).__name__), img_size)
        return _to_canvas(_rgb(self.base) + float(self.amplitude) * field[..., None])

    def _field(self, rng: np.random.Generator, img_size: int) -> NDArray[np.float32]:
        """Return the summed, normalized and optionally posterized value-noise field in ``[-1, 1]``.

        One uniform lattice is drawn per octave and upsampled bicubically, so the draw count is the octave count
        exactly. Weights halve per octave and the sum is divided by their total, which is what bounds the result rather
        than a second pass over the data.

        """
        from PIL import Image

        field = np.zeros((img_size, img_size), dtype=np.float32)
        weights = 0.0
        for octave in range(self.octaves):
            cells = math.ceil(float(self.frequency) * 2**octave) + 1
            lattice = rng.uniform(-1.0, 1.0, size=(cells, cells)).astype(np.float32)
            upsampled = Image.fromarray(lattice, mode="F").resize((img_size, img_size), Image.Resampling.BICUBIC)
            weight = 0.5**octave
            field += np.asarray(upsampled, dtype=np.float32) * weight
            weights += weight
        field = np.clip(field / weights, -1.0, 1.0)
        if self.quantize is None:
            return field
        levels = int(self.quantize) - 1
        return np.asarray(np.rint((field + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0, dtype=np.float32)
