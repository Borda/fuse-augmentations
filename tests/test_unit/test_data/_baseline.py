"""Build the byte-level snapshot the generator's pixel and label output is pinned against.

:mod:`._golden` pins *unit-space geometry* — what a shape looks like before it is placed. This
snapshot pins the other end: the pixels and labels a fully configured run actually emits. The two
answer different questions, and only this one can refuse a change that moves a placement, a fill, or
a visibility flag while every outline stays put.

It exists because the backgrounds/degradations work adds knobs that must be provably inert when
unset. A test comparing two configurations built from the *same* working tree — which is what
``test_asymmetry_jitter_default_leaves_placement_unchanged`` does, correctly, for its own narrower
question — cannot detect a regression that moved both. Only a digest committed before the change
can. This module is therefore snapshotted first, from untouched code, and any later commit that
legitimately moves a byte regenerates it in the same change that moved it.

Regenerate deliberately, never incidentally::

    FUSE_REGEN_BASELINE=1 uv run pytest tests/test_unit/test_data/test_baseline_digests.py

The digest covers more than the pixels: the class id, the outline, the axis-aligned box, the angle
and the landmark table all feed it, so a run that renders identically but relabels an object still
fails. Every float enters as little-endian ``float64``, so the digest is a property of the numbers
rather than of their repr.

"""

from __future__ import annotations

import hashlib

import numpy as np

from fuse_augmentations.data.animals import AnimalShape
from fuse_augmentations.data.backgrounds import (
    GradientBackground,
    ImpulseNoiseBackground,
    NoiseBackground,
    TextureBackground,
)
from fuse_augmentations.data.config import ClassMode, Color, SyntheticConfig, Task
from fuse_augmentations.data.degradations import (
    JPEG,
    ColorCast,
    Contrast,
    GaussianBlur,
    GaussianNoise,
    Quantize,
    Vignette,
)
from fuse_augmentations.data.generator import SyntheticGenerator
from fuse_augmentations.data.letters import LetterShape
from fuse_augmentations.data.sample import Sample

#: Samples drawn per configuration. More than one on purpose: a single sample cannot see a knob that
#: perturbs the shared :class:`numpy.random.Generator` across a stream, which is exactly the failure
#: mode the side-stream RNG contract exists to prevent.
_STREAM_LENGTH = 3

#: The configurations the digest covers, as ``name -> (config, seed)``. Spelled out one by one rather
#: than built as a product: the product would silently include ``Task.KEYPOINTS`` with primitive
#: shapes, which :class:`SyntheticConfig` rejects outright, and small explicit cases are easier to
#: extend than a filtered cartesian sweep.
#:
#: **These entries were snapshotted before any background, degradation, distractor or occluder knob
#: existed**, so they carry a guarantee the entries below cannot: that the knobs are inert when
#: unset. Regenerating one is a claim that the *old* behaviour legitimately moved, which is a much
#: larger claim than regenerating a feature entry. Never add a configuration that sets a new knob
#: here — it would dilute exactly that distinction.
_MATRIX: dict[str, tuple[SyntheticConfig, int]] = {
    "detection-shape-primitives-64": (SyntheticConfig(img_size=64), 0),
    "detection-shape-primitives-192": (SyntheticConfig(img_size=192), 0),
    "detection-color-primitives-64": (SyntheticConfig(img_size=64, class_mode=ClassMode.COLOR), 1),
    "detection-shapecolor-primitives-64": (SyntheticConfig(img_size=64, class_mode=ClassMode.SHAPE_COLOR), 1),
    "segmentation-shape-primitives-64": (SyntheticConfig(img_size=64, task=Task.SEGMENTATION), 0),
    "obb-shape-primitives-64": (SyntheticConfig(img_size=64, task=Task.OBB), 2),
    "keypoints-shape-animals-64": (SyntheticConfig(img_size=64, task=Task.KEYPOINTS, shapes=tuple(AnimalShape)), 0),
    "keypoints-shape-animals-192": (SyntheticConfig(img_size=192, task=Task.KEYPOINTS, shapes=tuple(AnimalShape)), 3),
    "detection-shape-animals-64": (SyntheticConfig(img_size=64, shapes=tuple(AnimalShape)), 4),
    "detection-shape-letters-64": (SyntheticConfig(img_size=64, shapes=tuple(LetterShape)), 5),
    "detection-norotate-primitives-64": (SyntheticConfig(img_size=64, rotate=False), 0),
    "detection-jitter-primitives-64": (SyntheticConfig(img_size=64, asymmetry_jitter=0.2), 0),
    "detection-dense-small-primitives-64": (
        SyntheticConfig(img_size=64, min_objects=3, max_objects=8, min_size_ratio=0.05, max_size_ratio=0.2),
        6,
    ),
    "detection-customfill-primitives-64": (SyntheticConfig(img_size=64, colors=((255, 215, 0), Color.RED)), 7),
}


#: Configurations exercising the knobs added by the backgrounds-and-difficulty work. These were
#: snapshotted from the code that implements them, so they prove **stability**, not inertness: a
#: digest here moving means a rendering change, deliberate or not, and a reviewer decides which. That
#: is a weaker guarantee than :data:`_MATRIX`'s and a necessary one — nothing in that matrix touches
#: :mod:`~fuse_augmentations.data.backgrounds` or
#: :mod:`~fuse_augmentations.data.degradations` at all, so a regression inside either module moved no
#: digest before these existed.
#:
#: :class:`~fuse_augmentations.data.backgrounds.ImageBackground` is deliberately absent: it reads
#: files this package does not ship, so pinning it would pin a fixture directory rather than the
#: renderer. Its own tests cover it against pictures they write themselves.
_FEATURE_MATRIX: dict[str, tuple[SyntheticConfig, int]] = {
    "bg-gradient-sampled-64": (SyntheticConfig(img_size=64, background=GradientBackground()), 0),
    "bg-gradient-radial-64": (
        SyntheticConfig(img_size=64, background=GradientBackground(radial=True, stops=((20, 20, 60), (200, 200, 240)))),
        1,
    ),
    "bg-noise-64": (SyntheticConfig(img_size=64, background=NoiseBackground(sigma=24.0)), 0),
    "bg-impulse-64": (SyntheticConfig(img_size=64, background=ImpulseNoiseBackground(amount=0.08)), 2),
    "bg-texture-64": (SyntheticConfig(img_size=64, background=TextureBackground(octaves=3, frequency=6.0)), 0),
    "bg-texture-quantized-64": (
        SyntheticConfig(img_size=64, background=TextureBackground(octaves=2, frequency=4.0, quantize=5)),
        3,
    ),
    "degrade-chain-64": (
        SyntheticConfig(
            img_size=64,
            degrade=(GaussianBlur(radius=0.8), JPEG(quality=55), Contrast(factor=0.6), Quantize(levels=12)),
        ),
        0,
    ),
    "degrade-noise-cast-vignette-64": (
        SyntheticConfig(
            img_size=64,
            degrade=(GaussianNoise(sigma=9.0), ColorCast(gain=(1.15, 1.0, 0.85)), Vignette(strength=0.4)),
        ),
        4,
    ),
    "distractors-64": (SyntheticConfig(img_size=64, min_objects=2, max_objects=4, distractors=5), 0),
    "occluders-keypoints-96": (
        SyntheticConfig(
            img_size=96,
            task=Task.KEYPOINTS,
            shapes=tuple(AnimalShape)[:4],
            min_objects=3,
            max_objects=3,
            min_size_ratio=0.2,
            max_size_ratio=0.35,
            occluders=5,
        ),
        0,
    ),
    "everything-96": (
        SyntheticConfig(
            img_size=96,
            background=TextureBackground(octaves=2, frequency=5.0),
            min_objects=2,
            max_objects=4,
            distractors=4,
            occluders=3,
            degrade=(GaussianBlur(radius=0.5), JPEG(quality=70)),
        ),
        5,
    ),
}


def _floats(values: object) -> bytes:
    """Return a flat float sequence as little-endian ``float64`` bytes.

    Args:
        values: Anything :func:`numpy.asarray` accepts as a float sequence, including an empty one.

    Returns:
        The values as ``<f8`` bytes, so the digest is fixed by the numbers rather than by the
        platform's native byte order or by a float's decimal repr.

    """
    return np.asarray(values, dtype="<f8").reshape(-1).tobytes()


def _feed(digest: hashlib._Hash, sample: Sample) -> None:
    """Fold one sample's pixels and every label field it carries into a running digest.

    Args:
        digest: The hash object to update in place.
        sample: The sample to absorb.

    Each field is length-prefixed by the update order alone: the annotation count and the landmark
    count both enter as their own byte blocks, so two samples that differ only in how their fields
    split cannot collide.

    """
    digest.update(sample.image.tobytes())
    digest.update(_floats([sample.width, sample.height, len(sample.annotations)]))
    for annotation in sample.annotations:
        digest.update(_floats([annotation.class_id, annotation.angle]))
        digest.update(annotation.class_name.encode("utf-8"))
        digest.update(_floats(annotation.polygon))
        digest.update(_floats(annotation.bbox_xyxy))
        keypoints = annotation.keypoints or ()
        digest.update(_floats([len(keypoints)]))
        digest.update(_floats([value for triple in keypoints for value in triple]))


def baseline_names() -> tuple[str, ...]:
    """Return the configuration names the snapshot covers, in matrix order.

    Returns:
        The keys :func:`build_baseline` produces, both matrices together. Exposed separately so the
        check can parametrize over them without generating every image at collection time.

    """
    return tuple(_MATRIX) + tuple(_FEATURE_MATRIX)


def build_baseline() -> dict[str, str]:
    """Return the ``configuration name -> hex digest`` snapshot over the whole matrix.

    Returns:
        One SHA-256 hex digest per entry of :data:`_MATRIX` and :data:`_FEATURE_MATRIX`, taken over
        a :data:`_STREAM_LENGTH`-sample stream from that configuration's own seed. The two matrices
        answer different questions — see each one's own note — and are kept apart for that reason
        even though they share a file.

    """
    baseline: dict[str, str] = {}
    for name, (config, seed) in (*_MATRIX.items(), *_FEATURE_MATRIX.items()):
        digest = hashlib.sha256()
        for sample in SyntheticGenerator(config).generate(_STREAM_LENGTH, seed=seed):
            _feed(digest, sample)
        baseline[name] = digest.hexdigest()
    return baseline
