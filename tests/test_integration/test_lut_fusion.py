"""Correctness + contract tests for lookup-table fusion of pointwise non-linear ops.

Gamma, solarize, and posterize are per-channel non-linear scalar maps. A contiguous run of
them collapses into a single :class:`~fuse_augmentations.affine.segment.FusedLUTSegment` that
composes the maps into one lookup and applies it once. This module verifies:

- **uint8 exact-by-enumeration** — on the Albumentations native (uint8 NumPy) path the fused
  lookup is bit-exact against a native sequential chain over all 256 byte values in every
  channel. The Kornia/TorchVision integer-tensor path composes each op's byte map by exact
  integer indexing, matching a sequential per-op byte-map chain.
- **float interpolation tolerance** — on floating tensors the composed map is sampled on a
  uniform 1024-point grid and applied by gather + linear interpolation. This is *not* exact:
  the tests assert parity within a documented interpolation tolerance (never ``>=`` native).
  Smooth maps (gamma) stay within ~2 uint8 levels; a discontinuous map (posterize step,
  off-centre solarize threshold) smears one grid cell, so a small fraction of pixels near the
  discontinuity may differ by up to one step height while the mean error stays tiny.
- **fusion plan / warp accounting** — a fused run reports as ``lut(...)`` (never passthrough)
  and credits ``n_warps_saved = n - 1``.
- **barriers preserved** — cross-channel ops (saturation/hue) and runtime-histogram ops
  (equalize) stay on the passthrough path and never fold into a lookup.
- **pickle round-trip** — a pipeline containing a ``FusedLUTSegment`` survives ``pickle`` (the
  DataLoader-worker contract).

Determinism is achieved with degenerate parameter ranges (e.g. ``gamma_limit=(70, 70)``), so the
fused and reference paths realise the identical map independent of RNG.

"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import (
    _ALBUMENTATIONS_AVAILABLE,
    _KORNIA_AVAILABLE,
    _TORCHVISION_V2_AVAILABLE,
)
from fuse_augmentations.affine.segment import FusedLUTSegment
from fuse_augmentations.types import TransformCategory

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters.kornia import KorniaAdapter
if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu
if _TORCHVISION_V2_AVAILABLE:
    import torchvision.transforms.v2 as tv2

    from fuse_augmentations.adapters.torchvision import TorchVisionAdapter

# Documented float interpolation tolerances (uint8 levels, i.e. fraction of 255) for the
# K=1024-entry interp-LUT path, measured against an exact composition of the same ops:
#   - smooth maps (gamma): max error stays within ~2 levels (steep gamma<1 near black).
#   - discontinuous maps (posterize step, off-centre solarize): the discontinuity is smeared
#     across one grid cell, so the MEAN error is tiny but up to ~1/K of pixels at the step edge
#     may differ by up to one step height. Parity is therefore gated on mean + outlier fraction,
#     never on max, and never claimed to beat native precision.
_GAMMA_FLOAT_MAX_LEVELS = 2.5
_DISCONTINUOUS_FLOAT_MEAN_LEVELS = 0.5
_DISCONTINUOUS_FLOAT_OUTLIER_FRACTION = 0.02  # <=2% of pixels may exceed 2 levels at a discontinuity

pytestmark = pytest.mark.integration


def _channel_ramp_uint8(size: int = 16) -> np.ndarray:
    """Return an ``(size, size, 3)`` uint8 image whose every channel spans all 256 byte values."""
    ramp = np.arange(256, dtype=np.uint8).reshape(size, size)
    return np.stack([ramp, ramp, ramp], axis=-1)


def _sequential_byte_map_reference(adapter, transforms, ramp: torch.Tensor) -> torch.Tensor:
    """Apply each op's own 256-entry byte map to ``ramp`` in sequence (snap to bytes between ops).

    Independent of :class:`FusedLUTSegment`'s internal threading, so equality is a genuine
    cross-check that the fused integer-tensor path composes byte maps without loss.

    """
    levels = 256
    grid = (torch.arange(levels, dtype=torch.float32) / (levels - 1)).view(1, 1, levels).expand(1, 3, levels)
    reference = ramp.clone()
    for transform in transforms:
        params = adapter.sample_params(transform, (1, 3, 16, 16), torch.device("cpu"))
        mapped = adapter.build_lut(transform, params, grid.clone())
        byte_lut = (mapped * (levels - 1)).round().clamp(0, levels - 1).long()
        idx = reference.long().reshape(1, 3, -1)
        reference = torch.gather(byte_lut, 2, idx).reshape(1, 3, 16, 16).to(ramp.dtype)
    return reference


# ---------------------------------------------------------------------------
# uint8 exact-by-enumeration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
class TestAlbumentationsUint8BitExact:
    """The Albumentations native uint8 path is bit-exact vs a native sequential chain."""

    @staticmethod
    def _fixed_ops() -> list[object]:
        """Return gamma/solarize/posterize with degenerate (deterministic) parameter ranges."""
        return [
            albu.RandomGamma(gamma_limit=(70, 70), p=1.0),
            albu.Solarize(threshold_range=(0.4, 0.4), p=1.0),
            albu.Posterize(num_bits=3, p=1.0),
        ]

    def test_fused_lut_bit_exact_over_all_bytes_and_channels(self):
        """The fused composed lookup equals the native sequential chain for every byte, every channel."""
        image = _channel_ramp_uint8()
        fused = Compose(self._fixed_ops())(image=image.copy())["image"]

        sequential = image.copy()
        for transform in self._fixed_ops():
            sequential = transform(image=sequential)["image"]

        assert np.array_equal(fused, sequential)

    def test_fused_run_reports_single_lookup(self):
        """The three ops collapse into one ``lut(...)`` segment saving two passes."""
        pipe = Compose(self._fixed_ops())
        assert pipe.fusion_plan == "lut(RandomGamma, Solarize, Posterize)"
        assert pipe.n_warps_saved == 2


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        pytest.param(
            lambda: [
                kornia_aug.RandomGamma((0.7, 0.7), (1.0, 1.0), p=1.0),
                kornia_aug.RandomSolarize((0.3, 0.3), (0.0, 0.0), p=1.0),
                kornia_aug.RandomPosterize((3, 3), p=1.0),
            ],
            "lut(RandomGamma, RandomSolarize, RandomPosterize)",
            id="kornia-gamma-solarize-posterize",
        ),
    ],
)
def test_kornia_integer_tensor_path_matches_sequential_byte_maps(factory, expected):
    """The integer-tensor 256-LUT composes each op's byte map by exact integer indexing."""
    adapter = KorniaAdapter()
    transforms = factory()
    ramp = torch.from_numpy(_channel_ramp_uint8()).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 16, 16) uint8

    segment = FusedLUTSegment(list(transforms), adapter)
    fused = segment(ramp.clone())
    reference = _sequential_byte_map_reference(adapter, transforms, ramp)

    assert torch.equal(fused, reference)
    assert Compose(factory()).fusion_plan == expected


@pytest.mark.skipif(not _TORCHVISION_V2_AVAILABLE, reason="torchvision v2 required")
def test_torchvision_integer_tensor_path_matches_sequential_byte_maps():
    """TorchVision solarize+posterize integer-tensor lookups compose by exact integer indexing."""
    adapter = TorchVisionAdapter()
    transforms = [tv2.RandomSolarize(0.4, p=1.0), tv2.RandomPosterize(3, p=1.0)]
    ramp = torch.from_numpy(_channel_ramp_uint8()).permute(2, 0, 1).unsqueeze(0)

    segment = FusedLUTSegment(list(transforms), adapter)
    fused = segment(ramp.clone())
    reference = _sequential_byte_map_reference(adapter, transforms, ramp)

    assert torch.equal(fused, reference)


# ---------------------------------------------------------------------------
# float interpolation tolerance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
class TestFloatInterpolationTolerance:
    """The float interp-LUT path stays within a documented tolerance of an exact composition."""

    @staticmethod
    def _exact_composition(adapter, transforms, image: torch.Tensor) -> torch.Tensor:
        """Return the ops composed exactly (no interpolation) on ``image`` — the parity reference."""
        batch, channels, height, width = image.shape
        values = image.reshape(batch, channels, height * width)
        for transform in transforms:
            params = adapter.sample_params(transform, image.shape, image.device)
            values = adapter.build_lut(transform, params, values)
        return values.reshape(batch, channels, height, width)

    def test_smooth_gamma_within_two_levels(self):
        """A steep gamma<1 map stays within ~2 uint8 levels of the exact map (max error)."""
        adapter = KorniaAdapter()
        transforms = [kornia_aug.RandomGamma((0.5, 0.5), (1.0, 1.0), p=1.0)]
        image = torch.rand(4, 3, 64, 64)

        fused = FusedLUTSegment(list(transforms), adapter)(image.clone())
        exact = self._exact_composition(adapter, transforms, image)

        max_levels = (fused - exact).abs().max().item() * 255
        assert max_levels <= _GAMMA_FLOAT_MAX_LEVELS

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(lambda: [kornia_aug.RandomSolarize((0.3, 0.3), (0.1, 0.1), p=1.0)], id="solarize"),
            pytest.param(lambda: [kornia_aug.RandomPosterize((3, 3), p=1.0)], id="posterize"),
        ],
    )
    def test_discontinuous_map_within_documented_tolerance(self, factory):
        """A discontinuous map keeps a tiny mean error with only a small outlier fraction."""
        adapter = KorniaAdapter()
        transforms = factory()
        image = torch.rand(4, 3, 64, 64)

        fused = FusedLUTSegment(list(transforms), adapter)(image.clone())
        exact = self._exact_composition(adapter, transforms, image)

        diff_levels = (fused - exact).abs() * 255
        mean_levels = diff_levels.mean().item()
        outlier_fraction = (diff_levels > 2.0).float().mean().item()
        assert mean_levels <= _DISCONTINUOUS_FLOAT_MEAN_LEVELS
        assert outlier_fraction <= _DISCONTINUOUS_FLOAT_OUTLIER_FRACTION

    def test_chain_matches_native_backend_within_tolerance(self):
        """A gamma∘solarize chain stays within tolerance of the native Kornia sequential chain."""
        adapter = KorniaAdapter()
        transforms = [
            kornia_aug.RandomGamma((1.3, 1.3), (1.0, 1.0), p=1.0),
            kornia_aug.RandomSolarize((0.5, 0.5), (0.0, 0.0), p=1.0),
        ]
        image = torch.rand(4, 3, 64, 64)

        fused = FusedLUTSegment(list(transforms), adapter)(image.clone())
        native = image.clone()
        for transform in transforms:
            native = transform(native)

        diff_levels = (fused - native).abs() * 255
        assert diff_levels.mean().item() <= _DISCONTINUOUS_FLOAT_MEAN_LEVELS
        assert (diff_levels > 2.0).float().mean().item() <= _DISCONTINUOUS_FLOAT_OUTLIER_FRACTION


# ---------------------------------------------------------------------------
# fusion plan / warp accounting
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
class TestFusionPlan:
    """A contiguous lookup run reports as fused, never passthrough."""

    def test_run_has_no_passthrough_and_saves_a_pass(self):
        """A gamma∘solarize run fuses (no passthrough) and saves at least one pass."""
        pipe = Compose([
            kornia_aug.RandomGamma((0.8, 1.2), p=1.0),
            kornia_aug.RandomSolarize(0.1, 0.1, p=1.0),
        ])
        assert "passthrough" not in pipe.fusion_plan
        assert pipe.n_warps_saved >= 1

    def test_descriptor_kind_is_lut(self):
        """The structured descriptor reports one ``lut`` segment over the three ops."""
        pipe = Compose([
            kornia_aug.RandomGamma((0.8, 1.2), p=1.0),
            kornia_aug.RandomSolarize(0.1, 0.1, p=1.0),
            kornia_aug.RandomPosterize(4, p=1.0),
        ])
        descriptors = pipe.fusion_plan_descriptors
        assert len(descriptors) == 1
        assert descriptors[0].kind == "lut"
        assert descriptors[0].n_warps_saved == 2

    def test_lut_run_is_split_by_a_colour_op(self):
        """A colour (matrix) op between lookups splits the run: lookups never fold into colour."""
        pipe = Compose([
            kornia_aug.RandomGamma((0.8, 1.2), p=1.0),
            kornia_aug.RandomBrightness((0.9, 1.1), p=1.0),
            kornia_aug.RandomSolarize(0.1, 0.1, p=1.0),
        ])
        kinds = [descriptor.kind for descriptor in pipe.fusion_plan_descriptors]
        assert kinds == ["lut", "color", "lut"]


# ---------------------------------------------------------------------------
# barriers preserved (negative tests)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
class TestBarriersPreserved:
    """Cross-channel and runtime-histogram ops must never fold into a lookup table."""

    def test_saturation_stays_passthrough(self):
        """Saturation is cross-channel (POINTWISE) and remains a passthrough barrier."""
        adapter = KorniaAdapter()
        saturation = kornia_aug.RandomSaturation((0.8, 1.2), p=1.0)
        assert adapter.category(saturation) == TransformCategory.POINTWISE

        pipe = Compose([saturation, kornia_aug.RandomSaturation((0.8, 1.2), p=1.0)])
        assert "passthrough" in pipe.fusion_plan
        assert "lut" not in pipe.fusion_plan

    def test_equalize_stays_passthrough(self):
        """Equalize needs a runtime per-image histogram and cannot pre-compose; it stays a barrier."""
        with pytest.warns(UserWarning, match="Unknown Kornia transform"):
            pipe = Compose([kornia_aug.RandomEqualize(p=1.0), kornia_aug.RandomEqualize(p=1.0)])
        assert "passthrough" in pipe.fusion_plan
        assert "lut" not in pipe.fusion_plan


# ---------------------------------------------------------------------------
# pickle round-trip (DataLoader-worker contract)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
def test_pickle_roundtrip_preserves_lut_segment():
    """A pipeline containing a FusedLUTSegment survives pickle and still fuses + runs."""
    pipe = Compose([
        kornia_aug.RandomGamma((0.8, 1.2), p=1.0),
        kornia_aug.RandomSolarize(0.1, 0.1, p=1.0),
    ])
    restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301 -- trusted, self-produced bytes

    assert restored.fusion_plan == pipe.fusion_plan
    assert restored.n_warps_saved == pipe.n_warps_saved
    assert restored.fusion_plan_descriptors[0].kind == "lut"

    output = restored(torch.rand(2, 3, 16, 16))
    assert output.shape == (2, 3, 16, 16)
