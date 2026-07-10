"""Integration tests for the passthrough boundary policy.

Covers the four passthrough-boundary behaviours and the passthrough/auxiliary policy:

- **Batched CPU boundary**: an Albumentations passthrough segment performs exactly
  one device-to-host and one host-to-device transfer for the whole batch, not one
  of each per sample, and the batched result is bit-identical to a per-sample loop.
- **Opt-in substitution**: ``substitute_passthrough=True`` swaps a registered
  non-fusible op (Albumentations ``GaussianBlur``) for a torch-native equivalent
  (Kornia ``RandomGaussianBlur``), keeping the pipeline on-device (no NumPy
  round-trip) and emitting a ``UserWarning``. Default off leaves outputs unchanged.
- **CPU-passthrough visibility**: ``fusion_plan`` marks passthrough segments with
  ``[CPU passthrough]`` when the pipeline is configured on a non-CPU device.
- **Machine-readable reasons**: ``fusion_plan_descriptors`` carry ``barrier``,
  ``split_reason``, and ``refused`` fields, while the human ``fusion_plan`` string
  form stays backward compatible.
- **Coordinate-changing policy**: a coordinate-changing passthrough (elastic/grid/
  optical distortion) with auxiliary targets raises ``ValueError``; a kernel
  passthrough (blur) with auxiliary targets does not warn.

Requires albumentations; substitution and CPU-passthrough checks additionally
require kornia.

"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import fuse_augmentations.adapters.albumentations as albu_adapter_mod
from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.adapters.albumentations import AlbumentationsAdapter

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required"),
]

BATCH, CHANNELS, HEIGHT, WIDTH = 4, 3, 24, 24


def _image(batch: int = BATCH) -> torch.Tensor:
    """Return a deterministic ``(batch, 3, 24, 24)`` float32 image batch."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(batch, CHANNELS, HEIGHT, WIDTH, generator=gen)


def _mask() -> torch.Tensor:
    """Return a deterministic ``(BATCH, 1, 24, 24)`` float mask batch."""
    gen = torch.Generator().manual_seed(1)
    return torch.randint(0, 3, (BATCH, 1, HEIGHT, WIDTH), generator=gen).float()


class TestBatchedCpuBoundary:
    """The Albumentations passthrough boundary transfers the whole batch once each way."""

    def test_one_device_to_host_transfer_per_segment(self, monkeypatch: pytest.MonkeyPatch):
        """A batched passthrough does exactly one ``.cpu()`` (D2H) call for the whole batch."""
        calls = {"cpu": 0}
        original_cpu = torch.Tensor.cpu

        def _spy_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            calls["cpu"] += 1
            return original_cpu(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "cpu", _spy_cpu)
        pipe = Compose([albu.GaussianBlur(p=1.0)])
        np.random.seed(0)
        pipe(_image(batch=8))

        assert calls["cpu"] == 1

    def test_one_host_to_device_materialization_per_segment(self, monkeypatch: pytest.MonkeyPatch):
        """A batched passthrough stacks the host results once (single H2D materialization)."""
        calls = {"stack": 0}
        original_stack = np.stack

        def _spy_stack(*args: object, **kwargs: object) -> np.ndarray:
            calls["stack"] += 1
            return original_stack(*args, **kwargs)

        monkeypatch.setattr(albu_adapter_mod.np, "stack", _spy_stack)
        pipe = Compose([albu.GaussianBlur(p=1.0)])
        np.random.seed(0)
        pipe(_image(batch=8))

        assert calls["stack"] == 1

    def test_batched_boundary_numerics_match_per_sample_loop(self):
        """The batched boundary result is bit-identical to an explicit per-sample loop."""
        image = _image()
        transform = albu.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1.0, 1.0), p=1.0)
        np.random.seed(7)
        batched = AlbumentationsAdapter.call_nonfused(transform, image)

        np.random.seed(7)
        per_sample = []
        for idx in range(image.shape[0]):
            image_np = image[idx].permute(1, 2, 0).cpu().numpy()
            output_np = transform(image=image_np)["image"]
            per_sample.append(torch.as_tensor(np.ascontiguousarray(output_np)).permute(2, 0, 1))
        expected = torch.stack(per_sample).to(dtype=image.dtype)

        torch.testing.assert_close(batched, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required for substitution target")
class TestSubstitutePassthrough:
    """``substitute_passthrough=True`` routes a registered op to a torch-native path."""

    def test_substitution_avoids_numpy_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        """With substitution on, the blur runs torch-native — the image never converts to numpy.

        The spy counts ``.numpy()`` conversions (the signature of the Albumentations host
        boundary) rather than ``.cpu()``: the substituted torch-native backend may move tensors
        internally for its own bookkeeping (kornia 0.8.2 caches a detached CPU copy of the
        output image; 0.8.3 does not), and that is not the round-trip this test guards against.

        """
        calls = {"numpy": 0}
        original_numpy = torch.Tensor.numpy

        def _spy_numpy(self: torch.Tensor, *args: object, **kwargs: object) -> object:
            calls["numpy"] += 1
            return original_numpy(self, *args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipe = Compose([albu.GaussianBlur(p=1.0)], substitute_passthrough=True)
        monkeypatch.setattr(torch.Tensor, "numpy", _spy_numpy)
        pipe(_image())

        assert calls["numpy"] == 0

    def test_substitution_emits_warning(self):
        """Constructing with ``substitute_passthrough=True`` warns about the behaviour change."""
        with pytest.warns(UserWarning, match="substitute_passthrough=True replaced"):
            Compose([albu.GaussianBlur(p=1.0)], substitute_passthrough=True)

    def test_substitution_replaces_op_in_plan(self):
        """The substituted op appears as the torch-native target in the fusion plan."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pipe = Compose([albu.GaussianBlur(p=1.0)], substitute_passthrough=True)

        assert "RandomGaussianBlur" in pipe.fusion_plan
        assert "GaussianBlur" not in pipe.fusion_plan.replace("RandomGaussianBlur", "")

    def test_default_off_keeps_original_op(self):
        """Default (``substitute_passthrough=False``) leaves the Albumentations op in place."""
        pipe = Compose([albu.GaussianBlur(p=1.0)])

        assert "passthrough(GaussianBlur)" in pipe.fusion_plan

    def test_unregistered_op_is_not_substituted(self):
        """An op with no registered substitution is kept even when substitution is on."""
        pipe = Compose([albu.MotionBlur(p=1.0)], substitute_passthrough=True)

        assert "passthrough(MotionBlur)" in pipe.fusion_plan


class TestFusionPlanCpuPassthrough:
    """``fusion_plan`` surfaces the CPU-passthrough poison pill on non-CPU pipelines."""

    def test_cpu_pipeline_has_no_marker(self):
        """On a CPU pipeline the passthrough entry carries no ``[CPU passthrough]`` marker."""
        pipe = Compose([albu.GaussianBlur(p=1.0)])

        assert "[CPU passthrough]" not in pipe.fusion_plan
        assert "passthrough(GaussianBlur)" in pipe.fusion_plan

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
    def test_mps_pipeline_flags_cpu_passthrough(self):
        """On an MPS pipeline the passthrough entry is annotated ``[CPU passthrough]``."""
        pipe = Compose([albu.HorizontalFlip(p=1.0), albu.GaussianBlur(p=1.0)]).to("mps")

        plan = pipe.fusion_plan
        assert "passthrough(GaussianBlur) [CPU passthrough]" in plan
        # The fused/exact segment is not a passthrough, so it carries no marker.
        assert "exact(HorizontalFlip) [CPU passthrough]" not in plan


class TestPassthroughDescriptors:
    """``fusion_plan_descriptors`` carry machine-readable reasons; the string stays compatible."""

    def test_kernel_passthrough_descriptor_reasons(self):
        """A blur passthrough descriptor reports a spatial-kernel barrier and not-fusible refusal."""
        pipe = Compose([albu.GaussianBlur(p=1.0)])
        descriptor = pipe.fusion_plan_descriptors[0]

        assert descriptor.kind == "passthrough"
        assert descriptor.barrier == "spatial_kernel"
        assert descriptor.refused == "not_fusible"
        assert descriptor.split_reason is None

    def test_coordinate_changing_passthrough_descriptor_barrier(self):
        """A geometric distortion passthrough descriptor reports a coordinate-change barrier."""
        pipe = Compose([albu.ElasticTransform(p=1.0)])
        descriptor = pipe.fusion_plan_descriptors[0]

        assert descriptor.barrier == "coordinate_change"
        assert descriptor.refused == "not_fusible"

    def test_descriptor_reasons_survive_to_dict(self):
        """``to_dict`` exposes the new machine-readable fields for downstream consumers."""
        pipe = Compose([albu.GaussianBlur(p=1.0)])
        as_dict = pipe.fusion_plan_descriptors[0].to_dict()

        assert as_dict["barrier"] == "spatial_kernel"
        assert as_dict["refused"] == "not_fusible"
        assert as_dict["split_reason"] is None

    def test_human_fusion_plan_string_unchanged_on_cpu(self):
        """The human-readable ``fusion_plan`` string form is unchanged (additive descriptors)."""
        pipe = Compose([albu.HorizontalFlip(p=1.0), albu.GaussianBlur(p=1.0)])

        assert pipe.fusion_plan == "exact(HorizontalFlip) → passthrough(GaussianBlur)"


class TestPassthroughAuxCorrectness:
    """Coordinate-changing passthrough with aux raises; kernel passthrough with aux is silent."""

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(lambda: albu.ElasticTransform(p=1.0), id="elastic-distortion"),
            pytest.param(lambda: albu.CoarseDropout(p=1.0), id="coarse-dropout"),
            pytest.param(lambda: albu.XYMasking(num_masks_x=1, mask_x_length=8, p=1.0), id="xy-masking"),
        ],
    )
    def test_coordinate_changing_passthrough_with_aux_raises(self, transform_factory):
        """A coordinate-dependent passthrough (distortion or region-zeroing) plus aux raises ``ValueError``."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0), transform_factory()],
            data_keys=["input", "mask"],
        )
        with pytest.raises(ValueError, match="Coordinate-changing passthrough"):
            pipe(_image(), _mask())

    def test_kernel_passthrough_with_aux_does_not_warn(self):
        """A kernel passthrough plus auxiliary targets does not warn — aux legitimately skips it."""
        pipe = Compose(
            [albu.HorizontalFlip(p=1.0), albu.GaussianBlur(p=1.0)],
            data_keys=["input", "mask"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pipe(_image(), _mask())
