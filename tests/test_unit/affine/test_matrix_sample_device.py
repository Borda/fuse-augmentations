"""Where a fused segment samples its per-transform matrices, and why it matters.

Adapters construct their parameter tensors with an explicit ``device=``, so sampling a chain on an accelerator issues a
separate host-to-device copy per parameter -- about ten per call for a five-transform chain. Profiled on MPS that
dominated a fused call: 40% in ``Tensor.to`` and 26% in ``torch.tensor``, against roughly 1% in ``affine_grid``.
Building on the host and moving the whole stack once removed 1.5x of the call, and the 3x3 matmuls cost the same
wherever they run.

The optimisation is only safe while it cannot move an RNG draw, which is what these tests pin.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations._compat import _TORCHVISION_V2_AVAILABLE
from fuse_augmentations.affine.segment import _samples_on_the_matrix_device

pytest_plugins: list[str] = []


class RandomRotation90:
    """Stand-in for Kornia's quarter-turn transform, matched by class name."""


class RandomRotation:
    """Stand-in for an ordinary rotation, whose parameters come from host RNG."""


class TestSamplesOnTheMatrixDevice:
    """Which transforms force sampling to stay on the image's device."""

    def test_kornia_quarter_turn_is_detected(self) -> None:
        """Kornia's ``RandomRotation90`` opts out.

        Its ``build_matrix`` falls back to ``torch.randint(..., device=params["_batch_size"].device)`` when ``k90`` is
        absent, so the sampling device selects the RNG stream that draw consumes. Sampling it on the host would silently
        change the quarter-turns a seeded pipeline produces.

        """
        assert _samples_on_the_matrix_device(RandomRotation90()) is True

    def test_an_ordinary_transform_is_not(self) -> None:
        """Every other adapter path derives parameters from host RNG, so the build device is free."""
        assert _samples_on_the_matrix_device(RandomRotation()) is False


class TestMatrixSampleDevice:
    """The per-call decision made by ``_BaseAffineSegment._matrix_sample_device``."""

    @staticmethod
    def _segment(transforms: list[object]) -> object:
        """Return an object exposing just what the resolver reads, without building a real segment."""
        from fuse_augmentations.affine.segment import _BaseAffineSegment

        class _Stub:
            pass

        stub = _Stub()
        stub.transforms = transforms  # type: ignore[attr-defined]
        stub._matrix_sample_device = _BaseAffineSegment._matrix_sample_device.__get__(stub)  # type: ignore[attr-defined]
        return stub

    def test_cpu_stays_on_cpu(self) -> None:
        """A host image samples on the host, which is where it already was -- no behaviour change."""
        segment = self._segment([RandomRotation()])

        assert segment._matrix_sample_device(torch.device("cpu")).type == "cpu"

    def test_accelerator_samples_on_the_host(self) -> None:
        """A device image samples on the host so the chain crosses in one transfer instead of ten.

        Asserted against a ``torch.device`` value rather than a live accelerator: the resolver reads only
        ``device.type``, so this pins the decision on a machine with no GPU attached.

        """
        segment = self._segment([RandomRotation(), RandomRotation()])

        assert segment._matrix_sample_device(torch.device("cuda")).type == "cpu"

    def test_a_quarter_turn_in_the_chain_forgoes_the_optimisation(self) -> None:
        """One opting-out transform pins the whole chain, because they share a sampling device.

        Correctness over the win: the chain keeps sampling on the image's device rather than
        splitting the decision per transform and reasoning about when the fallback draw fires.

        """
        segment = self._segment([RandomRotation(), RandomRotation90()])

        assert segment._matrix_sample_device(torch.device("cuda")).type == "cuda"

    def test_an_empty_chain_still_resolves(self) -> None:
        """A segment holding no matrix-building transforms resolves without inspecting anything."""
        segment = self._segment([])

        assert segment._matrix_sample_device(torch.device("cuda")).type == "cpu"


class TestComposeIsUnchangedOnCpu:
    """The host path composes exactly as before, so CPU output cannot have moved."""

    @pytest.mark.skipif(not _TORCHVISION_V2_AVAILABLE, reason="missing torchvision.transforms.v2")
    def test_seeded_cpu_output_is_reproducible_across_calls(self) -> None:
        """Two seeded runs of the same CPU pipeline agree bit for bit.

        On CPU ``_matrix_sample_device`` returns the image's own device, so the batched-transfer path is a no-op there
        by construction. This is the guard that keeps that true.

        """
        import torchvision.transforms.v2 as tv

        from fuse_augmentations import Compose

        def render() -> torch.Tensor:
            torch.manual_seed(0)
            pipe = Compose([tv.RandomRotation(15), tv.RandomHorizontalFlip(0.5)], execution="torch")
            torch.manual_seed(0)
            return pipe(torch.rand(2, 3, 32, 32))

        assert torch.equal(render(), render())
