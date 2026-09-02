"""Caller-owned randomness: ``generator=`` seeds every pipeline-owned draw (FA-1).

The direct-parameter engine samples geometry, colour factors and per-transform probability gates itself, so a caller-
supplied ``torch.Generator`` can own the whole stream. Backend transforms sample inside their own libraries, which no
``torch.Generator`` reaches; those pipelines refuse the argument instead of quietly drawing from the global stream,
which would report a reproducibility they do not have.

"""

from __future__ import annotations

import pickle

import pytest
import torch

from fuse_augmentations import FusedCompose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE


@pytest.fixture
def image() -> torch.Tensor:
    """Return a fixed ``(4, 3, 32, 32)`` float32 batch with structure in every channel."""
    ramp = torch.linspace(0.0, 1.0, 32)
    return (ramp[None, :] * ramp[:, None]).expand(4, 3, 32, 32).contiguous()


def _pipeline(seed: int | None, **kwargs: object) -> FusedCompose:
    """Build a rotation+flip+brightness pipeline seeded by ``seed`` (``None`` = global stream)."""
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    params: dict[str, object] = {
        "rotation": (-30.0, 30.0),
        "translate_x": (-4.0, 4.0),
        "hflip_p": 0.5,
        "brightness": 0.4,
        "generator": generator,
    }
    params.update(kwargs)
    return FusedCompose.from_params(**params)  # type: ignore[arg-type]


class TestSeededPipelineReproducibility:
    """A caller-owned generator fully determines the sampled parameters and the output."""

    def test_same_seed_gives_identical_output(self, image: torch.Tensor) -> None:
        """Two pipelines seeded identically produce bit-identical images.

        This is the contract downstream depends on: a dataloader worker that rebuilds the
        pipeline from a seed must replay the same augmentation, without touching the
        process-wide torch stream that other components also draw from.

        """
        first = _pipeline(0)(image)
        second = _pipeline(0)(image)

        assert torch.equal(first, second)

    def test_same_seed_gives_identical_parameters(self, image: torch.Tensor) -> None:
        """Identically seeded pipelines compose the identical transform matrix.

        Equal output could in principle hide unequal parameters that happen to render the same; asserting on the
        composed matrix checks the sampling itself, not the pixels.

        """
        _, first = _pipeline(0)(image, return_matrix=True)
        _, second = _pipeline(0)(image, return_matrix=True)

        assert torch.equal(first, second)

    def test_different_seeds_give_different_output(self, image: torch.Tensor) -> None:
        """Different seeds diverge, so the generator is actually driving the draws.

        Without this the reproducibility test above would also pass for a pipeline that ignored the generator and
        sampled a constant.

        """
        assert not torch.equal(_pipeline(0)(image), _pipeline(1)(image))

    def test_global_stream_run_between_seeded_runs_perturbs_neither(self, image: torch.Tensor) -> None:
        """An unseeded run interleaved between two seeded runs changes neither of them.

        The seeded and global streams must be genuinely independent: a caller that seeds
        its augmentation cannot be perturbed by unrelated code — or by another pipeline —
        consuming the global stream between its own calls.

        """
        first = _pipeline(0)(image)
        torch.manual_seed(7)
        _pipeline(None)(image)
        torch.rand(1024)
        second = _pipeline(0)(image)

        assert torch.equal(first, second)

    def test_seeded_run_ignores_global_seed(self, image: torch.Tensor) -> None:
        """Re-seeding the global stream between two identically seeded runs changes nothing.

        Complements the interleaving case: there the global stream advanced, here it is
        reset outright, which is what a training loop does at epoch boundaries.

        """
        torch.manual_seed(1)
        first = _pipeline(3)(image)
        torch.manual_seed(999)
        second = _pipeline(3)(image)

        assert torch.equal(first, second)

    def test_flip_gate_follows_the_generator(self, image: torch.Tensor) -> None:
        """The per-transform probability gate is seeded too, not only the parameter draw.

        A flip is decided by a Bernoulli gate rather than a sampled magnitude, so it is the draw most easily left on the
        global stream while everything else looks reproducible. A batch of 4 at ``p=0.5`` makes an unseeded gate
        visible.

        """
        flip_only: dict[str, object] = {"rotation": None, "translate_x": None, "brightness": None, "hflip_p": 0.5}
        torch.manual_seed(11)
        first = _pipeline(5, **flip_only)(image)
        torch.manual_seed(12)
        second = _pipeline(5, **flip_only)(image)

        assert torch.equal(first, second)

    def test_default_none_still_follows_the_global_seed(self, image: torch.Tensor) -> None:
        """``generator=None`` keeps the historical global-stream behaviour.

        The default path must be untouched by FA-1: an existing caller seeding
        ``torch.manual_seed`` still gets reproducible output with no code change.

        """
        torch.manual_seed(4)
        first = _pipeline(None)(image)
        torch.manual_seed(4)
        second = _pipeline(None)(image)

        assert torch.equal(first, second)

    def test_pickle_round_trip_resumes_the_generator_state(self, image: torch.Tensor) -> None:
        """An unpickled pipeline resumes the stream where the pickled one stood.

        DataLoader workers receive the pipeline by pickle, so the restored generator must carry its state rather than
        silently reverting to the global stream — every worker then replays the same augmentation and must be re-seeded
        per worker.

        """
        pipe = _pipeline(21)
        pipe(image)
        restored = pickle.loads(pickle.dumps(pipe))  # noqa: S301 -- trusted, self-produced bytes

        assert restored.generator is not None
        assert torch.equal(restored(image), pipe(image))

    def test_native_backend_accepts_a_generator(self, image: torch.Tensor) -> None:
        """``backend="native"`` is the same direct-parameter engine and honours the generator.

        The native backend routes through ``from_config``, a different code path than backend-free ``from_params``, so
        it needs its own check that the generator survives the detour rather than being dropped on the way.

        """

        def run() -> torch.Tensor:
            gen = torch.Generator().manual_seed(2)
            return FusedCompose.from_params(rotation=(-20.0, 20.0), backend="native", generator=gen)(image)

        assert torch.equal(run(), run())

    def test_identity_pipeline_accepts_a_generator(self, image: torch.Tensor) -> None:
        """A pipeline with no sampled parameters accepts a generator and stores it.

        The identity path builds no segments at all; it must not raise merely because a generator was supplied, so
        callers can pass one unconditionally from a config.

        """
        pipe = FusedCompose.from_params(generator=torch.Generator().manual_seed(0))

        assert torch.equal(pipe(image), image)


class TestBackendRandomnessIsRejected:
    """Backend-sampled pipelines refuse a generator instead of degrading to the global stream."""

    @pytest.mark.integration
    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
    def test_kornia_transforms_reject_a_generator(self) -> None:
        """A Kornia pipeline raises at construction when handed a generator.

        Kornia samples inside ``generate_parameters`` from the global torch stream. Accepting the generator here would
        silently produce runs that look seeded and are not.

        """
        import kornia.augmentation as kornia_aug

        with pytest.raises(ValueError, match="generator="):
            FusedCompose([kornia_aug.RandomAffine(degrees=30.0, p=1.0)], generator=torch.Generator())

    @pytest.mark.integration
    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required")
    def test_albumentations_transforms_reject_a_generator(self) -> None:
        """An Albumentations pipeline raises at construction when handed a generator.

        Albumentations draws from ``np.random`` and its own ``py_random``; neither is reachable from a
        ``torch.Generator``.

        """
        import albumentations as albu

        with pytest.raises(ValueError, match="generator="):
            FusedCompose([albu.HorizontalFlip(p=0.5)], generator=torch.Generator())

    @pytest.mark.integration
    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required")
    def test_torchvision_transforms_reject_a_generator(self) -> None:
        """A TorchVision pipeline raises at construction when handed a generator.

        TorchVision samples in its own parameter hooks, which take no generator argument.

        """
        from torchvision.transforms import v2

        with pytest.raises(ValueError, match="generator="):
            FusedCompose([v2.RandomHorizontalFlip(p=0.5)], generator=torch.Generator())

    @pytest.mark.integration
    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")
    def test_from_params_with_a_backend_rejects_a_generator(self) -> None:
        """``from_params(backend="kornia", ...)`` raises rather than dropping the generator.

        The factory delegates to real backend transform objects, so it inherits the same limitation as constructing them
        directly — and must fail the same way.

        """
        with pytest.raises(ValueError, match="generator="):
            FusedCompose.from_params(rotation=(-10.0, 10.0), backend="kornia", generator=torch.Generator())

    def test_the_error_names_the_supported_path(self) -> None:
        """The rejection explains which construction path does support a generator.

        A bare "unsupported" leaves the caller guessing; the message must point at ``from_params`` so the fix is obvious
        from the traceback alone.

        """
        from fuse_augmentations._random import reject_backend_randomness

        with pytest.raises(ValueError, match="from_params"):
            reject_backend_randomness(torch.Generator(), "a backend draw")


class TestGeneratorDeviceHop:
    """A CPU generator seeds a pipeline whose tensors live on an accelerator."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_cpu_generator_seeds_an_mps_pipeline(self, image: torch.Tensor) -> None:
        """A CPU generator drives an MPS batch and stays reproducible.

        ``torch`` rejects a CPU generator against an accelerator tensor outright, so the draw happens on the generator's
        device and is copied across. Without that hop the common case — one CPU-seeded config, tensors on the GPU —
        would raise.

        """
        mps_image = image.to("mps")

        def run() -> torch.Tensor:
            gen = torch.Generator().manual_seed(8)
            return FusedCompose.from_params(rotation=(-30.0, 30.0), hflip_p=0.5, generator=gen)(mps_image)

        assert torch.equal(run(), run())
