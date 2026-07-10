"""Multiprocess DataLoader stress and pickle-survival tests for fused pipelines.

The public contract under test (README "Training Loop" section):

- Pipelines are pickle-safe so they survive ``pickle.dumps``/``pickle.loads``,
  which is exactly the round-trip a ``torch.utils.data.DataLoader`` performs
  when it spawns worker processes (``num_workers > 0``).
- A round-tripped pipeline still produces valid outputs.
- Stochastic transforms (``p < 1``) drawing from the global torch RNG get an
  independent draw stream per worker, because PyTorch derives each worker's
  base seed from the parent seed -- so different workers do not emit identical
  augmentation streams for the same dataset index.
- Because the worker seeds derive from the parent ``torch.manual_seed``, the
  whole augmentation stream is reproducible across two loader runs seeded
  identically, and differs when seeded differently. This is the PyTorch
  DataLoader seeding contract, exercised end to end through the fused pipeline.

Design notes:

- On macOS the default multiprocessing start method is ``spawn``, so every
  ``Dataset`` and ``worker_init_fn`` referenced by the loader must live at
  module top level (picklable). No closures are used for that reason.
- The primary coverage runs on a *backend-free* ``Compose.from_params`` pipeline
  so the stress and RNG tests execute even with no optional deps installed
  (kornia / torchvision / albumentations). The pickle round-trip additionally
  covers the kornia-, torchvision-, and albumentations-backed pipelines, each
  guarded by a ``skipif`` on the corresponding availability flag. The module
  therefore collects cleanly with zero optional deps: conditionally imported
  names are never instantiated at class or module scope.
- The library samples ``p < 1`` probability from *both* the global torch RNG
  (kornia / torchvision / from_params paths) and, for the Albumentations path,
  ``numpy.random``. PyTorch reseeds only the torch RNG per worker, not numpy,
  so per-worker independence is asserted on the torch-RNG (from_params) path
  only; the numpy caveat is documented rather than asserted.

"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fuse_augmentations import Compose, FusedCompose
from fuse_augmentations._compat import (
    _ALBUMENTATIONS_AVAILABLE,
    _KORNIA_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)

pytestmark = pytest.mark.integration

# Small tensors keep multiprocess CI runtime bounded (spawn + fork of 4 workers
# dominates cost, not the compute). 3x32x32 mirrors a CIFAR-scale sample.
CHANNELS, HEIGHT, WIDTH = 3, 32, 32
NUM_WORKERS = 4
NUM_ITERS = 100  # 4 workers x ~100 samples; module target < 60s (see test timing)
# Fail fast: a hung worker should error, not hang CI. Generous vs. observed
# spawn cost (~a few seconds) yet far below any pytest-level stall.
LOADER_TIMEOUT_S = 60.0


class _AugmentedDataset(Dataset):
    """Dataset that applies a fresh fused pipeline inside ``__getitem__``.

    A new ``Compose.from_params`` pipeline is built per item on purpose: it is
    cheap, backend-free (no optional deps), and exercises the per-sample draw
    from the global torch RNG that per-worker seeding must make independent.
    Returns ``(output_tensor, matrix_signature)`` where the signature is the
    flattened, rounded ``transform_matrix`` -- a compact fingerprint of the
    augmentation actually applied, comparable across workers and runs.

    """

    def __init__(self, length: int) -> None:
        self._length = length
        # Fixed input content so any output/matrix difference is attributable to
        # the RNG stream, not to differing pixels.
        generator = torch.Generator().manual_seed(0)
        self._images = torch.rand(length, CHANNELS, HEIGHT, WIDTH, generator=generator)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pipeline = Compose.from_params(
            rotation=(-20.0, 20.0),
            hflip_p=0.5,  # p < 1 -> stochastic; per-worker streams must diverge
            randomness="per_sample",
        )
        image = self._images[index : index + 1]
        output = pipeline(image)
        matrix = pipeline.transform_matrix
        # transform_matrix is (batch, 3, 3); this path always fuses an affine.
        signature = matrix.reshape(-1).to(torch.float64) if matrix is not None else torch.empty(0)
        return output.reshape(-1), signature


class _WorkerIdDataset(Dataset):
    """Reports which worker processed each index, plus the augmentation signature.

    Used to prove that concurrent workers draw independent RNG streams: two
    workers handling different indices of identical input content must not
    yield byte-identical augmentation matrices across their whole assigned slice.

    """

    def __init__(self, length: int) -> None:
        self._length = length
        generator = torch.Generator().manual_seed(0)
        self._images = torch.rand(length, CHANNELS, HEIGHT, WIDTH, generator=generator)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[int, int, torch.Tensor]:
        pipeline = Compose.from_params(
            rotation=(-20.0, 20.0),
            hflip_p=0.5,
            randomness="per_sample",
        )
        pipeline(self._images[index : index + 1])
        matrix = pipeline.transform_matrix
        signature = matrix.reshape(-1).to(torch.float64) if matrix is not None else torch.empty(0)
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else -1
        return index, worker_id, signature


def _drain(loader: DataLoader, deadline_s: float) -> list:
    """Consume a loader, failing fast if it stalls past ``deadline_s`` wall time."""
    start = time.monotonic()
    batches = []
    for batch in loader:
        batches.append(batch)
        if time.monotonic() - start > deadline_s:
            pytest.fail(f"DataLoader did not drain within {deadline_s:.0f}s -- suspect a hung worker")
    return batches


class TestDataLoaderStress:
    """Multiprocess DataLoader driving a fused pipeline inside ``__getitem__``."""

    def test_multiworker_stress_produces_valid_finite_outputs(self) -> None:
        """4 workers x ~100 samples all yield finite, correctly shaped outputs."""
        dataset = _AugmentedDataset(NUM_ITERS)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        batches = _drain(loader, LOADER_TIMEOUT_S)

        assert len(batches) == NUM_ITERS
        expected_numel = CHANNELS * HEIGHT * WIDTH
        for output, _signature in batches:
            flat = output.reshape(-1)
            assert flat.numel() == expected_numel
            assert torch.isfinite(flat).all()

    def test_multiworker_stream_survives_full_pass_without_worker_crash(self) -> None:
        """Every index is delivered exactly once across all workers (no drops)."""
        dataset = _WorkerIdDataset(NUM_ITERS)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        batches = _drain(loader, LOADER_TIMEOUT_S)

        seen_indices = sorted(int(index) for index, _wid, _sig in batches)
        assert seen_indices == list(range(NUM_ITERS))


class TestPickleRoundTrip:
    """Pipelines survive the pickle round-trip a spawn-mode worker performs."""

    def test_from_params_pipeline_survives_pickle_and_stays_usable(self) -> None:
        """Backend-free pipeline round-trips and still produces valid output."""
        pipeline = Compose.from_params(rotation=(-15.0, 15.0), hflip_p=0.5)
        image = torch.rand(2, CHANNELS, HEIGHT, WIDTH)

        restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301 -- trusted, self-produced bytes
        output = restored(image)

        assert isinstance(restored, FusedCompose)
        assert output.shape == image.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
    def test_kornia_pipeline_survives_pickle_and_stays_usable(self) -> None:
        """Kornia-backed pipeline round-trips and still produces valid output."""
        import kornia.augmentation as kornia_aug

        pipeline = Compose([kornia_aug.RandomRotation(degrees=25.0, p=1.0)])
        image = torch.rand(2, CHANNELS, HEIGHT, WIDTH)

        restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301 -- trusted, self-produced bytes
        output = restored(image)

        assert output.shape == image.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="missing torchvision")
    def test_torchvision_pipeline_survives_pickle_and_stays_usable(self) -> None:
        """TorchVision-backed pipeline round-trips and still produces valid output."""
        import torchvision.transforms.v2 as tv_transforms

        pipeline = Compose([tv_transforms.RandomRotation(degrees=25)])
        image = torch.rand(2, CHANNELS, HEIGHT, WIDTH)

        restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301 -- trusted, self-produced bytes
        output = restored(image)

        assert output.shape == image.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
    def test_albumentations_pipeline_survives_pickle_and_stays_usable(self) -> None:
        """Albumentations-backed pipeline round-trips and still produces valid output."""
        import albumentations as albu

        pipeline = Compose([albu.Rotate(limit=25, p=1.0)])
        image = np.random.rand(HEIGHT, WIDTH, CHANNELS).astype(np.float32)

        restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301 -- trusted, self-produced bytes
        result = restored(image=image)

        assert isinstance(result, dict)
        assert result["image"].shape == image.shape
        assert np.isfinite(result["image"]).all()


class TestPerWorkerRandomness:
    """Concurrent workers must draw independent RNG streams for p < 1 transforms."""

    def test_workers_do_not_produce_identical_augmentation_streams(self) -> None:
        """No two of 4 workers emit byte-identical matrices across their slice.

        Scenario: identical input content is augmented with a stochastic
        (``hflip_p=0.5``) fused pipeline across 4 spawned workers; each worker's
        sequence of transform matrices must differ from every other worker's,
        proving PyTorch's per-worker torch-RNG reseeding reaches the pipeline.

        """
        torch.manual_seed(1234)
        dataset = _WorkerIdDataset(NUM_ITERS)
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        batches = _drain(loader, LOADER_TIMEOUT_S)

        streams: dict[int, list[tuple[float, ...]]] = {}
        for _index, worker_id, signature in batches:
            key = int(worker_id)
            streams.setdefault(key, []).append(tuple(signature.reshape(-1).tolist()))

        assert len(streams) == NUM_WORKERS, "expected all workers to contribute samples"

        worker_ids = sorted(streams)
        for outer in range(len(worker_ids)):
            for inner in range(outer + 1, len(worker_ids)):
                left = streams[worker_ids[outer]]
                right = streams[worker_ids[inner]]
                assert left != right, (
                    f"workers {worker_ids[outer]} and {worker_ids[inner]} produced "
                    "identical augmentation streams -- per-worker RNG not independent"
                )

    def test_loader_stream_is_reproducible_under_fixed_parent_seed(self) -> None:
        """Same parent seed -> identical augmentation stream; different seed -> different.

        Scenario: PyTorch derives worker base seeds from the parent
        ``torch.manual_seed``. Two loader runs seeded identically must yield the
        same per-index matrices; a run seeded differently must diverge on every
        index. This is the documented DataLoader seeding contract observed end to
        end through the fused pipeline (the library itself exposes no seed API).

        """
        run_a = _collect_indexed_signatures(seed=7)
        run_b = _collect_indexed_signatures(seed=7)
        run_c = _collect_indexed_signatures(seed=8)

        assert run_a.keys() == run_b.keys() == run_c.keys()
        for index in run_a:
            assert run_a[index] == run_b[index], f"index {index} not reproducible under fixed seed"
        differing = sum(1 for index in run_a if run_a[index] != run_c[index])
        assert differing == len(run_a), "different parent seeds must change every index's augmentation"


def _collect_indexed_signatures(seed: int) -> dict[int, tuple[float, ...]]:
    """Run a small multiprocess loader under ``seed`` and return index -> matrix signature."""
    torch.manual_seed(seed)
    length = 20  # smaller than the stress pass: two determinism runs stay well under budget
    dataset = _WorkerIdDataset(length)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    batches = _drain(loader, LOADER_TIMEOUT_S)
    return {int(index): tuple(signature.reshape(-1).tolist()) for index, _wid, signature in batches}
