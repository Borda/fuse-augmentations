"""Tests for the torch IterableDataset wrapper (fuse_augmentations.data.datasets)."""

from __future__ import annotations

from torch.utils.data import DataLoader

from fuse_augmentations.data import SyntheticIterableDataset
from fuse_augmentations.data.sample import Sample


def test_len_reports_budget():
    ds = SyntheticIterableDataset(num_images=7, img_size=32, seed=0)
    assert len(ds) == 7


def test_iter_yields_samples():
    ds = SyntheticIterableDataset(num_images=4, img_size=32, seed=0)
    samples = list(ds)
    assert len(samples) == 4
    assert all(isinstance(s, Sample) for s in samples)


def test_iter_is_reproducible():
    ds = SyntheticIterableDataset(num_images=3, img_size=32, seed=5)
    first = [ann.bbox_xyxy for s in ds for ann in s.annotations]
    second = [ann.bbox_xyxy for s in ds for ann in s.annotations]
    assert first == second


def test_dataloader_batches_cover_all_samples():
    ds = SyntheticIterableDataset(num_images=5, img_size=32, seed=0)
    loader = DataLoader(ds, batch_size=2, collate_fn=list)
    batches = list(loader)
    assert sum(len(b) for b in batches) == 5


def test_multiworker_shards_cover_all_samples():
    ds = SyntheticIterableDataset(num_images=6, img_size=32, seed=0)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=list)
    assert sum(len(b) for b in loader) == 6


def test_forwarded_config_kwargs_normalize_a_raw_fill():
    """`**config_kwargs` reaches `SyntheticConfig`, so a raw `(r, g, b)` fill normalizes on the streaming path too.

    This wrapper forwards its keywords verbatim, which is the reason `SyntheticConfig` coerces at construction rather
    than trusting its callers — `class_mode="shape"` arrives here as a bare string for exactly the same reason. A fill
    is the third spelling to go through that boundary, and the only one whose failure would surface far away: an un-
    normalized tuple has no `.rgb`, so it would raise inside the generator's draw call rather than at construction.

    """
    ds = SyntheticIterableDataset(num_images=1, img_size=32, seed=0, colors=((255, 215, 0),))

    assert [fill.label for fill in ds.config.colors] == ["ffd700"]
    assert len(list(ds)) == 1
