"""PyTorch ``IterableDataset`` streaming synthetic samples for training feeds.

Wraps :meth:`~fuse_augmentations.data.generator.SyntheticGenerator.generate`
so samples can be pulled straight into a ``DataLoader`` with no disk round-trip.
The dataset is worker-shard aware: under multi-worker loading each worker produces
a disjoint, deterministically-seeded slice of the requested count.

Because object annotations are ragged (a variable number per image), pass a custom
``collate_fn`` to ``DataLoader`` (e.g. ``collate_fn=list``) — the default collate
cannot stack variable-length targets.

Examples:
    ```pycon
    >>> from torch.utils.data import DataLoader
    >>> from fuse_augmentations.data.datasets import SyntheticIterableDataset
    >>> ds = SyntheticIterableDataset(num_images=4, img_size=32, seed=0)
    >>> loader = DataLoader(ds, batch_size=2, collate_fn=list)
    >>> batches = list(loader)
    >>> sum(len(b) for b in batches)
    4

    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data import IterableDataset, get_worker_info

from fuse_augmentations.data.config import SyntheticConfig
from fuse_augmentations.data.generator import SyntheticGenerator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fuse_augmentations.data.sample import Sample


class SyntheticIterableDataset(IterableDataset["Sample"]):
    """Stream synthetic :class:`Sample` objects into a PyTorch training loop.

    Args:
        num_images: Total number of samples the dataset yields per epoch.
        config: Full :class:`SyntheticConfig`; when given, ``config_kwargs`` are ignored.
        seed: Base seed for reproducible streams; ``None`` uses fresh entropy.
        **config_kwargs: Extra :class:`SyntheticConfig` fields (e.g. ``img_size``,
            ``class_mode``) used only when ``config`` is not supplied.

    Examples:
        ```pycon
        >>> from fuse_augmentations.data.datasets import SyntheticIterableDataset
        >>> ds = SyntheticIterableDataset(num_images=3, img_size=32, seed=1)
        >>> samples = list(ds)
        >>> len(samples)
        3

        ```

    """

    def __init__(
        self,
        num_images: int,
        config: SyntheticConfig | None = None,
        seed: int | None = None,
        **config_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to SyntheticConfig
    ) -> None:
        """Store the sample budget and build the underlying generator."""
        super().__init__()
        self.num_images = num_images
        self.config = config or SyntheticConfig(**config_kwargs)
        self.seed = seed
        self._generator = SyntheticGenerator(self.config)

    def __len__(self) -> int:
        """Return the per-epoch sample count."""
        return self.num_images

    def _worker_shard(self) -> tuple[int, int | None]:
        """Return this worker's ``(count, seed)`` slice of the total budget."""
        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            return self.num_images, self.seed
        per, remainder = divmod(self.num_images, info.num_workers)
        count = per + (1 if info.id < remainder else 0)
        seed = None if self.seed is None else self.seed + info.id
        return count, seed

    def __iter__(self) -> Iterator[Sample]:
        """Yield this worker's deterministically-seeded slice of samples."""
        count, seed = self._worker_shard()
        return self._generator.generate(count, seed=seed)
