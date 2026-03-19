"""Backend adapters for fuse-augmentations.

Each adapter implements the ``TransformAdapter`` protocol to bridge
framework-specific transforms to the fused affine engine.

Example:
    >>> from fuse_augmentations.adapters import KorniaAdapter
    >>> adapter = KorniaAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...KorniaAdapter...>

"""

from fuse_augmentations.adapters._albumentations import AlbumentationsAdapter
from fuse_augmentations.adapters._kornia import KorniaAdapter
from fuse_augmentations.adapters._torchvision import TorchVisionAdapter

__all__ = ["AlbumentationsAdapter", "KorniaAdapter", "TorchVisionAdapter"]
