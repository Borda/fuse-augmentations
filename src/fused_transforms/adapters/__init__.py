"""Backend adapters for vision-synth.

Each adapter implements the ``TransformAdapter`` protocol to bridge
framework-specific transforms to the fused affine engine.

Examples:
    ```pycon
    >>> from fused_transforms.adapters import KorniaAdapter
    >>> adapter = KorniaAdapter()
    >>> adapter  # doctest: +ELLIPSIS
    <...KorniaAdapter...>

    ```

"""

from fused_transforms._backend import register_adapter
from fused_transforms.adapters.albumentations import AlbumentationsAdapter
from fused_transforms.adapters.kornia import KorniaAdapter
from fused_transforms.adapters.torchvision import TorchVisionAdapter

__all__ = ["AlbumentationsAdapter", "KorniaAdapter", "TorchVisionAdapter", "register_adapter"]
