"""Short-name re-export package for fuse-augmentations.

Provides the canonical ``import fuse_aug`` entry point as specified in §19
of the project spec. All public symbols live in ``fuse_augmentations``; this
package simply re-exports them.

Example:
    >>> import fuse_aug
    >>> fuse_aug.__name__
    'fuse_aug'
"""

from fuse_augmentations import *  # noqa: F401, F403
from fuse_augmentations import __version__  # noqa: F401

__all__ = ["__version__"]
