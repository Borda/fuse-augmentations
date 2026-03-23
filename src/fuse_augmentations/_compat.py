"""Optional-dependency availability flags.

Import these instead of repeating try/except blocks in every module.

"""

from __future__ import annotations

try:
    import kornia as _kornia  # noqa: F401

    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False

try:
    import torchvision as _torchvision  # noqa: F401

    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

try:
    import albumentations as _albumentations  # noqa: F401

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False
