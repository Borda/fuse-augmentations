"""Optional-dependency availability flags.

Import these instead of repeating try/except blocks in every module.

"""

from __future__ import annotations

import importlib.util

try:
    import kornia  # noqa: F401

    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False

try:
    import torchvision  # noqa: F401

    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

_TORCHVISION_V2_AVAILABLE = _TORCHVISION_AVAILABLE and importlib.util.find_spec("torchvision.transforms.v2") is not None

try:
    import albumentations  # noqa: F401

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False

try:
    import cv2  # noqa: F401

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
