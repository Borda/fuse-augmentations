"""Demo tests."""

from fuse_augmentations.utils import demo_func


def test_demo_func() -> None:
    """Test the demo function."""
    assert demo_func(1, 2.0) == 3.0
    assert demo_func(0, 0.0) == 0.0
    assert demo_func(-1, 1.0) == 0.0
