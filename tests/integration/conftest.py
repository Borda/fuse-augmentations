"""Integration-test fixtures — require kornia or other external backend."""

import pytest

kornia = pytest.importorskip("kornia", reason="kornia>=0.6.12 required for integration tests")
