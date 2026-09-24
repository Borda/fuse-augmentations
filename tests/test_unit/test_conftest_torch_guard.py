"""Regression test for the `tests/conftest.py` torch-import guard's exception-narrowing contract.

`conftest.py` wraps `import torch` in `try/except ModuleNotFoundError`, narrowed from a bare
`ImportError` because a torch install that is *present but broken* (a shared library failing to load,
which raises a plain `ImportError`, not the `ModuleNotFoundError` subclass raised for a genuinely
absent package) must not be silently downgraded to `torch = None`. That downgrade would make every
tensor fixture `importorskip` and green a pytester CI job that actually ran zero augmentation tests.
This pins both branches of the narrowed guard directly against the shipped file, each in a fresh
subprocess with a stub `torch` module so the fake failure cannot leak into the running test session.

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = """
import sys
sys.path.insert(0, {fake_dir!r})
sys.path.insert(1, {tests_dir!r})
try:
    import conftest
except Exception as exc:
    print(f"RAISED:{{type(exc).__name__}}")
else:
    print(f"IMPORTED:torch_is_none={{conftest.torch is None}}")
"""


def _run_with_stub_torch(tmp_path: Path, exception_name: str) -> str:
    """Run `import conftest` in a subprocess where `import torch` raises `exception_name`; return stdout."""
    tests_dir = Path(__file__).resolve().parents[1]
    fake_dir = tmp_path / "fake_torch"
    fake_dir.mkdir()
    (fake_dir / "torch.py").write_text(f"raise {exception_name}('simulated torch import failure')\n")
    script = _SCRIPT.format(fake_dir=str(fake_dir), tests_dir=str(tests_dir))
    result = subprocess.run(  # noqa: S603 - fixed script, no untrusted input
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_absent_torch_still_degrades_to_none(tmp_path: Path) -> None:
    """A genuinely absent torch (`ModuleNotFoundError`) still degrades `conftest.torch` to `None`.

    This is the guard's intended, unchanged behaviour: an environment that never installed torch (the torch-free
    `synth_datasets` CI leg, or a torch-free dev checkout) collects the test session without torch, exactly as it did
    before the F8 narrowing.

    """
    output = _run_with_stub_torch(tmp_path, "ModuleNotFoundError")

    assert output == "IMPORTED:torch_is_none=True"


def test_broken_torch_import_propagates_instead_of_degrading(tmp_path: Path) -> None:
    """A present-but-broken torch (plain `ImportError`, e.g. a missing native shared library) is not swallowed.

    Before the F8 fix this branch was caught by a bare `except ImportError` and downgraded to
    `torch = None` identically to the absent case, so a broken install silently ran zero augmentation
    tests instead of failing collection loudly. The narrowed `except ModuleNotFoundError` must let this
    propagate.

    """
    output = _run_with_stub_torch(tmp_path, "ImportError")

    assert output == "RAISED:ImportError"
