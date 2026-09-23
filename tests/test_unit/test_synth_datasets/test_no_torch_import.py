"""Structural guarantee: `import synth_datasets` never pulls in torch."""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
import sys, tempfile, synth_datasets
assert "torch" not in sys.modules, "importing synth_datasets pulled in torch"
with tempfile.TemporaryDirectory() as tmp:
    synth_datasets.generate_dataset(tmp, num_images=2, fmt="yolo", task="detection", seed=0)
assert "torch" not in sys.modules, "generate_dataset pulled in torch"
print("OK")
"""


def test_importing_synth_datasets_does_not_import_torch() -> None:
    """`import synth_datasets` and dataset generation never touch `sys.modules['torch']`.

    Runs in a fresh subprocess so an already-torch-loaded test session cannot mask a regression: if a
    future change makes `synth_datasets` (or the parent `fuse_augmentations` package it must never
    touch) import torch as a side effect, this is the only way to catch it even in an environment
    where torch happens to be installed.

    """
    result = subprocess.run(  # noqa: S603 - fixed script, no untrusted input
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
