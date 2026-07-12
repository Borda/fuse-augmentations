---
title: Install fuse-augmentations
description: Install the PyTorch fusion engine alone or with Kornia, TorchVision, and Albumentations adapters, then verify the environment.
---

# Install fuse-augmentations

The base package requires Python 3.10 or newer and PyTorch 2.2 or newer. Kornia, TorchVision, and Albumentations are optional because the native builder can create a useful pipeline without them.

!!! note "Project maturity" The package is currently classified Alpha. Pin versions in production or research environments and validate the exact pipeline after upgrades.

## Base installation

```bash
python -m pip install fuse-augmentations
```

This installs NumPy, Pillow, SciPy, PyTorch, and the package itself. It is enough for [`Compose.from_params`](quickstart.md).

## Optional backends

Install only the adapter ecosystems you use:

=== "Kornia"

````
```bash
python -m pip install "fuse-augmentations[kornia]"
```
````

=== "TorchVision"

````
```bash
python -m pip install "fuse-augmentations[torchvision]"
```
````

=== "Albumentations"

````
```bash
python -m pip install "fuse-augmentations[albumentations]"
```
````

=== "All adapters"

````
```bash
python -m pip install "fuse-augmentations[all]"
```
````

The extras enable adapter support; they do not make every upstream transform or parameter combination fusible. Check the [capability tables](../concepts/capabilities.md).

## Verify the installation

```bash
python -c "import torch, fuse_augmentations; print(fuse_augmentations.__version__, torch.__version__)"
```

Both import namespaces expose the same objects:

```python
from fuse_aug import Compose as ShortCompose
from fuse_augmentations import Compose

assert ShortCompose is Compose
```

Long-form documentation uses `fuse_augmentations` so the import matches the distribution name.

## Build these docs locally

The repository uses `uv` for its locked development environment:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

Run the release-style documentation gate with:

```bash
uv run --group docs mkdocs build --strict
```

The generated site is written to `site/`. The docs dependency group is not installed for package users.
