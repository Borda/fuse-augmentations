---
title: Install fuse-augmentations
description: Install fuse-augmentations for synthetic COCO/YOLO dataset generation and PyTorch image augmentation, with optional backend adapters.
---

# Install fuse-augmentations

The base package requires Python 3.10 or newer. PyTorch is only required for the image-augmentation engine — it is the `torch` extra, not a base dependency. Kornia, TorchVision, and Albumentations are optional on top of that because the native builder can create a useful augmentation pipeline without them.

!!! note "Project maturity"

    The package is currently classified Alpha. Pin versions in production or research environments and validate the exact pipeline after upgrades.

## Base installation

```bash
python -m pip install fuse-augmentations
```

This installs NumPy, Pillow, and the package itself — no torch. It is enough for [synthetic dataset generation](../datasets/index.md) via `import synth_datasets`, which never imports torch. Generating COCO or YOLO data requires no optional extra, source images, or model download.

## Image augmentation (needs torch)

`Compose`, `FusedCompose`, `AugmentationSequential`, and everything else at the `fuse_augmentations` package root need the `torch` extra:

```bash
python -m pip install "fuse-augmentations[torch]"
```

This is enough for [`Compose.from_params`](quickstart.md) with no optional adapter backend.

## Optional backends

Install only the adapter ecosystems you use — each of these already includes the `torch` extra:

=== "Kornia"

    ```bash
    python -m pip install "fuse-augmentations[kornia]"
    ```

=== "TorchVision"

    ```bash
    python -m pip install "fuse-augmentations[torchvision]"
    ```

=== "Albumentations"

    ```bash
    python -m pip install "fuse-augmentations[albumentations]"
    ```

=== "All adapters"

    ```bash
    python -m pip install "fuse-augmentations[all]"
    ```

The extras enable adapter support; they do not make every upstream transform or parameter combination fusible. Check the [capability tables](../concepts/capabilities.md).

## Verify the installation

```bash
python -c "import torch, fuse_augmentations"
```

The same smoke check is executable in the generated documentation test suite:

```python
import torch
import fuse_augmentations

assert torch.__version__
assert fuse_augmentations.__version__
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
