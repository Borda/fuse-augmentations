"""Compare native backends against fuse-augmentations on fixed geometry recipes.

The demo defaults to scikit-image's bundled ``coins`` sample. It renders three
fixed, three-step geometry recipes for each supported live backend: Kornia,
TorchVision v2, and Albumentations. Every recipe fixes its rotation, scale, and
shear values, so native and fused paths receive the same transform parameters.
Pixel differences are therefore caused by sequential versus composed resampling,
not different random draws.

Run:
    uv run --all-extras --group benchmark python examples/visualize_resampling_degradation.py

Render a TorchVision case:
    uv run --all-extras --group benchmark python examples/visualize_resampling_degradation.py \
        --backend torchvision --case camera-jitter

"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn.functional import interpolate

from fuse_augmentations import Compose, ReorderPolicy

Backend = Literal["kornia", "torchvision", "albumentations"]
_BACKENDS: tuple[Backend, ...] = ("kornia", "torchvision", "albumentations")
_INPUT_SIZE = 512
_IMAGE_NAMES = ("astronaut", "camera", "chelsea", "coffee", "coins")
_SEQ_ACCENT = "#ff00ff"  # native sequential border/label = pure magenta (255, 0, 255), the overlay tint
_FUSED_ACCENT = "#00ff00"  # fused border/label = pure green (0, 255, 0), the overlay channel


@dataclass(frozen=True)
class Recipe:
    """Describe one fixed three-step geometry recipe shared across backends."""

    name: str
    stem: str
    steps: tuple[str, ...]
    rotation: float
    scale: float
    shear: float


@dataclass(frozen=True)
class DemoCase:
    """Describe one backend-specific realization of a fixed geometry recipe."""

    backend: Backend
    recipe: Recipe
    transforms: tuple[object, ...]


_RECIPES = (
    Recipe("Framing", "framing", ("rotate +20°", "scale 0.89", "x-shear -10°"), 20.0, 0.89, -10.0),
    Recipe("Camera jitter", "camera-jitter", ("rotate -16°", "scale 0.88", "x-shear -8°"), -16.0, 0.88, -8.0),
    Recipe("Off-axis jitter", "off-axis-jitter", ("rotate +13°", "scale 0.92", "x-shear -12°"), 13.0, 0.92, -12.0),
)


def _load_image(image_name: str) -> torch.Tensor:
    """Load one bundled scikit-image sample and resize it to the demo input size."""
    from skimage import data as skimage_data

    image_loaders = {
        "astronaut": skimage_data.astronaut,
        "camera": skimage_data.camera,
        "chelsea": skimage_data.chelsea,
        "coffee": skimage_data.coffee,
        "coins": skimage_data.coins,
    }
    image_np = np.asarray(image_loaders[image_name]())
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., None], 3, axis=-1)
    image = torch.from_numpy(image_np[..., :3].copy()).permute(2, 0, 1).unsqueeze(0).float()
    if image.max() > 1.0:
        image = image / 255.0
    return interpolate(image, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=True)


def _build_cases(backend: Backend) -> tuple[DemoCase, ...]:
    """Create the three equivalent fixed-parameter recipes for one backend."""
    cases = []
    for recipe in _RECIPES:
        if backend == "kornia":
            import kornia.augmentation as K

            transforms = (
                K.RandomRotation((recipe.rotation, recipe.rotation), p=1.0),
                K.RandomAffine((0.0, 0.0), scale=(recipe.scale, recipe.scale), p=1.0),
                K.RandomAffine((0.0, 0.0), shear=(recipe.shear, recipe.shear), p=1.0),
            )
        elif backend == "torchvision":
            import torchvision.transforms.v2 as T

            transforms = (
                T.RandomRotation((recipe.rotation, recipe.rotation)),
                T.RandomAffine((0.0, 0.0), scale=(recipe.scale, recipe.scale)),
                T.RandomAffine((0.0, 0.0), shear=(recipe.shear, recipe.shear)),
            )
        else:
            import albumentations as A

            transforms = (
                A.Rotate(limit=(recipe.rotation, recipe.rotation), p=1.0),
                A.Affine(scale=(recipe.scale, recipe.scale), p=1.0),
                A.Affine(scale=(1.0, 1.0), shear={"x": (recipe.shear, recipe.shear), "y": (0.0, 0.0)}, p=1.0),
            )
        cases.append(DemoCase(backend, recipe, transforms))
    return tuple(cases)


def _to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a BCHW image tensor into a clipped HWC tensor for Matplotlib."""
    return tensor[0].permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0)


def _channel_overlay(sequential: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """Encode sequential luminance in magenta and fused luminance in green (overlap reads white)."""
    sequential_luminance = _to_image(sequential).mean(dim=-1)
    fused_luminance = _to_image(fused).mean(dim=-1)
    return torch.stack((sequential_luminance, fused_luminance, sequential_luminance), dim=-1)


def _run_pair(case: DemoCase, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run one backend's native sequence and fused path with fixed parameters."""
    if case.backend == "kornia":
        import kornia.augmentation as K

        sequential = K.AugmentationSequential(*copy.deepcopy(case.transforms))(image)
    elif case.backend == "torchvision":
        import torchvision.transforms.v2 as T

        sequential = T.Compose(list(copy.deepcopy(case.transforms)))(image)
    else:
        import albumentations as A

        native_hwc = A.Compose(list(copy.deepcopy(case.transforms)))(image=_to_image(image).numpy())["image"]
        sequential = torch.from_numpy(native_hwc.copy()).permute(2, 0, 1).unsqueeze(0).to(image)

    fused = Compose(copy.deepcopy(case.transforms), reorder=ReorderPolicy.NONE)
    fused_output = fused(image)
    if sequential.shape != fused_output.shape:
        msg = f"native and fused shapes differ: {tuple(sequential.shape)} != {tuple(fused_output.shape)}"
        raise RuntimeError(msg)
    return sequential, fused_output, fused.n_warps_saved


def _load_plotting() -> tuple[object, object]:
    """Load Matplotlib only when the visualization CLI is executed."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    return plt, Patch


def _render_case(image: torch.Tensor, case: DemoCase, output_path: Path) -> None:
    """Render one 2x2 comparison figure for a replayed augmentation collection."""
    plt, Patch = _load_plotting()
    sequential, fused, _ = _run_pair(case, image)
    figure = plt.figure(figsize=(9, 9), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1, 0.12, 1))
    axes = np.array([
        [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
        [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
    ])
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    panels = (
        (_to_image(image), "Input", None),
        (_channel_overlay(sequential, fused), "Overlay", None),
        (_to_image(sequential), "Native composite: 3 resamples", _SEQ_ACCENT),
        (_to_image(fused), "Fused composite: 1 resample", _FUSED_ACCENT),
    )
    for axis, (panel, title, accent) in zip(axes.flat, panels, strict=True):
        axis.imshow(panel)
        if accent is None:
            axis.set_axis_off()
            axis.set_title(title, weight="bold")
            continue
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(3.0)
        axis.set_title(title, weight="bold", color=accent)
    legend_axis.legend(
        handles=[
            Patch(facecolor="#ff00ff", label="Native composite only"),
            Patch(facecolor="#00ff00", label="Fused composite only"),
            Patch(facecolor="white", edgecolor="0.3", label="Perfect overlap"),
        ],
        loc="center",
        ncols=3,
        framealpha=0.85,
        fontsize=9,
    )
    figure.suptitle(
        f"{case.backend.title()} — {case.recipe.name}\n" + " → ".join(case.recipe.steps),
        fontsize=14,
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=90, bbox_inches="tight", format="webp", pil_kwargs={"quality": 85})
    plt.close(figure)


def render(output_dir: Path, image_name: str, backend: Backend, selected_case: str) -> None:
    """Render one named backend-specific 2x2 comparison figure."""
    cases = {case.recipe.stem: case for case in _build_cases(backend)}
    _render_case(
        _load_image(image_name),
        cases[selected_case],
        output_dir / f"sequential-vs-fused-{backend}-{selected_case}.webp",
    )


def main(
    output_dir: str = "docs/assets/images",
    image: str = "coins",
    backend: Backend = "kornia",
    case: str = "framing",
) -> None:
    """Generate one deterministic native-versus-fused comparison figure.

    Args:
        output_dir: Directory for the generated WebP figure.
        image: Bundled scikit-image sample to transform.
        backend: Native backend to compare (kornia, torchvision, albumentations).
        case: Named scenario stem (framing, camera-jitter, off-axis-jitter).

    Examples:
        >>> callable(main)
        True

    """
    cases = tuple(recipe.stem for recipe in _RECIPES)
    assert image in _IMAGE_NAMES, f"--image must be one of {_IMAGE_NAMES}, got {image!r}"
    assert backend in _BACKENDS, f"--backend must be one of {_BACKENDS}, got {backend!r}"
    assert case in cases, f"--case must be one of {cases}, got {case!r}"
    render(Path(output_dir), image, backend, case)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
