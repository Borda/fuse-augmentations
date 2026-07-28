"""Animate native sequential resampling degradation against a single fused warp.

The static companion (``examples/visualize_resampling_degradation.py``) renders one 2x2
figure per recipe. This script turns the same fixed geometry recipes into a
side-by-side animation: the left panel replays the native backend one transform
at a time, resampling on every step, while the right panel holds the input until
``fuse_augmentations.Compose`` fires the whole recipe as one warp. A running
resample counter on each panel makes the "3 resamples versus 1" story literal,
and a closing channel overlay shows where the sequential path drifted.

Frames are rendered at a fixed canvas size and written as an animated WebP
(smaller than GIF, matching the existing WebP assets); ``--format gif`` is
available as a fallback.

Run one case:
    uv run --all-extras --group benchmark python examples/animate_resampling_degradation.py \
        --backend torchvision --case camera-jitter

Render every backend and case:
    uv run --all-extras --group benchmark python examples/animate_resampling_degradation.py --all_cases

"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Allow ``python examples/animate_resampling_degradation.py`` to import its sibling module
# by putting the repository root (this file's parent's parent) on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.visualize_resampling_degradation import (
    _BACKENDS,
    _IMAGE_NAMES,
    _RECIPES,
    Backend,
    DemoCase,
    _build_cases,
    _channel_overlay,
    _load_image,
    _to_image,
)
from fuse_augmentations import Compose, ReorderPolicy

_SEQ_COLOR = "#ff00ff"  # native sequential accent = pure magenta (255, 0, 255), the overlay tint
_FUSED_COLOR = "#00ff00"  # fused accent = pure green (0, 255, 0), the overlay channel
_CANVAS_DPI = 100


@dataclass(frozen=True)
class _Frame:
    """One rendered animation frame paired with its display duration."""

    image: object  # PIL.Image.Image, imported lazily
    duration_ms: int


def _cumulative_sequential(case: DemoCase, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply each native transform in turn, capturing the state after every resample.

    Returns:
        States ``(input, after_step_1, ..., after_step_n)`` — one extra resample
        per element beyond the input, exactly what the native sequential path pays.

    """
    states = [image]
    current = image
    if case.backend == "kornia" or case.backend == "torchvision":
        for transform in copy.deepcopy(case.transforms):
            current = transform(current)
            states.append(current)
    else:
        import albumentations as A

        current_hwc = _to_image(image).numpy()
        for transform in copy.deepcopy(case.transforms):
            current_hwc = A.Compose([transform])(image=current_hwc)["image"]
            current = torch.from_numpy(current_hwc.copy()).permute(2, 0, 1).unsqueeze(0).to(image)
            states.append(current)
    return tuple(states)


def _fused_output(case: DemoCase, image: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Run the fused single-warp path and report how many warps it saved."""
    fused = Compose(copy.deepcopy(case.transforms), reorder=ReorderPolicy.NONE)
    return fused(image), fused.n_warps_saved


def _load_plotting() -> tuple[object, object]:
    """Load Matplotlib (Agg) and PIL only when rendering frames."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    return plt, Image


def _new_canvas(plt: object) -> object:
    """Create a fixed-size constrained figure so every frame shares pixel dims."""
    return plt.figure(figsize=(9.0, 5.4), dpi=_CANVAS_DPI, layout="constrained")


def _fig_to_image(figure: object, Image: object) -> object:
    """Rasterize a Matplotlib figure to an RGB PIL image at the canvas size."""
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())
    return Image.fromarray(buffer).convert("RGB")


def _resample_label(count: int) -> str:
    """Format a resample counter with correct pluralization."""
    return f"{count} resample" if count == 1 else f"{count} resamples"


def _draw_panel(axis: object, panel: torch.Tensor, title: str, accent: str) -> None:
    """Show one image panel with an accent-colored border and bold title."""
    axis.imshow(panel)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(3.0)
    axis.set_title(title, weight="bold", fontsize=11, color=accent)


def _render_pair(
    plt: object,
    Image: object,
    case: DemoCase,
    left: torch.Tensor,
    right: torch.Tensor,
    seq_count: int,
    fused_count: int,
    left_caption: str,
    right_caption: str,
) -> object:
    """Render the two-panel comparison frame with a separate caption under each subplot."""
    figure = _new_canvas(plt)
    figure.suptitle(
        f"{case.backend.title()} — {case.recipe.name}\n" + " → ".join(case.recipe.steps),
        fontsize=13,
        weight="bold",
    )
    left_axis, right_axis = figure.subplots(1, 2)
    _draw_panel(left_axis, _to_image(left), f"Native composite · {_resample_label(seq_count)}", _SEQ_COLOR)
    _draw_panel(right_axis, _to_image(right), f"Fused composite · {_resample_label(fused_count)}", _FUSED_COLOR)
    left_axis.set_xlabel(left_caption, fontsize=10, style="italic")
    right_axis.set_xlabel(right_caption, fontsize=10, style="italic")
    image = _fig_to_image(figure, Image)
    plt.close(figure)
    return image


def _tint(image_tensor: torch.Tensor, rgb: tuple[float, float, float]) -> torch.Tensor:
    """Render one image's luminance through an RGB weight so it reads as a single hue.

    Args:
        image_tensor: A BCHW image tensor.
        rgb: Per-channel weights, e.g. ``(1, 0, 1)`` for magenta or ``(0, 1, 0)`` for green.

    Returns:
        An HWC tensor with luminance scaled into each channel by ``rgb``.

    Examples:
        >>> import torch
        >>> magenta = _tint(torch.ones(1, 3, 4, 4), (1, 0, 1))
        >>> bool((magenta[..., 1] == 0).all()) and bool((magenta[..., 0] == magenta[..., 2]).all())
        True

    """
    luminance = _to_image(image_tensor).mean(dim=-1)
    return torch.stack([luminance * rgb[0], luminance * rgb[1], luminance * rgb[2]], dim=-1)


def _render_split(plt: object, Image: object, case: DemoCase, sequential: torch.Tensor, fused: torch.Tensor) -> object:
    """Render the channel-split frame: sequential in magenta, fused in green, before the overlay."""
    figure = _new_canvas(plt)
    figure.suptitle(
        f"{case.backend.title()} — {case.recipe.name}\nsplit into channels before overlay",
        fontsize=13,
        weight="bold",
    )
    left_axis, right_axis = figure.subplots(1, 2)
    _draw_panel(left_axis, _tint(sequential, (1, 0, 1)), "Native composite → magenta", _SEQ_COLOR)
    _draw_panel(right_axis, _tint(fused, (0, 1, 0)), "Fused composite → green", _FUSED_COLOR)
    left_axis.set_xlabel("3-resample result, tinted magenta", fontsize=10, style="italic")
    right_axis.set_xlabel("1-warp result, tinted green — stacks to white where they agree", fontsize=10, style="italic")
    image = _fig_to_image(figure, Image)
    plt.close(figure)
    return image


def _render_overlay(
    plt: object, Image: object, case: DemoCase, sequential: torch.Tensor, fused: torch.Tensor
) -> object:
    """Render the closing full-width channel overlay of both final states."""
    figure = _new_canvas(plt)
    figure.suptitle(
        f"{case.backend.title()} — {case.recipe.name}\noverlay: 3 resamples vs 1",
        fontsize=13,
        weight="bold",
    )
    axis = figure.subplots(1, 1)
    _draw_panel(
        axis,
        _channel_overlay(sequential, fused),
        "White = match · magenta = sequential drift · green = fused",
        "#666666",
    )
    axis.set_xlabel(
        "Sequential softens with every resample; fused keeps a single sample.",
        fontsize=11,
        style="italic",
    )
    image = _fig_to_image(figure, Image)
    plt.close(figure)
    return image


def _native_caption(step_index: int, total: int, step_label: str) -> str:
    """Describe the native (left) panel at one timeline step — one resample added per step.

    The fused panel reaches its final image on the very first tick, so these captions
    never imply fusion is as slow as the whole native chain.

    Args:
        step_index: 1-based index of the current native step.
        total: Total number of native steps in the recipe.
        step_label: Human-readable label for the current native transform.

    Returns:
        The caption for the native (left) panel at this step.

    Examples:
        >>> _native_caption(1, 3, "rotate +20°")
        'Resample 1/3: rotate +20°'

    """
    return f"Resample {step_index}/{total}: {step_label}"


def _fused_caption(step_index: int) -> str:
    """Describe the fused (right) panel at a given step — done on tick one, then holding.

    Args:
        step_index: 1-based index of the current native step.

    Returns:
        The caption for the fused (right) panel at this step.

    Examples:
        >>> _fused_caption(1)
        'One warp — final image already'
        >>> _fused_caption(3)
        'Still done — no extra resampling'

    """
    return "One warp — final image already" if step_index == 1 else "Still done — no extra resampling"


def _build_frames(plt: object, Image: object, case: DemoCase, image: torch.Tensor) -> list[_Frame]:
    """Assemble the ordered frame list telling the resample-count story.

    The fused panel jumps to its final image on the first step tick — the same moment the native chain finishes only its
    first resample — then holds. This avoids the illusion that the single fused warp takes as long as all the sequential
    steps combined.

    """
    states = _cumulative_sequential(case, image)
    fused, _warps = _fused_output(case, image)
    total = len(case.recipe.steps)
    frames: list[_Frame] = [
        _Frame(
            _render_pair(
                plt,
                Image,
                case,
                states[0],
                states[0],
                0,
                0,
                "Starting image — nothing resampled yet",
                "Starting image — nothing resampled yet",
            ),
            1500,
        )
    ]
    for step_index, step_label in enumerate(case.recipe.steps, start=1):
        # Fused reaches its final image on the first tick and stays there.
        frames.append(
            _Frame(
                _render_pair(
                    plt,
                    Image,
                    case,
                    states[step_index],
                    fused,
                    step_index,
                    1,
                    _native_caption(step_index, total, step_label),
                    _fused_caption(step_index),
                ),
                1300 if step_index == total else 1150,
            )
        )
    frames.append(_Frame(_render_split(plt, Image, case, states[-1], fused), 1800))
    frames.append(_Frame(_render_overlay(plt, Image, case, states[-1], fused), 2200))
    return frames


def _save_animation(frames: list[_Frame], output_path: Path, image_format: str) -> None:
    """Write frames as an animated WebP or GIF loop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [frame.image for frame in frames]
    durations = [frame.duration_ms for frame in frames]
    save_kwargs = {"save_all": True, "append_images": images[1:], "duration": durations, "loop": 0}
    if image_format == "webp":
        save_kwargs["quality"] = 88
        save_kwargs["method"] = 6
    else:
        save_kwargs["disposal"] = 2
        save_kwargs["optimize"] = True
    images[0].save(output_path, format=image_format.upper(), **save_kwargs)


def render(output_dir: Path, image_name: str, backend: Backend, selected_case: str, image_format: str) -> Path:
    """Render one named backend/case animation and return its path."""
    plt, Image = _load_plotting()
    cases = {case.recipe.stem: case for case in _build_cases(backend)}
    case = cases[selected_case]
    frames = _build_frames(plt, Image, case, _load_image(image_name))
    output_path = output_dir / f"animated-sequential-vs-fused-{backend}-{selected_case}.{image_format}"
    _save_animation(frames, output_path, image_format)
    return output_path


def main(
    output_dir: str = "docs/assets/images",
    image: str = "coins",
    backend: Backend = "kornia",
    case: str = "framing",
    image_format: str = "webp",
    all_cases: bool = False,
) -> None:
    """Generate one animation, or the full backend x case grid with ``--all_cases``.

    Args:
        output_dir: Directory for the generated animations.
        image: Bundled scikit-image sample to transform.
        backend: Native backend to compare (kornia, torchvision, albumentations).
        case: Named scenario stem (framing, camera-jitter, off-axis-jitter).
        image_format: Animation container, ``webp`` or ``gif``.
        all_cases: Render every backend and case instead of the single selection.

    Examples:
        >>> callable(main)
        True

    """
    cases = tuple(recipe.stem for recipe in _RECIPES)
    assert image in _IMAGE_NAMES, f"--image must be one of {_IMAGE_NAMES}, got {image!r}"
    assert image_format in ("webp", "gif"), f"--image_format must be one of ('webp', 'gif'), got {image_format!r}"
    output_path = Path(output_dir)
    if all_cases:
        for a_backend in _BACKENDS:
            for recipe in _RECIPES:
                print(f"wrote {render(output_path, image, a_backend, recipe.stem, image_format)}")
        return
    assert backend in _BACKENDS, f"--backend must be one of {_BACKENDS}, got {backend!r}"
    assert case in cases, f"--case must be one of {cases}, got {case!r}"
    print(f"wrote {render(output_path, image, backend, case, image_format)}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
