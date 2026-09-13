---
title: Customization and extension
description: Reproducibility and tuning knobs, custom splits and fills, extending the shape vocabulary, and editing the packaged SVG assets.
---

# Customization and extension

## Reproducibility and tuning

Pass `seed=` for byte-identical output; all randomness flows through a single `numpy.random.Generator`. Tune content through `generate_dataset(**config_kwargs)` or a full `SyntheticConfig`:

```python
import tempfile

from fuse_augmentations.data import SplitRatios, generate_dataset

with tempfile.TemporaryDirectory() as out_dir:
    counts = generate_dataset(
        out_dir,
        num_images=200,
        fmt="yolo",
        task="obb",
        split_ratios=SplitRatios(train=0.8, val=0.2, test=0.0),
        img_size=64,
        max_objects=15,
        seed=42,
    )

print(counts)
```

<details>
<summary>Per-split image counts for the 80/20 OBB split</summary>

```
{'train': 160, 'val': 40}
```

</details>

`SyntheticConfig` knobs: `img_size`, `min_objects`/`max_objects`, `min_size_ratio`/`max_size_ratio`, `overlap_iou`, `boundary_tolerance`, `rotate`, `asymmetry_jitter`, `background`, `class_mode`, `task`, `shapes`, `colors`. Overlapping candidates (IoU above `overlap_iou`) and out-of-bounds candidates (more than `boundary_tolerance` outside the frame) are rejected during placement.

`task` and `class_mode` are config fields only. `generate_dataset` takes no `task=` argument of its own — pass it as a keyword and it flows into the config, or set it on a `SyntheticConfig` you build yourself, but not both. Earlier releases accepted it in both places and cross-checked them, which meant a `None` sentinel, a conflict error, and a paragraph explaining which one won; one owner removes all three.

### Choosing a background

`background` accepts a plain fill — a `Color`, an `(r, g, b)` triple, or a `Fill` — or one of the background types, which carry their own parameters and render themselves. The mode *is* the type: a `NoiseBackground` has a `sigma` and a `TextureBackground` does not, so there is no combination of mode and parameter that has to be rejected by hand.

| type                     | parameters                                              | what it changes                                                     |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------- |
| `SolidBackground`        | `color`                                                 | one flat fill — the default, and what a bare triple normalizes to   |
| `GradientBackground`     | `stops`, `direction`, `radial`                          | a linear or radial ramp, so intensity is no longer one global value |
| `NoiseBackground`        | `base`, `sigma`                                         | per-pixel Gaussian grain, which removes the trivial edge detector   |
| `ImpulseNoiseBackground` | `base`, `amount`, `salt_ratio`                          | salt-and-pepper pixels, heavy-tailed and blur-resistant             |
| `TextureBackground`      | `base`, `amplitude`, `frequency`, `octaves`, `quantize` | value noise at a chosen scale, so false positives become possible   |

Every background draws from a side stream of its own, never from the placement stream, so switching one on at a fixed seed leaves every object exactly where it was:

```python
from fuse_augmentations.data import NoiseBackground, SyntheticConfig, SyntheticGenerator

flat = SyntheticConfig(img_size=64, max_objects=3)
noisy = SyntheticConfig(img_size=64, max_objects=3, background=NoiseBackground(sigma=24.0))

placements = [[a.bbox_xyxy for a in s.annotations] for s in SyntheticGenerator(flat).generate(3, seed=1)]
unmoved = [[a.bbox_xyxy for a in s.annotations] for s in SyntheticGenerator(noisy).generate(3, seed=1)]

print(placements == unmoved)
print(type(SyntheticConfig().background).__name__)
```

<details>
<summary>The same placements under a different canvas</summary>

```
True
SolidBackground
```

</details>

`config.background` reads back as a background object whichever spelling went in, so a caller that wants the flat fill asks for `config.background.color.rgb` rather than for the field itself. Keep `NoiseBackground.sigma` at or below `32` for any run scored against a rasterized-ink oracle: the oracle finds ink by colour distance, Gaussian noise is unbounded, and its tail — not its mean — is what starts producing false ink above that.

### Breaking left/right symmetry

Most shapes are drawn mirror-symmetric about their own vertical axis in canonical orientation (every geometric shape, every symbol, most letters), so their oriented bounding box otherwise always shows identical left/right margins — real oriented objects (vehicles, ships) rarely are. `asymmetry_jitter` (default `0.0`, a fraction in `[0, 0.5)`) narrows a randomly chosen half — left or right of that axis, before rotation — of each placed object by up to that fraction, independently per instance. The animal silhouettes and roughly two-thirds of the letters are already asymmetric on their own (a letter's own strokes rarely balance left-right the way a symbol's outline is authored to), so the jitter is redundant orientation variety for them rather than the sole source of it — it still applies uniformly to every shape but `circle`, which is excluded for the separate reason below:

```pycon
>>> from fuse_augmentations.data.config import SyntheticConfig
>>> SyntheticConfig(asymmetry_jitter=0.15).asymmetry_jitter
0.15

```

`circle` is always excluded — it never rotates either, so an unrotated skew would bias every circle toward the same absolute image direction instead of varying with a random orientation. Under `Task.KEYPOINTS` the same draw skews the polygon and the landmark table together, so a shape and its keypoints never drift apart. `0.0` (the default) draws exactly the RNG sequence this package always has, so existing seeded configurations are unaffected.

### Custom splits

`SplitRatios` names train/val/test because that is what almost every caller wants, not because the set is closed. `SplitRatios.custom` takes any names at all:

```python
from fuse_augmentations.data import SplitRatios

holdout = SplitRatios.custom({"train": 0.6, "calib": 0.2, "test": 0.2})
print(holdout.to_dict())
```

<details>
<summary>Custom split fractions</summary>

```
{'train': 0.6, 'calib': 0.2, 'test': 0.2}
```

</details>

The arithmetic is unchanged: fractions must be non-negative and sum to 1, or construction raises.

### Custom fill colors

`colors` accepts a named `Color`, any 8-bit `(r, g, b)` triple, or an explicit `Fill`. All three are normalized to `Fill` at construction, so `config.colors` reads back as `Fill` objects whichever spelling went in — the same one-type-inside rule `task` and `class_mode` already follow.

A `Fill` carries the RGB to draw with and, when it came from a named `Color`, that name. A raw triple has no name, so `Fill.label` falls back to the hex value — which is what keeps class naming well defined under `ClassMode.COLOR` and `ClassMode.SHAPE_COLOR` without inventing color names:

```python
from fuse_augmentations.data import (
    ClassMode,
    Color,
    DEFAULT_SHAPES,
    SyntheticConfig,
    class_names,
)

gold = SyntheticConfig(colors=((255, 215, 0), Color.RED))
print([fill.label for fill in gold.colors])
print(class_names(ClassMode.SHAPE_COLOR, DEFAULT_SHAPES, gold.colors)[:2])
```

<details>
<summary>Class names for a custom fill</summary>

```
['ffd700', 'red']
['ffd700_square', 'red_square']
```

</details>

## Extending the vocabulary

Every shape family — the analytic primitives, the animals, the symbols, the letters — is registered once in `fuse_augmentations.data.families`, and every other module reads that registry rather than naming the families itself:

```python
from fuse_augmentations.data import SHAPE_FAMILIES, family_of
from fuse_augmentations.data.animals import AnimalShape

summary = [(f.name, len(f.members), f.has_keypoints) for f in SHAPE_FAMILIES]
print(summary)
print(family_of(AnimalShape.DUCK).name)
```

<details>
<summary>The registered families</summary>

```
[('primitives', 4, False), ('animals', 12, True), ('symbols', 7, True), ('letters', 26, True)]
animals
```

</details>

A `ShapeFamily` carries its members, an outline accessor, and — for a keypoint-bearing family — its schema and landmark placer. Adding a fifth family means writing the module and appending one entry: the enum derives from `ShapeEnum`, so `Shape` covers it with no second edit. It used to mean editing six places that each encoded the family list differently, where missing one failed quietly: forget the landmark dispatch and the family generated no keypoints at all, which the writer then serialized as a structurally valid all-zero table.

### Registering an output format

`OutputFormat` is a closed enum, but the writer table behind it is not. `register_writer` accepts any key, and `generate_dataset(fmt=...)` will then resolve it:

```python
from fuse_augmentations.data import (
    ClassMode,
    DEFAULT_SHAPES,
    Task,
    YoloWriter,
    class_vocabulary,
    get_writer,
    register_writer,
)


class UltralyticsWriter(YoloWriter):
    """A YOLO writer with house conventions layered on."""


register_writer("ultralytics", UltralyticsWriter)

vocabulary = class_vocabulary(ClassMode.SHAPE, DEFAULT_SHAPES)
writer = get_writer("ultralytics", Task.DETECTION, vocabulary)
print(type(writer).__name__)
```

<details>
<summary>The registered writer</summary>

```
UltralyticsWriter
```

</details>

### Editing the packaged assets

All three asset-backed families are read by one parser (`fuse_augmentations.data.svgio`) and edited by one tool:

```bash
python examples/edit_shape_keypoints.py duck      # an animal silhouette
python examples/edit_shape_keypoints.py anchor    # a symbol outline
python examples/edit_shape_keypoints.py r         # a letter stroke graph
```

Press and hold a point to drag it, release to drop, `s` to save back into the SVG, `q` to quit. Saving rewrites the point group and everything derived from it — an outline family's skeleton, a letter's stroke and cut coordinates — so the file always renders as what it loads as.

## End-to-end example

`examples/generate_synthetic_dataset.py` writes every format × task combination and prints the per-split counts:

```bash
python examples/generate_synthetic_dataset.py --outdir shapes_out --num-images 50 --seed 0
```
