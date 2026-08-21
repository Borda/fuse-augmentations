# Project-specific agent guidance

This file adds only `fuse-augmentations`-specific guidance to [Borda's shared agent defaults](https://github.com/Borda/.github/blob/main/AGENTS.md). Follow the shared defaults unless this file or the repository configuration says otherwise.

For the contributor workflow, use [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md), especially:

- [development setup and quality checks](.github/CONTRIBUTING.md#development-setup);
- [tests and quality assurance](.github/CONTRIBUTING.md#tests-and-quality-checks);
- [executable documentation examples](.github/CONTRIBUTING.md#executable-documentation-examples);
- [pull-request checklist](.github/CONTRIBUTING.md#pull-requests).

## Project-specific facts

- Python support starts at 3.10. Use `uv` and treat `pyproject.toml` and `.github/workflows/` as the source of truth.
- The package uses a `src/` layout and supports optional Kornia, TorchVision, and Albumentations backends.
- Documentation Python examples in `README.md` and `docs/**/*.md` are generated into ignored `tests/integration/` modules with `.github/scripts/generate_doc_tests.py`. Write them as plain ```` ```python ```` script blocks (`print(...)` + a following output fence in `<details>`) — never as ```` ```pycon ```` `>>>` sessions, which are reserved for `src/**` docstrings.
- Run `python .github/scripts/generate_doc_tests.py` before `python -m pytest tests/integration -q` when changing executable documentation.
- CI regenerates the documentation suite in the all-extras job. Do not commit generated files; there is intentionally no Makefile wrapper.
- When you edit any shape definition under `src/fuse_augmentations/data/` (`geometry.py`, `animals.py`, `symbols.py`, `landmarks.py`) — a new/changed outline, keypoint table, or the OBB/centroid logic they share — validate the change (`uv run pytest tests/test_unit/test_data -q` and `uv run pytest --doctest-modules src/fuse_augmentations/data -q`), then regenerate whichever previews the change actually affects: `python examples/render_shape_reference.py` (static field-guide PNGs under `docs/assets/shape-references/`, one `<prefix><shape>.png` per shape) and `python examples/animate_synthetic_dataset.py --shapes <family> --task all` (animated clips under `docs/assets/datasets/`) for the affected family. Look at the regenerated images before calling the change done — a mis-placed keypoint or a wrong outline is visible there, not just in a passing test.
- `examples/render_shape_reference.py`'s blue box is each shape's plain axis-aligned **detection** box at its fixed, upright reference orientation — not `polygon_to_obb`'s minimum-area **OBB**. The two differ for any shape whose true minimum-area OBB is not axis-aligned at that pose (e.g. the `arrow`, whose OBB is a diamond flush to its barb tips) or whose OBB candidates tie (any acute or right triangle, for instance — see `GeomShape.TRIANGLE`'s docstring). The real, rotated OBB is what the generator's actually-rotated samples carry — see the `--task obb` animated previews and `docs/guides/synthetic-datasets.md`'s "Tasks" section for that. Do not "fix" the reference script to draw the rotated OBB instead; that was tried and reverted because it produces a tilted or arbitrarily-tied box for a shape shown in a fixed pose, which is confusing rather than informative for a field-guide reference. `SymbolShape` has no plain-triangle member for the same tie reason — don't add one back without solving it.
