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
- Documentation Python examples in `README.md` and `docs/**/*.md` are generated into ignored `tests/integration/` modules with `.github/scripts/generate_doc_tests.py`.
- Run `python .github/scripts/generate_doc_tests.py` before `python -m pytest tests/integration -q` when changing executable documentation.
- CI regenerates the documentation suite in the all-extras job. Do not commit generated files; there is intentionally no Makefile wrapper.
