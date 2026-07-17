# Contributing Guide

Thank you for contributing to `fuse-augmentations`. Bug reports, documentation improvements, focused fixes, and reviewed pull requests are welcome. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

> [!NOTE]
>
> Project configuration is the source of truth. If this guide conflicts with `pyproject.toml`, `.pre-commit-config.yaml`, or a workflow under `.github/workflows/`, follow the configuration and open an issue so this guide can be corrected.

## Before you start

Please read the relevant documentation, search existing issues, and inspect the current implementation and tests before starting work. Open an issue before implementing a new public API, architecture change, or substantial feature so the scope can be discussed first.

Bug fixes and small documentation changes can usually proceed directly when their scope is clear. Keep every change focused and explain its motivation in the pull request.

## Ways to contribute

- Report a reproducible bug or documentation error.
- Improve examples, API documentation, or benchmark explanations.
- Add or improve tests, including edge cases and regression coverage.
- Fix an approved bug or implement an approved feature.
- Review pull requests and help answer questions in issues or discussions.

## Reporting bugs and proposing features

Open an issue with the [bug report](ISSUE_TEMPLATE/bug_report.yml) or [feature request](ISSUE_TEMPLATE/feature_request.yml) form; each lists the required fields. In short: search existing issues first, give a minimal reproduction for bugs, and describe the user problem, proposed interface, and alternatives for features.

Wait for maintainer feedback before implementing a new public API, architecture change, or substantial feature, then keep the implementation, tests, and docs in one focused pull request.

## Development setup

This is a Python 3.10+ project managed with `uv`.

```bash
uv sync --all-extras --group dev
```

For documentation builds, also install the docs group:

```bash
uv sync --all-extras --group dev --group docs
uv run --group docs mkdocs build --strict
```

## Tests and quality checks

Run the focused checks for the files you change, then the relevant full checks:

```bash
python -m pytest . -v --cov=fuse_augmentations
ruff check .
ruff format --check .
python -m mypy
pre-commit run --all-files
```

The CI matrix tests Python 3.10, 3.11, 3.12, 3.13, and 3.14, the supported optional backends, and the all-extras configuration. Keep tests deterministic and include specific assertions that would fail for plausible but incorrect behavior.

No CI runner has GPU access: tests marked `gpu` (`pytest -m gpu`) are skipped in every workflow and never exercise a real CUDA device, so CUDA numeric correctness has no automated verification. If you change GPU-specific code, run the `gpu`-marked tests locally on a CUDA-capable machine before opening a pull request.

## Executable documentation examples

Python examples in `README.md` and `docs/**/*.md` are generated into ignored pytest modules. Keep each example runnable from a clean generated test context:

- include imports, deterministic seeds, and all required setup in the block or an explicitly shared setup block;
- assert the user-visible contract, including output shape, dtype, and important invariants;
- use `print(...)` only for meaningful information a reader should see when running the example, such as a fusion plan, capability summary, descriptor, or recorded metadata;
- do not print smoke-test messages such as `print("imports ok")`;
- when a block depends on names from an earlier block, put `<!--phmdoctest-share-names-->` immediately before the setup block;
- when printing, place the exact output in an unlabelled Markdown fence inside a collapsible `<details>` block immediately after the Python fence. The summary must describe what the output represents, not use a generic label such as `Expected output`;
- make printed output deterministic by sorting sets and mappings, normalizing environment-specific values, and avoiding version/device output unless the example deliberately tests it;
- mark only genuinely non-standalone examples with `# phmdoctest:skip`, and explain why in the surrounding prose. Installed optional backends are tested in the all-extras CI job and should not be skipped merely because they are optional dependencies.

Example:

```python
print(pipe.fusion_plan)
```

<details>
<summary>Fusion plan for the configured pipeline</summary>

```
fused(_DirectParamTransform, _DirectFlipTransform)
```

</details>

Generate and run the documentation tests with:

```bash
python .github/scripts/generate_doc_tests.py
python -m pytest tests/integration -q
```

For a README-only check, use the original focused command:

```bash
python -m phmdoctest README.md -s "phmdoctest:skip" --outfile tests/integration/test_readme.py
python -m pytest tests/integration/test_readme.py -q
```

The generator scans `README.md` and all Markdown files below `docs/`, removes stale generated modules, and fails if a non-skipped Python block contains `print(...)` without an exact output fence. CI regenerates the complete suite in the all-extras test job. Do not commit files under `tests/integration/`; the directory is intentionally ignored.

## Pull requests

The [pull request template](PULL_REQUEST_TEMPLATE.md) carries the full checklist; complete it when you open a PR. Use a descriptive branch name such as `fix/123-short-description`, `feat/123-short-description`, `docs/update-examples`, or `test/add-regression-case`.

## Review expectations

Review comments should explain why an issue matters and distinguish required changes from suggestions. Reviewers should check correctness, tests, documentation, reproducibility, compatibility, performance, and security as applicable. Defer automatically fixable style issues to `ruff` or pre-commit.

## Attribution

This guide is adapted from [Borda's shared contributing guide](https://github.com/Borda/.github/blob/main/.github/CONTRIBUTING.md) for this project's Python, CI, and executable-documentation workflow.
