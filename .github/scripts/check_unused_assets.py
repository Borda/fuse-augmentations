"""Fail when a file under ``docs/assets/`` is referenced by nothing.

``mkdocs build --strict`` guards one direction: a page pointing at a missing image fails the build.
It says nothing about the other direction, where an image nobody points at sits in the repository
being cloned forever. Both are ways for the assets and the pages to drift apart, and generated
previews make the second one easy: rename a knob, re-run its generator, and the file written under
the old name stays behind looking exactly like a file still in use.

A reference is any occurrence of the asset's repository-relative path (``assets/<...>``) in a
Markdown file, ``mkdocs.yml``, or the README. That substring is what every form of reference has in
common -- a relative link from a page, an absolute ``raw.githubusercontent.com`` URL in the README,
or a path in the site configuration -- so one check covers all three without parsing any of them.

Referencing one asset from several pages is fine and is not what this looks for: the check is
existence of at least one reference, not uniqueness.

Run it directly, or let the ``check-unused-assets`` pre-commit hook run it::

    python .github/scripts/check_unused_assets.py

"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "docs" / "assets"
#: Where a reference may live. Stylesheets and scripts are searched too, since a background image or
#: a favicon is referenced from CSS rather than from any page.
SEARCHED_SUFFIXES = (".md", ".yml", ".yaml", ".css", ".js", ".html", ".txt")
#: Files that exist to be served rather than to be pointed at, so nothing references them by path.
EXEMPT = {"docs/assets/stylesheets/extra.css"}


def _haystack() -> str:
    """Return every searchable text file in the repository, concatenated.

    One string rather than a per-file scan: the question is only whether a path occurs anywhere, and
    the corpus is small enough that reading it once beats walking it per asset.

    """
    chunks = []
    for path in sorted(REPO.rglob("*")):
        if path.suffix not in SEARCHED_SUFFIXES or not path.is_file():
            continue
        if any(part.startswith(".") and part != ".github" for part in path.relative_to(REPO).parts):
            continue  # generated sites, caches and virtualenvs are not references
        if "site" in path.relative_to(REPO).parts or "node_modules" in path.relative_to(REPO).parts:
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def main() -> int:
    """Return 0 when every asset is referenced, 1 otherwise, naming the orphans."""
    if not ASSETS.is_dir():
        print(f"no {ASSETS.relative_to(REPO)} directory; nothing to check")
        return 0

    corpus = _haystack()
    orphans = []
    for asset in sorted(ASSETS.rglob("*")):
        if not asset.is_file() or asset.name == ".DS_Store":
            continue
        relative = asset.relative_to(REPO).as_posix()
        if relative in EXEMPT:
            continue
        # The docs-relative form is what a page writes; it is a suffix of the repository-relative
        # form, so searching for it also matches an absolute README URL ending in the same path.
        if relative.removeprefix("docs/") not in corpus:
            orphans.append(relative)

    if orphans:
        print(f"{len(orphans)} asset(s) under docs/assets/ are referenced by no page:")
        for orphan in orphans:
            print(f"  {orphan}")
        print("\nReference each from a page, or delete it -- an asset nobody points at is cloned forever.")
        return 1

    print(f"all {sum(1 for a in ASSETS.rglob('*') if a.is_file() and a.name != '.DS_Store')} assets are referenced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
