"""Validate that docstring doctest prompts are wrapped in fenced ```pycon blocks.

Griffe's Google docstring parser only auto-highlights bare ``>>>`` lines under an
exact ``Examples:`` section title, and silently mis-renders them (nested
blockquotes) otherwise. An explicit ```pycon fence sidesteps that pitfall
entirely and is what `pycon_copy.js` looks for to strip prompts on copy.

Checks, per docstring:

- every ``>>> `` doctest-prompt line sits inside a ```pycon fenced block;
- that fenced block has a blank line immediately before its closing ``` ```` ``` ````.

Usage::

    python .github/scripts/check_doctest_fences.py [FILES...]

With no arguments, scans every ``src/**/*.py`` file. Exits non-zero and prints
one violation per offending line if any file fails.

"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^\s*```\s*(\w*)\s*$")
DOCTEST_RE = re.compile(r"^\s*>>>\s")


def _docstring_constants(tree: ast.Module) -> list[ast.Constant]:
    """Return the string-literal AST node for every module/class/function docstring."""
    constants = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants.append(value)
    return constants


def _check_docstring(lines: list[str], start_lineno: int) -> list[tuple[int, str]]:
    """Return (line_number, message) violations for one docstring's lines.

    Examples:
        ```pycon
        >>> _check_docstring(["```pycon", "    >>> 1 + 1", "    2", "", "```"], 10)
        []
        >>> _check_docstring(["    >>> 1 + 1"], 20)
        [(20, "'>>>' doctest line is not inside a ```pycon fence")]
        >>> _check_docstring(["```python", "    >>> 1 + 1", "```"], 30)
        [(31, "'>>>' doctest line is inside ```python, expected ```pycon")]
        >>> _check_docstring(["```pycon", "    >>> 1 + 1", "```"], 40)
        [(42, 'closing ``` ``` needs a blank line above it')]

        ```

    """
    violations: list[tuple[int, str]] = []
    in_fence = False
    fence_lang = ""
    doctest_in_fence = False

    for offset, line in enumerate(lines):
        lineno = start_lineno + offset
        fence_match = FENCE_RE.match(line)
        if fence_match:
            if not in_fence:
                in_fence, fence_lang = True, fence_match.group(1)
            else:
                if doctest_in_fence and (offset == 0 or lines[offset - 1].strip() != ""):
                    violations.append((lineno, "closing ``` ``` needs a blank line above it"))
                in_fence, fence_lang, doctest_in_fence = False, "", False
            continue

        if DOCTEST_RE.match(line):
            if not in_fence:
                violations.append((lineno, "'>>>' doctest line is not inside a ```pycon fence"))
            elif fence_lang != "pycon":
                got = f"```{fence_lang}" if fence_lang else "```"
                violations.append((lineno, f"'>>>' doctest line is inside {got}, expected ```pycon"))
            else:
                doctest_in_fence = True

    return violations


def check_file(path: Path) -> list[str]:
    """Return formatted 'path:line: message' violations for one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    messages = []
    for constant in _docstring_constants(tree):
        docstring_lines = constant.value.splitlines()
        # constant.lineno is the line of the opening quote; docstring text starts there.
        for lineno, message in _check_docstring(docstring_lines, constant.lineno):
            messages.append(f"{path}:{lineno}: {message}")
    return messages


def main(argv: list[str]) -> int:
    """Check the given files, or every ``src/**/*.py`` file if none are given."""
    paths = [Path(p) for p in argv] if argv else sorted(Path("src").rglob("*.py"))
    violations = [msg for path in paths for msg in check_file(path)]
    for msg in violations:
        print(msg)
    if violations:
        print(f"\n{len(violations)} doctest fence violation(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
