"""Pin minimum dependency versions in pyproject.toml for oldest-compatible CI testing.

Replaces all '>=' specifiers with '==' in [project.dependencies] and [dependency-groups]
sections of pyproject.toml, preserving formatting and comments via tomlkit.

Usage::

    >>> import os
    >>> os.path.exists("pyproject.toml")  # doctest: +SKIP
    True

"""

import tomlkit


def _replace_min_versions(proj_file: str = "pyproject.toml") -> None:
    """Replace all '>=' with '==' in dependency sections of pyproject.toml.

    Handles both [project.dependencies] and all [dependency-groups] groups.

    Args:
        proj_file: Path to the pyproject.toml file.

    """
    with open(proj_file, encoding="utf-8") as f:
        content = f.read()
    doc = tomlkit.parse(content)

    # Pin [project.dependencies]
    deps = doc.get("project", {}).get("dependencies")
    if deps:
        for i, req in enumerate(deps):
            deps[i] = req.replace(">=", "==")

    # Pin [dependency-groups]
    # Entries can be strings (requirements) or dicts (include-group references per PEP 735)
    groups = doc.get("dependency-groups", {})
    for group_name in groups:
        for i, req in enumerate(groups[group_name]):
            if isinstance(req, str):
                groups[group_name][i] = req.replace(">=", "==")

    with open(proj_file, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


if __name__ == "__main__":
    _replace_min_versions()
