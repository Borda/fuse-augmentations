"""Pin minimum dependency versions in pyproject.toml for oldest-compatible CI testing.

Replaces all '>=' specifiers with '==' in [project.dependencies] and [dependency-groups]
sections of pyproject.toml, preserving formatting and comments via tomlkit.

Usage::

    >>> changed = _replace_min_versions("pyproject.toml")  # doctest: +SKIP
    >>> print(changed)  # doctest: +SKIP
    ['numpy==1.26', 'pytest==8.0']

"""

import tomlkit


def _replace_min_versions(proj_file: str = "pyproject.toml") -> list:
    """Replace all '>=' with '==' in dependency sections of pyproject.toml.

    Handles both [project.dependencies] and all [dependency-groups] groups.

    Args:
        proj_file: Path to the pyproject.toml file.

    Returns:
        List of pinned requirement strings that were changed.

    """
    with open(proj_file, encoding="utf-8") as f:
        content = f.read()
    doc = tomlkit.parse(content)
    changed = []

    # Pin [project.dependencies]
    deps = doc.get("project", {}).get("dependencies")
    if deps:
        for i, req in enumerate(deps):
            if ">=" in req:
                deps[i] = req.replace(">=", "==")
                changed.append(deps[i])

    # Pin [dependency-groups]
    # Entries can be strings (requirements) or dicts (include-group references per PEP 735)
    groups = doc.get("dependency-groups", {})
    for group_name in groups:
        for i, req in enumerate(groups[group_name]):
            if isinstance(req, str) and ">=" in req:
                groups[group_name][i] = req.replace(">=", "==")
                changed.append(groups[group_name][i])

    with open(proj_file, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))

    return changed


if __name__ == "__main__":
    _changed = _replace_min_versions()
    print(f"Pinned {len(_changed)} requirements: {_changed}")
