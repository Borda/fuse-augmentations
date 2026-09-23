"""Pin minimum dependency versions in pyproject.toml for oldest-compatible CI testing.

Replaces all '>=' specifiers with '==' in [project.dependencies], [project.optional-dependencies],
and [dependency-groups] sections of pyproject.toml, preserving formatting and comments via tomlkit.

The CLI uses ``fire``, which ships in the ``cli`` extra::

    pip install "fuse-augmentations[cli]"

Usage::

    python .github/scripts/min_deps.py --proj-file pyproject.toml

    >>> changed = _replace_min_versions("pyproject.toml")
    >>> print(changed)
    ['numpy==1.26', 'pytest==8.0']

"""

import tomlkit


def _replace_min_versions(proj_file: str = "pyproject.toml") -> list[str]:
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
    changed: list[str] = []

    # Pin [project.dependencies]
    project = doc.get("project", {})
    deps = project.get("dependencies")
    if deps:
        for i, req in enumerate(deps):
            if ">=" in req:
                deps[i] = req.replace(">=", "==")
                changed.append(deps[i])

    # Pin [project.optional-dependencies] (extras, e.g. "torch", "kornia", "all")
    optional_deps = project.get("optional-dependencies", {})
    for extra_name in optional_deps:
        for i, req in enumerate(optional_deps[extra_name]):
            if ">=" in req:
                optional_deps[extra_name][i] = req.replace(">=", "==")
                changed.append(optional_deps[extra_name][i])

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


def main(proj_file: str = "pyproject.toml") -> None:
    """Pin every '>=' dependency specifier in ``proj_file`` to '=='.

    Args:
        proj_file: Path to the pyproject.toml file.

    Examples:
        >>> main("pyproject.toml")  # doctest: +SKIP

    """
    changed = _replace_min_versions(proj_file)
    print(f"Pinned {len(changed)} requirements: {changed}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
