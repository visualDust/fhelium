"""Resolve the FHElium package version from distribution metadata."""

import tomllib
from importlib.metadata import version
from pathlib import Path


def _resolve_version() -> str:
    source_pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if source_pyproject.is_file():
        with source_pyproject.open("rb") as stream:
            project_version = tomllib.load(stream)["project"]["version"]
        if not isinstance(project_version, str) or not project_version:
            raise ValueError(
                "pyproject.toml contains no FHElium project version"
            )
        return project_version
    # Installed packages must carry dist-info; a source checkout must carry
    # its adjacent pyproject.toml.
    return version("fhelium")


__version__ = _resolve_version()


__all__ = ["__version__"]
