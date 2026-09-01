#!/usr/bin/env python3

"""Prepare clangd's compilation database after an editable native build.

Scikit-build-core writes ``compile_commands.json`` under an ABI-specific build
folder, and an isolated uv build records Torch headers from a temporary build
environment. This script selects the database for the active CPython
interpreter, replaces those temporary Torch paths with the active environment's
installed Torch paths, and atomically writes the ignored
``build/compile_commands.json`` consumed by ``.clangd``.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch

_POSIX_TORCH_ROOT = re.compile(
    r"/[^\s\"]*?/site-packages/torch(?=/include(?:[/\s\"]|$))"
)
_WINDOWS_TORCH_ROOT = re.compile(
    r"[A-Za-z]:[^\"]*?[\\/]site-packages[\\/]torch"
    r"(?=[\\/]include(?:[\\/\s\"]|$))",
    re.IGNORECASE,
)


def _rewrite_torch_roots(text: str, torch_root: Path) -> tuple[str, int]:
    """Replace serialized Torch package roots and return the replacement count."""

    replacement = str(torch_root)
    text, posix_count = _POSIX_TORCH_ROOT.subn(lambda _match: replacement, text)
    text, windows_count = _WINDOWS_TORCH_ROOT.subn(
        lambda _match: replacement, text
    )
    return text, posix_count + windows_count


def main() -> None:
    """Copy and repair the compile database for the active CPython environment."""

    interpreter = f"cp{sys.version_info.major}{sys.version_info.minor}"
    candidates = tuple(
        Path("build").glob(
            f"{interpreter}-{interpreter}-*/editable/compile_commands.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "native build did not generate compile_commands.json for "
            f"{interpreter}"
        )
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    torch_root = Path(torch.__file__).resolve().parent
    torch_include = torch_root / "include"
    if not torch_include.is_dir():
        raise FileNotFoundError(
            f"active Torch include directory does not exist: {torch_include}"
        )

    database = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(database, list) or not database:
        raise ValueError(f"compile database is empty or malformed: {source}")
    rewritten_count = 0
    for entry in database:
        if not isinstance(entry, dict):
            raise ValueError(
                f"compile database entry is not an object: {entry!r}"
            )
        command = entry.get("command")
        if isinstance(command, str):
            entry["command"], count = _rewrite_torch_roots(command, torch_root)
            rewritten_count += count
        arguments = entry.get("arguments")
        if isinstance(arguments, list):
            for index, argument in enumerate(arguments):
                if not isinstance(argument, str):
                    raise ValueError(
                        "compile database arguments must be strings"
                    )
                arguments[index], count = _rewrite_torch_roots(
                    argument, torch_root
                )
                rewritten_count += count

    if rewritten_count == 0:
        raise ValueError(
            f"compile database contains no Torch include path to normalize: {source}"
        )
    serialized = json.dumps(database, indent=2) + "\n"
    remaining_roots = {
        *(_POSIX_TORCH_ROOT.findall(serialized)),
        *(_WINDOWS_TORCH_ROOT.findall(serialized)),
    }
    if remaining_roots != {str(torch_root)}:
        raise ValueError(
            "compile database retains Torch roots other than the active "
            f"environment: {sorted(remaining_roots)}"
        )

    destination = Path("build/compile_commands.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    shutil.copymode(source, temporary)
    temporary.replace(destination)
    print(f"Updated {destination} from {source}")
    print(f"Torch headers: {torch_include}")


if __name__ == "__main__":
    main()
