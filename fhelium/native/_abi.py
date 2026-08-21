"""Native build/runtime ABI metadata and validation."""

from __future__ import annotations

import json
import re
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1

SUPPORTED_NATIVE_BACKENDS = frozenset({"cpu", "cuda"})


def manifest_native_backends(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the validated execution backends recorded by a manifest."""

    value = manifest.get("native_backends")
    if not isinstance(value, list) or not value:
        raise ValueError("native_backends must be a non-empty JSON array")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("native_backends entries must be strings")
    backends = tuple(value)
    if len(set(backends)) != len(backends):
        raise ValueError("native_backends entries must be unique")
    unknown = set(backends) - SUPPORTED_NATIVE_BACKENDS
    if unknown:
        raise ValueError(
            "native_backends contains unsupported entries: "
            + ", ".join(sorted(unknown))
        )
    return backends


def manifest_path(torchops_dir: Path, soabi: str) -> Path:
    return torchops_dir / f"_build_manifest.{soabi}.json"


def torch_cxx11_abi(torch_module: Any) -> bool:
    query = getattr(torch_module, "compiled_with_cxx11_abi", None)
    if callable(query):
        return bool(query())
    return bool(torch_module._C._GLIBCXX_USE_CXX11_ABI)


def python_abi_identity() -> dict[str, str]:
    """Return the CPython fields that identify a native extension ABI."""

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    soabi = sysconfig.get_config_var("SOABI") or ""
    if not soabi and sys.platform == "win32":
        match = re.fullmatch(r"\.(cp[0-9]+-win_amd64)\.pyd", ext_suffix)
        if match is not None:
            soabi = match.group(1)
    return {
        "implementation": sys.implementation.name,
        "soabi": soabi,
        "ext_suffix": ext_suffix,
    }


def runtime_identity(torch_module: Any) -> dict[str, Any]:
    """Return only the Python/Torch fields that govern binary compatibility."""

    return {
        "python": python_abi_identity(),
        "torch": {
            "version": str(torch_module.__version__),
            "cuda_version": torch_module.version.cuda,
            "cxx11_abi": torch_cxx11_abi(torch_module),
        },
    }


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read native ABI manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Native ABI manifest is not a JSON object: {path}")
    return value


def compatibility_mismatches(
    manifest: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    project_version: str | None,
) -> list[str]:
    """Compare the small set of identities not encoded by wheel tags."""

    mismatches: list[str] = []

    def compare(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(
                f"{label}: built={expected!r}, runtime={actual!r}"
            )

    compare(
        "manifest schema",
        manifest.get("manifest_schema_version"),
        MANIFEST_SCHEMA_VERSION,
    )
    try:
        manifest_native_backends(manifest)
    except ValueError as error:
        mismatches.append(f"Native backend metadata is invalid: {error}")
    if project_version is not None:
        compare(
            "FHElium version", manifest.get("project_version"), project_version
        )

    manifest_python = manifest.get("python")
    manifest_torch = manifest.get("torch")
    runtime_python = runtime.get("python")
    runtime_torch = runtime.get("torch")
    if not isinstance(manifest_python, Mapping):
        mismatches.append(
            "Python ABI identity is missing from the native manifest"
        )
    elif not isinstance(runtime_python, Mapping):
        mismatches.append("Python ABI identity is unavailable at runtime")
    else:
        for field in ("implementation", "soabi", "ext_suffix"):
            compare(
                f"Python {field}",
                manifest_python.get(field),
                runtime_python.get(field),
            )

    if not isinstance(manifest_torch, Mapping):
        mismatches.append(
            "Torch ABI identity is missing from the native manifest"
        )
    elif not isinstance(runtime_torch, Mapping):
        mismatches.append("Torch ABI identity is unavailable at runtime")
    else:
        for field in ("version", "cuda_version", "cxx11_abi"):
            compare(
                f"Torch {field}",
                manifest_torch.get(field),
                runtime_torch.get(field),
            )

    return mismatches
