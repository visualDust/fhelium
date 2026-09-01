"""Native Torch operator loading and runtime ABI validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from ._abi import (
    compatibility_mismatches,
    manifest_native_backends,
    manifest_path,
    python_abi_identity,
    read_manifest,
    runtime_identity,
)


@dataclass(frozen=True)
class NativeStatus:
    """Native-extension availability and diagnostic paths for this runtime.

    ``available`` is true only after binary discovery, ABI-manifest validation,
    PyTorch loading, and FakeTensor registration succeed. ``reason`` and
    ``details`` explain the recorded result without triggering a new load.
    """

    available: bool
    reason: str
    details: tuple[str, ...]
    binary_path: Path
    manifest_path: Path
    backends: tuple[str, ...]


class NativeExtensionError(ImportError):
    """Raised when native operators are unavailable or ABI-incompatible."""


_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _NATIVE_DIR.parents[1]
_TORCHOPS_DIR = _NATIVE_DIR / "torchops"
_PYTHON_ABI = python_abi_identity()
_SOABI = _PYTHON_ABI["soabi"]
_EXTENSION_SUFFIX = _PYTHON_ABI["ext_suffix"]
_OPS_PATH = _TORCHOPS_DIR / f"_ops{_EXTENSION_SUFFIX}"
_MANIFEST_PATH = manifest_path(_TORCHOPS_DIR, _SOABI)
_IS_SOURCE_CHECKOUT = all(
    path.exists()
    for path in (
        _SOURCE_ROOT / "pyproject.toml",
        _SOURCE_ROOT / "CMakeLists.txt",
        _SOURCE_ROOT / "csrc",
    )
)


def _project_version() -> str | None:
    try:
        return version("fhelium")
    except PackageNotFoundError:
        return None


def _diagnostic(status: NativeStatus) -> str:
    lines = [
        f"FHElium native extension is unavailable: {status.reason}",
        f"Expected binary: {status.binary_path}",
        f"Expected ABI manifest: {status.manifest_path}",
    ]
    lines.extend(f"  - {detail}" for detail in status.details)
    if _IS_SOURCE_CHECKOUT:
        lines.extend(
            [
                "Rebuild the editable native extension:",
                "  python -m pip install --editable . --no-build-isolation "
                "--no-cache-dir --verbose",
            ]
        )
    else:
        lines.extend(
            [
                "Remove any cached FHElium wheel and rebuild against the "
                "current Torch environment:",
                "  python -m pip cache remove fhelium",
                "  python -m pip install --no-build-isolation --no-cache-dir "
                "--verbose fhelium",
            ]
        )
    return "\n".join(lines)


def _unavailable(reason: str, *details: str) -> NativeStatus:
    status = NativeStatus(
        available=False,
        reason=reason,
        details=tuple(details),
        binary_path=_OPS_PATH,
        manifest_path=_MANIFEST_PATH,
        backends=(),
    )
    if not _IS_SOURCE_CHECKOUT:
        raise NativeExtensionError(_diagnostic(status))
    return status


def _initialize_native() -> NativeStatus:
    if not _SOABI or not _EXTENSION_SUFFIX:
        return _unavailable(
            "the current Python interpreter has no SOABI/EXT_SUFFIX"
        )
    if not _OPS_PATH.is_file():
        return _unavailable("the current-ABI native operator file is missing")
    if not _MANIFEST_PATH.is_file():
        return _unavailable("the native ABI manifest is missing")

    try:
        manifest = read_manifest(_MANIFEST_PATH)
    except ValueError as error:
        return _unavailable("the native ABI manifest is invalid", str(error))

    mismatches = compatibility_mismatches(
        manifest,
        runtime=runtime_identity(torch),
        project_version=_project_version(),
    )
    if mismatches:
        return _unavailable(
            "the native extension was built for a different Python/Torch ABI",
            *mismatches,
        )

    try:
        backends = manifest_native_backends(manifest)
    except ValueError as error:
        return _unavailable(
            "the native ABI manifest has invalid backend metadata", str(error)
        )

    try:
        torch.ops.load_library(str(_OPS_PATH))
    except Exception as error:
        return _unavailable(
            "PyTorch could not load the native operator file",
            f"{type(error).__name__}: {error}",
        )

    return NativeStatus(
        available=True,
        reason="native Python/Torch ABI matches the runtime",
        details=(),
        binary_path=_OPS_PATH,
        manifest_path=_MANIFEST_PATH,
        backends=backends,
    )


_NATIVE_STATUS = _initialize_native()


def native_status() -> NativeStatus:
    """Return the immutable native-extension status recorded during import."""

    return _NATIVE_STATUS


def native_available() -> bool:
    """Return whether native operators loaded successfully for this runtime."""

    return _NATIVE_STATUS.available


def require_native() -> None:
    """Raise ``NativeExtensionError`` unless native operators are available."""

    if not _NATIVE_STATUS.available:
        raise NativeExtensionError(_diagnostic(_NATIVE_STATUS))


def native_backend_available(backend: str) -> bool:
    """Return whether the loaded extension implements ``backend``."""

    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    return _NATIVE_STATUS.available and backend in _NATIVE_STATUS.backends


def require_native_backend(backend: str) -> None:
    """Raise unless the loaded extension implements ``backend``."""

    require_native()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    if backend not in _NATIVE_STATUS.backends:
        enabled = ", ".join(_NATIVE_STATUS.backends)
        raise NativeExtensionError(
            f"FHElium was built for native backends [{enabled}], but this "
            f"operation requires {backend}. Rebuild with "
            f"-DFHELIUM_NATIVE_BACKENDS={backend.upper()} or "
            "-DFHELIUM_NATIVE_BACKENDS=CPU+CUDA."
        )


if _NATIVE_STATUS.available:
    try:
        from fhelium.native.wrapper import (
            fake_op,  # supports FakeTensor/custom-op registration # noqa: F401
        )
    except Exception as error:
        _NATIVE_STATUS = _unavailable(
            "native operators loaded, but FakeTensor registration failed",
            f"{type(error).__name__}: {error}",
        )


__all__ = [
    "NativeExtensionError",
    "NativeStatus",
    "native_available",
    "native_backend_available",
    "native_status",
    "require_native",
    "require_native_backend",
]
