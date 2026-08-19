"""Native-extension availability and ABI diagnostics."""

from fhelium.native.runtime import (
    NativeExtensionError,
    NativeStatus,
    native_backend_available,
    native_available,
    native_status,
    require_native,
    require_native_backend,
)

__all__ = [
    "NativeExtensionError",
    "NativeStatus",
    "native_backend_available",
    "native_available",
    "native_status",
    "require_native",
    "require_native_backend",
]
