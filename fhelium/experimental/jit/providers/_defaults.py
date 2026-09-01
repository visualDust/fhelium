"""Compose the current Eager-backed adapter with neutral JIT components.

The Eager-backed entry is one current adapter, not a generic continuation of
Compile.
"""

from ._eager import eager_backend_provider
from ._rns_ntt_triton import triton_backend_provider
from ._registry import BackendProvider


def default_backend_providers() -> tuple[BackendProvider, ...]:
    """Construct the current Eager-backed CPU/CUDA and Triton provider set."""

    return (
        eager_backend_provider("cpu"),
        eager_backend_provider("cuda"),
        triton_backend_provider(),
    )


__all__ = ["default_backend_providers"]
