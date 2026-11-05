"""Compile generated NTT kernels with a scoped Triton pass policy."""

from __future__ import annotations

from types import FunctionType, SimpleNamespace
from typing import Any, cast

from triton._C.libtriton import passes
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler.compiler import compile as _standard_compile
from triton.compiler.compiler import make_backend
from triton.runtime.jit import JITFunction


def _bind_globals(function: Any, **bindings: Any) -> Any:
    """Give one compiler function private bindings without changing its code."""
    bound = FunctionType(
        function.__code__,
        {**function.__globals__, **bindings},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def _omit_coalesce(_pass_manager: Any) -> None:
    pass


# Triton 3.7.1's Coalesce pass recomputes a transitive dataflow slice for each
# tensor-pointer memory operation. Generated NTT joins can connect thousands
# of accesses through shared arithmetic and indices, making this analysis
# dominate compilation. Representative endpoint, rotation-join and matrix-join
# kernels produce identical LLVM IR, PTX and cubin when this pass is omitted.
# This does not assume every NTT memory access is coalesced.
#
# Bind a private pass namespace to the CUDA compiler function: only the
# Coalesce registration becomes a no-op; the other passes retain their order.
# Triton's global functions, pass namespace and inspection hook stay intact.
# This adapter depends on Triton's make_ttgir globals, compile backend factory
# and JIT create_binder interface. On Triton upgrades, recheck those interfaces,
# pass registration and generated-code equivalence before extending this policy.
_ntt_passes = SimpleNamespace(
    **{
        **vars(passes),
        "ttgpuir": SimpleNamespace(
            **{**vars(passes.ttgpuir), "add_coalesce": _omit_coalesce}
        ),
    }
)


class _NttCudaBackend(CUDABackend):
    make_ttgir = staticmethod(
        _bind_globals(CUDABackend.make_ttgir, passes=_ntt_passes)
    )

    def hash(self):
        # Keep policy-specific artifacts separate from ordinary Triton output.
        # Change the suffix when the compilation policy changes.
        return super().hash() + ":fhelium-ntt-no-coalesce-v1"


def _make_ntt_backend(target: Any) -> Any:
    backend = make_backend(target)
    # Preserve HIP and independently supplied compiler backends.
    if type(backend) is CUDABackend:
        return _NttCudaBackend(target)
    return backend


_compile_ntt = _bind_globals(_standard_compile, make_backend=_make_ntt_backend)


class NttJITFunction(JITFunction):
    """Bind generated NTT functions to the NTT-specific compiler and cache."""

    def create_binder(self):
        kernels, keys, target, backend, binder = super().create_binder()
        # Each device cache needs the same backend identity used by compilation,
        # including when Triton dispatches compilation asynchronously. Keep the
        # ordinary argument binder: parameter specialization is unchanged.
        if type(backend) is CUDABackend:
            backend = _NttCudaBackend(cast(GPUTarget, target))
        self.compile = _compile_ntt
        return kernels, keys, target, backend, binder
