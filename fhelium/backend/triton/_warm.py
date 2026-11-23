"""Compile the Triton kernels of a linked executable before its first call."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cache
from typing import Any

import torch


@cache
def _executor() -> ThreadPoolExecutor:
    # Triton releases the GIL inside its compiler passes, so threads overlap compilations.
    return ThreadPoolExecutor(thread_name_prefix="fhelium-triton-compile")


def warm_executable(
    executable: Any, device: torch.device | None = None
) -> dict:
    """Compile every Triton kernel of ``executable`` without launching it.

    Triton dispatch preparation reads physical layouts or reconstructs polynomial
    extents from component, batch, prime-row and ring-dimension state. This
    function drives the resulting kernel objects through Triton's compile path
    with dtype placeholders for pointer operands and the planned scalar and
    constexpr values. Dispatches with unresolved shape, dtype or device are
    skipped; assumed contiguous layouts are checked and re-bound at execution.

    Returns ``compiled`` (new Triton compilations, including disk-cache
    loads), ``reused`` (kernels already present in the process cache),
    ``skipped`` (dispatches without a prepared layout), ``dispatches`` (Triton
    dispatches visited) and ``seconds``. The kernels of one executable compile
    concurrently in a shared thread pool, and the call returns when all of
    them have finished; when the caller has already activated Triton's
    asynchronous compile mode, the compilations join that mode instead. Each
    kernel is loaded into the current CUDA context at its first launch.
    """
    from ._pointwise import PreparedFusion
    from ._tensor import _PreparedTensorFusion

    from triton.runtime import _async_compile

    counts = {"compiled": 0, "reused": 0, "skipped": 0, "dispatches": 0}
    start = time.perf_counter()
    seen: set[int] = set()
    pending = [executable]
    concurrent = (
        _async_compile.AsyncCompileMode(_executor())
        if _async_compile.active_mode.get() is None
        else nullcontext()
    )
    with (
        torch.cuda.device(device) if device is not None else nullcontext(),
        concurrent,
    ):
        while pending:
            for dispatch in pending.pop().dispatch_table.operations.values():
                implementation = dispatch.implementation
                called = getattr(implementation, "executable", None)
                if called is not None and id(called) not in seen:
                    # Shared function bodies are linked once for all their calls.
                    seen.add(id(called))
                    pending.append(called)
                    continue
                if not isinstance(
                    implementation, (PreparedFusion, _PreparedTensorFusion)
                ):
                    continue
                counts["dispatches"] += 1
                if id(implementation) in seen:
                    continue
                seen.add(id(implementation))
                result = implementation.warm()
                if result is None:
                    counts["skipped"] += 1
                    continue
                compiled, reused = result
                counts["compiled"] += compiled
                counts["reused"] += reused
    counts["seconds"] = time.perf_counter() - start
    return counts
