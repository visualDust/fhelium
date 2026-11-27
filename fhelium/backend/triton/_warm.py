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
    executable: Any,
    device: torch.device | None = None,
    *,
    runner: bool = True,
) -> dict:
    """Compile every Triton kernel of ``executable``'s lowered schedules and build their Runners.

    Lowering records each kernel launch with its signature: dtype placeholders for pointer operands and the planned scalar and constexpr values. This function drives those kernels through Triton's compile path, including the schedules of called functions. An executable whose inputs have no declared layouts is lowered at its first call per layout; until then it has no recorded launches.

    Returns ``compiled`` (new Triton compilations, including disk-cache loads), ``reused`` (kernels already present in the process cache), ``launches`` (recorded launches visited) and ``seconds``. The kernels compile concurrently in a shared thread pool, and the call returns when all of them have finished; when the caller has already activated Triton's asynchronous compile mode, the compilations join that mode instead. Without an enclosing asynchronous mode, the schedules' Runners are then built from the compiled kernel images, which they load at construction. A schedule's Runner builds the Runners of the functions it calls; ``runner=False`` builds none, for a caller that executes ``executable`` through other Runners such as Graph Runners. An executable restored from an artifact compiles nothing; its Runners load the stored kernel images.
    """
    from triton.runtime import _async_compile

    from fhelium.backend._artifact import RestoredProgramExecutable

    counts: dict[str, float] = {"compiled": 0, "reused": 0, "launches": 0}
    start = time.perf_counter()
    if isinstance(executable, RestoredProgramExecutable):
        # An artifact's Runners load its stored kernel images.
        if runner:
            for lowered in executable._lowered.values():
                lowered.function.runner()
        counts["seconds"] = time.perf_counter() - start
        return counts
    seen: set[object] = set()
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
            current = pending.pop()
            for dispatch in current.dispatch_table.operations.values():
                called = getattr(dispatch.implementation, "executable", None)
                if called is not None and id(called) not in seen:
                    # Shared function bodies are linked once for all their calls.
                    seen.add(id(called))
                    pending.append(called)
            for lowered in current._lowered.values():
                for kernel, signature in lowered.launches:
                    counts["launches"] += 1
                    key = (id(kernel), signature)
                    if key in seen:
                        continue
                    seen.add(key)
                    if kernel.warm(*signature):
                        counts["compiled"] += 1
                    else:
                        counts["reused"] += 1
    if isinstance(concurrent, _async_compile.AsyncCompileMode):
        if runner:
            # Building a schedule's Runner builds the Runners it calls.
            for lowered in executable._lowered.values():
                lowered.function.runner()
    counts["seconds"] = time.perf_counter() - start
    return counts
