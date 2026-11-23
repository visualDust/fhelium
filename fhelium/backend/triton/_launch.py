"""Prepare launchers for FHElium-generated Triton kernels."""

from __future__ import annotations

from typing import Any

from triton.runtime._async_compile import FutureKernel


class PreparedKernel:
    """Bind one compiled kernel while keeping launch operands and stream live.

    Generated kernels omit pointer-alignment specialization unless their data
    interface guarantees it. Layout and dtype requirements are checked by the
    owning operation preparation, rather than rediscovered by Triton on hits.

    ``warm`` compiles and loads the kernel from a launch signature without
    launching it: pointer operands are given as ``torch.dtype`` placeholders
    and scalar/constexpr operands as their values. The binding established by
    ``warm`` is reused by the first ``__call__`` with the same signature. Under
    Triton's asynchronous compile mode ``warm`` only submits the compilation;
    the first ``__call__`` binds the finished kernel from Triton's cache.

    ``source`` is the ``(compile, text)`` pair that produced ``function``; it
    lets a persisted launcher compile the same function again.
    """

    def __init__(
        self,
        function: Any,
        grid: tuple[int, ...],
        *,
        num_warps: int = 4,
        source: tuple[Any, str] | None = None,
        **compile_options,
    ):
        self.function = function
        self.source = source
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.num_warps = num_warps
        self.compile_options = compile_options
        self._prepared = None

    def __reduce__(self):
        if self.source is None:
            raise TypeError("PreparedKernel has no recorded source")
        compile_function, text = self.source
        return (
            _restore,
            (
                compile_function,
                text,
                self.grid,
                self.num_warps,
                self.compile_options,
            ),
        )

    def _options(self):
        from triton import knobs

        return (
            getattr(self.function, "debug", None),
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
        )

    def _bind(self, args, *, warmup: bool):
        from triton.runtime.jit import MockTensor

        if warmup:
            args = tuple(map(MockTensor.wrap_dtype, args))
        binary = self.function.run(
            *args,
            grid=self.grid,
            warmup=True,
            num_warps=self.num_warps,
            **self.compile_options,
        )
        if isinstance(binary, FutureKernel):
            return binary
        if binary is None:
            # The Triton interpreter has no compiled binary; run it directly.
            def launcher(*args, stream=None):
                return self.function.run(
                    *args,
                    grid=self.grid,
                    warmup=False,
                    num_warps=self.num_warps,
                    **self.compile_options,
                )

        else:
            launcher = binary[self.grid]
        self._prepared = (self._options(), launcher)
        return binary

    def warm(self, *signature) -> bool:
        """Compile and load this kernel for ``signature``; report a new compilation."""
        from triton import knobs

        if knobs.runtime.interpret:
            return False
        from triton.runtime import driver

        device = driver.active.get_current_device()
        kernel_cache = self.function.device_caches[device][0]
        before = len(kernel_cache)
        self._bind(signature, warmup=True)
        return len(kernel_cache) > before

    def __call__(self, *args, stream):
        from triton import knobs

        cached = self._prepared
        if cached is None or cached[0] != self._options():
            self._bind(args, warmup=False)
            launcher = self._prepared[1]
        else:
            launcher = cached[1]
            for hook in self.function.pre_run_hooks:
                hook(
                    *args,
                    num_warps=self.num_warps,
                    debug=getattr(self.function, "debug", None)
                    or knobs.runtime.debug,
                    instrumentation_mode=knobs.compilation.instrumentation_mode,
                    **self.compile_options,
                )
        return launcher(*args, stream=stream)


def _restore(compile_function, text, grid, num_warps, compile_options):
    return PreparedKernel(
        compile_function(text),
        grid,
        num_warps=num_warps,
        source=(compile_function, text),
        **compile_options,
    )
