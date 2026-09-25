"""Prepare launchers for FHElium-generated Triton kernels."""

from __future__ import annotations

from typing import Any


class PreparedKernel:
    """Bind one compiled kernel while keeping launch operands and stream live.

    Generated kernels omit pointer-alignment specialization unless their data
    interface guarantees it. Layout and dtype requirements are checked by the
    owning operation preparation, rather than rediscovered by Triton on hits.
    """

    def __init__(
        self,
        function: Any,
        grid: tuple[int, ...],
        *,
        num_warps: int = 4,
        **compile_options,
    ):
        self.function = function
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.num_warps = num_warps
        self.compile_options = compile_options
        self._prepared = None

    def __call__(self, *args, stream):
        from triton import knobs

        options = (
            self.function.debug,
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
        )
        cached = self._prepared
        if cached is None or cached[0] != options:
            binary = self.function.run(
                *args,
                grid=self.grid,
                warmup=True,
                num_warps=self.num_warps,
                **self.compile_options,
            )
            launcher = binary[self.grid]
            self._prepared = (options, launcher)
        else:
            launcher = cached[1]
            for hook in self.function.pre_run_hooks:
                hook(
                    *args,
                    num_warps=self.num_warps,
                    debug=self.function.debug or knobs.runtime.debug,
                    instrumentation_mode=knobs.compilation.instrumentation_mode,
                    **self.compile_options,
                )
        return launcher(*args, stream=stream)
