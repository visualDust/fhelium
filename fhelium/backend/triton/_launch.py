"""Prepare launchers for FHElium-generated Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from triton.runtime._async_compile import FutureKernel


@dataclass(frozen=True)
class KernelRecord:
    """Loadable image and launch facts of one compiled Triton kernel.

    ``image`` is the device binary (``cubin`` on CUDA, ``hsaco`` on ROCm) for ``arch``; ``ptx`` is the CUDA fallback source. ``parameter_types`` lists the kernel's runtime parameters in Triton type strings, and ``runtime_positions`` their positions among the call arguments, which also include ``constexpr`` values. Launches append the global and profile scratch pointers after the runtime parameters.
    """

    name: str
    arch: str
    image_kind: str
    image: bytes
    ptx: str
    num_warps: int
    warp_size: int
    num_ctas: int
    shared: int
    parameter_types: tuple[str, ...]
    runtime_positions: tuple[int, ...]
    global_scratch_size: int
    global_scratch_align: int
    profile_scratch_size: int
    profile_scratch_align: int
    triton_version: str

    def arguments(self) -> tuple[object, ...]:
        """Return the arguments of ``Runner.kernel`` for this record."""
        return (
            self.name,
            torch.frombuffer(bytearray(self.image), dtype=torch.uint8),
            self.image_kind,
            self.num_warps,
            self.warp_size,
            self.num_ctas,
            self.shared,
            list(self.parameter_types),
            [
                self.global_scratch_size,
                self.global_scratch_align,
                self.profile_scratch_size,
                self.profile_scratch_align,
            ],
            self.triton_version,
        )


def kernel_record(binary: Any) -> KernelRecord:
    """Extract the record of a Triton ``CompiledKernel``."""
    import triton

    metadata = binary.metadata
    if metadata.launch_cooperative_grid or metadata.launch_pdl:
        raise ValueError(
            f"Kernel {metadata.name} requires a cooperative or programmatic-dependent launch"
        )
    types = tuple(binary.src.signature.values())
    positions = tuple(
        index for index, kind in enumerate(types) if kind != "constexpr"
    )
    kind = "hsaco" if metadata.backend_name == "hip" else "cubin"
    return KernelRecord(
        name=metadata.name,
        arch=str(metadata.target.arch),
        image_kind=kind,
        image=binary.asm[kind],
        ptx=binary.asm.get("ptx", ""),
        num_warps=metadata.num_warps,
        warp_size=metadata.target.warp_size,
        num_ctas=metadata.num_ctas,
        shared=metadata.shared,
        parameter_types=tuple(types[i] for i in positions),
        runtime_positions=positions,
        global_scratch_size=metadata.global_scratch_size,
        global_scratch_align=metadata.global_scratch_align,
        profile_scratch_size=metadata.profile_scratch_size,
        profile_scratch_align=metadata.profile_scratch_align,
        triton_version=triton.__version__,
    )


class PreparedKernel:
    """Bind one compiled kernel while keeping launch operands and stream live.

    Generated kernels omit pointer-alignment specialization unless their data
    interface guarantees it. Layout and dtype requirements are checked by the
    owning operation preparation, rather than rediscovered by Triton on hits.

    Lowering records each call as a launch with its signature: pointer
    operands as ``torch.dtype`` placeholders and scalar and constexpr operands
    as their values. ``warm`` compiles the kernel for a signature without
    launching it; under Triton's asynchronous compile mode it only submits the
    compilation. ``record`` returns the :class:`KernelRecord` of a signature,
    compiling it or reading it from Triton's cache.

    ``source`` is the ``(compile, text)`` pair that produced ``function``; it
    lets a persisted launcher compile the same function again.

    Floating-point multiply-add contraction is enabled unless
    ``enable_fp_fusion`` is supplied, which fixes the floating-point results of
    generated kernels across Triton releases with different defaults.
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
        self.compile_options = {"enable_fp_fusion": True, **compile_options}
        self._records: dict[tuple[object, ...], KernelRecord] = {}

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

    def _binary(self, signature: tuple[object, ...], *, wait: bool) -> Any:
        """Compile for ``signature``; return None while a submitted compilation is pending."""
        from triton.runtime.jit import MockTensor

        binary = self.function.run(
            *map(MockTensor.wrap_dtype, signature),
            grid=self.grid,
            warmup=True,
            num_warps=self.num_warps,
            **self.compile_options,
        )
        if isinstance(binary, FutureKernel):
            if not wait:
                return None
            binary = binary.result()
        return binary

    def warm(self, *signature) -> bool:
        """Compile this kernel for ``signature``; report a new compilation."""
        from triton import knobs

        if knobs.runtime.interpret:
            return False
        from triton.runtime import driver

        # Triton annotates driver.active as DriverBase; the active GPU driver
        # defines get_current_device.
        device = driver.active.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]
        kernel_cache = self.function.device_caches[device][0]
        before = len(kernel_cache)
        self._binary(signature, wait=False)
        return len(kernel_cache) > before

    def record(self, *signature) -> KernelRecord:
        """Return the loadable record of this kernel compiled for ``signature``."""
        record = self._records.get(signature)
        if record is None:
            record = self._records[signature] = self._record(signature)
        return record

    def _record(self, signature: tuple[object, ...]) -> KernelRecord:
        record = kernel_record(self._binary(signature, wait=True))
        # The Runner passes Tensor buffers as pointers and other values as
        # scalars, so the compiled parameters must follow the recorded launch.
        launched = tuple(
            isinstance(signature[i], torch.dtype)
            for i in record.runtime_positions
        )
        if launched != tuple(t.startswith("*") for t in record.parameter_types):
            raise ValueError(
                f"Kernel {record.name} was compiled for another launch signature"
            )
        return record

    def __call__(self, *args, stream):
        from fhelium.backend.lowering import active_builder

        builder = active_builder()
        if builder is None:
            raise RuntimeError("Prepared kernels launch from lowered schedules")
        # Lowering records the launch; the Runner supplies its stream.
        builder.launch(
            self,
            args,
            device=next(a.device for a in args if hasattr(a, "device")),
        )
        return None


def _restore(compile_function, text, grid, num_warps, compile_options):
    return PreparedKernel(
        compile_function(text),
        grid,
        num_warps=num_warps,
        source=(compile_function, text),
        **compile_options,
    )
