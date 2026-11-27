"""Encode Runner schedules as FHElium artifact entries and read and write artifact containers.

An artifact is a container file whose manifest describes entries, kernels, constants and materials; ``libfhelium`` loads it as ``torch.classes.fhelium.Artifact`` (C++ ``fhelium::Artifact``), whose header ``fhelium/artifact.h`` specifies the format. :class:`ArtifactWriter` encodes linked schedules into entries: their buffer tables and builder calls become int64 tensors, kernels become cubin images for the current device and the requested targets plus PTX, link-time constants become stored tensors, and material and state buffers name the material symbols the caller assigns. The caller completes the manifest with identity, configuration, signature and material records and writes it with :func:`save_artifact`.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch

ARTIFACT_FORMAT = "fhelium-artifact"
ARTIFACT_SCHEMA_VERSION = 1


class ExportError(ValueError):
    """A schedule cannot be stored in an artifact."""


def _io() -> Any:
    import fhelium.native.wrapper  # noqa: F401  (loads libfhelium_torch)

    return torch.ops.fhelium_io


def native_identity() -> dict[str, str]:
    """Return the identity of the loaded native library and Triton that artifacts record and loaders compare."""
    import triton

    identity = json.loads(_io().native_identity())
    identity["triton_version"] = triton.__version__
    return identity


def save_container(
    path: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a container file readable by owner only."""
    _io().save_container(
        os.fspath(path), dict(tensors), dict(metadata), overwrite
    )


def load_container(
    path: str | os.PathLike[str], device: torch.device | str = "cpu"
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return the tensors on ``device`` and the metadata of a container file."""
    tensors, metadata = _io().load_container(
        os.fspath(path), torch.device(device)
    )
    return dict(tensors), dict(metadata)


def read_container_header(
    path: str | os.PathLike[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return the metadata and the tensor header (name to dtype, shape and offsets) of a container file without reading its tensors."""
    metadata, header = _io().read_container_header(os.fspath(path))
    return dict(metadata), json.loads(header)


def save_artifact(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    *,
    overwrite: bool = False,
) -> None:
    """Write an artifact; ``manifest`` receives the native identity unless it records one."""
    body = dict(manifest)
    body.setdefault("identity", native_identity())
    save_container(
        path,
        tensors,
        {
            "fhelium.format": ARTIFACT_FORMAT,
            "fhelium.schema_version": str(ARTIFACT_SCHEMA_VERSION),
            "fhelium.manifest": json.dumps(body, separators=(",", ":")),
        },
        overwrite=overwrite,
    )


def load_artifact(
    path: str | os.PathLike[str], device: torch.device | str
) -> Any:
    """Load an artifact for execution on ``device`` as ``torch.classes.fhelium.Artifact``."""
    _io()
    return torch.classes.fhelium.Artifact(os.fspath(path), torch.device(device))


Binding = Mapping[object, str] | Callable[[object], str | None]
"""Material symbol of a :class:`~fhelium.backend.runner.Material` or :class:`~fhelium.backend.runner.State` buffer binding."""


class ArtifactWriter:
    """Collect artifact entries with their kernels and constants.

    ``targets`` names further kernel architectures, such as ``"sm_90"``, whose cubins are compiled offline; every kernel also stores the cubin of the device it was compiled for and its PTX.
    """

    def __init__(self, targets: Iterable[str] = ()) -> None:
        self.targets = tuple(targets)
        self._entries: dict[str, dict[str, Any]] = {}
        self._kernels: list[dict[str, Any]] = []
        self._kernel_ids: dict[tuple[int, tuple[object, ...]], int] = {}
        self._constants: list[dict[str, Any]] = []
        self._constant_ids: dict[tuple[object, ...], int] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._keep: list[object] = []
        self._callees: dict[int, str] = {}
        self._python: list[str] = []

    def entry(self, name: str, schedule: Any, *, binding: Binding) -> str:
        """Encode ``schedule`` as entry ``name`` and return the name.

        ``schedule`` is a :class:`~fhelium.backend.runner.Schedule` or a :class:`~fhelium.backend.runner.ScheduleFunction`, whose planned schedule (arena placement and, for a graphed function, Graph segments) is stored. ``binding`` maps each material and state buffer binding to its material symbol; a :class:`~fhelium.backend.runner.Material` it does not map keeps its own symbol. Called schedules become entries ``<name>/call<k>`` with the same binding. A schedule with Python calls raises :class:`ExportError` that names their functions.
        """
        self._python = []
        self._entry(name, schedule, binding)
        if self._python:
            raise ExportError(
                "Schedules with Python calls cannot be exported: "
                + ", ".join(sorted(set(self._python)))
            )
        return name

    def build(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Return the manifest fields ``entries``, ``kernels`` and ``constants`` and the tensors they name."""
        return (
            {
                "entries": dict(self._entries),
                "kernels": list(self._kernels),
                "constants": list(self._constants),
            },
            dict(self._tensors),
        )

    # Entries -------------------------------------------------------------

    def _entry(self, name: str, source: Any, binding: Binding) -> None:
        from fhelium.backend.runner import (
            CalledSchedule,
            Constant,
            Material,
            ObjectRef,
            PythonCallee,
            Schedule,
        )

        if isinstance(source, Schedule):
            schedule, plan = source, None
        else:
            schedule = source._schedule()
            plan = source.plan
        strings: dict[str, int] = {}

        def string(text: str) -> int:
            return strings.setdefault(text, len(strings))

        buffers: list[int] = [len(schedule.buffers)]
        for storage, sizes, strides, dtype, device in schedule.buffers:
            buffers.append(string(storage))
            for values in (sizes, strides):
                if values is None:
                    buffers.append(-1)
                else:
                    buffers.append(len(values))
                    buffers.extend(values)
            buffers.append(-1 if dtype is None else string(str(dtype)))
            buffers.append(-1 if device is None else string(str(device)))

        code: list[int] = [len(schedule.instructions)]

        def value(item: object) -> None:
            if item is None:
                code.append(0)
            elif isinstance(item, bool):
                code.extend((3, int(item)))
            elif isinstance(item, int):
                code.extend((1, item))
            elif isinstance(item, float):
                code.extend(
                    (2, struct.unpack("<q", struct.pack("<d", item))[0])
                )
            elif isinstance(item, str):
                code.extend((4, string(item)))
            elif isinstance(item, (list, tuple)):
                code.extend((5, len(item)))
                for element in item:
                    value(element)
            elif isinstance(item, torch.dtype):
                code.extend((6, string(str(item))))
            elif isinstance(item, torch.device):
                code.extend((7, string(str(item))))
            elif isinstance(item, torch.memory_format):
                code.extend((10, string(str(item).removeprefix("torch."))))
            elif isinstance(item, torch.layout):
                code.extend((11, string(str(item).removeprefix("torch."))))
            else:
                raise ExportError(
                    f"Entry {name} has a builder argument of type {type(item).__name__}"
                )

        callees: list[str] = []
        for instruction in schedule.instructions:
            method = cast(str, instruction[0])
            arguments: list[Any] = list(instruction[1:])
            if method == "launch":
                reference, device, grid, kinds, values, signature = arguments
                kernel = schedule.objects[cast(ObjectRef, reference).index]
                index, positions = self._kernel(kernel, tuple(signature))
                code.extend((string(method), 5, 8, index))
                for item in (
                    device,
                    grid,
                    [kinds[i] for i in positions],
                    [values[i] for i in positions],
                ):
                    value(item)
                continue
            if method == "call":
                callee = cast(
                    CalledSchedule,
                    schedule.objects[cast(ObjectRef, arguments[0]).index],
                )
                function = callee.function
                callee_name = self._callees.get(id(function))
                if callee_name is None:
                    callee_name = f"{name}/call{len(callees)}"
                    self._callees[id(function)] = callee_name
                    self._keep.append(function)
                    self._entry(callee_name, function, binding)
                callees.append(callee_name)
                code.extend(
                    (string(method), len(arguments), 9, string(callee_name))
                )
                for item in arguments[1:]:
                    value(item)
                continue
            if method == "python_call":
                callee = cast(
                    PythonCallee,
                    schedule.objects[cast(ObjectRef, arguments[0]).index],
                )
                function = callee.function
                self._python.append(
                    getattr(function, "__qualname__", repr(function))
                )
                continue
            code.extend((string(method), len(arguments)))
            for item in arguments:
                value(item)

        bindings: list[list[Any]] = []
        for buffer, bound in schedule.bindings:
            if isinstance(bound, Constant):
                bindings.append(
                    [buffer, {"constant": self._constant(bound.tensor)}]
                )
                continue
            symbol = binding(bound) if callable(binding) else binding.get(bound)
            if symbol is None and isinstance(bound, Material):
                symbol = bound.symbol
            if symbol is None:
                raise ExportError(
                    f"Entry {name} binds buffer {buffer} to {bound!r}, which has no material symbol"
                )
            bindings.append([buffer, {"material": symbol}])

        outputs: list[list[Any]] = []
        for kind, item in schedule.outputs:
            if kind == "value":
                try:
                    json.dumps(item)
                except TypeError:
                    raise ExportError(
                        f"Entry {name} returns a constant {type(item).__name__}"
                    ) from None
            outputs.append([kind, item])

        prefix = f"entry.{name}"
        self._tensors[f"{prefix}.buffers"] = torch.tensor(
            buffers, dtype=torch.int64
        )
        self._tensors[f"{prefix}.code"] = torch.tensor(code, dtype=torch.int64)
        arena_bytes = None if plan is None else dict(plan.arena_bytes)
        output_bytes = None if plan is None else plan.output_bytes
        self._entries[name] = {
            "strings": list(strings),
            "buffers": f"{prefix}.buffers",
            "code": f"{prefix}.code",
            "int_registers": schedule.int_registers,
            "float_registers": schedule.float_registers,
            "inputs": [list(item) for item in schedule.inputs],
            "read_in_place": list(schedule.read_in_place),
            "output_buffers": list(schedule.output_buffers),
            "outputs": outputs,
            "bindings": bindings,
            "graphs": bool(schedule.graphs),
            "callees": callees,
            "arena_bytes": arena_bytes,
            "output_bytes": output_bytes,
            "working_set": [],
            "groups": [
                {"symbol": group.symbol, "size": group.size}
                for group in schedule.groups
            ],
            "streams": [
                instruction[2]
                for instruction in schedule.instructions
                if instruction[0] == "stream"
            ]
            or None,
        }

    # Kernels and constants -------------------------------------------------

    def _kernel(
        self, kernel: Any, signature: tuple[object, ...]
    ) -> tuple[int, tuple[int, ...]]:
        key = (id(kernel), signature)
        record = kernel.record(*signature)
        index = self._kernel_ids.get(key)
        if index is not None:
            return index, record.runtime_positions
        self._keep.append(kernel)
        index = self._kernel_ids[key] = len(self._kernels)
        prefix = f"kernel.{index}"
        images: dict[str, str] = {}
        architecture = _architecture(record.image_kind, record.arch)
        images[architecture] = self._bytes(
            f"{prefix}.{architecture}", record.image
        )
        if record.ptx:
            images["ptx"] = self._bytes(f"{prefix}.ptx", record.ptx.encode())
        for target in self.targets:
            if target not in images:
                images[target] = self._bytes(
                    f"{prefix}.{target}",
                    _compile_for(kernel, signature, target),
                )
        self._kernels.append(
            {
                "name": record.name,
                "num_warps": record.num_warps,
                "warp_size": record.warp_size,
                "num_ctas": record.num_ctas,
                "shared": record.shared,
                "parameter_types": list(record.parameter_types),
                "scratch": [
                    record.global_scratch_size,
                    record.global_scratch_align,
                    record.profile_scratch_size,
                    record.profile_scratch_align,
                ],
                "triton_version": record.triton_version,
                "images": images,
            }
        )
        return index, record.runtime_positions

    def _bytes(self, name: str, data: bytes) -> str:
        self._tensors[name] = torch.frombuffer(
            bytearray(data), dtype=torch.uint8
        )
        return name

    def _constant(self, tensor: torch.Tensor) -> int:
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.device,
            tensor.dtype,
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
        )
        index = self._constant_ids.get(key)
        if index is not None:
            return index
        self._keep.append(tensor)
        index = self._constant_ids[key] = len(self._constants)
        # The stored tensor is the span of storage the view reads.
        span = (
            0
            if tensor.numel() == 0
            else 1
            + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        )
        flat = tensor.detach().as_strided(
            (span,), (1,), tensor.storage_offset()
        )
        name = f"const.{index}"
        self._tensors[name] = flat.cpu().clone()
        self._constants.append(
            {
                "tensor": name,
                "sizes": list(tensor.shape),
                "strides": list(tensor.stride()),
                "offset": 0,
                "device": "cpu" if tensor.device.type == "cpu" else "cuda",
            }
        )
        return index


def _architecture(image_kind: str, arch: str) -> str:
    return f"sm_{arch}" if image_kind == "cubin" else arch


def _compile_for(
    kernel: Any, signature: tuple[object, ...], target: str
) -> bytes:
    """Compile ``kernel`` for ``signature`` on architecture ``target`` without a device of that architecture."""
    import triton
    from triton.backends.compiler import GPUTarget

    binary = kernel._binary(signature, wait=True)
    if target.startswith("sm_"):
        backend, arch, warp_size, kind = "cuda", int(target[3:]), 32, "cubin"
    else:
        backend, arch, warp_size, kind = "hip", target, 64, "hsaco"
    options = {
        key: item
        for key, item in binary.metadata._asdict().items()
        if key
        in (
            "num_warps",
            "num_ctas",
            "num_stages",
            "maxnreg",
            "enable_fp_fusion",
            "launch_cooperative_grid",
            "launch_pdl",
            "debug",
            "sanitize_overflow",
        )
    }
    compiled = triton.compile(
        binary.src,
        target=GPUTarget(backend, arch, warp_size),
        options=options,
    )
    return compiled.asm[kind]


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactWriter",
    "ExportError",
    "load_artifact",
    "load_container",
    "native_identity",
    "read_container_header",
    "save_artifact",
    "save_container",
]
