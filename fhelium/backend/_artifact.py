"""Persist linked Program executables without IR, Tensor data or live resources.

An artifact records the prepared host source of a :class:`ProgramExecutable` together with its prepared implementations, invocations and resource requirements. Registered implementations are recorded by their registry position, materials by symbol and bound resources by requirement position, so restoring an artifact against the same registry composition, current materials and Backend workspace reproduces the executable that linking produces, without decoding or linking the Program. Prepared implementations persist through ordinary pickling; generated Python closures declare how they are rebuilt with :func:`rebuild_with`.

Artifacts are pickles and execute code when loaded. Load them only from a cache directory trusted like the installed package.
"""

from __future__ import annotations

import contextvars
import io
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, is_dataclass
from functools import cached_property, partial
from types import FunctionType, MappingProxyType

import torch
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from .execution import (
    OperationBackend,
    ProgramDispatchTable,
    ProgramExecutable,
    _index_value,
    _tensor_region_results,
)

ARTIFACT_VERSION = 1


class ArtifactIneligible(Exception):
    """The executable retains an object that an artifact cannot record."""


def rebuild_with(function: Callable, factory: Callable, *args, **kwargs):
    """Record that the generated ``function`` is reproduced by ``factory(*args, **kwargs)``."""
    function.__fhelium_rebuild__ = (factory, args, kwargs)
    return function


def _call(factory, args, kwargs):
    return factory(*args, **kwargs)


def _mapping_proxy(mapping: dict) -> MappingProxyType:
    return MappingProxyType(mapping)


def _operator(namespace: str, name: str, overload: str | None):
    packet = getattr(getattr(torch.ops, namespace), name)
    return packet if overload is None else getattr(packet, overload)


@dataclass(frozen=True)
class Material:
    """A host-source global bound to the material of ``symbol``."""

    symbol: str


@dataclass(frozen=True)
class Resources:
    """A host-source global bound to linked resources by requirement position."""

    indices: tuple[int, ...]
    single: bool = False


@dataclass(frozen=True)
class _Context:
    implementations: tuple[object, ...]
    externals: tuple[object, ...]
    backend: OperationBackend
    materials: Mapping[str, object]
    relink: Callable[[], ProgramExecutable] | None


_CONTEXT: contextvars.ContextVar[_Context] = contextvars.ContextVar(
    "fhelium_artifact_context"
)


def registry_identity(registry) -> tuple[object, ...] | None:
    """Describe registered implementations, or return None when one is defined outside FHElium.

    Application-defined implementations are excluded because their source is not part of the package fingerprint that keys persistent artifacts.
    """
    described = []
    for implementation in registry.implementations:
        kind = type(implementation)
        if not kind.__module__.startswith("fhelium."):
            return None
        described.append(
            (
                kind.__module__,
                kind.__qualname__,
                getattr(implementation, "name", None),
                repr(implementation) if is_dataclass(implementation) else None,
            )
        )
    return tuple(described)


class _Pickler(pickle.Pickler):
    def __init__(self, stream, implementations, externals):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self._implementations = {
            id(item): index for index, item in enumerate(implementations)
        }
        self._externals = {
            id(item): index for index, item in enumerate(externals)
        }

    def persistent_id(self, obj):
        index = self._implementations.get(id(obj))
        if index is not None:
            return ("implementation", index)
        index = self._externals.get(id(obj))
        if index is not None:
            return ("external", index)
        # Tensor data and IR never enter artifacts.
        if isinstance(
            obj, (torch.Tensor, Operation, SSAValue, Block, Region, Attribute)
        ):
            raise ArtifactIneligible(
                f"prepared execution retains {type(obj).__name__}"
            )
        return None

    def reducer_override(self, obj):
        if isinstance(obj, ProgramExecutable):
            return (_restore_executable, (_executable_state(obj),))
        if isinstance(obj, FunctionType):
            rebuild = getattr(obj, "__fhelium_rebuild__", None)
            return NotImplemented if rebuild is None else (_call, rebuild)
        if isinstance(obj, MappingProxyType):
            return (_mapping_proxy, (dict(obj),))
        if isinstance(obj, torch._ops.OpOverloadPacket):
            namespace, name = obj._qualified_op_name.split("::")
            return (_operator, (namespace, name, None))
        if isinstance(obj, torch._ops.OpOverload):
            namespace, name = obj._schema.name.split("::")
            return (_operator, (namespace, name, obj._overloadname))
        return NotImplemented


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, context: _Context):
        super().__init__(stream)
        self._context = context

    def persistent_load(self, pid):
        kind, index = pid
        if kind == "implementation":
            return self._context.implementations[index]
        return self._context.externals[index]


def export_executable(
    executable: ProgramExecutable,
    registry,
    *,
    externals: tuple[object, ...] = (),
) -> bytes:
    """Serialize a linked executable; raise ArtifactIneligible when it cannot be recorded.

    ``externals`` lists live objects, such as Engines, that prepared implementations reference and that the restoring caller supplies in the same order.
    """
    stream = io.BytesIO()
    try:
        _Pickler(stream, registry.implementations, externals).dump(
            (ARTIFACT_VERSION, executable)
        )
    except (pickle.PicklingError, TypeError, AttributeError) as error:
        raise ArtifactIneligible(f"{type(error).__name__}: {error}") from None
    return stream.getvalue()


def restore_executable(
    payload: bytes,
    backend: OperationBackend,
    materials: Mapping[str, object],
    *,
    externals: tuple[object, ...] = (),
    relink: Callable[[], ProgramExecutable] | None = None,
) -> ProgramExecutable:
    """Restore an artifact exported with the same registry composition.

    ``materials`` supplies current Tensors by symbol and ``backend`` resolves resource requirements. ``relink`` links the Program on request when the restored executable's ``program`` is inspected.
    """
    context = _Context(
        tuple(backend.registry.implementations),
        externals,
        backend,
        materials,
        relink,
    )
    token = _CONTEXT.set(context)
    try:
        version, executable = _Unpickler(io.BytesIO(payload), context).load()
    finally:
        _CONTEXT.reset(token)
    if version != ARTIFACT_VERSION:
        raise ArtifactIneligible("artifact version differs")
    return executable


def _executable_state(executable: ProgramExecutable) -> dict[str, object]:
    from fhelium.ir._structure import _AttributeText

    text = _AttributeText()
    return {
        "source": executable.host_source,
        "bound": executable._bound,
        "requirements": executable._requirements,
        "dispatches": tuple(executable.dispatch_table.operations.values()),
        "manifest": dict(executable.manifest),
        "schedules": executable._schedule_symbols,
        "inputs": tuple(text(t).decode() for t in executable.input_types),
        "outputs": tuple(text(t).decode() for t in executable.output_types),
        "callee": executable._callee,
    }


def _parse_types(texts: tuple[str, ...]) -> tuple[Attribute, ...]:
    if not texts:
        return ()
    from xdsl.dialects.builtin import ArrayAttr
    from xdsl.parser import Parser

    from fhelium.ir import create_dialect_context

    parsed = Parser(
        create_dialect_context(), "[" + ", ".join(texts) + "]", "<artifact>"
    ).parse_attribute()
    assert isinstance(parsed, ArrayAttr)
    return tuple(parsed.data)


def link_resources(
    backend: OperationBackend, requirements
) -> tuple[object, ...]:
    """Resolve requirements against the Backend workspace as linking does."""
    workspace = backend.workspace
    bindings = workspace.named_resources
    bound = frozenset(bindings.symbols)
    missing = tuple(r for r in requirements if r.symbol not in bound)
    if missing and workspace.materializer is not None:
        bindings = bindings.overlay(workspace.materializer.materialize(missing))
    return bindings.resolve_all(tuple(requirements))


def _restore_executable(state: dict[str, object]) -> ProgramExecutable:
    from fhelium.native.wrapper import ntt_ops

    context = _CONTEXT.get()
    resources = link_resources(context.backend, state["requirements"])
    materials = context.materials
    for even, odd in state["schedules"]:
        ntt_ops.validate_indexed_schedule(
            materials[even], materials[odd], 2 * materials[even].size(1)
        )
    namespace: dict[str, object] = {
        "__fhelium_codegen_index": _index_value,
        "__fhelium_codegen_tensor_results": _tensor_region_results,
    }
    material_globals = {}
    for name, value in state["bound"].items():
        if isinstance(value, Material):
            namespace[name] = materials[value.symbol]
            material_globals[name] = value.symbol
        elif isinstance(value, Resources):
            selected = tuple(resources[i] for i in value.indices)
            namespace[name] = selected[0] if value.single else selected
        else:
            namespace[name] = value
    exec(
        compile(state["source"], "<fhelium-prepared-program>", "exec"),
        namespace,
    )
    executable = object.__new__(RestoredProgramExecutable)
    executable.__dict__.update(
        {
            "dispatch_table": ProgramDispatchTable(
                dict(enumerate(state["dispatches"]))
            ),
            "resources": resources,
            "_requirements": state["requirements"],
            "resource_indices": MappingProxyType({}),
            "bound_resources": MappingProxyType({}),
            "bound_materials": MappingProxyType({}),
            "_call": namespace.pop("__fhelium_codegen_execute"),
            "host_source": state["source"],
            "_material_globals": material_globals,
            "_bound": state["bound"],
            "_schedule_symbols": state["schedules"],
            "_callee": state["callee"],
            "_manifest": MappingProxyType(state["manifest"]),
            "_input_texts": state["inputs"],
            "_output_texts": state["outputs"],
            "_relink": context.relink,
        }
    )
    for index, dispatch in enumerate(state["dispatches"]):
        attach = getattr(dispatch.implementation, "attach_region", None)
        if attach is not None:
            attach(partial(_linked_implementation, executable, index))
    return executable


def _linked_implementation(executable: RestoredProgramExecutable, index: int):
    linked = executable.linked()
    return tuple(linked.dispatch_table.operations.values())[
        index
    ].implementation


class RestoredProgramExecutable(ProgramExecutable):
    """A ProgramExecutable restored from an artifact.

    Execution, material rebinding and dispatch inspection use the restored state. The linked Program is produced on first access to ``program`` by linking the cached Program again; a restored fusion that meets an operand layout without a prepared plan obtains its region the same way.
    """

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def linked(self) -> ProgramExecutable:
        """Return the executable that linking the cached Program produces for this function."""
        if self._relink is None:
            raise RuntimeError("This restored executable has no Program source")
        linked = self._relink()
        if self._callee is None:
            return linked
        for dispatch in linked.dispatch_table.operations.values():
            if getattr(dispatch.implementation, "symbol", None) == self._callee:
                return dispatch.implementation.executable
        raise RuntimeError(f"Called function {self._callee!r} is not linked")

    @cached_property
    def program(self):
        return self.linked().program

    @cached_property
    def input_types(self):
        return _parse_types(self._input_texts)

    @cached_property
    def output_types(self):
        return _parse_types(self._output_texts)

    @property
    def manifest(self):
        return self._manifest


__all__ = [
    "ArtifactIneligible",
    "Material",
    "Resources",
    "RestoredProgramExecutable",
    "export_executable",
    "link_resources",
    "rebuild_with",
    "registry_identity",
    "restore_executable",
]
