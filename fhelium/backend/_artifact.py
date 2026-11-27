"""Store linked Program executables as artifact entries and restore them from loaded artifacts.

An executable is stored as two entries of an artifact (:mod:`fhelium.native.artifact`): its planned Runner schedule, ``<name>``, and the same schedule planned with CUDA Graph segments, ``<name>.graphs``; called schedules become entries of their own. Kernels are stored as device images, link-time constants as Tensors, material buffers by symbol and state buffers by the symbol ``state.<requirement>.<field>`` of the resource requirement whose live value owns them. The executable record returned with the entries holds what restoring needs without the Program: resource requirements, NTT schedule symbols, the dispatch manifest and the input and output types as text. Restoring binds current material Tensors and the state Tensors of the resources the Backend workspace resolves, so it reproduces the executable that linking produces without decoding, linking or lowering the Program. Executables lowered per runtime input layout and schedules with Python calls are not stored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import is_dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, cast

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map
from xdsl.ir import Attribute

from fhelium.compile.passes.backend._prepare_host import (
    LoweredProgram,
    static_input_layout,
)

from .execution import (
    OperationBackend,
    ProgramDispatchTable,
    ProgramExecutable,
)
from .resources import ResourceRequirement
from .runner import (
    CalledSchedule,
    LoadedArtifact,
    Material,
    Resources,
    ScheduleFunction,
    State,
)


class ArtifactIneligible(Exception):
    """The executable cannot be stored as artifact entries."""


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


def _state_symbol(
    requirements: tuple[ResourceRequirement, ...], binding: State
) -> str:
    resource = binding.resource
    if not isinstance(resource, Resources):
        raise ArtifactIneligible(
            f"a state buffer reads {type(resource).__name__}.{binding.field}, which no resource requirement names"
        )
    return f"state.{requirements[resource.indices[0]].symbol}.{binding.field}"


def _schedule_materials(
    function: ScheduleFunction, found: dict[str, torch.Tensor]
) -> None:
    """Collect the material Tensors that ``function`` and its callees bind, by symbol."""
    schedule = function.schedule
    for _, binding in schedule.bindings:
        if isinstance(binding, Material):
            value = cast(torch.Tensor, function.materials[binding.symbol])
            if found.setdefault(binding.symbol, value) is not value:
                raise ArtifactIneligible(
                    f"material {binding.symbol} names different Tensors in called schedules"
                )
    for item in schedule.objects:
        if isinstance(item, CalledSchedule):
            _schedule_materials(item.function, found)


def material_record(
    symbol: str, tensor: torch.Tensor, binding: str = "caller"
) -> dict[str, object]:
    """Return the artifact material record of a Tensor the loader binds by symbol."""
    return {
        "symbol": symbol,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": tensor.device.type,
        "binding": binding,
    }


def export_executable(
    writer: Any,
    executable: ProgramExecutable,
    name: str,
    *,
    symbols: Callable[[str, torch.Tensor], str] | None = None,
) -> dict[str, Any]:
    """Store ``executable`` as entries ``name`` and ``name.graphs`` of ``writer`` and return its executable record.

    ``symbols`` names the artifact material of each material symbol and Tensor the executable binds; by default a material keeps its symbol. The record lists the bound material Tensors under ``materials`` (artifact symbol to Tensor), the executable's symbols under ``symbols`` (executable symbol to artifact symbol) and the state buffers under ``states`` (symbol to requirement position and field); the caller turns them into manifest material records.
    """
    from fhelium.ir._structure import _AttributeText
    from fhelium.native.artifact import ExportError

    if executable._static is None:
        raise ArtifactIneligible("the executable is lowered per input layout")
    function = executable._lowered[executable._static].function
    requirements = executable._requirements
    states: dict[str, tuple[int, str]] = {}
    bound_materials: dict[str, torch.Tensor] = {}
    _schedule_materials(function, bound_materials)
    names = {
        symbol: symbol if symbols is None else symbols(symbol, tensor)
        for symbol, tensor in bound_materials.items()
    }

    def binding(bound: object) -> str | None:
        if isinstance(bound, State):
            symbol = _state_symbol(requirements, bound)
            states[symbol] = (
                cast(Resources, bound.resource).indices[0],
                bound.field,
            )
            return symbol
        if isinstance(bound, Material):
            return names[bound.symbol]
        return None

    graphed = ScheduleFunction(
        function.schedule,
        function.materials,
        function.resources,
        graphs=True,
        shared=True,
    )
    try:
        writer.entry(name, function, binding=binding)
        writer.entry(f"{name}.graphs", graphed, binding=binding)
    except ExportError as error:
        raise ArtifactIneligible(str(error)) from None
    text = _AttributeText()
    return {
        "entry": name,
        "requirements": [[r.symbol, r.kind] for r in requirements],
        "schedules": [list(pair) for pair in executable._schedule_symbols],
        "manifest": _plain(dict(executable.manifest)),
        "inputs": [text(t).decode() for t in executable.input_types],
        "outputs": [text(t).decode() for t in executable.output_types],
        "materials": {names[s]: t for s, t in bound_materials.items()},
        "symbols": names,
        "states": {symbol: list(item) for symbol, item in states.items()},
    }


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _parse_types(texts: list[str]) -> tuple[Attribute, ...]:
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


def load_device(record: Mapping[str, Any]) -> torch.device:
    """Return the device an executable record runs on: that of its first CUDA input, else CPU."""
    for layout in map(static_input_layout, _parse_types(record["inputs"])):
        device = getattr(layout, "device", None)
        if device is not None and device.type == "cuda":
            return device
    return torch.device("cpu")


def restore_executable(
    loaded: LoadedArtifact,
    record: Mapping[str, Any],
    backend: OperationBackend,
    materials: Mapping[str, torch.Tensor],
    *,
    relink: Callable[[], ProgramExecutable] | None = None,
    graphs: bool | str = False,
) -> ProgramExecutable:
    """Restore the executable of ``record`` from ``loaded``, binding current materials and the Backend's resources.

    ``materials`` supplies Tensors by the executable's material symbols; symbols it omits read the artifact's bound, built or embedded materials. ``relink`` links the Program on request when the restored executable's ``program`` or ``dispatch_table`` is inspected. ``graphs`` calls the entry planned with CUDA Graph segments: True replays every segment and ``"auto"`` measures each segment's replay against direct execution.
    """
    from fhelium.native.wrapper import ntt_ops

    requirements = tuple(
        ResourceRequirement(symbol, kind)
        for symbol, kind in record["requirements"]
    )
    resources = link_resources(backend, requirements)
    names = record.get("symbols", {})
    materials = {names.get(s, s): t for s, t in materials.items()}

    def material(symbol: str) -> torch.Tensor:
        name = names.get(symbol, symbol)
        return (
            materials[name]
            if name in materials
            else loaded.artifact.material(name)
        )

    for even, odd in record["schedules"]:
        ntt_ops.validate_indexed_schedule(
            material(even), material(odd), 2 * material(even).size(1)
        )
    states = {}
    for symbol, (index, field) in record["states"].items():
        resource = resources[index]
        states[symbol] = getattr(getattr(resource, "value", resource), field)
    loaded.bind({**materials, **states})
    function = loaded.function(record["entry"])
    call = (
        function.graphed(None, adaptive=graphs == "auto")
        if graphs
        else function
    )
    inputs = _parse_types(record["inputs"])
    layouts = tuple(static_input_layout(t) for t in inputs)
    executable = object.__new__(RestoredProgramExecutable)
    executable.__dict__.update(
        {
            "resources": resources,
            "_requirements": requirements,
            "resource_indices": MappingProxyType({}),
            "bound_resources": MappingProxyType({}),
            "bound_materials": MappingProxyType({}),
            "_call": call,
            "_lowered": {layouts: LoweredProgram(None, function, ())},
            "_static": layouts,
            "_schedule_symbols": tuple(
                tuple(pair) for pair in record["schedules"]
            ),
            "_callee": None,
            "_manifest": MappingProxyType(record["manifest"]),
            "input_types": inputs,
            "_output_texts": record["outputs"],
            "_relink": relink,
        }
    )
    return executable


class RestoredProgramExecutable(ProgramExecutable):
    """A ProgramExecutable restored from an artifact.

    Execution and material rebinding use the artifact's entries. The linked Program and its dispatch table are produced on first access by linking the cached Program again.
    """

    _relink: Callable[[], ProgramExecutable] | None
    _manifest: Mapping[str, object]
    _output_texts: list[str]

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return True if self is other else NotImplemented

    def linked(self) -> ProgramExecutable:
        """Return the executable that linking the cached Program produces."""
        if self._relink is None:
            raise RuntimeError("This restored executable has no Program source")
        return self._relink()

    @cached_property
    def program(self):
        return self.linked().program

    @cached_property
    def dispatch_table(self) -> ProgramDispatchTable:  # type: ignore[override]
        return self.linked().dispatch_table

    @cached_property
    def output_types(self):
        return _parse_types(self._output_texts)

    @property
    def manifest(self):
        return self._manifest


__all__ = [
    "ArtifactIneligible",
    "RestoredProgramExecutable",
    "export_executable",
    "link_resources",
    "load_device",
    "material_record",
    "registry_identity",
    "restore_executable",
]


class _BuilderRecorder(TorchFunctionMode):
    """Run table construction and record it into an ExecBuilder.

    A registered ``fhelium_*`` operator call is recorded as a call of that operator with its computed results; any other PyTorch call that reads a recorded result is replayed on the placeholders under the builder's tracing, which records its ATen operations.
    """

    def __init__(self, builder: Any) -> None:
        super().__init__()
        self.builder = builder
        self.placed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _placeholder(self, value: object) -> object:
        entry = (
            self.placed.get(id(value))
            if isinstance(value, torch.Tensor)
            else None
        )
        return value if entry is None else entry[1]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        from .lowering import isolated

        kwargs = kwargs or {}
        with isolated():
            output = func(*args, **kwargs)
        results = tree_flatten(output)[0]
        if not any(isinstance(leaf, torch.Tensor) for leaf in results):
            return output
        placed_args, placed_kwargs = tree_map(self._placeholder, (args, kwargs))
        operator = _builder_operator(func)
        if operator is not None:
            with isolated():
                recorded = self.builder.record_call(
                    operator, placed_args, placed_kwargs, output
                )
        elif any(
            isinstance(leaf, torch.Tensor) and id(leaf) in self.placed
            for leaf in tree_flatten((args, kwargs))[0]
        ):
            try:
                recorded = func(*placed_args, **placed_kwargs)
            except Exception:
                # Data-dependent Python code: its results stay unrecorded.
                return output
        else:
            return output
        replayed = tree_flatten(recorded)[0]
        if len(replayed) == len(results):
            for real, fake in zip(results, replayed):
                if isinstance(real, torch.Tensor) and isinstance(
                    fake, torch.Tensor
                ):
                    self.placed[id(real)] = (real, fake)
        return output

    def result(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Return the placeholder of a recorded result or of a view into one."""
        entry = self.placed.get(id(tensor))
        if entry is not None:
            return entry[1]
        storage = tensor.untyped_storage().data_ptr()
        for real, fake in self.placed.values():
            if real.untyped_storage().data_ptr() == storage:
                return self.builder.view_of(fake, tensor)
        return None


def _builder_operator(func: object) -> Any:
    if isinstance(func, torch._ops.OpOverloadPacket):
        if func.overloads() != ["default"]:
            return None
        func = func.default
    if isinstance(func, torch._ops.OpOverload) and func.namespace.startswith(
        "fhelium_"
    ):
        return func
    return None


def export_tables(
    writer: Any,
    tables: Mapping[str, tuple[Mapping[str, object], torch.Tensor]],
    builders: Callable[
        [], Callable[[str, Mapping[str, object]], torch.Tensor | None]
    ],
    name: str = "tables",
) -> tuple[dict[str, dict[str, object]], dict[str, torch.Tensor]]:
    """Store the computation that reproduces ``tables`` as entry ``name`` of ``writer``.

    ``tables`` maps material symbols to their provider selection and bound Tensor. ``builders()`` returns a function that recomputes one table from fresh providers, which hold no cached tables; registered ``fhelium_*`` operator calls and the PyTorch operations applied to their results are recorded while it runs. The recorded entry takes no inputs and runs once at load; it is executed here and keeps the tables it reproduces bitwise. Returns the material record of every table, ``built`` with its entry output position or ``embedded`` when its value is not reproduced by recorded operations, and the container Tensors of the embedded tables.
    """
    from .lowering import compile_fragment, isolated

    excluded: set[str] = set()
    while True:
        outputs: list[str] = []

        def lower(builder, values):
            recorder = _BuilderRecorder(builder)
            build = builders()
            results = []
            with recorder:
                built = {
                    symbol: build(symbol, selection)
                    for symbol, (selection, _) in tables.items()
                    if symbol not in excluded
                }
            for symbol, tensor in built.items():
                with isolated():
                    placed = None if tensor is None else recorder.result(tensor)
                if placed is not None:
                    outputs.append(symbol)
                    results.append(placed)
            return results

        fragment = compile_fragment(lower, ())
        if not outputs:
            break
        produced = fragment.function()
        wrong = {
            symbol
            for symbol, value in zip(outputs, produced, strict=True)
            if not (
                isinstance(value, torch.Tensor)
                and value.shape == tables[symbol][1].shape
                and value.dtype == tables[symbol][1].dtype
                and torch.equal(value.cpu(), tables[symbol][1].cpu())
            )
        }
        del produced
        if not wrong:
            writer.entry(name, fragment.function, binding=lambda bound: None)
            break
        excluded |= wrong
    records: dict[str, dict[str, object]] = {}
    embedded: dict[str, torch.Tensor] = {}
    for symbol, (selection, tensor) in tables.items():
        if symbol in outputs:
            records[symbol] = {
                **material_record(symbol, tensor, "built"),
                "entry": name,
                "output": outputs.index(symbol),
                "builder": selection.get("kind"),
            }
        else:
            records[symbol] = {
                **embedded_record(symbol, tensor, embedded),
                "builder": selection.get("kind"),
            }
    return records, embedded


def embedded_record(
    symbol: str, tensor: torch.Tensor, tensors: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Add ``tensor`` to the container ``tensors`` and return its ``embedded`` material record.

    A view that covers part of its storage is recorded as ``view`` (sizes, strides, offset in elements) of one flat base holding the whole storage, so views of one storage are stored and loaded once.
    """
    storage = tensor.untyped_storage()
    element = tensor.element_size()
    covers = (
        tensor.is_contiguous() and tensor.numel() * element == storage.nbytes()
    )
    if covers or storage.nbytes() % element:
        name = f"material.{symbol}"
        tensors[name] = tensor.contiguous()
        return {**material_record(symbol, tensor, "embedded"), "tensor": name}
    name = f"storage.{tensor.device}.{storage.data_ptr():x}.{str(tensor.dtype)[6:]}"
    if name not in tensors:
        tensors[name] = tensor.as_strided(
            (storage.nbytes() // element,), (1,), 0
        )
    return {
        **material_record(symbol, tensor, "embedded"),
        "tensor": name,
        "view": {
            "sizes": list(tensor.shape),
            "strides": list(tensor.stride()),
            "offset": tensor.storage_offset(),
        },
    }
