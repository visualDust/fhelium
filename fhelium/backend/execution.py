"""Resolve operation dispatch, link resources, and execute Programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Attribute, Block, Operation

from fhelium.ir import (
    OperationEffect,
    Program,
    value_role,
)
from fhelium.ir.dialects import ckks, core
from fhelium.values._validation import validate_integral_tensor

from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    ModulusBasis,
    Plaintext,
    PlaintextRepresentation,
    PolynomialDomain,
    ResidueRepresentation,
)

from .implementation import (
    OperationImplementation,
    OperationImplementationRegistry,
    OperationInvocation,
    operation_invocation,
    requested_implementation,
)
from .lowering import Layout, layout_key
from .resources import (
    BoundResource,
    ResourceBindings,
    ResourceMaterializer,
    ResourceRequirement,
)
from .workspace import BackendWorkspace

if TYPE_CHECKING:
    from fhelium.compile import Compilation
    from fhelium.compile.passes.backend._prepare_host import LoweredProgram
    from fhelium.compile._pipeline import Pipeline


def _default_operation_registry() -> OperationImplementationRegistry:
    from .assembly import create_builtin_operation_registry

    return create_builtin_operation_registry(ntt_backend_name=None)


@dataclass(frozen=True)
class OperationDispatch:
    """Hold one operation's resolved Backend call before resource linking."""

    invocation: OperationInvocation
    implementation: OperationImplementation
    requirements: tuple[ResourceRequirement, ...]
    effect: OperationEffect
    in_place: bool = False
    prepared_regions: bool = False


_DispatchKeyT = TypeVar("_DispatchKeyT")


@dataclass(frozen=True)
class ProgramDispatchTable(Generic[_DispatchKeyT]):
    """Map executable Program operations to resolved Backend calls.

    A linked executable keys dispatches by Program operation; an executable restored from an artifact keys them by dispatch position.
    """

    operations: Mapping[_DispatchKeyT, OperationDispatch]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operations",
            MappingProxyType(dict(self.operations)),
        )


@dataclass(frozen=True, init=False)
class OperationBackend:
    """Own operation implementations and one live Backend workspace."""

    registry: OperationImplementationRegistry
    workspace: BackendWorkspace

    def __init__(
        self,
        registry: OperationImplementationRegistry | None = None,
        workspace: BackendWorkspace | None = None,
        *,
        named_resources: ResourceBindings | None = None,
        materializer: ResourceMaterializer | None = None,
    ) -> None:
        base = BackendWorkspace() if workspace is None else workspace
        selected_workspace = BackendWorkspace(
            named_resources=(
                base.named_resources
                if named_resources is None
                else named_resources
            ),
            materializer=(
                base.materializer if materializer is None else materializer
            ),
        )
        object.__setattr__(
            self,
            "registry",
            _default_operation_registry() if registry is None else registry,
        )
        object.__setattr__(self, "workspace", selected_workspace)

    @property
    def named_resources(self) -> ResourceBindings:
        """Return named low-level resources configured before linking."""

        return self.workspace.named_resources

    def with_named_resources(
        self,
        resources: ResourceBindings,
        *,
        override: bool = True,
    ) -> OperationBackend:
        """Return this implementation set with an extended resource table."""

        return OperationBackend(
            self.registry,
            self.workspace.with_named_resources(
                resources,
                override=override,
            ),
        )

    def diagnostics(
        self,
        operation: Operation,
        *,
        implementation: str | None = None,
        in_place: bool = False,
    ) -> tuple[str, ...]:
        """Report implementation or resource binding failures."""

        bindings = self.named_resources
        invocation = operation_invocation(operation)
        try:
            selected = self.registry.resolve(
                operation,
                requested=requested_implementation(operation, implementation),
                in_place=in_place,
            )
            bindings.resolve_all(selected.resource_requirements(invocation))
        except (KeyError, TypeError, ValueError) as error:
            return (str(error),)
        return ()

    def link(
        self,
        compilation: Compilation,
        *,
        pipeline: Pipeline | None = None,
        clone: bool = True,
    ) -> ProgramExecutable:
        """Link one Compilation with this workspace into an executable.

        Linking transforms a copy of the Program. ``clone=False`` transfers the Program to the executable, which then owns it; the caller must not use that Program afterwards.
        """

        from fhelium.compile import Compilation, CompileWorkspace
        from fhelium.compile.passes.backend import backend_linking_pipeline

        selected_pipeline = (
            backend_linking_pipeline(self) if pipeline is None else pipeline
        )
        linking_workspace = CompileWorkspace(compilation.workspace)
        for transient in (
            ProgramDispatchTable,
            ProgramExecutable,
            ResourceBindings,
        ):
            linking_workspace.pop(transient, None)
        request = Compilation(
            compilation.program,
            linking_workspace,
            material_bindings=dict(compilation.material_bindings),
        )
        result = selected_pipeline.run(request, clone=clone)
        executable = result.workspace.get(ProgramExecutable)
        if not isinstance(executable, ProgramExecutable):
            raise RuntimeError(
                "Backend linking pipeline did not produce a "
                "ProgramExecutable; resolve operations and link all external "
                "references"
            )
        return executable


@dataclass(frozen=True)
class ProgramExecutable:
    """Execute the exec schedule lowered from one linked Program.

    Every dispatched operation is lowered once to exec operations for the Program's input layouts; when the argument types leave a layout unknown, the Program is lowered once per runtime input layout. Each schedule is built into a libfhelium Runner, which executes it and releases its temporaries when the call returns. ``lowered`` exposes each schedule's exec Program and Runner function. Tensor storage and aliases retain ordinary PyTorch ownership.
    """

    program: Program
    dispatch_table: ProgramDispatchTable[Operation]
    resources: tuple[BoundResource, ...]
    resource_indices: Mapping[Operation, tuple[int, ...]]
    bound_resources: Mapping[Operation, BoundResource]
    bound_materials: Mapping[core.MaterialRefOp, object]
    _requirements: tuple[ResourceRequirement, ...] = field(
        default=(), repr=False, compare=False
    )
    _schedule_symbols: tuple[tuple[str, str], ...] = field(
        default=(), repr=False, compare=False
    )
    _callee: str | None = field(default=None, repr=False, compare=False)
    _call: Callable[..., tuple[object, ...]] = field(
        init=False, repr=False, compare=False
    )
    _lowered: dict[tuple[object, ...], LoweredProgram] = field(
        init=False, repr=False, compare=False
    )
    _static: tuple[object, ...] | None = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        from fhelium.compile.passes.backend._prepare_host import (
            static_input_layout,
        )

        layouts = tuple(
            static_input_layout(value_type) for value_type in self.input_types
        )
        object.__setattr__(self, "_lowered", {})
        object.__setattr__(self, "_static", None)
        if all(layout is not None for layout in layouts):
            lowered = self._lower(layouts)
            object.__setattr__(self, "_static", layouts)
            object.__setattr__(self, "_call", lowered.function)
        else:
            object.__setattr__(self, "_call", self._call_by_layout)

    def _lower(self, layouts: tuple[object, ...]) -> LoweredProgram:
        from fhelium.compile.passes.backend._prepare_host import lower_program

        lowered = lower_program(self, layouts)  # type: ignore[arg-type]
        self._lowered[layouts] = lowered
        return lowered

    def _call_by_layout(self, *payloads: object) -> tuple[object, ...]:
        """Lower once per runtime input layout when Program types leave layouts unknown."""
        from fhelium.compile.passes.backend._prepare_host import (
            runtime_input_layout,
        )

        layouts = tuple(runtime_input_layout(value) for value in payloads)
        lowered = self._lowered.get(layouts)
        if lowered is None:
            lowered = self._lower(layouts)
        return lowered.function(*payloads)

    def _graph_invoke(
        self, key: object, *, adaptive: bool = False
    ) -> Callable[..., object]:
        """Return ``_invoke`` executing through Runners that capture Graph segments.

        ``key`` selects separate Runners and captures for different execution conditions; ``adaptive`` lets each segment keep its graph only when its replays are faster than its direct execution (see :meth:`~fhelium.backend.runner.ScheduleFunction.graphed`).
        """

        def call(*payloads: object) -> tuple[object, ...]:
            if self._static is not None:
                lowered = self._lowered[self._static]
            else:
                from fhelium.compile.passes.backend._prepare_host import (
                    runtime_input_layout,
                )

                layouts = tuple(runtime_input_layout(v) for v in payloads)
                lowered = self._lowered.get(layouts) or self._lower(layouts)
            return lowered.function.graphed(key, adaptive=adaptive)(*payloads)

        def invoke(*payloads: object) -> object:
            outputs = call(*payloads)
            public_outputs = tuple(
                adapt(output)
                for adapt, output in zip(
                    self._output_adapters, outputs, strict=True
                )
            )
            return (
                public_outputs[0]
                if len(public_outputs) == 1
                else public_outputs
            )

        return invoke

    def graph_stats(self, key: object) -> dict[str, int]:
        """Return Graph and storage counts of the Graph Runners for ``key``.

        ``staged_copies`` counts live-in Tensors copied into Runner storage before replays; ``detached_outputs`` counts results that the caller still referenced when a later execution was about to overwrite them and that moved to fresh storage. ``variants`` counts the captured Graphs the Runners hold; a segment keeps up to four, one for each set of addresses of inputs it reads in place from other Runners' storage, ``private_pool_bytes`` the caching-allocator memory still reserved by the Graph pool, which all Runners sharing an arena on one device use, ``live_out_copies`` the copies of segment results that a Graph allocated in its pool and that are read after the segment, moved into Runner-owned storage after each replay, ``arena_bytes`` their arena requirement and ``output_bytes`` their output regions.
        """
        names = (
            "segments",
            "captures",
            "replays",
            "staged_copies",
            "detached_outputs",
            "variants",
            "private_pool_bytes",
            "live_out_copies",
        )
        totals: dict[str, int] = dict.fromkeys(
            (*names, "arena_bytes", "output_bytes"), 0
        )
        for lowered in self._lowered.values():
            function = lowered.function._graphed.get(key)
            if function is None or function._runner is None:
                continue
            for name, count in zip(
                names, function._runner.graph_stats(), strict=True
            ):
                if name == "private_pool_bytes":
                    # Runners on one Arena share one pool and report its size.
                    totals[name] = max(totals[name], count)
                else:
                    totals[name] += count
            totals["arena_bytes"] += function._runner.arena_bytes()
            totals["output_bytes"] += function._runner.output_bytes()
        return totals

    def overlap_stats(self) -> dict[str, float]:
        """Return overlap planning and CPU worker lane counts of the Runners built so far.

        ``moved_transfers`` counts transfers issued before their original position, ``h2d_transfers`` and ``d2h_transfers`` the asynchronous transfers, ``stream_collectives`` the collectives on communication streams and ``planned_tasks`` the CPU tasks of the schedules. ``tasks_run``, ``task_seconds`` (worker wall time), ``join_wait_seconds`` (host time blocked in joins) and ``event_waits`` accumulate over executions.
        """
        totals: dict[str, float] = dict.fromkeys(
            (
                "moved_transfers",
                "h2d_transfers",
                "d2h_transfers",
                "stream_collectives",
                "planned_tasks",
                "tasks_run",
                "task_seconds",
                "join_wait_seconds",
                "event_waits",
            ),
            0,
        )
        for lowered in self._lowered.values():
            # Functions restored from an artifact run its planned schedule
            # and record no plan.
            functions = (
                lowered.function,
                *getattr(lowered.function, "_graphed", {}).values(),
            )
            for function in functions:
                report = getattr(
                    getattr(function, "plan", None), "overlap", None
                )
                if report is None:
                    continue
                totals["moved_transfers"] += len(report.moved)
                for direction, count in report.transfers:
                    name = (
                        "stream_collectives"
                        if direction == "comm"
                        else f"{direction}_transfers"
                    )
                    totals[name] += count
                totals["planned_tasks"] += report.tasks
                for runner in (function._runner, function._private):
                    if runner is None:
                        continue
                    tasks, seconds, joins, waits, *_ = runner.lane_stats()
                    totals["tasks_run"] += tasks
                    totals["task_seconds"] += seconds
                    totals["join_wait_seconds"] += joins
                    totals["event_waits"] += waits
        return totals

    def segment_stats(self, key: object) -> list[dict[str, float | str]]:
        """Return one record per Graph segment of the Graph Runners for ``key``.

        ``nodes``, ``capture_seconds`` and ``instantiate_seconds`` describe the latest capture (host wall time); ``direct_seconds`` and ``replay_seconds`` are device times of the latest direct execution and replay, recorded only by Runners built with segment profiling and NaN otherwise. ``choice`` is ``"graph"`` or ``"direct"`` once a segment has decided how it executes and ``"measuring"`` before; segments that always capture report ``"graph"``. ``decision_direct_seconds`` and ``decision_replay_seconds`` are the device times the decision compared: the first direct execution and the median of three replays, NaN until the decision.
        """
        names = (
            "nodes",
            "capture_seconds",
            "instantiate_seconds",
            "direct_seconds",
            "replay_seconds",
            "variants",
            "choice",
            "decision_direct_seconds",
            "decision_replay_seconds",
        )
        choices = ("measuring", "graph", "direct")
        records: list[dict[str, float | str]] = []
        for lowered in self._lowered.values():
            function = lowered.function._graphed.get(key)
            if function is None or function._runner is None:
                continue
            for row in function._runner.segment_stats():
                record: dict[str, float | str] = dict(
                    zip(names, row, strict=True)
                )
                record["choice"] = choices[int(row[6])]
                records.append(record)
        return records

    def release_graphs(self, key: object) -> None:
        """Drop the Graph Runners, captures and arenas selected by ``key``."""
        for lowered in self._lowered.values():
            lowered.function._graphed.pop(key, None)

    @property
    def lowered(self) -> tuple[LoweredProgram, ...]:
        """Return the exec schedules lowered so far, one per input layout."""
        return tuple(self._lowered.values())

    @property
    def schedule_text(self) -> str:
        """Return the exec IR text of the first lowered schedule."""
        lowered = self.lowered
        program = lowered[0].program if lowered else None
        return "" if program is None else str(program)

    def rebind_materials(
        self, bindings: Mapping[str, object]
    ) -> ProgramExecutable:
        """Reuse lowered schedules and resources with new materials.

        The owner supplies bindings with the layouts of the replaced materials and manages outstanding execution lifetimes. Only material buffer bindings change; lowering and kernel compilation are retained.
        """
        from dataclasses import replace

        materials = {}
        for operation, value in self.bound_materials.items():
            # Linking binds only material references that name a symbol.
            assert operation.symbol is not None
            materials[operation] = bindings.get(operation.symbol.data, value)
        rebound = {}
        for key, lowered in self._lowered.items():
            rebound[key] = replace(
                lowered, function=lowered.function.with_materials(bindings)
            )
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        object.__setattr__(
            result, "bound_materials", MappingProxyType(materials)
        )
        object.__setattr__(result, "_lowered", rebound)
        object.__setattr__(
            result,
            "_call",
            result._call_by_layout
            if self._static is None
            else rebound[self._static].function,
        )
        return result

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "operations": [
                    {
                        "operation": operation.name,
                        "implementation": dispatch.implementation.name,
                        "effect": dispatch.effect,
                        "in_place": dispatch.in_place,
                    }
                    for operation, dispatch in (
                        self.dispatch_table.operations.items()
                    )
                ],
                "resources": [
                    {
                        "symbol": resource.symbol,
                        "kind": resource.kind,
                    }
                    for resource in self.resources
                ],
                "fallback": False,
            }
        )

    @cached_property
    def _entry_block(self) -> Block:
        return self.program.single_block("main")

    @cached_property
    def input_types(self) -> tuple[Attribute, ...]:
        """Return the represented types of the Program inputs."""
        return tuple(argument.type for argument in self._entry_block.args)

    @cached_property
    def output_types(self) -> tuple[Attribute, ...]:
        """Return the represented types of the Program results."""
        return tuple(self.program.function("main").function_type.outputs)

    @cached_property
    def _input_adapters(self) -> tuple[Callable[[object], object], ...]:
        adapters = []
        for index, value_type in enumerate(self.input_types):
            label = f"Program input {index}"
            adapt = _prepare_boundary_input(value_type, label=label)
            layout = None if self._static is None else self._static[index]
            if isinstance(layout, Layout):
                adapt = _check_layout(adapt, layout, label=label)
            adapters.append(adapt)
        return tuple(adapters)

    @cached_property
    def _output_adapters(self) -> tuple[Callable[[object], object], ...]:
        # Built only after the numerical execution in run. Incomplete output
        # state remains an execution-time error, not a linking-time requirement.
        return tuple(
            _prepare_boundary_output(
                result_type, label=f"Program result {index}"
            )
            for index, result_type in enumerate(self.output_types)
        )

    def run(self, *inputs: object) -> object:
        count = len(self.input_types)
        if len(inputs) != count:
            raise ValueError(
                f"Program executable requires {count} inputs, got {len(inputs)}"
            )
        payloads = tuple(
            adapt(value)
            for adapt, value in zip(self._input_adapters, inputs, strict=True)
        )
        return self._invoke(*payloads)

    def _invoke(self, *payloads: object) -> object:
        """Execute Tensor payloads already adapted by the calling interface."""
        outputs = self._call(*payloads)
        public_outputs = tuple(
            adapt(output)
            for adapt, output in zip(
                self._output_adapters, outputs, strict=True
            )
        )
        return public_outputs[0] if len(public_outputs) == 1 else public_outputs


def _check_layout(
    adapt: Callable[[object], object], layout: Layout, *, label: str
) -> Callable[[object], object]:
    """Require the Tensor layout for which the schedule was lowered."""
    expected = (
        torch.Size(layout.sizes),
        layout.strides,
        layout.dtype,
        layout.device,
    )

    def checked(value: object) -> object:
        payload = adapt(value)
        if (
            isinstance(payload, torch.Tensor)
            and layout_key(payload) != expected
        ):
            raise ValueError(
                f"{label} has layout {layout_key(payload)!r}, but the "
                f"executable was lowered for {expected!r}"
            )
        return payload

    return checked


def _boundary_kind(value_type: Attribute) -> str | None:
    role = value_role(value_type)
    if role == "encrypted":
        return "ciphertext"
    if role == "plaintext":
        return "plaintext"
    return None


def _state(value_type: Attribute, *, label: str) -> Mapping[str, Attribute]:
    state = getattr(value_type, "state", None)
    data = getattr(state, "data", None)
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} has no represented CKKS state")
    return data


def _integer_state(
    state: Mapping[str, Attribute],
    name: str,
    *,
    label: str,
) -> int:
    attribute = state.get(name)
    if not isinstance(attribute, IntegerAttr):
        raise ValueError(f"{label} lacks concrete {name!r} state")
    return int(attribute.value.data)


def _float_state(
    state: Mapping[str, Attribute],
    name: str,
    *,
    label: str,
) -> float:
    attribute = state.get(name)
    if not isinstance(attribute, FloatAttr):
        raise ValueError(f"{label} lacks concrete {name!r} state")
    return float(attribute.value.data)


def _string_state(
    state: Mapping[str, Attribute],
    name: str,
    *,
    label: str,
) -> str:
    attribute = state.get(name)
    if not isinstance(attribute, StringAttr) or attribute.data == "unknown":
        raise ValueError(f"{label} lacks concrete {name!r} state")
    return attribute.data


def _optional_string_state(
    state: Mapping[str, Attribute],
    name: str,
    *,
    label: str,
) -> str | None:
    attribute = state.get(name)
    if attribute is None:
        return None
    if not isinstance(attribute, StringAttr) or attribute.data == "unknown":
        raise ValueError(f"{label} lacks concrete {name!r} state")
    return attribute.data


def _prime_ids_state(
    state: Mapping[str, Attribute],
    *,
    label: str,
    required: bool,
) -> tuple[int, ...]:
    attribute = state.get("prime_ids")
    if attribute is None and not required:
        return ()
    if not isinstance(attribute, ArrayAttr) or any(
        not isinstance(item, IntegerAttr) for item in attribute
    ):
        raise ValueError(f"{label} lacks concrete 'prime_ids' state")
    return tuple(int(cast(IntegerAttr, item).value.data) for item in attribute)


def _component_state(
    state: Mapping[str, Attribute],
    *,
    label: str,
) -> int | None:
    attribute = state.get("components")
    if attribute is None:
        attribute = state.get("component_count")
    if attribute is None:
        return None
    if not isinstance(attribute, IntegerAttr):
        raise ValueError(f"{label} lacks concrete 'components' state")
    return int(attribute.value.data)


def _ciphertext_state(
    value_type: Attribute,
    *,
    label: str,
) -> tuple[int, float, tuple[int, ...], str, str, str, int | None]:
    state = _state(value_type, label=label)
    return (
        _integer_state(state, "depth", label=label),
        _float_state(state, "scale", label=label),
        _prime_ids_state(state, label=label, required=True),
        _string_state(state, "polynomial_domain", label=label),
        _string_state(state, "basis", label=label),
        _string_state(state, "residue_representation", label=label),
        _component_state(state, label=label),
    )


def _plaintext_state(
    value_type: Attribute,
    *,
    label: str,
) -> tuple[
    int,
    float,
    str,
    str | None,
    str | None,
    str | None,
    tuple[int, ...],
]:
    state = _state(value_type, label=label)
    depth = _integer_state(state, "depth", label=label)
    scale = _float_state(state, "scale", label=label)
    representation = _string_state(state, "representation", label=label)
    domain = _optional_string_state(state, "polynomial_domain", label=label)
    basis = _optional_string_state(state, "basis", label=label)
    residues = _optional_string_state(
        state, "residue_representation", label=label
    )
    prime_ids = _prime_ids_state(
        state,
        label=label,
        required=representation == "rns",
    )
    if representation == "slots":
        if domain is not None or basis is not None or residues is not None:
            raise ValueError(f"{label} has RNS state for slots representation")
    elif representation in {
        "integer_coefficients",
        "approximate_coefficients",
    }:
        if domain != "coefficient":
            raise ValueError(f"{label} lacks concrete coefficient-domain state")
        if basis is not None or residues is not None:
            raise ValueError(
                f"{label} has RNS state for coefficient representation"
            )
    elif representation == "rns":
        if domain is None or basis is None or residues is None:
            raise ValueError(f"{label} lacks concrete RNS plaintext state")
    else:
        raise ValueError(
            f"{label} has unsupported plaintext representation "
            f"{representation!r}"
        )
    if representation != "rns" and prime_ids:
        raise ValueError(
            f"{label} has prime IDs for non-RNS plaintext representation"
        )
    return (
        depth,
        scale,
        representation,
        domain,
        basis,
        residues,
        prime_ids,
    )


def _require_state_match(
    expected: object,
    actual: object,
    name: str,
    *,
    label: str,
) -> None:
    if expected != actual:
        raise ValueError(
            f"{label} declares {name}={expected!r}, "
            f"but the runtime value has {actual!r}"
        )


def _identity_value(value: object) -> object:
    return value


def _prepare_boundary_input(
    value_type: Attribute,
    *,
    label: str,
) -> Callable[[object], object]:
    """Decode one fixed input type into a reusable public-value adapter."""

    if isinstance(value_type, ckks.EvaluationKeyType):

        def key_input(value: object) -> torch.Tensor:
            from fhelium.values import KeySwitchKey

            if isinstance(value, KeySwitchKey):
                return value.data
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(
                f"{label} requires an evaluation-key Tensor or KeySwitchKey"
            )

        return key_input
    kind = _boundary_kind(value_type)
    compressed = isinstance(value_type, ckks.CompressedPlaintextType)
    if kind is None and not compressed:
        return _identity_value
    state = _state(value_type, label=label)
    if kind == "ciphertext" or compressed:
        checks: list[tuple[str, str, object]] = []
        fields = (
            ("depth", "depth"),
            ("scale", "scale"),
            ("prime_ids", "prime_ids"),
            ("polynomial_domain", "polynomial_domain"),
            ("basis", "modulus_basis"),
            ("residue_representation", "residue_representation"),
        )
        if compressed:
            fields += (
                ("compression_layout", "compression_layout"),
                ("ring_dimension", "ring_dimension"),
            )
        for name, runtime_field in fields:
            attribute = state.get(name)
            if attribute is None or (
                isinstance(attribute, StringAttr)
                and attribute.data == "unknown"
            ):
                continue
            if isinstance(attribute, IntegerAttr):
                expected: object = int(attribute.value.data)
            elif isinstance(attribute, FloatAttr):
                expected = float(attribute.value.data)
            elif isinstance(attribute, StringAttr):
                expected = attribute.data
            elif isinstance(attribute, ArrayAttr) and all(
                isinstance(item, IntegerAttr) for item in attribute
            ):
                expected = tuple(
                    int(cast(IntegerAttr, item).value.data)
                    for item in attribute
                )
            else:
                raise ValueError(f"{label} has malformed {name!r} state")
            checks.append((name, runtime_field, expected))
        components = _component_state(state, label=label)
        if components is not None:
            checks.append(("components", "component_count", components))
        fixed_checks = tuple(checks)

        expected_type = CompressedPlaintext if compressed else Ciphertext

        def polynomial_input(value: object) -> object:
            if not isinstance(value, expected_type):
                raise TypeError(f"{label} requires a {expected_type.__name__}")
            for name, runtime_field, expected in fixed_checks:
                actual = getattr(value, runtime_field)
                if expected != actual:
                    raise ValueError(
                        f"{label} declares {name}={expected!r}, "
                        f"but the runtime value has {actual!r}"
                    )
            return value.data

        return polynomial_input

    representation_attribute = state.get("representation")
    represented = (
        representation_attribute.data
        if isinstance(representation_attribute, StringAttr)
        and representation_attribute.data != "unknown"
        else None
    )

    def plaintext_input(value: object) -> object:
        if not isinstance(value, Plaintext):
            raise TypeError(f"{label} requires a Plaintext")
        representation = (
            value.representation if represented is None else represented
        )
        payload = value.message if representation == "slots" else value.data
        if payload is None:
            raise ValueError(f"{label} has no active Plaintext Tensor")
        return payload

    return plaintext_input


def _prepare_compressed_output(fields):
    """Prepare public compressed results while checking each Tensor payload.

    The first result validates and normalizes metadata. Later results reuse
    those fields while checking extents and the optional implicit Tensor ABI.
    This adapter is shared by ordinary and structured Compile outputs.
    """
    fields = dict(fields)
    fields["prime_ids"] = tuple(fields["prime_ids"])
    prepared = False

    def wrap(data: object, implicit_data: object = None) -> CompressedPlaintext:
        nonlocal prepared
        if not prepared:
            result = CompressedPlaintext(
                data=cast(torch.Tensor, data),
                implicit_data=cast(torch.Tensor | None, implicit_data),
                **fields,
            )
            fields.update(
                scale=result.scale,
                depth=result.depth,
                prime_ids=result.prime_ids,
            )
            prepared = True
            return result
        data = validate_integral_tensor(
            data, value_name="Compressed plaintext output"
        )
        if (
            data.ndim < 2
            or not data.numel()
            or data.size(-2) != len(fields["prime_ids"])
        ):
            raise ValueError(
                "Compressed plaintext output has incompatible extents"
            )
        unique = data.size(-1)
        if unique >= fields["ring_dimension"] or unique & (unique - 1):
            raise ValueError(
                "Compressed plaintext output has incompatible compact extent"
            )
        if fields["compression_layout"] == "strided_sparse":
            implicit_data = validate_integral_tensor(
                implicit_data,
                value_name="Compressed plaintext output implicit_data",
            )
            if (
                implicit_data.shape != data.shape[:-1]
                or implicit_data.dtype != data.dtype
                or implicit_data.device != data.device
            ):
                raise ValueError(
                    "Compressed plaintext output implicit_data must match the payload's batch/limb shape, dtype and device"
                )
        elif implicit_data is not None:
            raise ValueError(
                "Only strided_sparse compressed output accepts implicit_data"
            )
        return CompressedPlaintext._from_fields(
            data=data, implicit_data=implicit_data, **fields
        )

    return wrap


def _prepare_boundary_output(
    value_type: Attribute,
    *,
    label: str,
) -> Callable[[object], object]:
    """Decode one result's represented state after its first execution."""

    if isinstance(value_type, ckks.CompressedPlaintextType):
        state = _state(value_type, label=label)
        depth, scale, _, domain, basis, residues, prime_ids = _plaintext_state(
            value_type, label=label
        )
        fields = {
            "depth": depth,
            "scale": scale,
            "prime_ids": prime_ids,
            "polynomial_domain": domain,
            "modulus_basis": basis,
            "residue_representation": residues,
            "ring_dimension": _integer_state(
                state, "ring_dimension", label=label
            ),
            "compression_layout": cast(
                StringAttr, state["compression_layout"]
            ).data,
        }
        return _prepare_compressed_output(fields)
    kind = _boundary_kind(value_type)
    if kind is None:
        return _identity_value
    if kind == "ciphertext":
        depth, scale, prime_ids, domain, basis, residues, components = (
            _ciphertext_state(value_type, label=label)
        )

        construct: Callable[..., Ciphertext] = Ciphertext
        prepared_construct = Ciphertext._from_fields

        def ciphertext_output(value: object) -> object:
            nonlocal construct
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{label} requires a Tensor payload")
            if components is not None:
                actual_components = value.size(0) if value.ndim else 0
                _require_state_match(
                    components, actual_components, "components", label=label
                )
            validate_integral_tensor(value, value_name=label)
            if (
                value.ndim < 3
                or value.size(0) not in (2, 3)
                or value.size(-2) != len(prime_ids)
                or value.numel() == 0
            ):
                raise ValueError(f"{label} has incompatible ciphertext extents")
            result = construct(
                data=value,
                depth=depth,
                scale=scale,
                prime_ids=prime_ids,
                polynomial_domain=cast(PolynomialDomain, domain),
                modulus_basis=cast(ModulusBasis, basis),
                residue_representation=cast(ResidueRepresentation, residues),
            )
            construct = prepared_construct
            return result

        return ciphertext_output

    depth, scale, representation, domain, basis, residues, prime_ids = (
        _plaintext_state(value_type, label=label)
    )

    construct_plaintext: Callable[..., Plaintext] = Plaintext
    prepared_plaintext = Plaintext._from_fields

    def plaintext_output(value: object) -> object:
        nonlocal construct_plaintext
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} requires a Tensor payload")
        if representation != "slots":
            if representation == "approximate_coefficients":
                if (
                    value.dtype != torch.float64
                    or value.layout != torch.strided
                ):
                    raise TypeError(
                        f"{label} requires strided float64 coefficients"
                    )
            else:
                validate_integral_tensor(value, value_name=label)
            if value.ndim < (2 if representation == "rns" else 1):
                raise ValueError(f"{label} has incompatible plaintext extents")
            if representation == "rns" and value.size(-2) != len(prime_ids):
                raise ValueError(f"{label} has incompatible prime rows")
        if value.numel() == 0:
            raise ValueError(f"{label} has empty plaintext storage")
        result = construct_plaintext(
            message=value if representation == "slots" else None,
            depth=depth,
            scale=scale,
            data=None if representation == "slots" else value,
            representation=cast(PlaintextRepresentation, representation),
            polynomial_domain=cast(PolynomialDomain | None, domain),
            modulus_basis=cast(ModulusBasis | None, basis),
            residue_representation=cast(ResidueRepresentation | None, residues),
            prime_ids=prime_ids,
        )

        construct_plaintext = prepared_plaintext
        return result

    return plaintext_output


def _tensor_region_results(
    outputs: tuple[object, ...],
) -> tuple[torch.Tensor, ...]:
    if any(not isinstance(output, torch.Tensor) for output in outputs):
        raise TypeError("Operation combine regions must return Tensors")
    return cast(tuple[torch.Tensor, ...], outputs)


def _index_value(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    raise TypeError("Structured-control-flow indices must be integer values")


__all__ = [
    "OperationBackend",
    "OperationDispatch",
    "ProgramDispatchTable",
    "ProgramExecutable",
]
