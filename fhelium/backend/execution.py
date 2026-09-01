"""Resolve operation dispatch, link resources, and execute Programs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from fhelium.ir import (
    OperationEffect,
    Program,
    value_role,
)
from fhelium.ir.dialects import core, distributed
from fhelium.values import (
    Ciphertext,
    KeySwitchKey,
    ModulusBasis,
    Plaintext,
    PlaintextRepresentation,
    PolynomialDomain,
    PublicKey,
    ResidueRepresentation,
    SecretKey,
)

from .implementation import (
    OperationImplementation,
    OperationImplementationRegistry,
    OperationInvocation,
    RegionCallable,
    RegionOperationImplementation,
    operation_invocation,
    requested_implementation,
)
from .resources import (
    BoundResource,
    ResourceBindings,
    ResourceMaterializer,
    ResourceRequirement,
)
from .workspace import BackendWorkspace

if TYPE_CHECKING:
    from fhelium.compile import Compilation
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


@dataclass(frozen=True)
class ProgramDispatchTable:
    """Map executable Program operations to resolved Backend calls."""

    operations: Mapping[Operation, OperationDispatch]

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
        keys: Iterable[PublicKey | SecretKey | KeySwitchKey] | None = None,
        named_resources: ResourceBindings | None = None,
        materializer: ResourceMaterializer | None = None,
        material_overrides: Mapping[str, object] | None = None,
    ) -> None:
        base = BackendWorkspace() if workspace is None else workspace
        selected_workspace = BackendWorkspace(
            named_resources=(
                base.named_resources
                if named_resources is None
                else named_resources
            ),
            keys=tuple(base.keys if keys is None else keys),
            materializer=(
                base.materializer if materializer is None else materializer
            ),
            material_overrides=(
                base.material_overrides
                if material_overrides is None
                else material_overrides
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
    ) -> ProgramExecutable:
        """Link one Compilation with this workspace into an executable."""

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
        )
        result = selected_pipeline.run(request)
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
    """Execute one fixed Program with structured control-flow regions."""

    program: Program
    dispatch_table: ProgramDispatchTable
    resources: tuple[BoundResource, ...]
    resource_indices: Mapping[Operation, tuple[int, ...]]
    bound_resources: Mapping[Operation, BoundResource]
    bound_materials: Mapping[Operation, object]

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

    def run(self, *inputs: object) -> object:
        block = self.program.single_block("main")
        if len(inputs) != len(block.args):
            raise ValueError(
                f"Program executable requires {len(block.args)} inputs, "
                f"got {len(inputs)}"
            )
        payloads = tuple(
            _boundary_input(
                argument.type,
                value,
                label=f"Program input {index}",
            )
            for index, (argument, value) in enumerate(
                zip(block.args, inputs, strict=True)
            )
        )
        outputs = self._run_block(block, payloads)
        result_types = tuple(
            self.program.function("main").function_type.outputs
        )
        public_outputs = tuple(
            _boundary_output(
                result_type,
                output,
                label=f"Program result {index}",
            )
            for index, (result_type, output) in enumerate(
                zip(result_types, outputs, strict=True)
            )
        )
        return public_outputs[0] if len(public_outputs) == 1 else public_outputs

    def _run_region(
        self,
        region: Region,
        arguments: tuple[object, ...],
        enclosing_values: Mapping[SSAValue, object],
    ) -> tuple[object, ...]:
        blocks = tuple(region.blocks)
        if len(blocks) != 1:
            raise ValueError(
                "Structured operation regions require one block; arbitrary "
                "control-flow graphs are not supported"
            )
        return self._run_block(
            blocks[0],
            arguments,
            enclosing_values=enclosing_values,
        )

    def _run_block(
        self,
        block: Block,
        arguments: tuple[object, ...],
        *,
        enclosing_values: Mapping[SSAValue, object] | None = None,
    ) -> tuple[object, ...]:
        if len(arguments) != len(block.args):
            raise ValueError(
                f"Block requires {len(block.args)} inputs, got {len(arguments)}"
            )
        values: dict[SSAValue, object] = dict(enclosing_values or {})
        values.update(zip(block.args, arguments, strict=True))

        for operation in block.ops:
            if isinstance(operation, UnrealizedConversionCastOp):
                if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                    raise ValueError("Boundary casts must be one-to-one")
                values[operation.outputs[0]] = values[operation.inputs[0]]
                continue
            if isinstance(operation, core.MaterialRefOp):
                values[operation.value] = self.bound_materials[operation]
                continue
            if isinstance(operation, core.ResourceRefOp):
                values[operation.value] = self.bound_resources[operation]
                continue
            if isinstance(operation, ReturnOp):
                return tuple(values[value] for value in operation.arguments)
            if isinstance(operation, scf.YieldOp):
                return tuple(values[value] for value in operation.arguments)
            if isinstance(operation, distributed.YieldOp):
                return (values[operation.value],)
            if isinstance(operation, scf.ForOp):
                lower = _index_value(values[operation.lb])
                upper = _index_value(values[operation.ub])
                step = _index_value(values[operation.step])
                carried = tuple(values[value] for value in operation.iter_args)
                for index in range(lower, upper, step):
                    carried = self._run_region(
                        operation.body,
                        (index, *carried),
                        values,
                    )
                if len(carried) != len(operation.results):
                    raise ValueError(
                        "scf.for yielded a result count different from its "
                        "loop-carried result count"
                    )
                values.update(zip(operation.results, carried, strict=True))
                continue
            if isinstance(operation, scf.IfOp):
                selected = (
                    operation.true_region
                    if bool(values[operation.cond])
                    else operation.false_region
                )
                branch_results = self._run_region(selected, (), values)
                if len(branch_results) != len(operation.results):
                    raise ValueError(
                        "scf.if yielded a result count different from its "
                        "declared result count"
                    )
                values.update(
                    zip(operation.results, branch_results, strict=True)
                )
                continue
            arithmetic = _execute_arithmetic(operation, values)
            if arithmetic is not None:
                values.update(zip(operation.results, arithmetic, strict=True))
                continue

            dispatch = self.dispatch_table.operations[operation]
            payloads: list[torch.Tensor] = []
            operand_resources: list[BoundResource] = []
            for operand in operation.operands:
                value = values[operand]
                if isinstance(value, torch.Tensor):
                    payloads.append(value)
                elif isinstance(value, BoundResource):
                    operand_resources.append(value)
                else:
                    raise TypeError(
                        f"Operand for {operation.name!r} has unsupported "
                        f"runtime type {type(value).__name__}"
                    )
            # ResourceRef operands preserve the operation declaration order;
            # implementation-owned requirements follow in declaration order.
            resources = (
                *operand_resources,
                *(
                    self.resources[index]
                    for index in self.resource_indices[operation]
                ),
            )
            if operation.regions:
                implementation = dispatch.implementation
                if not isinstance(
                    implementation, RegionOperationImplementation
                ):
                    raise TypeError(
                        f"Implementation {implementation.name!r} does not "
                        f"execute regions owned by {operation.name!r}"
                    )
                region_callables = tuple(
                    _tensor_region_callable(self, region, values)
                    for region in operation.regions
                )
                outputs = implementation.execute_regions(
                    dispatch.invocation,
                    tuple(payloads),
                    resources,
                    region_callables,
                    in_place=dispatch.in_place,
                )
            else:
                outputs = dispatch.implementation.execute(
                    dispatch.invocation,
                    tuple(payloads),
                    resources,
                    in_place=dispatch.in_place,
                )
            if len(outputs) != len(operation.results):
                raise ValueError(
                    f"Implementation {dispatch.implementation.name!r} returned "
                    f"{len(outputs)} values for {len(operation.results)} results"
                )
            values.update(zip(operation.results, outputs, strict=True))
        raise ValueError("Program has no return operation")


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
        _integer_state(state, "level", label=label),
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
    level = _integer_state(state, "level", label=label)
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
        level,
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


def _boundary_input(
    value_type: Attribute,
    value: object,
    *,
    label: str,
) -> object:
    kind = _boundary_kind(value_type)
    if kind is None:
        return value
    if kind == "ciphertext":
        if not isinstance(value, Ciphertext):
            raise TypeError(f"{label} requires a Ciphertext")
        state = _state(value_type, label=label)
        represented: tuple[tuple[str, object, object], ...] = (
            ("level", state.get("level"), value.level),
            ("scale", state.get("scale"), value.scale),
            ("prime_ids", state.get("prime_ids"), value.prime_ids),
            (
                "polynomial_domain",
                state.get("polynomial_domain"),
                value.polynomial_domain,
            ),
            ("basis", state.get("basis"), value.modulus_basis),
            (
                "residue_representation",
                state.get("residue_representation"),
                value.residue_representation,
            ),
        )
        for name, attribute, actual in represented:
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
            _require_state_match(expected, actual, name, label=label)
        components = _component_state(state, label=label)
        if components is not None:
            _require_state_match(
                components,
                value.component_count,
                "components",
                label=label,
            )
        return value.data

    if not isinstance(value, Plaintext):
        raise TypeError(f"{label} requires a Plaintext")
    state = _state(value_type, label=label)
    representation_attribute = state.get("representation")
    representation = (
        representation_attribute.data
        if isinstance(representation_attribute, StringAttr)
        and representation_attribute.data != "unknown"
        else value.representation
    )
    payload = value.message if representation == "slots" else value.data
    if payload is None:
        raise ValueError(f"{label} has no active Plaintext Tensor")
    return payload


def _boundary_output(
    value_type: Attribute,
    value: object,
    *,
    label: str,
) -> object:
    kind = _boundary_kind(value_type)
    if kind is None:
        return value
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} requires a Tensor payload")
    if kind == "ciphertext":
        level, scale, prime_ids, domain, basis, residues, components = (
            _ciphertext_state(value_type, label=label)
        )
        if components is not None:
            actual_components = value.size(0) if value.ndim else 0
            _require_state_match(
                components,
                actual_components,
                "components",
                label=label,
            )
        return Ciphertext(
            data=value,
            level=level,
            scale=scale,
            prime_ids=prime_ids,
            polynomial_domain=cast(PolynomialDomain, domain),
            modulus_basis=cast(ModulusBasis, basis),
            residue_representation=cast(ResidueRepresentation, residues),
        )

    level, scale, representation, domain, basis, residues, prime_ids = (
        _plaintext_state(value_type, label=label)
    )
    return Plaintext(
        message=value if representation == "slots" else None,
        level=level,
        scale=scale,
        data=None if representation == "slots" else value,
        representation=cast(PlaintextRepresentation, representation),
        polynomial_domain=cast(PolynomialDomain | None, domain),
        modulus_basis=cast(ModulusBasis | None, basis),
        residue_representation=cast(
            ResidueRepresentation | None,
            residues,
        ),
        prime_ids=prime_ids,
    )


def _tensor_region_callable(
    executable: ProgramExecutable,
    region: Region,
    enclosing_values: Mapping[SSAValue, object],
) -> RegionCallable:
    def run(arguments: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        outputs = executable._run_region(
            region,
            tuple(arguments),
            enclosing_values,
        )
        if any(not isinstance(output, torch.Tensor) for output in outputs):
            raise TypeError("Operation combine regions must return Tensors")
        return tuple(outputs)  # type: ignore[return-value]

    return run


def _index_value(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    raise TypeError("Structured-control-flow indices must be integer values")


def _execute_arithmetic(
    operation: Operation,
    values: Mapping[SSAValue, object],
) -> tuple[object, ...] | None:
    operands = tuple(values[value] for value in operation.operands)
    if isinstance(operation, arith.ConstantOp):
        value = operation.value
        if isinstance(value, IntegerAttr | FloatAttr):
            return (value.value.data,)
        raise TypeError(
            "Structured execution supports scalar arithmetic constants"
        )
    if isinstance(operation, arith.AddiOp):
        return (_index_value(operands[0]) + _index_value(operands[1]),)
    if isinstance(operation, arith.SubiOp):
        return (_index_value(operands[0]) - _index_value(operands[1]),)
    if isinstance(operation, arith.MuliOp):
        return (_index_value(operands[0]) * _index_value(operands[1]),)
    if isinstance(operation, arith.DivUIOp | arith.DivSIOp):
        return (_index_value(operands[0]) // _index_value(operands[1]),)
    if isinstance(operation, arith.RemUIOp | arith.RemSIOp):
        return (_index_value(operands[0]) % _index_value(operands[1]),)
    if isinstance(operation, arith.MinUIOp):
        return (min(_index_value(operands[0]), _index_value(operands[1])),)
    if isinstance(operation, arith.MaxUIOp):
        return (max(_index_value(operands[0]), _index_value(operands[1])),)
    if isinstance(operation, arith.IndexCastOp):
        return (_index_value(operands[0]),)
    if isinstance(operation, arith.SelectOp):
        return (operands[1] if bool(operands[0]) else operands[2],)
    if isinstance(operation, arith.CmpiOp):
        lhs = _index_value(operands[0])
        rhs = _index_value(operands[1])
        predicates = (
            lambda: lhs == rhs,
            lambda: lhs != rhs,
            lambda: lhs < rhs,
            lambda: lhs <= rhs,
            lambda: lhs > rhs,
            lambda: lhs >= rhs,
            lambda: lhs < rhs,
            lambda: lhs <= rhs,
            lambda: lhs > rhs,
            lambda: lhs >= rhs,
        )
        return (predicates[operation.predicate.value.data](),)
    return None


__all__ = [
    "OperationBackend",
    "OperationDispatch",
    "ProgramDispatchTable",
    "ProgramExecutable",
]
