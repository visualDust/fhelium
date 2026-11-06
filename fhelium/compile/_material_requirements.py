"""Resolve implementation material requirements in Compile Programs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, StringAttr, i32
from xdsl.ir import Attribute, Operation
from xdsl.rewriter import InsertPoint, Rewriter

from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    requested_implementation,
)
from fhelium.config import CkksConfig
from fhelium.ir.dialects import ckks, core, rns

from ._materials import (
    _resolve_key_layouts,
    declare_material,
    rns_parameter_identity,
    parameter_request_identity,
    parameter_description,
)
from ._pipeline import DecisionRecord, PassResult, PassStats

if TYPE_CHECKING:
    from ._compilation import Compilation


def implementation_registry(
    compilation: Compilation,
    registry: OperationImplementationRegistry | None = None,
) -> OperationImplementationRegistry:
    """Use a local registry or the compilation's implementation registry."""
    if registry is not None:
        compilation.workspace.setdefault(
            OperationImplementationRegistry, registry
        )
        return registry
    selected = compilation.workspace.get(OperationImplementationRegistry)
    if not isinstance(selected, OperationImplementationRegistry):
        raise ValueError(
            "Material requirements need an OperationImplementationRegistry in CompileWorkspace"
        )
    return selected


def resolve_key_layouts(
    compilation: Compilation, operations: Iterable[Operation]
) -> int:
    """Resolve layouts for material symbols consumed by these operations."""
    symbols = set()
    for operation in operations:
        if (
            isinstance(operation, core.MaterialRefOp)
            and isinstance(operation.value.type, ckks.EvaluationKeyType)
            and operation.symbol is not None
        ):
            symbols.add(operation.symbol.data)
        for value in operation.operands:
            owner = value.owner
            if (
                isinstance(value.type, ckks.EvaluationKeyType)
                and isinstance(owner, core.MaterialRefOp)
                and owner.symbol is not None
            ):
                symbols.add(owner.symbol.data)
    if not symbols:
        return 0
    return _resolve_key_layouts(
        compilation,
        cast(
            Mapping[str, Mapping[str, object]],
            compilation.workspace.get("key_layouts", {}),
        ),
        symbols=symbols,
    )


def resolve_material_requirements(
    compilation: Compilation,
    operations: Iterable[Operation],
    registry: OperationImplementationRegistry | None = None,
) -> PassResult:
    """Complete declarations for the supplied operations using their implementations."""
    operations = tuple(operations)
    registry = implementation_registry(compilation, registry)
    program = compilation.program
    workspace = compilation.workspace
    layouts = resolve_key_layouts(compilation, operations)
    matched = transformed = layouts
    inserted = skipped = 0
    decisions = []
    for operation in operations:
        if not registry.supports(operation):
            continue
        implementation = registry.resolve(
            operation, requested=requested_implementation(operation)
        )
        declare = getattr(implementation, "tensor_requirements", None)
        if not callable(declare):
            continue
        matched += 1
        config_attr = operation.attributes.get("ckks_config")
        config = (
            CkksConfig.parse(json.loads(config_attr.data))
            if isinstance(config_attr, StringAttr)
            else workspace.get(CkksConfig)
        )
        requirements = (
            declare(operation, config)
            if isinstance(config, CkksConfig)
            else None
        )
        if requirements is None:
            skipped += 1
            decisions.append(
                DecisionRecord(
                    operation.name,
                    details=(
                        "operand preparation awaits configuration, placement, or implementation constraints",
                    ),
                )
            )
            continue
        attributes, descriptions = cast(
            tuple[dict[str, Attribute], dict[str, dict[str, object]]],
            requirements,
        )
        if not attributes and not descriptions:
            continue
        represented = operation.attributes.get("parameter_names")
        names = (
            [x.data for x in represented if isinstance(x, StringAttr)]
            if isinstance(represented, ArrayAttr)
            else []
        )
        supplied = tuple(getattr(operation, "parameters", ()))
        if len(names) != len(supplied):
            raise ValueError(
                f"{operation.name} supplied parameters need corresponding names"
            )
        state = getattr(
            getattr(operation.operands[0].type, "state", None), "data", {}
        )
        physical = {
            key: state[key] for key in ("dtype", "device") if key in state
        }
        refs = []
        new_values = []
        changed = False
        for name, description in descriptions.items():
            if name in names:
                continue
            kind = cast(str, description["kind"])
            typ = (
                rns.RnsParametersType().with_state(physical)
                if kind == "rns_parameters"
                else core.MessageType().with_state(physical)
            )
            declared = parameter_description(
                cast(CkksConfig, config),
                kind,
                **{k: v for k, v in description.items() if k != "kind"},
            )
            identity = (
                rns_parameter_identity(
                    cast(CkksConfig, config),
                    description.get("prime_ids"),
                    physical,
                )
                if kind == "rns_parameters"
                else parameter_request_identity(declared, physical)
            )
            created, value = declare_material(
                compilation,
                operation,
                f"{operation.name}/parameters/{name}",
                typ,
                declared,
                identity=identity,
            )
            refs.extend(created)
            new_values.append(value)
            names.append(name)
        if new_values:
            if refs:
                Rewriter.insert_op(refs, InsertPoint.before(operation))
            operation.operands = (*operation.operands, *new_values)
            segments = operation.properties.get("operandSegmentSizes")
            if isinstance(segments, DenseArrayBase):
                sizes = list(segments.iter_values())
                sizes[-1] += len(new_values)
                operation.properties["operandSegmentSizes"] = (
                    DenseArrayBase.from_list(i32, sizes)
                )
            operation.attributes["parameter_names"] = ArrayAttr(
                StringAttr(name) for name in names
            )
            inserted += len(refs)
            changed = True
        for name, value in attributes.items():
            existing = operation.attributes.get(name)
            if existing is None or existing == StringAttr("unknown"):
                operation.attributes[name] = value
                changed = True
        transformed += int(changed)

    return PassResult(
        program,
        PassStats(
            matched=matched,
            transformed=transformed,
            inserted=inserted,
            skipped=skipped,
        ),
        decisions=tuple(decisions),
    )
