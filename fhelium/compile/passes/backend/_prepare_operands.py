"""Prepare missing Tensor operands declared by selected whole implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    StringAttr,
    i32,
)
from xdsl.ir import Attribute
from xdsl.rewriter import InsertPoint, Rewriter

from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    requested_implementation,
)
from fhelium.config import CkksConfig
from fhelium.ir.dialects import core, rns

from ..._materials import (
    material_sources,
    declare_material,
    rns_parameter_identity,
    parameter_description,
    prepare_material_bindings,
)
from ..._pipeline import DecisionRecord, PassResult, PassStats
from .._operation_transforms import program_operations


if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext
    from fhelium.values import EvaluationKeySet


@dataclass(frozen=True)
class PrepareOperationOperandsPass:
    """Ask selected implementations for missing parameter Tensor requirements.

    A Backend's optional ``tensor_requirements(operation, config)`` returns
    operation attributes and named material descriptions. This pass adds only
    missing operands; supplied Tensors, assignments, and existing symbols remain
    unchanged. Optional caller-supplied resources and keys populate missing
    entries in the current Compilation's material table through the shared
    preparation utility. No keys are generated. Implementations can defer
    requirements when execution facts are insufficient.
    """

    registry: OperationImplementationRegistry
    resources: (
        CkksDeviceResources
        | RnsContext
        | NttContext
        | Iterable[CkksDeviceResources | RnsContext | NttContext]
    ) = ()
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = ()

    def __post_init__(self) -> None:
        from collections.abc import Mapping
        from fhelium.values import EvaluationKeySet

        object.__setattr__(self, "resources", material_sources(self.resources))
        if not isinstance(self.keys, (Mapping, EvaluationKeySet)):
            object.__setattr__(self, "keys", tuple(self.keys))

    name: str = "prepare-operation-operands"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        workspace = compilation.workspace
        matched = transformed = inserted = skipped = 0
        decisions = []
        for operation in program_operations(program):
            if not self.registry.supports(operation):
                continue
            implementation = self.registry.resolve(
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
                    else None
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
        if self.resources or self.keys:
            before = len(compilation.material_bindings)
            missing = prepare_material_bindings(
                compilation, resources=self.resources, keys=self.keys
            )
            added = len(compilation.material_bindings) - before
            if added or missing:
                decisions.append(
                    DecisionRecord(
                        "material-bindings",
                        details=(
                            f"bound={added}",
                            f"unresolved={len(missing)}",
                        ),
                    )
                )
            matched += added + len(missing)
            transformed += added
            skipped += len(missing)
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


__all__ = ["PrepareOperationOperandsPass"]
