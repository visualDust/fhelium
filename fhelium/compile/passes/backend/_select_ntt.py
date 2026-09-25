"""Select NTT schedules without replacing supplied numerical materials."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


import json
from dataclasses import dataclass
from typing import Any, cast

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Operation
from xdsl.rewriter import InsertPoint, Rewriter

from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    requested_implementation,
)
from fhelium.config import CkksConfig
from fhelium.ir.dialects import ckks, core, ntt

from ..._materials import material_symbol, parameter_description
from ..._pipeline import DecisionRecord, PassResult, PassStats
from .._operation_transforms import program_operations

_NTT_TYPES = tuple(
    cast(type[Operation], spec.operation_type) for spec in ntt.OPERATION_SPECS
)
_TRANSFORMS = (*_NTT_TYPES, ckks.ToNttOp, ckks.FromNttOp)


def _text(operation: Operation, name: str) -> str | None:
    value = operation.attributes.get(name)
    return (
        value.data
        if isinstance(value, StringAttr) and value.data != "unknown"
        else None
    )


def _integer(operation: Operation, name: str) -> int | None:
    value = operation.attributes.get(name)
    return int(value.value.data) if isinstance(value, IntegerAttr) else None


@dataclass(frozen=True)
class SelectNttImplementationsPass:
    """Ask the selected NTT implementation to complete missing schedule choices.

    Caller assignments and supplied Tensor operands remain authoritative.
    Missing transform tables become ordinary material references, whose data
    must be supplied separately. Operations with insufficient selection facts
    remain unchanged and report why their choice was deferred.
    """

    registry: OperationImplementationRegistry
    name: str = "select-ntt-implementations"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        workspace = compilation.workspace
        matched = transformed = inserted = skipped = 0
        decisions = []
        for operation in program_operations(program):
            if not isinstance(operation, _TRANSFORMS):
                continue
            matched += 1
            logical = isinstance(operation, _NTT_TYPES)
            requested = requested_implementation(operation)
            if not logical and requested is not None:
                skipped += 1
                continue
            operation_type = (
                type(operation)
                if logical
                else ntt.CoefficientStandardToNttMontgomeryOp
            )
            if operation_type in self.registry.operation_types:
                providers = (
                    self.registry.resolve_type(
                        operation_type, requested=requested if logical else None
                    ),
                )
            elif requested is not None:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        requested,
                        details=(
                            "assigned implementation is not registered here",
                        ),
                    )
                )
                continue
            else:
                # A region implementation can support NTT without registering
                # an individual-operation executor.
                providers = self.registry.implementations
            selectors = tuple(
                choose
                for provider in providers
                if callable(
                    choose := getattr(provider, "select_ntt_schedule", None)
                )
            )
            if not selectors:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        details=("no registered NTT schedule selector",),
                    )
                )
                continue
            state = getattr(
                getattr(operation.operands[0].type, "state", None), "data", {}
            )
            device = state.get("device")
            device_type = (
                device.data.split(":", 1)[0]
                if isinstance(device, StringAttr) and device.data != "unknown"
                else None
            )
            size = state.get("ring_dimension")
            if not isinstance(size, IntegerAttr):
                shape = state.get("shape")
                size = (
                    tuple(shape)[-1]
                    if isinstance(shape, ArrayAttr) and len(shape)
                    else None
                )
            log_n = (
                int(size.value.data).bit_length() - 1
                if isinstance(size, IntegerAttr) and size.value.data > 0
                else None
            )
            selected = None
            for choose in selectors:
                selected = choose(
                    selected=_text(operation, "ntt_backend"),
                    algorithm=_text(operation, "ntt_algorithm"),
                    group_width=_integer(operation, "ntt_group_width"),
                    radix=_integer(operation, "ntt_radix"),
                    tables=_text(operation, "ntt_table_layout"),
                    device_type=device_type,
                    log_n=log_n,
                )
                if selected is not None:
                    break
            if selected is None:
                skipped += 1
                decisions.append(
                    DecisionRecord(
                        operation.name,
                        details=(
                            "NTT selection awaits placement or algorithm constraints",
                        ),
                    )
                )
                continue
            selected = cast(dict[str, Any], selected)
            changed = False
            for key, value in (
                ("ntt_backend", selected["ntt_backend"]),
                ("ntt_table_layout", selected["ntt_table_layout"]),
            ):
                if operation.attributes.get(key) != StringAttr(value):
                    operation.attributes[key] = StringAttr(value)
                    changed = True
            if logical:
                count = selected["table_count"]
                existing = tuple(cast(ntt._NttOp, operation).tables)
                if len(existing) > count:
                    raise ValueError(
                        f"{operation.name} has more table operands than {selected['ntt_backend']} consumes"
                    )
                config_attr = operation.attributes.get("ckks_config")
                config = (
                    CkksConfig.parse(json.loads(config_attr.data))
                    if isinstance(config_attr, StringAttr)
                    else workspace.get(CkksConfig)
                )
                ids = state.get("prime_ids")
                prime_ids = (
                    [int(x.value.data) for x in ids]
                    if isinstance(ids, ArrayAttr)
                    and all(isinstance(x, IntegerAttr) for x in ids)
                    else []
                )
                inverse = isinstance(
                    operation,
                    (
                        ntt.NttMontgomeryToCoefficientStandardOp,
                        ntt.NttMontgomeryToCoefficientMontgomeryOp,
                    ),
                )
                references = []
                for index in range(len(existing) + 1, count + 1):
                    symbol = material_symbol(
                        operation,
                        f"ntt/{'inverse' if inverse else 'forward'}/{index}",
                    )
                    reference = core.MaterialRefOp(
                        core.MessageType(), symbol=symbol
                    )
                    references.append(reference)
                    # Reserve each symbol before creating the next reference.
                    fields = dict(
                        prime_ids=prime_ids,
                        direction="inverse" if inverse else "forward",
                        index=index,
                        ntt_backend=selected["ntt_backend"],
                    )
                    description = (
                        parameter_description(config, "ntt_table", **fields)
                        if isinstance(config, CkksConfig)
                        else {"kind": "ntt_table", **fields}
                    )
                    program.set_material_description(symbol, description)
                if references:
                    Rewriter.insert_op(
                        references, InsertPoint.before(operation)
                    )
                    operation.operands = (
                        *operation.operands,
                        *(r.value for r in references),
                    )
                    inserted += len(references)
                    changed = True
            if changed:
                transformed += 1
            else:
                skipped += 1
            decisions.append(
                DecisionRecord(
                    operation.name,
                    selected["ntt_backend"],
                    details=("existing Tensor operands preserved",),
                )
            )
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
