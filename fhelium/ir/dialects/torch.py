"""Registered structural support for preserved PyTorch calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_attr_def,
    result_def,
    var_operand_def,
)

from .._operation_catalog import (
    OperationSpec,
    argument_descriptor_diagnostics,
    decode_json_attribute,
    flat_operation,
    registered_operation_spec,
    required_string_attribute,
    unsupported_attributes,
)


@irdl_op_definition
class CallOp(IRDLOperation):
    """Preserve one encoded PyTorch function, method, or module call.

    Registration fixes the flat variadic-operand, single-result structure. The
    encoded target and argument descriptor remain inert until a selected
    consumer validates and authorizes them.
    """

    name = "torch.call"
    arguments = var_operand_def()
    result = result_def()
    kind = opt_attr_def(StringAttr, attr_name="fhelium.call.kind")
    target = opt_attr_def(StringAttr, attr_name="fhelium.call.target")
    argument_descriptor = opt_attr_def(
        StringAttr, attr_name="fhelium.call.arguments"
    )
    role = opt_attr_def(StringAttr, attr_name="fhelium.role")

    def __init__(
        self,
        operands: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        kind: str | StringAttr | None = None,
        target: str | StringAttr | None = None,
        argument_descriptor: str | StringAttr | None = None,
        role: str | StringAttr | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        for name, value in (
            ("fhelium.call.kind", kind),
            ("fhelium.call.target", target),
            ("fhelium.call.arguments", argument_descriptor),
            ("fhelium.role", role),
        ):
            attrs[name] = StringAttr(value) if isinstance(value, str) else value
        super().__init__(
            operands=[operands], result_types=[result_type], attributes=attrs
        )


def _call_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_operation(operation))
    unsupported = unsupported_attributes(
        operation,
        (
            "fhelium.call.target",
            "fhelium.call.kind",
            "fhelium.call.arguments",
            "fhelium.role",
        ),
    )
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if required_string_attribute(operation, "fhelium.call.target") is None:
        diagnostics.append("requires a nonempty string call target")
    kind = required_string_attribute(operation, "fhelium.call.kind")
    if kind not in {"function", "method", "module"}:
        diagnostics.append(f"unsupported call kind {kind!r}")
    descriptor, error = decode_json_attribute(
        operation, "fhelium.call.arguments"
    )
    if error is not None:
        diagnostics.append(error)
    elif not isinstance(descriptor, dict):
        diagnostics.append("call argument metadata must be a JSON object")
    else:
        diagnostics.extend(
            argument_descriptor_diagnostics(
                descriptor.get("args"), len(operation.operands)
            )
        )
        diagnostics.extend(
            argument_descriptor_diagnostics(
                descriptor.get("kwargs"), len(operation.operands)
            )
        )
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        CallOp,
        "auxiliary",
        effect="opaque",
        validator=_call_specification,
    ),
)
"""Semantic specifications owned by the structural Torch dialect."""


Torch = Dialect("torch", [CallOp], [])
"""Structural Torch call dialect used by FHElium source capture."""


__all__ = ["CallOp", "OPERATION_SPECS", "Torch"]
