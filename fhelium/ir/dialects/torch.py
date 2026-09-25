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
    traits_def,
)

from xdsl.traits import Pure

from .._operation_catalog import (
    OperationSpec,
    argument_descriptor_diagnostics,
    decode_json_attribute,
    flat_operation,
    registered_operation_spec,
    required_string_attribute,
    unsupported_attributes,
)
from .._dependencies import (
    DependencyKind,
    OperationDependencies,
    operand_dependencies,
)


class _CallOp(IRDLOperation):
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


@irdl_op_definition
class CallOp(_CallOp):
    """Preserve an opaque frontend call until an implementation is selected."""

    name = "torch.call"


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


TENSOR_FUNCTION_TARGETS = frozenset(
    {
        "torch.add",
        "torch.sub",
        "torch.mul",
        "torch.div",
        "torch.neg",
        "torch.pow",
        "torch.roll",
        "torch.matmul",
        "torch.linalg.vector_norm",
        "torch.sum",
        "torch.mean",
        "torch.sqrt",
        "torch.rsqrt",
        "torch.exp",
        "torch.log",
        "torch.sin",
        "torch.cos",
        "torch.tanh",
        "torch.sigmoid",
        "torch.relu",
        "torch.reshape",
        "torch.transpose",
        "torch.permute",
        "torch.clone",
        "torch.stack",
        "torch.cat",
        "torch.getitem",
        "torch.Tensor.to",
        "torch.Tensor.contiguous",
    }
)


@irdl_op_definition
class TensorCallOp(_CallOp):
    """Apply one supported out-of-place public Tensor operation.

    Targets have ordinary Torch semantics. No encrypted input interpretation,
    arbitrary importer, mutation through ``out``, or random sampling is implied.
    """

    name = "torch.tensor_call"
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe data reads in the registered target's Tensor coordinates."""
        target = required_string_attribute(self, "fhelium.call.target")
        if target in {
            "torch.add",
            "torch.sub",
            "torch.mul",
            "torch.div",
            "torch.pow",
        }:
            kind: DependencyKind = "reindexed"
        elif target in {
            "torch.neg",
            "torch.sqrt",
            "torch.rsqrt",
            "torch.exp",
            "torch.log",
            "torch.sin",
            "torch.cos",
            "torch.tanh",
            "torch.sigmoid",
            "torch.relu",
            "torch.clone",
            "torch.Tensor.to",
            "torch.Tensor.contiguous",
        }:
            kind = "element"
        elif target in {
            "torch.roll",
            "torch.reshape",
            "torch.transpose",
            "torch.permute",
            "torch.stack",
            "torch.cat",
        }:
            kind = "reindexed"
        elif target in {
            "torch.matmul",
            "torch.sum",
            "torch.mean",
            "torch.linalg.vector_norm",
        }:
            kind = "mixing"
        else:
            return OperationDependencies()
        return operand_dependencies(
            self, tuple(range(len(self.operands))), {"tensor_position": kind}
        )


def _tensor_call_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(_call_specification(operation))
    if required_string_attribute(operation, "fhelium.call.kind") != "function":
        diagnostics.append("Tensor calls require a registered function target")
    if (
        required_string_attribute(operation, "fhelium.call.target")
        not in TENSOR_FUNCTION_TARGETS
    ):
        diagnostics.append("unsupported pure Tensor function")
    descriptor, _ = decode_json_attribute(operation, "fhelium.call.arguments")
    if isinstance(descriptor, dict):
        kwargs = descriptor.get("kwargs")
        if isinstance(kwargs, dict):
            for key, value in kwargs.get("entries", ()):
                if key == "out" and value != {"kind": "literal", "value": None}:
                    diagnostics.append(
                        "Tensor calls cannot mutate an out operand"
                    )
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        TensorCallOp,
        "pointwise",
        effect="pure",
        validator=_tensor_call_specification,
    ),
    registered_operation_spec(
        CallOp,
        "auxiliary",
        effect="opaque",
        validator=_call_specification,
    ),
)
"""Semantic specifications owned by the structural Torch dialect."""


Torch = Dialect("torch", [CallOp, TensorCallOp], [])
"""Structural Torch call dialect used by FHElium source capture."""


__all__ = [
    "CallOp",
    "TensorCallOp",
    "TENSOR_FUNCTION_TARGETS",
    "OPERATION_SPECS",
    "Torch",
]
