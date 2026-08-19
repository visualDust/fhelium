"""PyTorch FX capture into the canonical mixed-dialect xDSL ``Program``."""

from __future__ import annotations

import inspect
import json
import operator
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import torch
from torch import fx
from torch.fx.proxy import TraceError
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Operation, SSAValue

from ._dialect import (
    MaterialRefOp,
    create_ir_context,
    create_operation,
    value_type,
)
from ._errors import JitInputError, JitTraceError
from ._ir import ValueRole, encode_literal, target_symbol
from ._program import Program
from ._specs import InputSpec
from ._workspace import Workspace

_MATERIALS_KEY = "materials"
ReturnT = TypeVar("ReturnT")


@dataclass(frozen=True)
class CaptureResult(Generic[ReturnT]):
    """Pair one captured Program with retained frontend and runtime state.

    ``program`` is the same canonical Program used by textual import and direct
    construction and contains serializable IR. Captured Tensor constants appear
    in the graph as symbolic material operations; their live values are retained
    in ``workspace['materials']``. ``function``, ``signature``, ``specs``, and
    ``fx_code`` are capture-result attachments. The Program itself remains
    source-independent.
    """

    program: Program
    workspace: MutableMapping[Any, Any]
    function: Callable[..., ReturnT]
    signature: inspect.Signature
    specs: Mapping[str, InputSpec]
    fx_code: str

    @property
    def runtime_signature(self) -> inspect.Signature:
        """Return the callable signature after specialized static inputs."""

        return self.signature.replace(
            parameters=[
                parameter
                for name, parameter in self.signature.parameters.items()
                if self.specs[name].role != "static"
            ]
        )

    def reference(self, *args: object, **kwargs: object) -> ReturnT:
        """Execute the captured Python callable with static inputs restored."""

        dynamic = self.runtime_signature.bind(*args, **kwargs)
        dynamic.apply_defaults()
        full = self.signature.bind_partial()
        for name in self.signature.parameters:
            spec = self.specs[name]
            full.arguments[name] = (
                spec.static_value
                if spec.role == "static"
                else dynamic.arguments[name]
            )
        return self.function(*full.args, **full.kwargs)


@dataclass(frozen=True)
class _Primitive:
    name: str
    form: str


_FUNCTION_PRIMITIVES: dict[object, _Primitive] = {
    operator.add: _Primitive("fhelium.semantic.add", "binary"),
    operator.iadd: _Primitive("fhelium.semantic.add", "binary"),
    torch.add: _Primitive("fhelium.semantic.add", "binary"),
    operator.sub: _Primitive("fhelium.semantic.subtract", "binary"),
    operator.isub: _Primitive("fhelium.semantic.subtract", "binary"),
    torch.sub: _Primitive("fhelium.semantic.subtract", "binary"),
    torch.subtract: _Primitive("fhelium.semantic.subtract", "binary"),
    operator.mul: _Primitive("fhelium.semantic.multiply", "binary"),
    operator.imul: _Primitive("fhelium.semantic.multiply", "binary"),
    torch.mul: _Primitive("fhelium.semantic.multiply", "binary"),
    torch.multiply: _Primitive("fhelium.semantic.multiply", "binary"),
    operator.neg: _Primitive("fhelium.semantic.negate", "unary"),
    torch.neg: _Primitive("fhelium.semantic.negate", "unary"),
    torch.negative: _Primitive("fhelium.semantic.negate", "unary"),
    torch.roll: _Primitive("fhelium.semantic.roll", "roll"),
}
_METHOD_PRIMITIVES: dict[str, _Primitive] = {
    "add": _Primitive("fhelium.semantic.add", "binary"),
    "__add__": _Primitive("fhelium.semantic.add", "binary"),
    "sub": _Primitive("fhelium.semantic.subtract", "binary"),
    "subtract": _Primitive("fhelium.semantic.subtract", "binary"),
    "__sub__": _Primitive("fhelium.semantic.subtract", "binary"),
    "mul": _Primitive("fhelium.semantic.multiply", "binary"),
    "multiply": _Primitive("fhelium.semantic.multiply", "binary"),
    "__mul__": _Primitive("fhelium.semantic.multiply", "binary"),
    "neg": _Primitive("fhelium.semantic.negate", "unary"),
    "negative": _Primitive("fhelium.semantic.negate", "unary"),
    "roll": _Primitive("fhelium.semantic.roll", "roll"),
}


def _validate_specs(
    signature: inspect.Signature,
    inputs: Mapping[str, InputSpec],
) -> dict[str, InputSpec]:
    parameters = signature.parameters
    variadic = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        raise JitInputError(
            "JIT functions cannot declare *args or **kwargs: "
            + ", ".join(variadic)
        )
    missing = [name for name in parameters if name not in inputs]
    extra = [name for name in inputs if name not in parameters]
    if missing or extra:
        raise JitInputError(
            "JIT input specifications differ from the function signature: "
            f"missing={missing}, extra={extra}"
        )
    specs = {name: inputs[name] for name in parameters}
    for name, spec in specs.items():
        if not isinstance(spec, InputSpec):
            raise JitInputError(
                f"JIT input {name!r} must use encrypted(), message(), "
                f"plaintext(), or static(), got {type(spec).__name__}"
            )
    return specs


def _fetch_attr(root: object, target: str) -> object:
    value = root
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec_data(spec: InputSpec) -> dict[str, object]:
    return {
        "role": spec.role,
        "level": spec.level,
        "scale": spec.scale,
        "slots": spec.slots,
        "batch_mode": spec.batch_mode,
        "static_value": encode_literal(spec.static_value),
    }


def _dependency_nodes(value: object) -> tuple[fx.Node, ...]:
    if isinstance(value, fx.Node):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(node for item in value for node in _dependency_nodes(item))
    if isinstance(value, Mapping):
        return tuple(
            node for item in value.values() for node in _dependency_nodes(item)
        )
    if isinstance(value, slice):
        return tuple(
            node
            for item in (value.start, value.stop, value.step)
            for node in _dependency_nodes(item)
        )
    return ()


def _result_role(dependencies: tuple[ValueRole, ...]) -> ValueRole:
    for role in ("encrypted", "plaintext", "message", "static"):
        if role in dependencies:
            return role  # type: ignore[return-value]
    return "static"


class _Emitter:
    def __init__(
        self,
        graph_module: fx.GraphModule,
        function: Callable[..., object],
        signature: inspect.Signature,
        specs: Mapping[str, InputSpec],
        workspace: MutableMapping[Any, Any],
    ) -> None:
        self.graph_module = graph_module
        self.function = function
        self.signature = signature
        self.specs = specs
        self.workspace = workspace
        self.ir_context = create_ir_context()
        self.values: dict[fx.Node, SSAValue] = {}
        self.roles: dict[fx.Node, ValueRole] = {}
        self.static_values: dict[fx.Node, object] = {}
        self.placeholder_index = 0
        self.runtime_index = 0
        self.output_descriptor: object | None = None
        self._output_ssa_values: tuple[SSAValue, ...] = ()

        material_values = workspace.setdefault(_MATERIALS_KEY, {})
        if not isinstance(material_values, MutableMapping):
            raise TypeError("workspace['materials'] must be a mutable mapping")
        self.materials: MutableMapping[str, object] = material_values

        runtime_specs = [
            spec for spec in specs.values() if spec.role != "static"
        ]
        self.block = Block(
            arg_types=[
                value_type(
                    spec.role,
                    {"input_spec": StringAttr(_json(_spec_data(spec)))},
                )
                for spec in runtime_specs
            ]
        )

    def emit(self) -> Program:
        for node in self.graph_module.graph.nodes:
            self.emit_node(node)
        if self.output_descriptor is None:
            raise JitTraceError("JIT capture produced no output")
        return_values = self._output_ssa_values
        self.block.add_op(ReturnOp(*return_values))
        runtime_names = [
            name for name, spec in self.specs.items() if spec.role != "static"
        ]
        module_attributes = {
            "fhelium.frontend": StringAttr("torch.fx"),
            "fhelium.program_name": StringAttr(
                getattr(self.function, "__name__", "main")
            ),
            "fhelium.input_names": ArrayAttr(
                StringAttr(name) for name in runtime_names
            ),
            "fhelium.input_specs": StringAttr(
                _json(
                    {
                        name: _spec_data(spec)
                        for name, spec in self.specs.items()
                    }
                )
            ),
            "fhelium.output_structure": StringAttr(
                _json(self.output_descriptor)
            ),
        }
        return Program.from_function(
            self.block,
            [value.type for value in return_values],
            name="main",
            module_attributes=module_attributes,
        )

    def emit_node(self, node: fx.Node) -> None:
        if node.op == "placeholder":
            self._emit_placeholder(node)
            return
        if node.op == "get_attr":
            self._emit_get_attr(node)
            return
        if node.op == "output":
            if len(node.args) != 1:
                raise JitTraceError("FX output must contain one root value")
            values: list[SSAValue] = []
            self.output_descriptor = self._encode_output(node.args[0], values)
            self._output_ssa_values = tuple(values)
            return
        if node.op in {"call_function", "call_method", "call_module"}:
            if self._emit_static_guard(node):
                return
            self._emit_call(node)
            return
        raise JitTraceError(
            f"Unsupported FX structural node kind {node.op!r} at {node.name!r}"
        )

    def _emit_placeholder(self, node: fx.Node) -> None:
        parameters = tuple(self.signature.parameters)
        if self.placeholder_index >= len(parameters):
            raise JitTraceError("FX produced an unexpected placeholder")
        name = parameters[self.placeholder_index]
        self.placeholder_index += 1
        spec = self.specs[name]
        self.roles[node] = spec.role
        if spec.role == "static":
            self.static_values[node] = spec.static_value
            return
        argument = self.block.args[self.runtime_index]
        self.runtime_index += 1
        argument.name_hint = name
        self.values[node] = argument

    def _emit_get_attr(self, node: fx.Node) -> None:
        value = _fetch_attr(self.graph_module, str(node.target))
        if isinstance(value, torch.Tensor):
            result = self._emit_tensor_material(
                value, symbol=f"capture/tensor/{node.target}"
            )
            self.values[node] = result
            self.roles[node] = "message"
            result.name_hint = node.name
            return
        self.static_values[node] = value
        result = self._emit_constant(value, name_hint=node.name)
        self.values[node] = result
        self.roles[node] = "static"

    def _emit_static_guard(self, node: fx.Node) -> bool:
        if node.op == "call_function" and node.target is operator.eq:
            dependencies = _dependency_nodes(node.args)
            if dependencies and all(
                item in self.static_values for item in dependencies
            ):
                left = self._static_argument(node.args[0])
                right = self._static_argument(node.args[1])
                self.static_values[node] = left == right
                self.roles[node] = "static"
                return True
        if node.op == "call_function" and node.target is torch._assert:
            condition = self._static_argument(node.args[0])
            if not bool(condition):
                message = (
                    node.args[1]
                    if len(node.args) > 1
                    else "static assertion failed"
                )
                raise JitInputError(str(message))
            self.static_values[node] = None
            self.roles[node] = "static"
            return True
        return False

    def _emit_call(self, node: fx.Node) -> None:
        dependency_roles: list[ValueRole] = []
        for dependency in _dependency_nodes((node.args, node.kwargs)):
            dependency_role = self.roles.get(dependency)
            if dependency_role is not None:
                dependency_roles.append(dependency_role)
        role = _result_role(tuple(dependency_roles))
        primitive = self._primitive(node)
        if role == "encrypted" and primitive is not None:
            operation = self._emit_primitive(node, primitive)
        else:
            operation = self._emit_torch_call(node, role)
        operation.results[0].name_hint = node.name
        self.block.add_op(operation)
        self.values[node] = operation.results[0]
        self.roles[node] = role

    def _primitive(self, node: fx.Node) -> _Primitive | None:
        if node.op == "call_function":
            return _FUNCTION_PRIMITIVES.get(node.target)
        if node.op == "call_method":
            return _METHOD_PRIMITIVES.get(str(node.target))
        return None

    def _emit_primitive(
        self, node: fx.Node, primitive: _Primitive
    ) -> Operation:
        if primitive.form == "roll":
            normalized = self._roll(node)
            if normalized is None:
                return self._emit_torch_call(node, "encrypted")
            source, shift, dimension = normalized
            return create_operation(
                self.ir_context,
                primitive.name,
                operands=[source],
                result_types=[value_type("encrypted")],
                attributes={
                    "shift": IntegerAttr(shift, 64),
                    **(
                        {}
                        if dimension is None
                        else {"dimension": IntegerAttr(dimension, 64)}
                    ),
                },
            )
        expected = 2 if primitive.form == "binary" else 1
        if len(node.args) != expected or node.kwargs:
            return self._emit_torch_call(node, "encrypted")
        operands = [
            self._as_ssa(argument, path=f"{node.name}/operand/{index}")
            for index, argument in enumerate(node.args)
        ]
        return create_operation(
            self.ir_context,
            primitive.name,
            operands=operands,
            result_types=[value_type("encrypted")],
            attributes={
                "fhelium.frontend.target": StringAttr(self._call_target(node))
            },
        )

    def _emit_torch_call(self, node: fx.Node, role: ValueRole) -> Operation:
        operands: list[SSAValue] = []
        arguments = self._encode_argument(
            node.args, operands, f"{node.name}/args"
        )
        keywords = self._encode_argument(
            node.kwargs, operands, f"{node.name}/kwargs"
        )
        call_kind = (
            "function"
            if node.op == "call_function"
            else "method"
            if node.op == "call_method"
            else "module"
        )
        return create_operation(
            self.ir_context,
            "torch.call",
            operands=operands,
            result_types=[value_type(role)],
            attributes={
                "fhelium.call.kind": StringAttr(call_kind),
                "fhelium.call.target": StringAttr(self._call_target(node)),
                "fhelium.call.arguments": StringAttr(
                    _json({"args": arguments, "kwargs": keywords})
                ),
                "fhelium.role": StringAttr(role),
            },
        )

    def _call_target(self, node: fx.Node) -> str:
        if node.op == "call_module":
            module = self.graph_module.get_submodule(str(node.target))
            return f"{node.target}:{target_symbol(type(module))}"
        return target_symbol(node.target)

    def _roll(self, node: fx.Node) -> tuple[SSAValue, int, int | None] | None:
        if not node.args:
            return None
        positional = list(node.args[1:])
        shifts = node.kwargs.get(
            "shifts", positional.pop(0) if positional else None
        )
        dims = node.kwargs.get(
            "dims", positional.pop(0) if positional else None
        )
        if positional or set(node.kwargs) - {"shifts", "dims"}:
            return None
        static_shifts = self._maybe_static(shifts)
        static_dims = self._maybe_static(dims)
        if isinstance(static_shifts, bool) or not isinstance(
            static_shifts, int
        ):
            return None
        if static_dims not in (None, -1):
            return None
        try:
            source = self._as_ssa(node.args[0], path=f"{node.name}/source")
        except JitTraceError:
            return None
        return source, static_shifts, None if static_dims is None else -1

    def _as_ssa(self, value: object, *, path: str) -> SSAValue:
        if isinstance(value, fx.Node):
            if value in self.values:
                return self.values[value]
            if value in self.static_values:
                return self._emit_constant(
                    self.static_values[value], name_hint="literal"
                )
            raise JitTraceError(f"FX value {value.name!r} is unresolved")
        if isinstance(value, torch.Tensor):
            return self._emit_tensor_material(
                value, symbol=f"capture/tensor/{path}"
            )
        return self._emit_constant(value, name_hint="literal")

    def _emit_constant(self, value: object, *, name_hint: str) -> SSAValue:
        if isinstance(value, torch.Tensor):
            return self._emit_tensor_material(
                value, symbol=f"capture/tensor/{name_hint}"
            )
        operation = create_operation(
            self.ir_context,
            "fhelium.constant",
            result_types=[value_type("static")],
            attributes={
                "fhelium.literal": StringAttr(_json(encode_literal(value)))
            },
        )
        operation.results[0].name_hint = name_hint
        self.block.add_op(operation)
        return operation.results[0]

    def _emit_tensor_material(
        self, value: torch.Tensor, *, symbol: str
    ) -> SSAValue:
        candidate = symbol
        suffix = 2
        while candidate in self.materials:
            candidate = f"{symbol}/{suffix}"
            suffix += 1
        snapshot = value.detach().clone()
        self.materials[candidate] = snapshot
        descriptor = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "layout": str(value.layout),
            "device": str(value.device),
        }
        operation = MaterialRefOp(
            value_type(
                "message",
                {"material": StringAttr(_json(descriptor))},
            ),
            symbol=candidate,
            kind="tensor",
            attributes={
                "fhelium.material.descriptor": StringAttr(_json(descriptor))
            },
        )
        self.block.add_op(operation)
        return operation.results[0]

    def _encode_argument(
        self,
        value: object,
        operands: list[SSAValue],
        path: str,
    ) -> object:
        if isinstance(value, fx.Node):
            if value in self.values:
                index = len(operands)
                operands.append(self.values[value])
                return {"kind": "ssa", "operand": index}
            if value in self.static_values:
                return {
                    "kind": "literal",
                    "value": encode_literal(self.static_values[value]),
                }
            raise JitTraceError(f"FX value {value.name!r} is unresolved")
        if isinstance(value, torch.Tensor):
            index = len(operands)
            operands.append(
                self._emit_tensor_material(
                    value, symbol=f"capture/tensor/{path}"
                )
            )
            return {"kind": "ssa", "operand": index}
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [
                    self._encode_argument(item, operands, f"{path}/{index}")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [
                    self._encode_argument(item, operands, f"{path}/{index}")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "entries": [
                    [
                        encode_literal(key),
                        self._encode_argument(item, operands, f"{path}/{key}"),
                    ]
                    for key, item in value.items()
                ],
            }
        if isinstance(value, slice):
            return {
                "kind": "slice",
                "start": self._encode_argument(
                    value.start, operands, f"{path}/start"
                ),
                "stop": self._encode_argument(
                    value.stop, operands, f"{path}/stop"
                ),
                "step": self._encode_argument(
                    value.step, operands, f"{path}/step"
                ),
            }
        return {"kind": "literal", "value": encode_literal(value)}

    def _encode_output(self, value: object, values: list[SSAValue]) -> object:
        if isinstance(value, fx.Node):
            if value in self.values:
                index = len(values)
                values.append(self.values[value])
                return {"kind": "ssa", "result": index}
            if value in self.static_values:
                result = self._emit_constant(
                    self.static_values[value], name_hint="output_literal"
                )
                index = len(values)
                values.append(result)
                return {"kind": "ssa", "result": index}
            raise JitTraceError(f"FX output {value.name!r} is unresolved")
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [self._encode_output(item, values) for item in value],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [self._encode_output(item, values) for item in value],
            }
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "entries": [
                    [encode_literal(key), self._encode_output(item, values)]
                    for key, item in value.items()
                ],
            }
        result = self._as_ssa(value, path="output")
        index = len(values)
        values.append(result)
        return {"kind": "ssa", "result": index}

    def _static_argument(self, value: object) -> object:
        if isinstance(value, fx.Node):
            if value not in self.static_values:
                raise JitTraceError(
                    f"Expected static FX value, got {value.name!r}"
                )
            return self.static_values[value]
        return value

    def _maybe_static(self, value: object) -> object:
        if isinstance(value, fx.Node) and value in self.static_values:
            return self.static_values[value]
        return value


def capture(
    function: Callable[..., ReturnT],
    *,
    inputs: Mapping[str, InputSpec],
    workspace: MutableMapping[Any, Any] | None = None,
) -> CaptureResult[ReturnT]:
    """Trace a Python callable into the canonical mixed-dialect Program.

    ``inputs`` declares every parameter's encrypted, message, plaintext, or
    static role. Capture specializes static values, records Tensor constants as
    symbolic materials in the retained Workspace, lowers recognized arithmetic
    to semantic FHElium operations, and preserves other FX calls as
    ``torch.call`` operations. This structural frontend accepts pure-public and
    partially lowered graphs. Readiness and execution evaluate CKKS
    state, handlers, bindings, services, parameters, and keys for a selected
    entry.
    """

    if not callable(function):
        raise TypeError(
            f"JIT trace expects a callable, got {type(function).__name__}"
        )
    signature = inspect.signature(function)
    specs = _validate_specs(signature, inputs)
    concrete_args = {
        name: spec.static_value
        for name, spec in specs.items()
        if spec.role == "static"
    }
    try:
        graph_module = fx.symbolic_trace(
            function,
            concrete_args=concrete_args or None,
        )
    except TraceError as error:
        raise JitTraceError(
            "JIT cannot capture data-dependent Python control flow, "
            f"iteration, or truth conversion: {error}"
        ) from error
    except Exception as error:
        raise JitTraceError(f"JIT FX capture failed: {error}") from error

    retained = Workspace() if workspace is None else workspace
    if not isinstance(retained, MutableMapping):
        raise TypeError("JIT capture workspace must be a mutable mapping")
    program = _Emitter(
        graph_module,
        function,
        signature,
        specs,
        retained,
    ).emit()
    return CaptureResult(
        program=program,
        workspace=retained,
        function=function,
        signature=signature,
        specs=MappingProxyType(dict(specs)),
        fx_code=graph_module.code,
    )


__all__ = ["CaptureResult", "capture"]
