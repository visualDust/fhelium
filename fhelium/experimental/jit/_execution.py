"""Readiness analysis and selected-entry Program execution.

``check_readiness`` performs a side-effect-free capability and executable-schema
check for one entry. ``run_program`` combines that report with runtime argument
binding and online-encryption requirements, then interprets only the selected
single-block entry. Import, export, and ordinary pass execution remain structural
operations and invoke neither operation handlers nor binding resolvers.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation, SSAValue

from fhelium.core import (
    Ciphertext,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    Plaintext,
    PublicKey,
    RotationKey,
)
from fhelium.engine import CkksEngine

from ._analysis import (
    RUNTIME_CKKS_OPERATION_NAMES,
    ProgramRequirements,
    analyze_requirements,
)
from ._dialect import (
    DIALECT_VERSION,
    DIALECT_VERSION_ATTRIBUTE,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTRIBUTE,
    MaterialRefOp,
    ResourceRefOp,
    operation_name,
    value_role,
)
from ._errors import JitError, JitInputError, JitPassError
from ._program import Program
from ._workspace import Workspace

OperationHandler = Callable[
    [Operation, tuple[object, ...], MutableMapping[Any, Any]], object
]
BindingResolver = Callable[
    [str, str | None, object, MutableMapping[Any, Any]], object
]
# OperationHandler receives extension IR, evaluated operands, and the retained
# Workspace. BindingResolver receives symbol, optional kind, raw binding, and
# that same Workspace when the reference operation executes.


@dataclass(frozen=True)
class ReadinessDiagnostic:
    """Describe one concrete missing or malformed execution requirement."""

    code: str
    message: str
    subject: str | None = None
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class ReadinessReport:
    """Carry selected-entry requirements and readiness diagnostics.

    ``runnable`` is true exactly when ``diagnostics`` contains no item whose
    severity is ``"error"``. Warnings, including a missing version marker, are
    preserved as evidence and permit execution.
    """

    requirements: ProgramRequirements
    diagnostics: tuple[ReadinessDiagnostic, ...]

    @property
    def runnable(self) -> bool:
        """Whether this report contains no execution-blocking diagnostic."""

        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def missing_operations(self) -> frozenset[str]:
        """Return operation or Torch targets lacking compatible handlers."""

        return frozenset(
            item.subject
            for item in self.diagnostics
            if item.code in {"missing-handler", "missing-torch-handler"}
            and item.subject is not None
        )

    @property
    def missing_materials(self) -> frozenset[str]:
        """Return symbolic material names lacking Workspace bindings."""

        return frozenset(
            item.subject
            for item in self.diagnostics
            if item.code == "missing-material" and item.subject is not None
        )

    @property
    def missing_resources(self) -> frozenset[str]:
        """Return symbolic resource names lacking Workspace bindings."""

        return frozenset(
            item.subject
            for item in self.diagnostics
            if item.code == "missing-resource" and item.subject is not None
        )


class ProgramNotReadyError(JitError, RuntimeError):
    """Reject one ``Program.run(...)`` request while preserving its full report."""

    def __init__(self, report: ReadinessReport) -> None:
        self.report = report
        detail = "; ".join(item.message for item in report.diagnostics)
        super().__init__(f"JIT Program is not ready to run: {detail}")


_SAFE_FUNCTION_TARGETS: dict[str, Callable[..., object]] = {}
for _function in (
    torch.add,
    torch.subtract,
    torch.sub,
    torch.multiply,
    torch.mul,
    torch.neg,
    torch.negative,
    torch.roll,
    torch.diagonal,
    torch.cat,
    torch.concat,
    torch.sin,
    torch.cos,
    torch.exp,
    torch.reshape,
    torch.flatten,
    torch.transpose,
    operator.add,
    operator.sub,
    operator.mul,
    operator.neg,
    operator.getitem,
):
    _module = getattr(_function, "__module__", type(_function).__module__)
    _qualname = getattr(_function, "__qualname__", None)
    _name = getattr(_function, "__name__", type(_function).__qualname__)
    if not isinstance(_qualname, str) or not _qualname:
        _qualname = _name
    _SAFE_FUNCTION_TARGETS[f"{_module}.{_qualname}"] = _function
    _SAFE_FUNCTION_TARGETS[f"{_module}.{_name}"] = _function
    if _module == "torch":
        _SAFE_FUNCTION_TARGETS[f"torch.{_name}"] = _function

_SAFE_TENSOR_METHODS = frozenset(
    {
        "add",
        "clone",
        "contiguous",
        "cos",
        "diagonal",
        "exp",
        "expand",
        "flatten",
        "mul",
        "neg",
        "permute",
        "repeat",
        "reshape",
        "sin",
        "squeeze",
        "sub",
        "transpose",
        "unsqueeze",
        "view",
    }
)


def _mapping(workspace: Mapping[Any, Any], key: str) -> Mapping[Any, Any]:
    value = workspace.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _handlers(workspace: Mapping[Any, Any]) -> Mapping[str, object]:
    return _mapping(workspace, "handlers")


def _torch_handlers(workspace: Mapping[Any, Any]) -> Mapping[str, object]:
    return _mapping(workspace, "torch_handlers")


def _string_attr(operation: Operation, *names: str) -> str | None:
    for name in names:
        value = operation.attributes.get(name)
        if isinstance(value, StringAttr) and value.data:
            return value.data
    return None


def _int_attr(operation: Operation, *names: str) -> int | None:
    for name in names:
        value = operation.attributes.get(name)
        if isinstance(value, IntegerAttr):
            return int(value.value.data)
    return None


def _torch_target(operation: Operation) -> str | None:
    return _string_attr(operation, "fhelium.call.target")


def _torch_call_kind(operation: Operation) -> str:
    return _string_attr(operation, "fhelium.call.kind") or "function"


def _resolve_torch_target(
    target: str,
    kind: str,
    workspace: Mapping[Any, Any],
) -> Callable[..., object] | None:
    candidate = _torch_handlers(workspace).get(target)
    if callable(candidate):
        return candidate
    if kind == "method" and target in _SAFE_TENSOR_METHODS:

        def call_tensor_method(*args: object, **kwargs: object) -> object:
            if not args or not isinstance(args[0], torch.Tensor):
                raise JitInputError(
                    f"Safe Torch method {target!r} requires a Tensor receiver"
                )
            return getattr(args[0], target)(*args[1:], **kwargs)

        return call_tensor_method
    return _SAFE_FUNCTION_TARGETS.get(target)


def check_readiness(
    program: Program,
    workspace: Mapping[Any, Any] | None = None,
    *,
    entry: str = "main",
) -> ReadinessReport:
    """Assess whether ``entry`` has the capabilities required to execute.

    The assessment verifies Workspace binding container types, resolver
    callability, Program schema and dialect versions, a single-block entry with
    one final return, entry argument and built-in operation schemas, cleared
    scheduling obligations, callable handlers for extension operations and
    Torch targets, symbolic material/resource membership, a CKKS engine, and
    required rotation/relinearization keys compatible with that engine.

    Readiness reserves the built-in runtime operation names for the interpreter;
    ``workspace['handlers']`` authorizes extension operations only and cannot
    override a built-in schema or implementation. ``torch.call`` operations
    that touch encrypted or plaintext values require a bound callable in
    ``workspace['torch_handlers']``. Pure-public Torch calls may instead use the
    runtime's audited target set.

    This function reads Program and Workspace state without mutation. Material
    and resource bindings are checked by name, while resolver and handler calls,
    runtime argument validation, online-public-key requirements, and operation
    execution occur only in ``run_program``.
    """

    if not isinstance(program, Program):
        raise TypeError("check_readiness expects a Program")
    if workspace is None:
        workspace = {}
    if not isinstance(workspace, Mapping):
        raise TypeError("workspace must be a mapping")

    requirements = analyze_requirements(program, entry=entry)
    diagnostics: list[ReadinessDiagnostic] = []
    for key in ("materials", "resources", "handlers", "torch_handlers"):
        if key in workspace and not isinstance(workspace[key], Mapping):
            diagnostics.append(
                ReadinessDiagnostic(
                    "malformed-workspace-binding",
                    f"workspace[{key!r}] must be a mapping when supplied.",
                    key,
                )
            )
    for key in ("material_resolver", "resource_resolver"):
        if key in workspace and not callable(workspace[key]):
            diagnostics.append(
                ReadinessDiagnostic(
                    "malformed-workspace-binding",
                    f"workspace[{key!r}] must be callable when supplied.",
                    key,
                )
            )
    handlers = _handlers(workspace)
    torch_handlers = _torch_handlers(workspace)

    for attribute_name, supported in (
        (SCHEMA_VERSION_ATTRIBUTE, SCHEMA_VERSION),
        (DIALECT_VERSION_ATTRIBUTE, DIALECT_VERSION),
    ):
        attribute = program.module.attributes.get(attribute_name)
        if attribute is None:
            diagnostics.append(
                ReadinessDiagnostic(
                    "unversioned-program",
                    f"Program omits {attribute_name!r}; current semantics are "
                    "selected by this readiness check.",
                    attribute_name,
                    "warning",
                )
            )
        elif (
            not isinstance(attribute, StringAttr) or attribute.data != supported
        ):
            diagnostics.append(
                ReadinessDiagnostic(
                    "unsupported-program-version",
                    f"Program {attribute_name!r} is unsupported; expected "
                    f"{supported!r}.",
                    attribute_name,
                )
            )

    try:
        from .passes.validate_executable_graph import validate_executable_graph

        validate_executable_graph(
            program,
            entry=entry,
            handled_operations=tuple(
                str(name)
                for name, handler in handlers.items()
                if callable(handler)
            ),
            handled_torch_targets=tuple(
                str(name)
                for name, handler in torch_handlers.items()
                if callable(handler)
            ),
        )
    except JitPassError as error:
        diagnostics.append(
            ReadinessDiagnostic(
                "invalid-executable-graph",
                str(error),
                entry,
            )
        )

    if requirements.return_count is None:
        diagnostics.append(
            ReadinessDiagnostic(
                "missing-entry",
                f"Entry function {entry!r} is missing, multi-block, or has no "
                "unique func.return.",
                entry,
            )
        )

    for malformed in requirements.malformed_references:
        diagnostics.append(
            ReadinessDiagnostic(
                "malformed-reference",
                f"Operation {malformed!r} lacks required symbolic metadata.",
                malformed,
            )
        )

    for operation_name_ in sorted(requirements.unknown_operations):
        if callable(handlers.get(operation_name_)):
            continue
        diagnostics.append(
            ReadinessDiagnostic(
                "missing-handler",
                f"Operation {operation_name_!r} has no execution handler.",
                operation_name_,
            )
        )

    try:
        entry_block = program.entry_block(entry)
    except (KeyError, ValueError):
        entry_block = None
    if entry_block is not None:
        for operation in entry_block.ops:
            if operation_name(operation) != "torch.call":
                continue
            target = _torch_target(operation)
            kind = _torch_call_kind(operation)
            if target is None:
                continue
            touches_fhe = any(
                value_role(value) in {"encrypted", "plaintext"}
                for value in (*operation.operands, *operation.results)
            )
            has_explicit = callable(torch_handlers.get(target))
            if (touches_fhe and not has_explicit) or (
                not touches_fhe
                and _resolve_torch_target(target, kind, workspace) is None
            ):
                diagnostics.append(
                    ReadinessDiagnostic(
                        "missing-torch-handler",
                        f"Torch target {target!r} has no compatible bound "
                        "or audited public execution handler.",
                        target,
                    )
                )

    materials = _mapping(workspace, "materials")
    for symbol in sorted(requirements.materials - set(materials)):
        diagnostics.append(
            ReadinessDiagnostic(
                "missing-material",
                f"Symbolic material {symbol!r} is not bound.",
                symbol,
            )
        )

    resources = _mapping(workspace, "resources")
    for symbol in sorted(requirements.resources - set(resources)):
        diagnostics.append(
            ReadinessDiagnostic(
                "missing-resource",
                f"Symbolic resource {symbol!r} is not bound.",
                symbol,
            )
        )

    engine = workspace.get("engine")
    if requirements.requires_engine and not isinstance(engine, CkksEngine):
        diagnostics.append(
            ReadinessDiagnostic(
                "missing-engine",
                "Explicit CKKS operations require workspace['engine'] to be "
                "a CkksEngine.",
                "engine",
            )
        )

    if requirements.rotation_steps or requirements.requires_relinearization:
        evaluation_keys = workspace.get("evaluation_keys")
        if not isinstance(evaluation_keys, EvaluationKeySet):
            diagnostics.append(
                ReadinessDiagnostic(
                    "missing-evaluation-keys",
                    "Rotation or relinearization operations require an "
                    "EvaluationKeySet in workspace['evaluation_keys'].",
                    "evaluation_keys",
                )
            )
        else:
            try:
                evaluation_keys.require(
                    EvaluationKeyRequirements(
                        rotation_steps=requirements.rotation_steps,
                        requires_relinearization=(
                            requirements.requires_relinearization
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    ReadinessDiagnostic(
                        "incomplete-evaluation-keys",
                        str(error),
                        "evaluation_keys",
                    )
                )
            else:
                if isinstance(engine, CkksEngine):
                    try:
                        for step in requirements.rotation_steps:
                            canonical_step = RotationKey.canonical_step(
                                step,
                                ring_dimension=engine.config.N,
                            )
                            if step != canonical_step:
                                raise ValueError(
                                    "JIT rotation steps must be canonical for "
                                    f"the consuming engine: {step} != "
                                    f"{canonical_step}"
                                )
                            engine.validate_key_switch_key(
                                evaluation_keys.rotations[step]
                            )
                        if requirements.requires_relinearization:
                            assert evaluation_keys.relinearization is not None
                            engine.validate_key_switch_key(
                                evaluation_keys.relinearization
                            )
                    except (KeyError, TypeError, ValueError) as error:
                        diagnostics.append(
                            ReadinessDiagnostic(
                                "incompatible-evaluation-keys",
                                str(error),
                                "evaluation_keys",
                            )
                        )

    return ReadinessReport(requirements, tuple(diagnostics))


def run_program(
    program: Program,
    *args: object,
    workspace: MutableMapping[Any, Any] | None = None,
    entry: str = "main",
    **kwargs: object,
) -> Any:
    """Readiness-check and interpret one selected Program entry.

    The run first computes ``check_readiness`` and rejects its error diagnostics
    through ``ProgramNotReadyError``. It then selects the entry's unique block,
    binds positional and named runtime arguments, and adds an engine-compatible
    PublicKey requirement when an encrypted entry argument receives a Tensor.
    Argument count/name failures raise ``TypeError`` only after the base
    readiness gate succeeds. Input materialization then enforces each role
    specification and performs online encryption where requested.

    Interpretation follows entry-block order. Material and resource reference
    operations fetch their raw Workspace binding and invoke the corresponding
    resolver at that point, once per encountered reference. Built-in constants,
    references, Torch calls, and CKKS operations use reserved interpreter paths;
    callable ``workspace['handlers']`` entries execute extension operations.
    Every handler and resolver receives the retained Workspace and may implement
    caller-defined caching or state updates.

    Captured output metadata reconstructs its Python tuple, list, or mapping
    shape. Without that metadata, zero return operands produce ``None``, one
    produces the value directly, and multiple produce a tuple. Accordingly the
    API result type is dynamic for imported or directly constructed Programs.
    """

    if workspace is None:
        workspace = Workspace()
    if not isinstance(workspace, MutableMapping):
        raise TypeError("workspace must be a mutable mapping")
    report = check_readiness(program, workspace, entry=entry)
    if not report.runnable:
        raise ProgramNotReadyError(report)
    try:
        block = program.entry_block(entry)
    except (KeyError, ValueError):
        raise ProgramNotReadyError(report) from None
    raw_inputs = _bind_argument_values(program, block.args, args, kwargs)
    runtime_diagnostics = list(report.diagnostics)
    requires_online_encryption = any(
        value_role(argument) == "encrypted" and isinstance(value, torch.Tensor)
        for argument, value in zip(block.args, raw_inputs)
    )
    if requires_online_encryption:
        public_key = workspace.get("public_key")
        if not isinstance(public_key, PublicKey):
            runtime_diagnostics.append(
                ReadinessDiagnostic(
                    "missing-public-key",
                    "Online encryption requires a PublicKey in "
                    "workspace['public_key'].",
                    "public_key",
                )
            )
        else:
            engine = workspace.get("engine")
            assert isinstance(engine, CkksEngine)
            try:
                engine.validate_public_key(public_key)
            except (TypeError, ValueError) as error:
                runtime_diagnostics.append(
                    ReadinessDiagnostic(
                        "incompatible-public-key",
                        str(error),
                        "public_key",
                    )
                )
    report = ReadinessReport(report.requirements, tuple(runtime_diagnostics))
    if not report.runnable:
        raise ProgramNotReadyError(report)

    bound_inputs = tuple(
        _materialize_entry_input(argument, value, workspace)
        for argument, value in zip(block.args, raw_inputs)
    )
    environment: dict[SSAValue, object] = dict(zip(block.args, bound_inputs))
    returned: tuple[object, ...] | None = None

    for operation in block.ops:
        if isinstance(operation, ReturnOp):
            returned = tuple(
                environment[value] for value in operation.arguments
            )
            break
        operands = tuple(environment[value] for value in operation.operands)
        results = _execute_operation(operation, operands, workspace)
        _store_results(operation, results, environment)

    if returned is None:
        raise RuntimeError(
            f"Entry function {entry!r} did not execute func.return"
        )
    structured = _reconstruct_output(program, returned)
    if structured is not _NO_OUTPUT_STRUCTURE:
        return structured
    if not returned:
        return None
    if len(returned) == 1:
        return returned[0]
    return returned


def _bind_argument_values(
    program: Program,
    block_arguments: Sequence[SSAValue],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> tuple[object, ...]:
    input_names = _program_input_names(program, block_arguments)
    if len(args) > len(input_names):
        raise TypeError(
            f"Program accepts {len(input_names)} inputs, got {len(args)} positional"
        )
    values: dict[str, object] = dict(zip(input_names, args))
    for name, value in kwargs.items():
        if name not in input_names:
            raise TypeError(f"Unknown Program input {name!r}")
        if name in values:
            raise TypeError(f"Program input {name!r} was supplied twice")
        values[name] = value
    missing = tuple(name for name in input_names if name not in values)
    if missing:
        raise TypeError(f"Missing Program inputs: {missing}")
    return tuple(values[name] for name in input_names)


def _program_input_names(
    program: Program,
    block_arguments: Sequence[SSAValue],
) -> tuple[str, ...]:
    attribute = program.module.attributes.get("fhelium.input_names")
    if isinstance(attribute, ArrayAttr) and all(
        isinstance(item, StringAttr) and item.data for item in attribute
    ):
        names = tuple(item.data for item in attribute)
        if len(names) == len(block_arguments):
            return names
    if isinstance(attribute, StringAttr):
        try:
            decoded_names = json.loads(attribute.data)
        except json.JSONDecodeError:
            decoded_names = None
        if (
            isinstance(decoded_names, list)
            and len(decoded_names) == len(block_arguments)
            and all(isinstance(name, str) and name for name in decoded_names)
        ):
            return tuple(decoded_names)
    return tuple(
        argument.name_hint or f"arg{index}"
        for index, argument in enumerate(block_arguments)
    )


def _materialize_entry_input(
    argument: SSAValue,
    value: object,
    workspace: Mapping[Any, Any],
) -> object:
    role = value_role(argument)
    if role == "plaintext" and not isinstance(value, Plaintext):
        raise JitInputError("Plaintext Program input must be a core Plaintext")
    if role != "encrypted":
        return value
    engine = workspace.get("engine")
    if not isinstance(engine, CkksEngine):
        raise JitInputError(
            "Encrypted Program inputs require a CkksEngine in "
            "workspace['engine']"
        )
    spec = _input_spec(argument)
    level = spec.get("level", 0)
    scale = spec.get("scale")
    if isinstance(level, bool) or not isinstance(level, int):
        raise RuntimeError("Encrypted input_spec level must be an integer")
    if scale is None:
        scale = engine.config.default_scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise RuntimeError("Encrypted input_spec scale must be real or null")
    expected_scale = float(scale)
    slots = spec.get("slots", "full")
    batch_mode = spec.get("batch_mode", "none")

    if isinstance(value, Ciphertext):
        if value.context_id != engine.context.context_id:
            raise JitInputError(
                "Encrypted input belongs to a different CKKS context"
            )
        if (
            value.device != engine.device
            or value.data.dtype != engine.config.torch_dtype
        ):
            raise JitInputError(
                "Encrypted input device or dtype differs from the engine"
            )
        if value.ring_dimension != engine.config.N:
            raise JitInputError(
                "Encrypted input ring dimension differs from the engine"
            )
        if value.level != level or value.scale != expected_scale:
            raise JitInputError(
                "Encrypted input state differs from its InputSpec: "
                f"level/scale={value.level}/{value.scale!r}, expected "
                f"{level}/{expected_scale!r}"
            )
        if slots != "full":
            raise JitInputError(
                "A fixed logical slot extent cannot be recovered from an "
                "opaque Ciphertext; supply a Tensor for online encryption or "
                "declare slots='full'"
            )
        if batch_mode == "none" and value.batch_shape:
            raise JitInputError(
                "Encrypted input declares batch_mode='none' but received "
                f"batch shape {tuple(value.batch_shape)}"
            )
        return value

    if not isinstance(value, torch.Tensor):
        raise JitInputError(
            "Encrypted Program input must be Tensor or Ciphertext"
        )
    if value.ndim < 1:
        raise JitInputError("Encrypted Tensor input requires a slot axis")
    expected_slots = engine.num_slots if slots == "full" else slots
    if isinstance(expected_slots, bool) or not isinstance(expected_slots, int):
        raise RuntimeError("Encrypted input_spec slots is malformed")
    if value.shape[-1] != expected_slots:
        raise JitInputError(
            "Encrypted Tensor slot extent differs from InputSpec: "
            f"{value.shape[-1]} != {expected_slots}"
        )
    if batch_mode == "none" and value.ndim != 1:
        raise JitInputError(
            "Encrypted Tensor declares batch_mode='none' and must have one "
            "slot axis"
        )
    public_key = workspace.get("public_key")
    if not isinstance(public_key, PublicKey):
        raise JitInputError(
            "Online encryption requires a PublicKey in workspace['public_key']"
        )
    return engine.encrypt_message(
        value,
        public_key,
        level=level,
        scale=expected_scale,
    )


def _input_spec(argument: SSAValue) -> dict[str, object]:
    state = getattr(argument.type, "state", None)
    if state is None or not hasattr(state, "data"):
        return {}
    encoded = state.data.get("input_spec")
    if not isinstance(encoded, StringAttr):
        return {}
    try:
        decoded = json.loads(encoded.data)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Encrypted input has malformed input_spec"
        ) from error
    return decoded if isinstance(decoded, dict) else {}


def _execute_operation(
    operation: Operation,
    operands: tuple[object, ...],
    workspace: MutableMapping[Any, Any],
) -> object:
    name = operation_name(operation)
    if name == "fhelium.constant":
        return _decode_json_attribute(operation, "fhelium.literal")
    if isinstance(operation, MaterialRefOp) or name == "fhelium.material.ref":
        symbol = _string_attr(operation, "symbol")
        assert symbol is not None
        binding = _mapping(workspace, "materials")[symbol]
        resolver = workspace.get("material_resolver")
        if callable(resolver):
            return resolver(
                symbol,
                _string_attr(operation, "kind"),
                binding,
                workspace,
            )
        return binding
    if isinstance(operation, ResourceRefOp) or name == "fhelium.resource.ref":
        symbol = _string_attr(operation, "symbol")
        assert symbol is not None
        binding = _mapping(workspace, "resources")[symbol]
        resolver = workspace.get("resource_resolver")
        if callable(resolver):
            return resolver(
                symbol,
                _string_attr(operation, "kind"),
                binding,
                workspace,
            )
        return binding
    if name == "torch.call":
        return _execute_torch_call(operation, operands, workspace)
    if name in RUNTIME_CKKS_OPERATION_NAMES:
        return _execute_ckks(operation, operands, workspace)
    handler = _handlers(workspace).get(name)
    if callable(handler):
        return handler(operation, operands, workspace)
    raise RuntimeError(f"No execution handler for operation {name!r}")


def _store_results(
    operation: Operation,
    result: object,
    environment: dict[SSAValue, object],
) -> None:
    count = len(operation.results)
    if count == 0:
        if result is not None:
            raise RuntimeError(
                f"Zero-result operation {operation_name(operation)!r} returned a value"
            )
        return
    if count == 1:
        environment[operation.results[0]] = result
        return
    if not isinstance(result, tuple) or len(result) != count:
        raise RuntimeError(
            f"Operation {operation_name(operation)!r} must return {count} values"
        )
    environment.update(zip(operation.results, result))


def _decode_json_attribute(operation: Operation, *names: str) -> object:
    encoded = _string_attr(operation, *names)
    if encoded is None:
        raise RuntimeError(
            f"Operation {operation_name(operation)!r} lacks JSON attribute {names}"
        )
    try:
        return _decode_literal(json.loads(encoded))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Operation {operation_name(operation)!r} has malformed JSON metadata"
        ) from error


def _decode_literal(value: object) -> object:
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "complex":
        return complex(value["real"], value["imag"])
    if kind == "ellipsis":
        return Ellipsis
    if kind == "torch.dtype":
        name = str(value["value"]).removeprefix("torch.")
        return getattr(torch, name)
    if kind == "torch.device":
        return torch.device(str(value["value"]))
    if kind == "torch.layout":
        name = str(value["value"]).removeprefix("torch.")
        return getattr(torch, name)
    return value


def _execute_torch_call(
    operation: Operation,
    operands: tuple[object, ...],
    workspace: Mapping[Any, Any],
) -> object:
    target = _torch_target(operation)
    if target is None:
        raise RuntimeError("torch.call lacks fhelium.call.target")
    kind = _torch_call_kind(operation)
    touches_fhe = any(
        value_role(value) in {"encrypted", "plaintext"}
        for value in (*operation.operands, *operation.results)
    )
    if touches_fhe and not callable(_torch_handlers(workspace).get(target)):
        raise RuntimeError(
            f"Torch target {target!r} touches FHE values and requires an "
            "torch_handlers binding"
        )
    callable_target = _resolve_torch_target(target, kind, workspace)
    if callable_target is None:
        raise RuntimeError(f"No Torch handler for target {target!r}")

    encoded = _string_attr(operation, "fhelium.call.arguments")
    if encoded is None:
        raise RuntimeError("torch.call lacks fhelium.call.arguments")
    try:
        descriptor = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "torch.call has malformed argument metadata"
        ) from error
    if not isinstance(descriptor, dict):
        raise RuntimeError("torch.call argument metadata must be an object")
    decoded_args = _decode_argument(descriptor.get("args"), operands)
    if not isinstance(decoded_args, tuple):
        raise RuntimeError("torch.call args descriptor must decode to tuple")
    positional = decoded_args
    decoded_kwargs = _decode_argument(descriptor.get("kwargs"), operands)
    if not isinstance(decoded_kwargs, Mapping):
        raise RuntimeError(
            "torch.call kwargs descriptor must decode to mapping"
        )
    keyword = {str(name): item for name, item in decoded_kwargs.items()}
    return callable_target(*positional, **keyword)


def _decode_argument(value: object, operands: tuple[object, ...]) -> object:
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "ssa":
        index = value.get("operand")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError(
                "SSA argument descriptor lacks an integer operand"
            )
        return operands[index]
    if kind == "literal":
        return _decode_literal(value.get("value"))
    if kind == "tuple":
        items = value.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("Sequence argument descriptor is malformed")
        return tuple(_decode_argument(item, operands) for item in items)
    if kind == "list":
        items = value.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("List argument descriptor is malformed")
        return [_decode_argument(item, operands) for item in items]
    if kind == "mapping":
        entries = value.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError("Mapping argument descriptor is malformed")
        return {
            _decode_argument(pair[0], operands): _decode_argument(
                pair[1], operands
            )
            for pair in entries
        }
    if kind == "slice":
        return slice(
            _decode_argument(value.get("start"), operands),
            _decode_argument(value.get("stop"), operands),
            _decode_argument(value.get("step"), operands),
        )
    return _decode_literal(value)


_NO_OUTPUT_STRUCTURE = object()


def _reconstruct_output(
    program: Program,
    values: tuple[object, ...],
) -> object:
    attribute = program.module.attributes.get("fhelium.output_structure")
    if not isinstance(attribute, StringAttr):
        return _NO_OUTPUT_STRUCTURE
    try:
        descriptor = json.loads(attribute.data)
    except json.JSONDecodeError as error:
        raise RuntimeError("Program has malformed output_structure") from error
    return _decode_output(descriptor, values)


def _decode_output(value: object, results: tuple[object, ...]) -> object:
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "ssa":
        index = value.get("result")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError("Output SSA descriptor lacks result index")
        return results[index]
    if kind == "tuple":
        return tuple(
            _decode_output(item, results) for item in value.get("items", [])
        )
    if kind == "list":
        return [
            _decode_output(item, results) for item in value.get("items", [])
        ]
    if kind == "mapping":
        entries = value.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError("Output mapping descriptor is malformed")
        return {
            _decode_literal(pair[0]): _decode_output(pair[1], results)
            for pair in entries
        }
    return _decode_literal(value)


def _execute_ckks(
    operation: Operation,
    operands: tuple[object, ...],
    workspace: MutableMapping[Any, Any],
) -> object:
    engine = workspace.get("engine")
    if not isinstance(engine, CkksEngine):
        raise RuntimeError("CKKS operation has no CkksEngine")
    name = operation_name(operation)

    if name.startswith("fhelium.ckks.prepare."):
        if len(operands) != 2 or not isinstance(operands[1], Ciphertext):
            raise RuntimeError(
                f"Preparation {name!r} requires public,ciphertext"
            )
        public = operands[0]
        ciphertext = operands[1]
        assert isinstance(ciphertext, Ciphertext)
        mode = _string_attr(operation, "scale_mode")
        action = _string_attr(operation, "operation")
        if mode == "runtime_plaintext_scale" and isinstance(public, Plaintext):
            scale = public.scale
        elif mode == "default_scale":
            scale = engine.config.default_scale
        elif action == "add":
            scale = ciphertext.scale
        else:
            scale = engine.config.default_scale
        return _prepare_public_operand(
            engine,
            public,
            ciphertext,
            operation=action or "multiply",
            scale=scale,
        )

    if name == "fhelium.ckks.negate":
        return engine.negate(_ciphertext(operands, 0, name))
    if name == "fhelium.ckks.rotate":
        ciphertext = _ciphertext(operands, 0, name)
        shift = _int_attr(operation, "shift")
        if shift is None:
            raise RuntimeError("CKKS rotate lacks shift")
        step = RotationKey.canonical_step(shift, ring_dimension=engine.config.N)
        if step == 0:
            return ciphertext
        keys = workspace.get("evaluation_keys")
        assert isinstance(keys, EvaluationKeySet)
        return engine.rotate_with_key(ciphertext, keys.rotations[step])
    if name == "fhelium.ckks.to_ntt":
        return engine.coefficient_domain_to_ntt_domain(
            _ciphertext(operands, 0, name)
        )
    if name == "fhelium.ckks.from_ntt":
        return engine.ntt_domain_to_coefficient_domain(
            _ciphertext(operands, 0, name)
        )
    if name in {
        "fhelium.ckks.add",
        "fhelium.ckks.subtract",
        "fhelium.ckks.multiply",
    }:
        left = _ciphertext(operands, 0, name)
        right = _ciphertext(operands, 1, name)
        if name == "fhelium.ckks.add":
            return engine.add(left, right)
        if name == "fhelium.ckks.subtract":
            return engine.subtract(left, right)
        return engine.multiply(left, right)
    if name in {
        "fhelium.ckks.add_plaintext",
        "fhelium.ckks.multiply_plaintext",
    }:
        ciphertext = _ciphertext(operands, 0, name)
        if len(operands) < 2 or not isinstance(operands[1], Plaintext):
            raise RuntimeError(f"{name} requires ciphertext,plaintext")
        method = (
            engine.add_plaintext
            if name == "fhelium.ckks.add_plaintext"
            else engine.multiply_plaintext
        )
        return method(ciphertext, operands[1])
    if name == "fhelium.ckks.relinearize":
        keys = workspace.get("evaluation_keys")
        assert isinstance(keys, EvaluationKeySet)
        assert keys.relinearization is not None
        return engine.relinearize(
            _ciphertext(operands, 0, name), keys.relinearization
        )
    if name == "fhelium.ckks.rescale":
        ciphertext = _ciphertext(operands, 0, name)
        condition = _string_attr(operation, "condition")
        if condition == "plaintext_scale_not_one":
            if len(operands) < 2 or not isinstance(operands[1], Plaintext):
                raise RuntimeError("Conditional rescale lacks plaintext")
            if operands[1].scale == 1.0:
                return ciphertext
        return engine.rescale_to_next_level(ciphertext)
    raise RuntimeError(f"Unhandled explicit CKKS operation {name!r}")


def _prepare_public_operand(
    engine: CkksEngine,
    public: object,
    ciphertext: Ciphertext,
    *,
    operation: str,
    scale: float,
) -> Plaintext:
    """Encode and prepare one public value under the graph policy."""

    if isinstance(public, Plaintext):
        if public.is_approximate_coefficients:
            raise ValueError(
                "approximate_coefficients Plaintext is decode-only"
            )
        if public.context_id not in (None, engine.context.context_id):
            raise ValueError("Plaintext belongs to a different CKKS context")
        if public.level != ciphertext.level:
            raise ValueError(
                "Plaintext and ciphertext levels differ at the explicit "
                f"level mismatch: {public.level} != {ciphertext.level}"
            )
        if operation == "add" and public.scale != ciphertext.scale:
            raise ValueError(
                "Plaintext addition requires equal scales: "
                f"{public.scale} != {ciphertext.scale}"
            )
        if public.is_slots:
            assert public.message is not None
            prepared_source = engine.encode(
                public.message,
                level=public.level,
                scale=public.scale,
            )
        elif public.is_integer_coefficients:
            prepared_source = public
        else:
            assert public.is_rns
            if public.modulus_basis != ciphertext.modulus_basis:
                raise ValueError(
                    "RNS Plaintext and ciphertext modulus bases differ"
                )
            if public.prime_ids != ciphertext.prime_ids:
                raise ValueError(
                    "RNS Plaintext and ciphertext prime identities differ"
                )
            prepared = public
            if operation == "add" and prepared.polynomial_domain == "ntt":
                prepared = engine.ntt_domain_to_coefficient_domain(prepared)
            if prepared.residue_representation == "standard":
                prepared = engine.standard_residues_to_montgomery_residues(
                    prepared
                )
            if (
                operation == "multiply"
                and prepared.polynomial_domain == "coefficient"
            ):
                prepared = engine.coefficient_domain_to_ntt_domain(prepared)
            return prepared
    else:
        prepared_source = engine.encode(
            public,
            level=ciphertext.level,
            scale=scale,
        )

    prepare = (
        engine.prepare_plaintext_for_addition
        if operation == "add"
        else engine.prepare_plaintext_for_multiplication
    )
    return prepare(
        prepared_source,
        modulus_basis=ciphertext.modulus_basis,
    )


def _ciphertext(
    operands: tuple[object, ...], index: int, operation_name_: str
) -> Ciphertext:
    if index >= len(operands) or not isinstance(operands[index], Ciphertext):
        raise RuntimeError(
            f"{operation_name_} operand {index} must be a Ciphertext"
        )
    return cast(Ciphertext, operands[index])


__all__ = [
    "BindingResolver",
    "OperationHandler",
    "ProgramNotReadyError",
    "ReadinessDiagnostic",
    "ReadinessReport",
    "check_readiness",
    "run_program",
]
