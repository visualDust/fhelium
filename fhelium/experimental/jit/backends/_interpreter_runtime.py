"""IR-interpreter coverage analysis and selected-entry execution.

``check_interpreter_coverage`` performs a side-effect-free capability and executable-schema
check for one entry. ``interpret_program`` combines that report with runtime argument
binding and online-encryption requirements, then interprets only the selected
single-block entry. Import, export, and ordinary pass execution remain structural
operations and invoke neither operation handlers nor binding resolvers.
"""

from __future__ import annotations

from fhelium.values.state import PolynomialDomain

import json
import operator
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation, SSAValue

from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    EvaluationKeyRequirements,
    EvaluationKeySet,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    RotationKeySet,
    SecretKey,
)
from fhelium.eager import Engine

from fhelium.ir import (
    DIALECT_VERSION,
    DIALECT_VERSION_ATTRIBUTE,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTRIBUTE,
    Program,
    operation_name,
    value_role,
)
from fhelium.ir.dialects import (
    ckks,
    core,
    semantic,
    torch as torch_dialect,
)

from .._analysis import (
    RUNTIME_CKKS_OPERATION_TYPES,
    ProgramRequirements,
    analyze_requirements,
    rotation_key_reference,
)
from .._errors import JitError, JitInputError, JitInterpreterError
from .._bindings import BindingResolver, OperationHandler, RuntimeBindings
# OperationHandler receives extension IR, evaluated operands, and the current
# interpreter workspace. BindingResolver receives symbol, optional kind, raw
# binding, and that same workspace when the reference operation executes.


@dataclass(frozen=True)
class InterpreterDiagnostic:
    """Describe one concrete missing or malformed execution requirement."""

    code: str
    message: str
    subject: str | None = None
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class InterpreterCoverage:
    """Carry selected-entry requirements and implementation diagnostics.

    ``covered`` is true when ``diagnostics`` contains no item whose
    severity is ``"error"``. Warnings, including a missing version marker, are
    recorded in ``diagnostics`` and permit execution.
    """

    requirements: ProgramRequirements
    diagnostics: tuple[InterpreterDiagnostic, ...]

    @property
    def covered(self) -> bool:
        """Whether this report contains no implementation-blocking diagnostic."""

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
        """Return symbolic material names lacking runtime bindings."""

        return frozenset(
            item.subject
            for item in self.diagnostics
            if item.code == "missing-material" and item.subject is not None
        )

    @property
    def missing_resources(self) -> frozenset[str]:
        """Return symbolic resource names lacking runtime bindings."""

        return frozenset(
            item.subject
            for item in self.diagnostics
            if item.code == "missing-resource" and item.subject is not None
        )


class InterpreterNotCoveredError(JitError, RuntimeError):
    """Reject one ``Program.run(...)`` request while preserving its full report."""

    def __init__(self, report: InterpreterCoverage) -> None:
        self.report = report
        detail = "; ".join(item.message for item in report.diagnostics)
        super().__init__(f"Interpreter does not cover the Program: {detail}")


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


def _evaluation_key_set(
    value: object,
    requirements: EvaluationKeyRequirements,
) -> EvaluationKeySet | None:
    """Normalize a key inventory or one directly usable primitive key."""

    if isinstance(value, EvaluationKeySet):
        return value
    if (
        isinstance(value, RelinearizationKey)
        and requirements.requires_relinearization
        and not requirements.rotation_steps
        and not requirements.requires_conjugation
    ):
        return EvaluationKeySet(relinearization=value)
    if (
        isinstance(value, RotationKey)
        and requirements.rotation_steps == {value.rotation_step}
        and not requirements.requires_relinearization
        and not requirements.requires_conjugation
    ):
        return EvaluationKeySet(
            rotations=RotationKeySet({value.rotation_step: value})
        )
    if (
        isinstance(value, RotationKeySet)
        and requirements.rotation_steps
        and not requirements.requires_relinearization
        and not requirements.requires_conjugation
    ):
        return EvaluationKeySet(rotations=value)
    return None


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


def _float_attr(operation: Operation, *names: str) -> float | None:
    for name in names:
        value = operation.attributes.get(name)
        if isinstance(value, FloatAttr):
            return float(value.value.data)
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


def check_interpreter_coverage(
    program: Program,
    bindings: RuntimeBindings,
    *,
    entry: str = "main",
) -> InterpreterCoverage:
    """Assess whether ``entry`` has the capabilities required to execute.

    The assessment verifies runtime binding container types, resolver
    callability, Program schema and dialect versions, a single-block entry with
    one final return, entry argument and built-in operation schemas, cleared
    concrete transition operations, callable handlers for extension operations and
    Torch targets, symbolic material/resource membership, an Eager Engine, and
    required rotation/relinearization keys compatible with that engine.

    The interpreter reserves its built-in runtime operation names;
    ``bindings.handlers`` authorizes extension operations only and cannot
    override a built-in schema or implementation. ``torch.call`` operations
    that touch encrypted or plaintext values require a bound callable in
    ``bindings.torch_handlers``. Pure-public Torch calls may instead use the
    interpreter's audited target set.

    This function reads Program and RuntimeBindings state without mutation. Material
    and resource bindings are checked by name, while resolver and handler calls,
    runtime argument validation, online-public-key requirements, and operation
    execution occur only in ``interpret_program``.
    """

    if not isinstance(program, Program):
        raise TypeError("check_interpreter_coverage expects a Program")
    if not isinstance(bindings, RuntimeBindings):
        raise TypeError("bindings must be RuntimeBindings")
    workspace = bindings.interpreter_workspace()

    requirements = analyze_requirements(program, entry=entry)
    diagnostics: list[InterpreterDiagnostic] = []
    for key in ("materials", "resources", "handlers", "torch_handlers"):
        if key in workspace and not isinstance(workspace[key], Mapping):
            diagnostics.append(
                InterpreterDiagnostic(
                    "malformed-runtime-binding",
                    f"Runtime binding {key!r} must be a mapping when supplied.",
                    key,
                )
            )
    for key in ("material_resolver", "resource_resolver"):
        if key in workspace and not callable(workspace[key]):
            diagnostics.append(
                InterpreterDiagnostic(
                    "malformed-runtime-binding",
                    f"Runtime binding {key!r} must be callable when supplied.",
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
                InterpreterDiagnostic(
                    "unversioned-program",
                    f"Program omits {attribute_name!r}; current semantics are "
                    "selected by this interpreter analysis.",
                    attribute_name,
                    "warning",
                )
            )
        elif (
            not isinstance(attribute, StringAttr) or attribute.data != supported
        ):
            diagnostics.append(
                InterpreterDiagnostic(
                    "unsupported-program-version",
                    f"Program {attribute_name!r} is unsupported; expected "
                    f"{supported!r}.",
                    attribute_name,
                )
            )

    try:
        from ._interpreter_schema import validate_interpreter_graph

        validate_interpreter_graph(
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
    except JitInterpreterError as error:
        diagnostics.append(
            InterpreterDiagnostic(
                "invalid-executable-graph",
                str(error),
                entry,
            )
        )

    if requirements.return_count is None:
        diagnostics.append(
            InterpreterDiagnostic(
                "missing-entry",
                f"Entry function {entry!r} is missing, multi-block, or has no "
                "unique func.return.",
                entry,
            )
        )

    for malformed in requirements.malformed_references:
        diagnostics.append(
            InterpreterDiagnostic(
                "malformed-reference",
                f"Operation {malformed!r} lacks required symbolic metadata.",
                malformed,
            )
        )

    for operation_name_ in sorted(requirements.unknown_operations):
        if callable(handlers.get(operation_name_)):
            continue
        diagnostics.append(
            InterpreterDiagnostic(
                "missing-handler",
                f"Operation {operation_name_!r} has no execution handler.",
                operation_name_,
            )
        )

    try:
        entry_block = program.single_block(entry)
    except (KeyError, ValueError):
        entry_block = None
    if entry_block is not None:
        for operation in entry_block.ops:
            if not isinstance(operation, torch_dialect.CallOp):
                continue
            target = _torch_target(operation)
            kind = _torch_call_kind(operation)
            if target is None:
                continue
            touches_fhe = any(
                value_role(value) in {"encrypted", "plaintext"}
                for value in (*operation.operands, *operation.results)
            )
            has_bound_handler = callable(torch_handlers.get(target))
            if (touches_fhe and not has_bound_handler) or (
                not touches_fhe
                and _resolve_torch_target(target, kind, workspace) is None
            ):
                diagnostics.append(
                    InterpreterDiagnostic(
                        "missing-torch-handler",
                        f"Torch target {target!r} has no compatible bound "
                        "or audited public execution handler.",
                        target,
                    )
                )

    materials = _mapping(workspace, "materials")
    for symbol in sorted(requirements.materials - set(materials)):
        diagnostics.append(
            InterpreterDiagnostic(
                "missing-material",
                f"Symbolic material {symbol!r} is not bound.",
                symbol,
            )
        )

    resources = _mapping(workspace, "resources")
    for symbol in sorted(requirements.resources - set(resources)):
        diagnostics.append(
            InterpreterDiagnostic(
                "missing-resource",
                f"Symbolic resource {symbol!r} is not bound.",
                symbol,
            )
        )

    engine = workspace.get("eager_engine")
    if requirements.requires_eager_engine and not isinstance(engine, Engine):
        diagnostics.append(
            InterpreterDiagnostic(
                "missing-eager-engine",
                "CKKS operations require bindings.eager_engine.",
                "eager_engine",
            )
        )

    if (
        requirements.rotation_steps
        or requirements.requires_relinearization
        or requirements.requires_conjugation
    ):
        evaluation_key_requirements = EvaluationKeyRequirements(
            rotation_steps=requirements.rotation_steps,
            requires_relinearization=requirements.requires_relinearization,
            requires_conjugation=requirements.requires_conjugation,
        )
        evaluation_keys = _evaluation_key_set(
            workspace.get("evaluation_keys"), evaluation_key_requirements
        )
        if evaluation_keys is None:
            diagnostics.append(
                InterpreterDiagnostic(
                    "missing-evaluation-keys",
                    "Rotation, relinearization, or conjugation operations "
                    "require matching evaluation-key material.",
                    "evaluation_keys",
                )
            )
        else:
            try:
                evaluation_keys.require(evaluation_key_requirements)
                if isinstance(engine, Engine):
                    for step in requirements.rotation_steps:
                        normalized_step = RotationKey.normalize_step(
                            step,
                            ring_dimension=engine.config.N,
                        )
                        if step != normalized_step:
                            raise ValueError(
                                "JIT rotation step is not normalized for the "
                                f"engine: {step} != {normalized_step}"
                            )
                        engine.validate_key_switch_key(
                            evaluation_keys.rotations[step]
                        )
                    if requirements.requires_relinearization:
                        assert evaluation_keys.relinearization is not None
                        engine.validate_key_switch_key(
                            evaluation_keys.relinearization
                        )
                    if requirements.requires_conjugation:
                        assert evaluation_keys.conjugation is not None
                        engine.validate_key_switch_key(
                            evaluation_keys.conjugation
                        )
            except (KeyError, TypeError, ValueError) as error:
                diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-evaluation-keys",
                        str(error),
                        "evaluation_keys",
                    )
                )

    if requirements.public_key_symbols:
        public_key = workspace.get("public_key")
        if not isinstance(public_key, PublicKey):
            diagnostics.append(
                InterpreterDiagnostic(
                    "missing-public-key",
                    "CKKS encrypt requires bindings.public_key.",
                    "public_key",
                )
            )
        elif isinstance(engine, Engine):
            try:
                engine.validate_public_key(public_key)
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-public-key",
                        str(error),
                        "public_key",
                    )
                )

    for symbol in sorted(requirements.secret_key_symbols):
        secret_key = resources.get(symbol)
        if not isinstance(secret_key, SecretKey):
            diagnostics.append(
                InterpreterDiagnostic(
                    "missing-secret-key-resource",
                    f"Secret-key resource {symbol!r} is not bound.",
                    symbol,
                )
            )
        elif isinstance(engine, Engine):
            try:
                engine.validate_secret_key(secret_key)
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-secret-key-resource",
                        str(error),
                        symbol,
                    )
                )
    for symbol in sorted(requirements.key_switch_symbols):
        key = resources.get(symbol)
        if key is None:
            diagnostics.append(
                InterpreterDiagnostic(
                    "missing-key-switch-resource",
                    f"Key-switch resource {symbol!r} is not bound.",
                    symbol,
                )
            )
        elif isinstance(engine, Engine):
            try:
                if type(key) is not KeySwitchKey:
                    raise TypeError(
                        f"Resource {symbol!r} must be an exact KeySwitchKey"
                    )
                engine.validate_key_switch_key(key)
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-key-switch-resource",
                        str(error),
                        symbol,
                    )
                )

    for symbol, step in requirements.rotation_key_symbols:
        key = resources.get(symbol)
        if type(key) is not RotationKey:
            diagnostics.append(
                InterpreterDiagnostic(
                    "missing-rotation-key-resource",
                    f"Rotation-key resource {symbol!r} is not bound.",
                    symbol,
                )
            )
        elif isinstance(engine, Engine):
            try:
                engine.validate_key_switch_key(key)
                normalized = RotationKey.normalize_step(
                    step,
                    ring_dimension=engine.ring_dimension,
                )
                if step != normalized or key.rotation_step != normalized:
                    raise ValueError(
                        f"Rotation resource {symbol!r} step differs from "
                        f"represented step {step}"
                    )
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-rotation-key-resource",
                        str(error),
                        symbol,
                    )
                )

    return InterpreterCoverage(requirements, tuple(diagnostics))


def interpret_program(
    program: Program,
    *args: object,
    bindings: RuntimeBindings,
    execution_device: torch.device,
    entry: str = "main",
    **kwargs: object,
) -> Any:
    """Check interpreter coverage and execute one selected Program entry.

    The run first computes ``check_interpreter_coverage`` and rejects its error diagnostics
    through ``InterpreterNotCoveredError``. It then selects the entry's unique block,
    binds positional and named runtime arguments, and requires a PublicKey
    accepted by `Engine` when an encrypted entry argument receives a Tensor.
    Argument count/name failures raise ``TypeError`` only after the base
    implementation coverage succeeds. Input materialization then enforces each role
    specification and performs online encryption where requested.

    Interpretation follows entry-block order. Material and resource reference
    operations fetch their raw runtime binding and invoke the corresponding
    resolver at that point, once per encountered reference. Built-in constants,
    references, Torch calls, and CKKS operations use reserved interpreter paths;
    callable ``bindings.handlers`` entries execute extension operations.
    Every handler and resolver receives the per-invocation interpreter workspace. Persistent
    application state can live in an object stored by
    ``RuntimeBindings.extensions``.

    Captured output metadata reconstructs its Python tuple, list, or mapping
    shape. Without that metadata, zero return operands produce ``None``, one
    produces the value directly, and multiple produce a tuple. Accordingly the
    API result type is dynamic for imported or directly constructed Programs.
    """

    if not isinstance(bindings, RuntimeBindings):
        raise TypeError("bindings must be RuntimeBindings")
    workspace = bindings.interpreter_workspace()
    workspace["execution_device"] = execution_device
    report = check_interpreter_coverage(program, bindings, entry=entry)
    if not report.covered:
        raise InterpreterNotCoveredError(report)
    try:
        block = program.single_block(entry)
    except (KeyError, ValueError):
        raise InterpreterNotCoveredError(report) from None
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
                InterpreterDiagnostic(
                    "missing-public-key",
                    "Online encryption requires bindings.public_key.",
                    "public_key",
                )
            )
        else:
            engine = workspace.get("eager_engine")
            assert isinstance(engine, Engine)
            try:
                engine.validate_public_key(public_key)
            except (TypeError, ValueError) as error:
                runtime_diagnostics.append(
                    InterpreterDiagnostic(
                        "incompatible-public-key",
                        str(error),
                        "public_key",
                    )
                )
    report = InterpreterCoverage(
        report.requirements, tuple(runtime_diagnostics)
    )
    if not report.covered:
        raise InterpreterNotCoveredError(report)

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
    engine = workspace.get("eager_engine")
    if not isinstance(engine, Engine):
        raise JitInputError(
            "Encrypted Program inputs require bindings.eager_engine"
        )
    execution_device = torch.device(workspace["execution_device"])
    spec = _input_spec(argument)
    depth = spec.get("depth", 0)
    scale = spec.get("scale")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise RuntimeError("Encrypted input_spec depth must be an integer")
    if scale is None:
        scale = engine.config.default_scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise RuntimeError("Encrypted input_spec scale must be real or null")
    expected_scale = float(scale)
    slots = spec.get("slots", "full")
    batch_mode = spec.get("batch_mode", "none")

    if isinstance(value, Ciphertext):
        if (
            value.device != execution_device
            or value.data.dtype != engine.dtype
        ):
            raise JitInputError(
                "Encrypted input device or dtype differs from the JIT execution"
            )
        if value.ring_dimension != engine.config.N:
            raise JitInputError(
                "Encrypted input ring dimension differs from the Eager Engine"
            )
        if value.depth != depth or value.scale != expected_scale:
            raise JitInputError(
                "Encrypted input state differs from its InputSpec: "
                f"depth/scale={value.depth}/{value.scale!r}, expected "
                f"{depth}/{expected_scale!r}"
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
        raise JitInputError("Online encryption requires bindings.public_key")
    return engine.encrypt_message(
        value,
        public_key,
        depth=depth,
        scale=expected_scale,
        device=execution_device,
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

    if isinstance(operation, UnrealizedConversionCastOp):
        if len(operands) != 1:
            raise RuntimeError(
                "IR interpretation supports one-to-one conversion casts"
            )
        return operands[0]
    if isinstance(operation, core.ConstantOp):
        return _decode_json_attribute(operation, "fhelium.literal")
    if isinstance(operation, core.MaterialRefOp):
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
    if isinstance(operation, core.ResourceRefOp):
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
    if isinstance(operation, torch_dialect.CallOp):
        return _execute_torch_call(operation, operands, workspace)
    if isinstance(operation, semantic.AddOp):
        if not all(isinstance(value, torch.Tensor) for value in operands):
            raise RuntimeError("Public semantic add requires Tensor operands")
        return torch.add(
            cast(torch.Tensor, operands[0]), cast(torch.Tensor, operands[1])
        )
    if isinstance(operation, semantic.SubtractOp):
        if not all(isinstance(value, torch.Tensor) for value in operands):
            raise RuntimeError(
                "Public semantic subtract requires Tensor operands"
            )
        return torch.subtract(
            cast(torch.Tensor, operands[0]), cast(torch.Tensor, operands[1])
        )
    if isinstance(operation, semantic.MultiplyOp):
        if not all(isinstance(value, torch.Tensor) for value in operands):
            raise RuntimeError(
                "Public semantic multiply requires Tensor operands"
            )
        return torch.multiply(
            cast(torch.Tensor, operands[0]), cast(torch.Tensor, operands[1])
        )
    if isinstance(operation, semantic.NegateOp):
        if not isinstance(operands[0], torch.Tensor):
            raise RuntimeError(
                "Public semantic negate requires a Tensor operand"
            )
        return torch.neg(operands[0])
    if isinstance(operation, semantic.RollOp):
        if not isinstance(operands[0], torch.Tensor):
            raise RuntimeError("Public semantic roll requires a Tensor operand")
        shift = _int_attr(operation, "shift")
        if shift is None:
            raise RuntimeError("Public semantic roll lacks shift")
        dimension = _int_attr(operation, "dimension")
        if dimension is None:
            return torch.roll(operands[0], shifts=shift)
        return torch.roll(operands[0], shifts=shift, dims=dimension)
    if isinstance(operation, RUNTIME_CKKS_OPERATION_TYPES):
        return _execute_ckks(operation, operands, workspace)
    handler = _handlers(workspace).get(name)
    if callable(handler):
        return handler(operation, operands, workspace)
    raise RuntimeError(f"No execution handler for operation {name!r}")


def _bind_program_inputs_raw(
    program: Program,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> tuple[object, ...]:
    """Bind Program arguments without imposing an execution-provider ABI."""

    block = program.single_block("main")
    return _bind_argument_values(program, block.args, args, kwargs)


def _materialize_interpreter_region_inputs(
    input_values: Sequence[SSAValue],
    args: tuple[object, ...],
    bindings: RuntimeBindings,
    source_input_is_entry: Sequence[bool],
    *,
    execution_device: torch.device | None = None,
) -> tuple[object, ...]:
    """Adapt only original entry arguments for one interpreter region.

    Produced region-boundary values pass through unchanged. Original entry
    arguments are adapted locally without replacing the Composite SSA slot, so
    fanout to another provider observes the raw object. A fanout to multiple
    interpreter regions may repeat materialization in this first version.
    """

    if len(source_input_is_entry) != len(input_values):
        raise ValueError(
            "Interpreter region source-input metadata is incomplete"
        )
    workspace = bindings.interpreter_workspace()
    if execution_device is not None:
        workspace["execution_device"] = torch.device(execution_device)
    return tuple(
        _materialize_entry_input(value, argument, workspace)
        if is_entry
        else argument
        for value, argument, is_entry in zip(
            input_values, args, source_input_is_entry, strict=True
        )
    )


def _execute_operation_region(
    operations: Sequence[Operation],
    input_values: Sequence[SSAValue],
    output_values: Sequence[SSAValue],
    args: tuple[object, ...],
    bindings: RuntimeBindings,
    *,
    execution_device: torch.device | None = None,
) -> tuple[object, ...]:
    """Execute one already-covered straight-line interpreter region."""

    if len(args) != len(input_values):
        raise TypeError(
            f"Interpreter region expected {len(input_values)} inputs, "
            f"got {len(args)}"
        )
    workspace = bindings.interpreter_workspace()
    if execution_device is not None:
        workspace["execution_device"] = torch.device(execution_device)
    environment: dict[SSAValue, object] = dict(
        zip(input_values, args, strict=True)
    )
    for operation in operations:
        operands = tuple(environment[value] for value in operation.operands)
        result = _execute_operation(operation, operands, workspace)
        _store_results(operation, result, environment)
    return tuple(environment[value] for value in output_values)


def _reconstruct_program_output(
    program: Program, values: tuple[object, ...]
) -> object:
    """Apply captured output metadata or conventional return arity."""

    structured = _reconstruct_output(program, values)
    if structured is not _NO_OUTPUT_STRUCTURE:
        return structured
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


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
    engine = workspace.get("eager_engine")
    if not isinstance(engine, Engine):
        raise RuntimeError("CKKS operation has no Engine")
    name = operation_name(operation)

    if isinstance(operation, ckks.EncodeOp):
        if len(operands) != 1:
            raise RuntimeError("CKKS encode requires one message")
        return engine.encode(
            operands[0],  # type: ignore[arg-type]
            depth=int(operation.depth.value.data),
            scale=float(operation.scale.value.data),
        )
    if isinstance(operation, ckks.DecodeOp):
        if len(operands) != 1 or not isinstance(operands[0], Plaintext):
            raise RuntimeError("CKKS decode requires one Plaintext")
        return engine.decode(
            operands[0],
            is_real=bool(int(operation.is_real.value.data)),
        )
    if isinstance(operation, ckks.IntegerCoefficientsToRnsOp):
        if len(operands) != 1 or not isinstance(operands[0], Plaintext):
            raise RuntimeError("Integer-to-RNS requires one Plaintext")
        basis = operation.modulus_basis.data
        if basis not in {"Q", "QP"}:
            raise RuntimeError("Integer-to-RNS basis is unsupported")
        return engine.integer_coefficients_to_rns(
            operands[0],
            modulus_basis=cast(Literal["Q", "QP"], basis),
        )
    if isinstance(operation, ckks.EncryptOp):
        if len(operands) != 1 or not isinstance(operands[0], Plaintext):
            raise RuntimeError("CKKS encrypt requires one Plaintext")
        public_key = workspace.get("public_key")
        if not isinstance(public_key, PublicKey):
            raise RuntimeError("CKKS encrypt lacks public-key material")
        return engine.encrypt(
            operands[0],
            public_key,
            output_domain=cast(PolynomialDomain, operation.output_domain.data),
        )
    if isinstance(operation, ckks.DecryptOp):
        ciphertext = _ciphertext(operands, 0, name)
        secret_key = _mapping(workspace, "resources").get(
            operation.key_symbol.data
        )
        if not isinstance(secret_key, SecretKey):
            raise RuntimeError("CKKS decrypt lacks secret-key material")
        return engine.decrypt(ciphertext, secret_key)

    if isinstance(
        operation,
        (
            ckks.PrepareAddMessageOp,
            ckks.PrepareAddPlaintextOp,
            ckks.PrepareAddStaticOp,
            ckks.PrepareMultiplyMessageOp,
            ckks.PrepareMultiplyPlaintextOp,
            ckks.PrepareMultiplyStaticOp,
        ),
    ):
        if len(operands) != 2 or not isinstance(operands[1], Ciphertext):
            raise RuntimeError(
                f"Preparation {name!r} requires public,ciphertext"
            )
        public = operands[0]
        ciphertext = operands[1]
        assert isinstance(ciphertext, Ciphertext)
        mode = _string_attr(operation, "scale_mode")
        action = _string_attr(operation, "operation")
        source_role = _string_attr(operation, "source_role")
        if action not in {"add", "multiply"}:
            raise RuntimeError("CKKS preparation action is unsupported")
        if source_role not in {"message", "plaintext", "static"}:
            raise RuntimeError("CKKS preparation source role is unsupported")
        if mode not in {
            "runtime_plaintext_scale",
            "default_scale",
            "ciphertext_scale",
        }:
            raise RuntimeError("CKKS preparation scale mode is unsupported")
        return engine.prepare_public_operand(
            public,
            ciphertext,
            operation=cast(Literal["add", "multiply"], action),
            source_role=cast(
                Literal["message", "plaintext", "static"], source_role
            ),
            scale_mode=cast(
                Literal[
                    "runtime_plaintext_scale",
                    "default_scale",
                    "ciphertext_scale",
                ],
                mode,
            ),
        )

    if isinstance(operation, ckks.NegateOp):
        return engine.negate(_ciphertext(operands, 0, name))
    if isinstance(operation, ckks.RotateManyOp):
        ciphertext = _ciphertext(operands, 0, name)
        if len(operands) != 1 + len(operation.keys):
            raise RuntimeError(
                "CKKS rotate-many operands do not match its key operands"
            )
        rotation_keys_for_many: list[RotationKey] = []
        for key_operand, key in zip(operation.keys, operands[1:], strict=True):
            symbol, step = rotation_key_reference(key_operand)
            if type(key) is not RotationKey:
                raise RuntimeError(
                    f"CKKS rotate-many lacks key resource {symbol!r}"
                )
            if key.rotation_step != step:
                raise RuntimeError(
                    f"CKKS rotation key {symbol!r} has another step"
                )
            rotation_keys_for_many.append(key)
        return tuple(
            engine.rotate_many_with_keys(
                ciphertext,
                rotation_keys_for_many,
                output_domain=cast(
                    PolynomialDomain, operation.output_domain.data
                ),
            )
        )
    if isinstance(operation, ckks.RotateOp):
        ciphertext = _ciphertext(operands, 0, name)
        if len(operands) != 2:
            raise RuntimeError(
                "CKKS rotate requires ciphertext and key operands"
            )
        symbol, step = rotation_key_reference(operation.key)
        key = operands[1]
        if type(key) is not RotationKey:
            raise RuntimeError(f"CKKS rotate lacks key resource {symbol!r}")
        if key.rotation_step != step:
            raise RuntimeError(f"CKKS rotation key {symbol!r} has another step")
        return engine.rotate_with_key(
            ciphertext,
            key,
            output_domain=cast(PolynomialDomain, operation.output_domain.data),
        )
    if isinstance(operation, ckks.ToNttOp):
        return engine.coefficient_domain_to_ntt_domain(
            _ciphertext(operands, 0, name)
        )
    if isinstance(operation, ckks.FromNttOp):
        return engine.ntt_domain_to_coefficient_domain(
            _ciphertext(operands, 0, name)
        )
    if isinstance(operation, ckks.ToMontgomeryResiduesOp):
        if len(operands) != 1 or not isinstance(operands[0], Plaintext):
            raise RuntimeError(f"{name} requires a plaintext")
        return engine.standard_residues_to_montgomery_residues(operands[0])
    if isinstance(operation, ckks.ToStandardResiduesOp):
        if len(operands) != 1 or not isinstance(operands[0], Plaintext):
            raise RuntimeError(f"{name} requires a plaintext")
        return engine.montgomery_residues_to_standard_residues(operands[0])
    if isinstance(
        operation,
        (ckks.AddOp, ckks.SubtractOp, ckks.MultiplyOp),
    ):
        left = _ciphertext(operands, 0, name)
        right = _ciphertext(operands, 1, name)
        if isinstance(operation, ckks.AddOp):
            return engine.add(left, right)
        if isinstance(operation, ckks.SubtractOp):
            return engine.subtract(left, right)
        return engine.multiply(left, right)
    if isinstance(
        operation,
        (
            ckks.AddScalarOp,
            ckks.MultiplyScalarOp,
            ckks.MultiplyIntegerScalarOp,
        ),
    ):
        ciphertext = _ciphertext(operands, 0, name)
        if isinstance(operation, ckks.MultiplyIntegerScalarOp):
            scalar = _int_attr(operation, "scalar")
            if scalar is None:
                raise RuntimeError("CKKS integer scalar operation lacks scalar")
            return engine.multiply_integer_scalar(ciphertext, scalar)
        scalar = _float_attr(operation, "scalar")
        scalar_scale = _float_attr(operation, "scalar_scale")
        if scalar is None or scalar_scale is None:
            raise RuntimeError(
                "CKKS real scalar operation lacks scalar metadata"
            )
        if isinstance(operation, ckks.AddScalarOp):
            return engine.add_scalar(
                ciphertext,
                scalar,
                scalar_scale=scalar_scale,
            )
        return engine.multiply_scalar(
            ciphertext,
            scalar,
            scalar_scale=scalar_scale,
        )
    if isinstance(
        operation,
        (
            ckks.AddPlaintextOp,
            ckks.MultiplyPlaintextOp,
            ckks.AddCompressedPlaintextOp,
            ckks.MultiplyCompressedPlaintextOp,
        ),
    ):
        ciphertext = _ciphertext(operands, 0, name)
        if len(operands) < 2 or not isinstance(
            operands[1],
            (Plaintext, CompressedPlaintext),
        ):
            raise RuntimeError(f"{name} requires ciphertext,plaintext")
        method = (
            engine.add_plaintext
            if isinstance(
                operation,
                (ckks.AddPlaintextOp, ckks.AddCompressedPlaintextOp),
            )
            else engine.multiply_plaintext
        )
        return method(ciphertext, operands[1])
    if isinstance(operation, ckks.RelinearizeOp):
        keys = _evaluation_key_set(
            workspace.get("evaluation_keys"),
            EvaluationKeyRequirements(requires_relinearization=True),
        )
        if keys is None or keys.relinearization is None:
            raise RuntimeError(
                "CKKS relinearize lacks matching relinearization-key material"
            )
        return engine.relinearize(
            _ciphertext(operands, 0, name),
            keys.relinearization,
            output_domain=cast(PolynomialDomain, operation.output_domain.data),
        )
    if isinstance(operation, ckks.SwitchKeyOp):
        key = _mapping(workspace, "resources").get(operation.key_symbol.data)
        if type(key) is not KeySwitchKey:
            raise RuntimeError(
                f"CKKS switch-key resource {operation.key_symbol.data!r} "
                "is not an exact KeySwitchKey"
            )
        return engine.switch_key(
            _ciphertext(operands, 0, name),
            key,
            output_domain=cast(PolynomialDomain, operation.output_domain.data),
        )
    if isinstance(operation, ckks.ConjugateOp):
        keys = _evaluation_key_set(
            workspace.get("evaluation_keys"),
            EvaluationKeyRequirements(requires_conjugation=True),
        )
        if keys is None or keys.conjugation is None:
            raise RuntimeError("CKKS conjugate lacks conjugation-key material")
        return engine.conjugate(
            _ciphertext(operands, 0, name),
            keys.conjugation,
            output_domain=cast(PolynomialDomain, operation.output_domain.data),
        )
    if isinstance(operation, ckks.RescaleOp):
        ciphertext = _ciphertext(operands, 0, name)
        rounding = _string_attr(operation, "rounding")
        if rounding not in {"nearest", "floor"}:
            raise RuntimeError("CKKS rescale lacks represented rounding")
        return engine.rescale_to_next_depth(
            ciphertext,
            rounding=cast(Literal["nearest", "floor"], rounding),
        )
    if isinstance(operation, ckks.ModSwitchOp):
        return engine.mod_switch_to_depth(
            _ciphertext(operands, 0, name),
            int(operation.target_depth.value.data),
        )
    if isinstance(operation, ckks.ReinterpretScaleOp):
        return engine.reinterpret_at_scale(
            _ciphertext(operands, 0, name),
            float(operation.scale.value.data),
        )
    raise RuntimeError(f"Unhandled built-in CKKS operation {name!r}")


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
    "InterpreterNotCoveredError",
    "InterpreterDiagnostic",
    "InterpreterCoverage",
    "check_interpreter_coverage",
    "interpret_program",
]
