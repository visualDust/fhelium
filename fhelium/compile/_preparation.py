"""Organize input specialization, Compile passes, and Backend linking."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
import dataclasses
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import torch

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, f64
from xdsl.dialects import scf
from xdsl.ir import Attribute, BlockArgument, OpResult, SSAValue
from xdsl.rewriter import Rewriter
from xdsl.traits import IsTerminator

from fhelium.ir import Program
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import OpenStateType
from fhelium.values import Ciphertext, CompressedPlaintext, Plaintext

from ._compilation import Compilation, DeferredCompilation
from ._lower_and_fuse import default_lower_and_fuse_pipeline
from ._pipeline import Pipeline
from ._specialization import CallSignature, prepare_match
from ._workspace import CompileWorkspace
from .frontend._eager_values import state_attributes

if TYPE_CHECKING:
    from fhelium.backend import OperationBackend, ProgramExecutable
    from fhelium.backend.resources import BoundResource
    from fhelium.values import EvaluationKeySet


@dataclass(frozen=True)
class LinkArtifact:
    """A linked executable recorded in a cache file and the provider selections of its generated materials."""

    record: Mapping[str, Any]
    selections: Mapping[str, Mapping[str, object] | None]


@dataclass(frozen=True)
class _CompiledVariant:
    signature: CallSignature
    source: Compilation | Callable[[], Compilation]
    compilation: Compilation
    input_names: tuple[str, ...]
    output_structure: object | None
    argument_types: tuple[type, ...]
    input_fields: Mapping[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    artifact: LinkArtifact | None = None


@dataclass(frozen=True)
class Specialization:
    """Pair an input-specialized computation with its linked executable.

    ``source`` records the computation before specialization. ``compilation`` contains the transformed Program and pass reports; ``backend`` and ``executable`` provide its linked execution. The executable retains its prepared bindings. A new plaintext-material generation reuses prepared dispatch and host bytecode while replacing material bindings; a changed Program requires preparation and linking.
    """

    _variant: _CompiledVariant
    backend: OperationBackend
    executable: ProgramExecutable

    @property
    def signature(self) -> CallSignature:
        return self._variant.signature

    @cached_property
    def source(self) -> Compilation:
        source = self._variant.source
        return source() if callable(source) else source

    @property
    def compilation(self) -> Compilation:
        return self._variant.compilation

    @cached_property
    def _result_count(self) -> int:
        return len(self.executable.output_types)

    @cached_property
    def _matches(self):
        return prepare_match(self.signature, self._variant.argument_types)

    @cached_property
    def _execute(self):
        """Adapt matched arguments and restore a fixed output structure.

        The returned function takes the argument values and, optionally, the executable entry point to call instead of ``executable._invoke``.
        """
        from fhelium.backend.execution import (
            _boundary_kind,
            _prepare_compressed_output,
        )
        from fhelium.values import KeySwitchKey
        from operator import attrgetter

        positions = {
            argument.name: i
            for i, argument in enumerate(self.signature.arguments)
        }
        namespace: dict[str, Any] = {"invoke": self.executable._invoke}
        operands = []
        fields = self._variant.input_fields
        for ordinal, (name, value_type, adapter) in enumerate(
            zip(
                self._variant.input_names,
                self.executable.input_types,
                self.executable._input_adapters,
                strict=True,
            )
        ):
            source_name, field = fields.get(name, (name, None))
            position = positions[source_name]
            typ = self._variant.argument_types[position]
            role = _boundary_kind(value_type)
            if field is not None:
                pass
            elif isinstance(
                value_type, ckks.CompressedPlaintextType
            ) and issubclass(typ, CompressedPlaintext):
                adapter = attrgetter("data")
            elif role == "ciphertext" and issubclass(typ, Ciphertext):
                adapter = attrgetter("data")
            elif role == "plaintext" and issubclass(typ, Plaintext):
                representation = self.signature.arguments[position].get(
                    "representation"
                )
                adapter = attrgetter(
                    "message" if representation == "slots" else "data"
                )
            elif isinstance(value_type, ckks.EvaluationKeyType) and issubclass(
                typ, KeySwitchKey
            ):
                adapter = attrgetter("data")
            namespace[f"adapt{ordinal}"] = adapter
            namespace[f"field{ordinal}"] = field
            actual = (
                f"values[{position}]"
                if field is None
                else f"getattr(values[{position}], field{ordinal})"
            )
            operands.append(f"adapt{ordinal}({actual})")

        def literal(value):
            name = f"literal{len(namespace)}"
            namespace[name] = _literal(value)
            return name

        def output(descriptor):
            kind = descriptor["kind"]
            if kind == "ssa":
                return (
                    "result"
                    if self._result_count == 1
                    else f"result[{descriptor['result']}]"
                )
            if kind == "compressed_plaintext":
                fields = dict(descriptor["fields"])
                fields["prime_ids"] = tuple(fields["prime_ids"])
                name = f"compressed_output{len(namespace)}"
                namespace[name] = _prepare_compressed_output(fields)
                return f"{name}({output(descriptor['data'])}, {output(descriptor['implicit'])})"
            if kind == "literal":
                return literal(descriptor["value"])
            if kind in {"tuple", "list"}:
                items = [output(item) for item in descriptor["items"]]
                if kind == "list":
                    return "[" + ", ".join(items) + "]"
                return (
                    "("
                    + ", ".join(items)
                    + ("," if len(items) == 1 else "")
                    + ")"
                )
            if kind == "mapping":
                return (
                    "{"
                    + ", ".join(
                        f"{literal(key)}: {output(item)}"
                        for key, item in descriptor["entries"]
                    )
                    + "}"
                )
            raise ValueError(f"Unsupported callable output descriptor {kind!r}")

        try:
            expression = (
                "result"
                if self._variant.output_structure is None
                else output(self._variant.output_structure)
            )
        finally:
            # Release the recursive closure's reference to itself.
            output = cast(Any, None)
        # ``invoke`` may be replaced by the executable's Graph route.
        source = f"def execute(values, invoke=invoke):\n    result = invoke({', '.join(operands)})\n    return {expression}\n"
        exec(compile(source, "<fhelium-matched-call>", "exec"), namespace)
        return namespace.pop("execute")


def fresh_workspace(workspace: Mapping[object, object]) -> CompileWorkspace:
    """Construct a workspace with a shallow copy of the caller's entries.

    Material payloads and other caller-provided mutable workspace entries remain
    shared.
    """

    result = CompileWorkspace(workspace)
    return result


def runtime_names(program: Program) -> tuple[str, ...]:
    arguments = tuple(program.single_block().args)
    represented = program.module.attributes.get("fhelium.input_names")
    if (
        isinstance(represented, ArrayAttr)
        and len(represented) == len(arguments)
        and all(isinstance(item, StringAttr) for item in represented)
    ):
        return tuple(
            item.data for item in represented if isinstance(item, StringAttr)
        )
    return tuple(
        argument.name_hint or f"arg{index}"
        for index, argument in enumerate(arguments)
    )


def input_fields(program: Program) -> dict[str, tuple[str, str]]:
    """Map additional Tensor inputs to their public argument storage field."""
    encoded = program.module.attributes.get("fhelium.input_fields")
    return json.loads(encoded.data) if isinstance(encoded, StringAttr) else {}


def field_values(
    fields: Mapping[str, tuple[str, str]], arguments: Mapping[str, object]
) -> dict[str, object]:
    """Add the storage-field Tensors named by ``fields`` to call arguments."""
    values = dict(arguments)
    for name, (source, attribute) in fields.items():
        values[name] = getattr(arguments[source], attribute)
    return values


def input_values(
    program: Program, arguments: Mapping[str, object]
) -> dict[str, object]:
    return field_values(input_fields(program), arguments)


def _refine(value: SSAValue, refined: Attribute) -> None:
    """Give ``value`` the refined input type and carry it to the values declared with the same type.

    Structured control flow carries the type of a value through the iteration arguments, body arguments and results of ``scf.for``, and from a region terminator to the corresponding results of its operation and the iteration arguments of ``scf.for``. Any other operation without a further operand of the value's former type passes the refined type to its results and to its region arguments declared with that type, such as the combine arguments of an all-reduce.
    """
    pending = [(value, refined)]
    while pending:
        current, new_type = pending.pop()
        old_type = current.type
        # A value reached along two paths is replaced on the first.
        if isinstance(current, OpResult):
            live = current.op.results[current.index]
        elif isinstance(current, BlockArgument):
            live = current.block.args[current.index]
        else:
            live = current
        if old_type == new_type or live is not current:
            continue
        current = Rewriter.replace_value_with_new_type(current, new_type)
        carried: list[SSAValue] = []
        for use in tuple(current.uses):
            owner = use.operation
            if isinstance(owner, scf.ForOp) and use.index >= 3:
                position = use.index - 3
                carried += [
                    owner.body.block.args[position + 1],
                    owner.results[position],
                ]
            elif owner.has_trait(IsTerminator):
                parent = owner.parent_op()
                if parent is not None and use.index < len(parent.results):
                    carried.append(parent.results[use.index])
                if isinstance(parent, scf.ForOp):
                    carried.append(parent.body.block.args[use.index + 1])
            elif not any(
                operand is not current and operand.type == old_type
                for operand in owner.operands
            ):
                carried += owner.results
                carried += [
                    argument
                    for region in owner.regions
                    if region.blocks.first is not None
                    for argument in region.blocks.first.args
                ]
        pending += [
            (item, new_type) for item in carried if item.type == old_type
        ]


def _input_specialization(
    compilation: Compilation, arguments: Mapping[str, object]
) -> Compilation:
    program = compilation.program.clone()
    block = program.single_block()
    names = runtime_names(program)
    arguments = input_values(program, arguments)
    for name, argument in zip(names, tuple(block.args), strict=True):
        represented = argument.type
        value = arguments[name]
        if isinstance(represented, ckks.CiphertextType) and not isinstance(
            value, Ciphertext
        ):
            raise TypeError(f"Program input {name!r} requires a Ciphertext")
        if isinstance(represented, ckks.PlaintextType) and not isinstance(
            value, Plaintext
        ):
            raise TypeError(f"Program input {name!r} requires a Plaintext")
        if not isinstance(represented, OpenStateType):
            continue
        actual = state_attributes(value)
        updates: dict[str, Attribute] = {}
        for field, value in actual.items():
            prior = represented.state.data.get(field)
            if (
                field == "scale"
                and isinstance(prior, StringAttr)
                and prior.data != "unknown"
            ):
                prior = FloatAttr(float(prior.data), f64)
                updates[field] = prior
            if prior is None or prior == StringAttr("unknown"):
                updates[field] = value
            elif prior != value:
                raise ValueError(
                    f"Program input {name!r} requires {field}={prior}, "
                    f"but this invocation has {value}"
                )
        if updates:
            _refine(argument, represented.with_state(updates))
    program.function().update_function_type()
    return Compilation(
        program,
        fresh_workspace(compilation.workspace),
        compilation.reports,
        dict(compilation.material_bindings),
    )


def prepare_variant(
    source: Compilation,
    arguments: Mapping[str, object],
    signature: CallSignature,
    *,
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None,
    backend: OperationBackend,
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
) -> tuple[_CompiledVariant, OperationBackend]:
    """Prepare a Program and select the Backend used to bind its execution."""

    from fhelium.backend.implementation import OperationImplementationRegistry

    # The specialization owns its Program copy and transfers it to the pipeline.
    request = _input_specialization(source, arguments)
    request.workspace[OperationImplementationRegistry] = backend.registry
    if keys:
        request.workspace["evaluation_keys"] = keys
    if pipeline is None:
        optimized = default_lower_and_fuse_pipeline(backend).run(
            request, clone=False
        )
    else:
        if not isinstance(pipeline, Pipeline):
            pipeline = pipeline(signature)
        optimized = pipeline.run(request, clone=False)
    input_names = runtime_names(optimized.program)
    if any(
        name not in input_values(optimized.program, arguments)
        for name in input_names
    ):
        raise ValueError(
            "The preparation pipeline changed the callable's input interface; "
            "transform the Compilation first and create a callable for its new interface"
        )
    output = optimized.program.module.attributes.get("fhelium.output_structure")
    return (
        _CompiledVariant(
            signature,
            source,
            optimized,
            input_names,
            json.loads(output.data) if isinstance(output, StringAttr) else None,
            tuple(
                type(arguments[argument.name])
                for argument in signature.arguments
            ),
            input_fields(optimized.program),
        ),
        backend,
    )


def captured_group_names(
    workspace: CompileWorkspace,
) -> dict[str, str | None] | None:
    """Return the c10d registry names of the captured process groups by resource symbol; None names the default group."""
    from .frontend._eager_capture import CAPTURED_GROUPS

    groups = workspace.get(CAPTURED_GROUPS)
    if not groups:
        return None
    return {
        group.symbol: None
        if (pg := cast(Any, group.value).group) is None
        else pg.group_name
        for group in groups
    }


def restored_groups(
    names: Mapping[str, str | None],
) -> tuple[BoundResource, ...]:
    """Bind the recorded group symbols to the live c10d groups registered under the recorded names."""
    from torch._C._distributed_c10d import _resolve_process_group

    from fhelium.backend.distributed import ProcessGroupExecutionResource
    from fhelium.backend.resources import BoundResource

    return tuple(
        BoundResource(
            symbol,
            "process-group",
            ProcessGroupExecutionResource(
                None
                if name is None
                else _resolve_process_group(cast(Any, name))
            ),
        )
        for symbol, name in names.items()
    )


def _with_groups(
    backend: OperationBackend, compilation: Compilation
) -> OperationBackend:
    """Add the process groups of captured collectives to the Backend's resources.

    A Program restored from a cache entry names the default process group as ``world``, which is bound to the live default group.
    """
    from fhelium.backend.resources import BoundResource, ResourceBindings

    from .frontend._eager_capture import CAPTURED_GROUPS

    groups = compilation.workspace.get(CAPTURED_GROUPS)
    if groups is None and torch.distributed.is_initialized():
        from fhelium.backend.distributed import ProcessGroupExecutionResource

        groups = (
            BoundResource(
                "world", "process-group", ProcessGroupExecutionResource()
            ),
        )
    # Groups the Backend already binds keep their binding.
    present = {
        resource.symbol
        for resource in backend.workspace.named_resources.resources
    }
    groups = tuple(
        group for group in groups or () if group.symbol not in present
    )
    if not groups:
        return backend
    return backend.with_named_resources(
        ResourceBindings(groups), override=False
    )


def bind_variant(
    variant: _CompiledVariant,
    backend: OperationBackend,
    *,
    reload: Callable[[], Program] | None = None,
) -> Specialization:
    """Link a variant, or restore its persisted executable when one is attached.

    With ``reload``, linking consumes the variant's Program in place and the returned Specialization obtains later Program requests from ``reload``.
    """
    compilation = variant.compilation
    backend = _with_groups(backend, compilation)
    if variant.artifact is None:
        if reload is None:
            return Specialization(variant, backend, backend.link(compilation))
        executable = backend.link(compilation, clone=False)
        deferred = DeferredCompilation(
            reload,
            compilation.workspace,
            compilation.reports,
            compilation.material_bindings,
        )
        return Specialization(
            replace(variant, compilation=deferred), backend, executable
        )
    from fhelium.backend._artifact import load_device, restore_executable
    from fhelium.backend.runner import LoadedArtifact, group_name

    linked = []

    def relink():
        if not linked:
            fresh = Compilation(
                compilation.program.clone()
                if not isinstance(compilation, DeferredCompilation)
                else compilation._load(),
                compilation.workspace,
                compilation.reports,
                compilation.material_bindings,
            )
            linked.append(backend.link(fresh, clone=False))
        return linked[0]

    record = variant.artifact.record
    executable = record["executable"]
    try:
        loaded = LoadedArtifact(record["path"], load_device(executable))
    except RuntimeError as error:
        if "identity mismatch" not in str(error):
            raise
        # Kernels and schedules from another native library or Triton
        # version are relinked from the Program.
        return Specialization(
            replace(variant, artifact=None), backend, relink()
        )
    # Collectives run on the process groups the Backend binds to their symbols.
    groups = {
        resource.symbol: resource
        for resource in backend.workspace.named_resources.resources
    }
    for symbol in sorted(
        {
            group["symbol"]
            for entry in loaded.manifest["entries"].values()
            for group in entry.get("groups", ())
        }
    ):
        loaded.bind_group(symbol, group_name(groups[symbol]))
    return Specialization(
        variant,
        backend,
        restore_executable(
            loaded,
            executable,
            backend,
            compilation.material_bindings,
            relink=relink,
        ),
    )


def _literal(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "complex":
        return complex(value["real"], value["imag"])
    return value


def restore_output(descriptor: Any, results: tuple[object, ...]) -> object:
    """Restore the callable's tuple/list/mapping output from Program results."""

    kind = descriptor["kind"]
    if kind == "compressed_plaintext":
        fields = dict(descriptor["fields"])
        fields["prime_ids"] = tuple(fields["prime_ids"])
        return CompressedPlaintext(
            data=cast(
                torch.Tensor, restore_output(descriptor["data"], results)
            ),
            implicit_data=cast(
                torch.Tensor, restore_output(descriptor["implicit"], results)
            ),
            **fields,
        )
    if kind == "ssa":
        return results[descriptor["result"]]
    if kind == "literal":
        return _literal(descriptor["value"])
    if kind == "tuple":
        return tuple(
            restore_output(item, results) for item in descriptor["items"]
        )
    if kind == "list":
        return [restore_output(item, results) for item in descriptor["items"]]
    if kind == "mapping":
        return {
            _literal(key): restore_output(item, results)
            for key, item in descriptor["entries"]
        }
    raise ValueError(f"Unsupported callable output descriptor {kind!r}")


__all__ = ["Specialization"]
