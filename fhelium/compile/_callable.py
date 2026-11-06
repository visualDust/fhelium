"""Own callable specialization and reuse of linked execution programs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from functools import update_wrapper
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)

from fhelium.ir import Program, value_role

from ._compilation import Compilation
from ._materials import material_sources, prepare_material_bindings
from ._pipeline import Pipeline
from ._preparation import (
    Specialization,
    _CompiledVariant,
    bind_variant,
    fresh_workspace,
    prepare_variant,
    runtime_names,
    input_fields,
    input_values,
)
from ._specialization import CallSignature, SpecializationMiss, describe_call
from ._workspace import CompileWorkspace
from .frontend._captured_callable import CapturedCallable
from .frontend._invocation import capture_invocation
from .frontend._objects import python_callable
from .frontend._specs import InputSpec

if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext
    from fhelium.values import EvaluationKeySet
    from fhelium.backend import OperationBackend

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _capture_origin(compilation: Compilation) -> CapturedCallable[Any] | None:
    """Use frontend signature facts only while the Program still matches them."""

    origin = compilation.workspace.get(CapturedCallable)
    if not isinstance(origin, CapturedCallable):
        return None
    names = tuple(origin.runtime_signature.parameters)
    arguments = tuple(compilation.program.single_block().args)
    represented_names = runtime_names(compilation.program)
    fields = input_fields(compilation.program)
    primary = [
        (name, argument)
        for name, argument in zip(represented_names, arguments, strict=True)
        if name not in fields
    ]
    if tuple(name for name, _ in primary) != names:
        return None
    if any(
        value_role(argument) != origin.input_specs[name].role
        for name, argument in primary
    ):
        return None
    return origin


def _prepare_argument_binding(signature: inspect.Signature):
    """Use Python's own parameter binding without allocating BoundArguments."""
    namespace = {}
    fields = []
    parameters = tuple(signature.parameters.values())
    keyword_only = False
    for index, parameter in enumerate(parameters):
        if (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not keyword_only
        ):
            fields.append("*")
            keyword_only = True
        field = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            default = f"default{index}"
            namespace[default] = parameter.default
            field += f"={default}"
        fields.append(field)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and (
            index + 1 == len(parameters)
            or parameters[index + 1].kind
            is not inspect.Parameter.POSITIONAL_ONLY
        ):
            fields.append("/")
    values = ", ".join(parameter.name for parameter in parameters)
    expression = "(" + values + ("," if len(parameters) == 1 else "") + ")"
    source = f"def bind({', '.join(fields)}):\n    return {expression}\n"
    exec(compile(source, "<fhelium-argument-binding>", "exec"), namespace)
    return namespace["bind"]


class CompiledCallable(Generic[_P, _R]):
    """A callable that caches input-specialized executables for one source computation.

    ``prepare`` creates and links a specialization for given arguments. Calls reuse a matching executable; a cache miss prepares a new specialization by default, or raises ``SpecializationMiss`` when ``on_miss='error'``. ``clear`` releases the cached specializations.
    """

    def __init__(
        self,
        source: Program | Compilation | Callable[_P, _R],
        *,
        backend: OperationBackend | None = None,
        resources: CkksDeviceResources
        | RnsContext
        | NttContext
        | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
        keys: Mapping[object, object]
        | Iterable[object]
        | EvaluationKeySet = (),
        pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
        on_miss: Literal["compile", "error"] = "compile",
        workspace: CompileWorkspace | None = None,
        inputs: Mapping[str, InputSpec] | None = None,
        material_names: Mapping[str, object] | None = None,
    ) -> None:
        """Record the source and options used to prepare specializations.

        Program sources are cloned. Workspace and material-binding dictionaries are shallow-copied; referenced Tensors and custom mutable entries remain shared. Python sources retain their signature, function metadata, and ``reference`` callable. Construct a new object after changing the source Program or pass policy.

        Args:
            source: Python callable, Program, or Compilation to specialize.
            backend: Implementation registry and execution resources. ``None`` selects the standard operation registry during preparation.
            resources: Numerical table providers used when preparing execution.
            keys: Existing key values or an inventory used to select layouts and bind execution.
            pipeline: Complete pass sequence, or a function from ``CallSignature`` to Pipeline. ``None`` selects ``default_lower_and_fuse_pipeline``.
            on_miss: ``"compile"`` prepares a missing specialization on invocation; ``"error"`` requires a matching specialization from ``prepare``.
            workspace: Shared capture and pass inputs for a Python callable or Program source. A Compilation supplies its own workspace.
            inputs: Input-role declarations selecting the FX frontend for Python capture.
            material_names: Stable symbols for fixed Tensor/value objects encountered during Python capture.
        """

        if on_miss not in ("compile", "error"):
            raise ValueError("on_miss must be 'compile' or 'error'")
        if material_names is not None and isinstance(
            source, (Program, Compilation)
        ):
            raise ValueError(
                "material_names applies to Python capture; Program symbols are already defined"
            )
        from fhelium.values import EvaluationKeySet

        self.backend: OperationBackend | None = backend
        self._resources = material_sources(resources)
        self._keys = (
            keys
            if isinstance(keys, (Mapping, EvaluationKeySet))
            else tuple(keys)
        )
        self.pipeline = pipeline
        self.on_miss: Literal["compile", "error"] = on_miss
        self.inputs = None if inputs is None else dict(inputs)
        self._material_names = (
            None if material_names is None else dict(material_names)
        )
        self._source: Compilation | Callable[_P, _R]
        self._workspace: CompileWorkspace | None
        self.reference: Callable[_P, _R] | None = None
        origin: CapturedCallable[Any] | None = None
        if isinstance(source, Compilation):
            if workspace is not None:
                raise ValueError("A Compilation already supplies its workspace")
            self._source = Compilation(
                source.program.clone(),
                fresh_workspace(source.workspace),
                source.reports,
                dict(source.material_bindings),
            )
            self._workspace = None
            origin = _capture_origin(self._source)
        elif isinstance(source, Program):
            self._source = Compilation(
                source.clone(),
                fresh_workspace({} if workspace is None else workspace),
            )
            self._workspace = None
        elif callable(source):
            python_callable(source)
            self._source = source
            self._workspace = fresh_workspace(
                {} if workspace is None else workspace
            )
            self.reference = source
        else:
            raise TypeError(
                "compile source must be a Python callable, Program, or Compilation"
            )
        if isinstance(self._source, Compilation):
            if inputs is not None:
                raise ValueError(
                    "inputs selects Python capture and cannot accompany a Program or Compilation"
                )
            if origin is not None:
                self.reference = cast(Callable[_P, _R], origin.reference)
                update_wrapper(self, origin.function, updated=())
                self.__signature__ = origin.runtime_signature
            else:
                self.__signature__ = inspect.Signature(
                    tuple(
                        inspect.Parameter(
                            name, inspect.Parameter.POSITIONAL_OR_KEYWORD
                        )
                        for name in runtime_names(self._source.program)
                        if name not in input_fields(self._source.program)
                    )
                )
        else:
            update_wrapper(self, self._source, updated=())
            self.__signature__ = inspect.signature(self._source)
            if any(
                parameter.kind
                in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
                for parameter in self.__signature__.parameters.values()
            ):
                raise TypeError(
                    "Compiled functions require a fixed Python signature"
                )
        self._variants: dict[CallSignature, _CompiledVariant] = {}
        self._prepared: dict[CallSignature, Specialization] = {}
        self._prepare_lock = RLock()
        self._bind_arguments = _prepare_argument_binding(self.__signature__)
        self._argument_names = tuple(self.__signature__.parameters)
        self._last_prepared: Specialization | None = None

    @property
    def specializations(self) -> tuple[Specialization, ...]:
        """Return successfully linked variants in preparation order."""

        return tuple(self._prepared.values())

    def _arguments(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        return dict(
            zip(
                self._argument_names,
                self._bind_arguments(*args, **kwargs),
                strict=True,
            )
        )

    def _capture(self, arguments: Mapping[str, object]) -> Compilation:
        if isinstance(self._source, Compilation):
            return self._source
        return capture_invocation(
            self._source,
            arguments,
            inputs=self.inputs,
            material_names=self._material_names,
            workspace=fresh_workspace(cast(CompileWorkspace, self._workspace)),
        )

    def _prepare(
        self, arguments: Mapping[str, object], signature: CallSignature
    ) -> Specialization:
        with self._prepare_lock:
            prepared = self._prepared.get(signature)
            if prepared is not None:
                return prepared
            variant = self._variants.get(signature)
            backend = self.backend
            if variant is None:
                source = self._capture(arguments)
                if backend is None:
                    from fhelium.backend.execution import OperationBackend

                    backend = OperationBackend()
                variant, backend = prepare_variant(
                    source,
                    arguments,
                    signature,
                    pipeline=self.pipeline,
                    backend=backend,
                    keys=self._keys,
                )
            if self._resources or self._keys:
                prepare_material_bindings(
                    variant.compilation,
                    resources=self._resources,
                    keys=self._keys,
                )
            prepared = bind_variant(variant, cast("OperationBackend", backend))
            physical_arguments = input_values(
                variant.compilation.program, arguments
            )
            for name, adapter in zip(
                variant.input_names,
                prepared.executable._input_adapters,
                strict=True,
            ):
                adapter(physical_arguments[name])
            self._variants[signature] = variant
            self._prepared[signature] = prepared
            return prepared

    def prepare(self, *args: _P.args, **kwargs: _P.kwargs) -> Specialization:
        """Prepare and return the specialization matching these arguments.

        Preparation captures the source when needed, runs its selected pipeline, and links the resulting Program. A matching cached specialization is returned directly. This method populates the cache under either ``on_miss`` policy.

        Device-kernel compilation may occur on first execution after linking. Warm representative calls before measuring steady-state execution or capturing device launches.
        """

        arguments = self._arguments(args, kwargs)
        return self._prepare(arguments, describe_call(arguments))

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Execute a matching specialization, preparing it on a permitted cache miss."""

        values = self._bind_arguments(*args, **kwargs)
        prepared = self._last_prepared
        if prepared is not None and prepared._matches(values):
            return cast(_R, prepared._execute(values))
        for candidate in self._prepared.values():
            if candidate is not prepared and candidate._matches(values):
                self._last_prepared = candidate
                return cast(_R, candidate._execute(values))
        if self.on_miss == "error":
            raise SpecializationMiss(
                "No prepared specialization matches this invocation; "
                "call prepare with these input conditions first"
            )
        arguments = dict(zip(self._argument_names, values, strict=True))
        prepared = self._prepare(arguments, describe_call(arguments))
        self._last_prepared = prepared
        return cast(_R, prepared._execute(values))

    def with_backend(
        self, backend: OperationBackend
    ) -> CompiledCallable[_P, _R]:
        """Relink a snapshot of compiled variants against another Backend.

        Call ``prepare`` to link the returned callable before execution when ``on_miss='error'``. Existing compilations retain their selected implementations and target/ABI requirements, which the new Backend must satisfy. Each callable prepares new input signatures independently.
        """

        rebound: CompiledCallable[_P, _R] = CompiledCallable(
            self._source,
            backend=backend,
            resources=self._resources,
            keys=self._keys,
            pipeline=self.pipeline,
            on_miss=self.on_miss,
            workspace=self._workspace,
            inputs=self.inputs,
            material_names=self._material_names,
        )
        with self._prepare_lock:
            rebound._variants = dict(self._variants)
        return rebound

    def clear(self) -> None:
        """Release this callable's specialization cache.

        Call this after changing captured Python constants so the next preparation captures their new values. Changes to input Tensor contents reuse the existing specialization while its recorded input conditions hold. Caller resources and compiled results retained by peers made with ``with_backend`` remain live.
        """

        with self._prepare_lock:
            self._variants.clear()
            self._prepared.clear()
            self._last_prepared = None


__all__ = ["CompiledCallable"]
