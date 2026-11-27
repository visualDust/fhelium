"""Own callable specialization and reuse of linked execution programs."""

from __future__ import annotations

import inspect
import time
import types
import weakref
from collections.abc import Callable, Iterable, Mapping
from functools import update_wrapper
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    overload,
)

import torch

from fhelium.ir import Program, value_role

from ._compilation import Compilation, DeferredCompilation
from ._binding_time import PREPARATION, PreparationPlan
from ._cache import (
    CacheEntry,
    CacheIneligible,
    CompileCache,
    SourceIdentity,
    artifact_key,
    build_manifest,
    cache_directory,
    entry_key,
    relocate_materials,
    source_identity,
    runtime_cacheable,
    supplied_keys,
    trace_entry_key,
)
from ._graph import _GraphRoute
from ._context import execution_context
from ._timing import ExecutionTiming
from ._streams import PreparedStreams
from ._materials import (
    bind_provided_materials,
    key_description,
    material_sources,
    prepare_material_bindings,
    provided_material_selections,
)
from ._pipeline import Pipeline
from ._preparation import (
    LinkArtifact,
    Specialization,
    _CompiledVariant,
    bind_variant,
    captured_group_names,
    fresh_workspace,
    restored_groups,
    prepare_variant,
    runtime_names,
    field_values,
    input_fields,
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


def _argument_tensors(value: object) -> tuple[Any, ...]:
    import torch

    from fhelium.serialization import ValueEnvelope
    from fhelium.values import TensorResident

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, TensorResident):
        return tuple(
            ValueEnvelope.from_value(value, compact_keys=False).tensors.values()
        )
    return ()


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
    return namespace.pop("bind")


def _record_fields(record: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        name: cast(tuple[str, str], tuple(item))
        for name, item in record["input_fields"].items()
    }


def _weak(value: object) -> Callable[[], object]:
    try:
        return weakref.ref(value)
    except TypeError:
        return lambda: value


def _state_tensor(executable: Any, index: int, name: str) -> torch.Tensor:
    resource = executable.resources[index]
    return getattr(getattr(resource, "value", resource), name)


class _CachedProgram:
    """Produce a Program from its encoded payload, then from the persistent cache entry."""

    def __init__(self, cache: CompileCache, key: str, payload: object) -> None:
        self._cache, self._key, self._payload = cache, key, payload

    def release(self) -> None:
        """Drop the retained payload once the cache entry holds it."""
        self._payload = None

    def __call__(self) -> Program:
        if isinstance(self._payload, dict):
            return Program.decode(self._payload, verify=False)
        entry = self._cache.load(self._key)
        if entry is None or entry.program is None:
            raise RuntimeError(f"Cached Program {self._key} is unavailable")
        return cast(Program, entry.program)


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
        cache: bool | str | Path = True,
        graph: Literal["auto"] | bool = "auto",
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
            cache: Persist prepared Programs across processes. ``True`` uses ``FHELIUM_CACHE_DIR`` or the user cache directory, a path selects a directory, ``False`` disables persistence. Python sources with the default compilation configuration are persisted. Functions within the statically checked subset are found before capture; other sources are captured and found by their captured Program, bound material layouts and call signature, which skips the pass pipeline. Material Tensors never enter the cache.
            graph: Execute CUDA specializations through CUDA Graphs of their schedules' capture-safe segments, captured at the second execution and replayed afterwards. ``"auto"`` lets each segment compare the device time of its first execution with the median of the three replays after its capture and keep its graph only when the replays are faster; a slower segment releases its graph and runs directly from then on. ``True`` keeps every segment's graph and ``False`` disables capture. ``"auto"`` and ``True`` fall back to prepared execution on CPU inputs, inputs that require gradients, or a failure before the first capture. Graph results occupy the specialization's output storage without a copy and move to fresh storage when a later execution would overwrite a result the caller still references.
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

        self._options: dict[str, Any] = dict(
            backend=backend,
            resources=resources,
            keys=keys,
            pipeline=pipeline,
            on_miss=on_miss,
            workspace=workspace,
            inputs=inputs,
            material_names=material_names,
            cache=cache,
            graph=graph,
        )
        self._name: str | None = None
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
        # Cache entries read for a specialization being built; a link that
        # cannot use their linked executable stores one with the same entry.
        self._loaded: dict[CallSignature, CacheEntry] = {}
        self._ready_streams: dict[CallSignature, PreparedStreams] = {}
        self._bind_arguments = _prepare_argument_binding(self.__signature__)
        self._argument_names = tuple(self.__signature__.parameters)
        self._last_prepared: Specialization | None = None
        directory = (
            cache_directory()
            if cache is True
            else None
            if cache is False
            else Path(cache)
        )
        standard = (
            not isinstance(self._source, Compilation)
            and pipeline is None
            and inputs is None
            and backend is None
            and workspace is None
            and not self._resources
        )
        self._cache = (
            CompileCache(directory)
            if standard and directory is not None
            else None
        )
        self._resident_hits = 0
        # Specialization stages, and "__cache__" for source-cache eligibility.
        self._stages: dict[CallSignature | str, dict[str, object]] = {}
        self._execution_lock = RLock()
        self._graphs = _GraphRoute(graph=graph)
        self._timing = ExecutionTiming()

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(
        self, instance: object, owner: type | None = None
    ) -> CompiledCallable[..., _R]: ...

    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> CompiledCallable[..., _R]:
        """Return the callable compiled for ``instance``'s bound method, or this callable on class access.

        Each instance keeps its own compiled callable, and therefore its own specializations, in its ``__dict__``. A copied instance whose ``__dict__`` holds the callable of another instance receives a new one.
        """
        if instance is None or type(self._source) is not types.FunctionType:
            return self
        try:
            stored = vars(instance)
        except TypeError:
            raise TypeError(
                "Compiled methods keep per-instance specializations in the instance __dict__"
            ) from None
        attribute = f"__fhelium_compiled_{self._name or f'{id(self):x}'}"
        bound = stored.get(attribute)
        if (
            type(bound) is not CompiledCallable
            or getattr(bound._source, "__self__", None) is not instance
        ):
            bound = CompiledCallable(
                types.MethodType(self._source, instance), **self._options
            )
            stored[attribute] = bound
        return bound

    def __set__(self, instance: object, value: object) -> None:
        raise AttributeError(
            "Compiled methods cannot be reassigned per instance"
        )

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
        workspace = fresh_workspace(cast(CompileWorkspace, self._workspace))
        return capture_invocation(
            self._source,
            arguments,
            inputs=self.inputs,
            material_names=self._material_names,
            workspace=workspace,
        )

    def _source_identity(self) -> SourceIdentity | None:
        if self._cache is None:
            return None
        try:
            return source_identity(
                self._source, material_names=self._material_names
            )
        except CacheIneligible as error:
            self._stages.setdefault("__cache__", {})["ineligible"] = str(error)
            return None

    def _load_cached(
        self,
        arguments: Mapping[str, object],
        signature: CallSignature,
        stages: dict[str, object],
        identity: SourceIdentity | None,
        backend: OperationBackend,
    ) -> _CompiledVariant | None:
        """Restore a prepared variant from the persistent cache when eligible."""
        if identity is None or self._cache is None:
            return None
        key = entry_key(identity, signature, self._keys)
        stages["cache_key"] = key
        started = time.perf_counter()
        entry = self._cache.load(key, decode=False)
        if entry is None:
            stages["cache_load_seconds"] = time.perf_counter() - started
            return None
        try:
            bindings, providers = relocate_materials(
                self._source,
                entry.manifest,
                material_names=self._material_names,
            )
        except (
            CacheIneligible,
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            stages["cache_miss_reason"] = f"{type(error).__name__}: {error}"
            return None
        from fhelium.backend.implementation import (
            OperationImplementationRegistry,
        )
        from fhelium.config import CkksConfig

        workspace = fresh_workspace(cast(CompileWorkspace, self._workspace))
        workspace[OperationImplementationRegistry] = backend.registry
        if entry.groups:
            from .frontend._eager_capture import CAPTURED_GROUPS

            workspace[CAPTURED_GROUPS] = restored_groups(entry.groups)
        if entry.preparation is not None:
            plan = PreparationPlan.restore(
                entry.preparation,
                source=self._source,
                arguments=arguments,
                bindings=bindings,
                material_names=self._material_names,
            )
            bindings.update(plan.bindings)
            workspace[PREPARATION] = plan
        self._loaded[signature] = entry
        program = _CachedProgram(self._cache, key, entry.program)
        linked = self._linked(entry, backend, stages)
        if linked is not None:
            artifact = LinkArtifact(linked, linked["selections"])
            keyed = self._bind_keys(bindings, linked)
            for engine, device in providers:
                workspace.setdefault(CkksConfig, engine.config)
                resources = engine._dispatcher_for(device).device_resources
                bind_provided_materials(
                    bindings, artifact.selections, (resources,)
                )
            if keyed and all(
                symbol in bindings for symbol in artifact.selections
            ):
                stages["cache_load_seconds"] = time.perf_counter() - started
                stages["cache"] = "hit"
                compilation: Compilation = DeferredCompilation(
                    program, workspace, material_bindings=bindings
                )
                return self._cached_variant(
                    signature, arguments, entry, compilation, artifact
                )
            stages["artifact"] = "unbound-materials"
        if entry.program is None:
            stages["cache_miss_reason"] = "no Program text for relinking"
            return None
        compilation = DeferredCompilation(
            program, workspace, material_bindings=bindings
        )
        for engine, device in providers:
            resources = engine._dispatcher_for(device).device_resources
            workspace.setdefault(CkksConfig, engine.config)
            prepare_material_bindings(compilation, resources=resources)
        stages["cache_load_seconds"] = time.perf_counter() - started
        stages["cache"] = "hit"
        return self._cached_variant(signature, arguments, entry, compilation)

    def _cached_variant(
        self,
        signature: CallSignature,
        arguments: Mapping[str, object],
        entry: CacheEntry,
        compilation: Compilation,
        artifact: LinkArtifact | None = None,
        source: Compilation | None = None,
    ) -> _CompiledVariant:
        return _CompiledVariant(
            signature,
            self._recapture(arguments) if source is None else source,
            compilation,
            entry.input_names,
            entry.output_structure,
            tuple(type(arguments[a.name]) for a in signature.arguments),
            (
                input_fields(compilation.program)
                if artifact is None
                else _record_fields(artifact.record)
            ),
            artifact,
        )

    def _recapture(
        self, arguments: Mapping[str, object]
    ) -> Callable[[], Compilation]:
        """Capture the source again for a variant restored without its captured Program."""
        held = {name: _weak(value) for name, value in arguments.items()}

        def source() -> Compilation:
            values = {name: value() for name, value in held.items()}
            if any(value is None for value in values.values()):
                raise RuntimeError(
                    "The arguments this specialization was prepared with are released; prepare it again to capture its source"
                )
            return self._capture(values)

        return source

    def _linked(
        self,
        entry: CacheEntry,
        backend: OperationBackend,
        stages: dict[str, object],
    ) -> dict[str, Any] | None:
        """The linked executable of ``entry`` recorded for the implementation registry of ``backend``."""
        registry = self._artifact_registry(backend, stages)
        if registry is None:
            return None
        if entry.linked is None or entry.linked["registry"] != registry:
            return None
        stages["artifact"] = "loaded"
        return entry.linked

    def _artifact_registry(
        self, backend: OperationBackend, stages: dict[str, object]
    ) -> str | None:
        """Key linked executables by the implementation registry of ``backend``."""
        registry = artifact_key(backend.registry)
        stages["artifact"] = "ineligible" if registry is None else "miss"
        if registry is not None:
            stages["artifact_registry"] = registry
        return registry

    def _store(
        self,
        key: str,
        entry: CacheEntry,
        prepared: Specialization,
        selections: Mapping[str, Mapping[str, object] | None] | None,
        stages: dict[str, object],
    ) -> None:
        """Write ``entry`` once, with the linked executable of ``prepared`` when the registry admits one."""
        part = (
            self._linked_part(prepared, selections, stages)
            if selections is not None and stages.get("artifact") == "miss"
            else None
        )
        parts: dict[str, Any] | None = None
        tensors: dict[str, torch.Tensor] | None = None
        if part is not None:
            linked, parts, tensors = part
            entry = replace(entry, linked=linked)
        started = time.perf_counter()
        try:
            cast(CompileCache, self._cache).store(key, entry, parts, tensors)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            stages["store"] = "write-failed"
            stages["store_error"] = f"{type(error).__name__}: {error}"
            if part is not None:
                stages["artifact"] = "write-failed"
        else:
            stages["store"] = "stored"
            if part is not None:
                stages["artifact"] = "stored"
        stages["store_seconds"] = time.perf_counter() - started

    def _bind_keys(
        self, bindings: dict[str, torch.Tensor], linked: Mapping[str, Any]
    ) -> bool:
        """Bind the recorded key symbols from the callable's keys with the recorded layouts; False when one has no unique match."""
        keys = supplied_keys(self._keys)
        for symbol, description in linked.get("keys", {}).items():
            if symbol in bindings:
                continue
            found = {
                id(key.data): key.data
                for key in keys
                if key_description(key) == description
            }
            if len(found) != 1:
                return False
            bindings[symbol] = next(iter(found.values()))
        return True

    def _linked_part(
        self,
        prepared: Specialization,
        selections: Mapping[str, Mapping[str, object] | None],
        stages: dict[str, object],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]] | None:
        """Export the linked executable of ``prepared`` as the cache file's artifact."""
        from fhelium.backend._artifact import (
            ArtifactIneligible,
            export_executable,
            material_record,
        )
        from fhelium.native.artifact import ArtifactWriter

        started = time.perf_counter()
        variant = prepared._variant
        try:
            writer = ArtifactWriter()
            record = export_executable(writer, prepared.executable, "main")
            parts, tensors = writer.build()
        except ArtifactIneligible as error:
            stages["artifact"] = f"ineligible: {error}"
            return None
        finally:
            stages["artifact_export_seconds"] = time.perf_counter() - started
        materials = record.pop("materials")
        parts["materials"] = {
            "caller": [
                material_record(symbol, tensor)
                for symbol, tensor in materials.items()
            ]
            + [
                material_record(
                    symbol, _state_tensor(prepared.executable, index, name)
                )
                for symbol, (index, name) in record["states"].items()
            ]
        }
        keys = {id(key.data): key for key in supplied_keys(self._keys)}
        linked = {
            "registry": stages["artifact_registry"],
            "selections": selections,
            "keys": {
                symbol: key_description(keys[id(tensor)])
                for symbol, tensor in materials.items()
                if id(tensor) in keys
            },
            "input_fields": {
                name: list(item) for name, item in variant.input_fields.items()
            },
            "executable": record,
        }
        return linked, parts, tensors

    def _load_traced(
        self,
        source: Compilation,
        arguments: Mapping[str, object],
        signature: CallSignature,
        stages: dict[str, object],
        backend: OperationBackend,
    ) -> _CompiledVariant | None:
        """Restore the prepared Program compiled from an identical captured Program."""
        started = time.perf_counter()
        try:
            key = trace_entry_key(
                source.program, source.material_bindings, signature, self._keys
            )
        except CacheIneligible as error:
            stages["trace"] = "ineligible"
            stages["trace_reason"] = str(error)
            return None
        finally:
            stages["trace_key_seconds"] = time.perf_counter() - started
        stages["trace_key"] = key
        from fhelium.backend.implementation import (
            OperationImplementationRegistry,
        )

        workspace = fresh_workspace(source.workspace)
        workspace[OperationImplementationRegistry] = backend.registry
        cache = cast(CompileCache, self._cache)
        started = time.perf_counter()
        entry = cache.load(key, decode=False)
        stages["trace_load_seconds"] = time.perf_counter() - started
        if entry is None:
            stages["trace"] = "miss"
            return None
        self._loaded[signature] = entry
        bindings = dict(source.material_bindings)
        linked = self._linked(entry, backend, stages)
        artifact = (
            LinkArtifact(linked, linked["selections"])
            if linked is not None and self._bind_keys(bindings, linked)
            else None
        )
        if artifact is None and entry.program is None:
            stages["trace"] = "miss"
            return None
        stages["trace"] = "hit"
        return self._cached_variant(
            signature,
            arguments,
            entry,
            DeferredCompilation(
                _CachedProgram(cache, key, entry.program),
                workspace,
                material_bindings=bindings,
            ),
            artifact,
            source,
        )

    def _cached_entry(
        self,
        variant: _CompiledVariant,
        stages: dict[str, object],
        identity: SourceIdentity | None,
    ) -> CacheEntry | None:
        """Describe a source-identity cache entry before linking consumes its Program."""
        compilation = variant.compilation
        try:
            # Symbols bound from the callable's keys are bound from them again.
            keys = {id(key.data) for key in supplied_keys(self._keys)}
            manifest = build_manifest(
                cast(SourceIdentity, identity),
                {
                    symbol: tensor
                    for symbol, tensor in compilation.material_bindings.items()
                    if id(tensor) not in keys
                },
                compilation.program.material_descriptions,
            )
            if manifest is None:
                stages["cache"] = "unlocatable-materials"
                return None
            return CacheEntry(
                compilation.program.encode(),
                manifest,
                variant.output_structure,
                variant.input_names,
                groups=captured_group_names(compilation.workspace),
                preparation=(
                    compilation.workspace[PREPARATION].descriptor(identity)
                    if PREPARATION in compilation.workspace
                    else None
                ),
            )
        except (ValueError, CacheIneligible) as error:
            stages["cache"] = "write-failed"
            stages["cache_error"] = f"{type(error).__name__}: {error}"
            return None

    def _warm(
        self,
        prepared: Specialization,
        arguments: Mapping[str, object],
        stages: dict[str, object],
    ) -> None:
        """Compile device kernels of a linked executable ahead of its first call."""
        devices = {
            tensor.device
            for value in arguments.values()
            for tensor in _argument_tensors(value)
        }
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return
        try:
            from fhelium.backend.triton import warm_executable
        except ImportError:
            return
        started = time.perf_counter()
        try:
            # Graph routing executes the top-level schedule through Graph Runners,
            # so its prepared Runner and arena are built only for a fallback.
            counts = warm_executable(
                prepared.executable,
                next(iter(devices)),
                runner=self._graphs.graph is False,
            )
        except Exception as error:
            stages["warm_error"] = f"{type(error).__name__}: {error}"
            raise
        else:
            stages["warm_kernels"] = counts
        finally:
            stages["warm_seconds"] = time.perf_counter() - started

    def _build(
        self, arguments: Mapping[str, object], signature: CallSignature
    ) -> Specialization:
        """Capture, transform, bind, and link one specialization."""
        stages: dict[str, object] = {
            "cache": "disabled" if self._cache is None else "miss"
        }
        backend = self.backend
        if backend is None:
            from fhelium.backend.execution import OperationBackend

            backend = OperationBackend()
        started = time.perf_counter()
        identity = (
            self._source_identity() if runtime_cacheable(arguments) else None
        )
        stages["identity_seconds"] = time.perf_counter() - started
        if self._cache is not None and identity is None:
            stages["cache"] = "ineligible"
        variant = self._variants.get(signature)
        if variant is not None:
            stages["cache"] = "resident"
            if signature in self._ready_streams:
                self._ready_streams[signature].use()
        if variant is None:
            variant = self._load_cached(
                arguments, signature, stages, identity, backend
            )
        if variant is None:
            started = time.perf_counter()
            source = self._capture(arguments)
            stages["capture_seconds"] = time.perf_counter() - started
            traced = None
            # Sources without a pre-capture identity are keyed by what they captured.
            if self._cache is not None and identity is None:
                traced = self._load_traced(
                    source, arguments, signature, stages, backend
                )
            if traced is not None:
                variant = traced
            else:
                started = time.perf_counter()
                variant, backend = prepare_variant(
                    source,
                    arguments,
                    signature,
                    pipeline=self.pipeline,
                    backend=backend,
                    keys=self._keys,
                )
                stages["pipeline_seconds"] = time.perf_counter() - started
                stages["passes"] = tuple(
                    (report.name, report.seconds)
                    for report in variant.compilation.reports
                )
                from fhelium.ir._operation_structure import STRUCTURE_LOOKUPS

                from .passes.lowering._select_execution import (
                    SelectionDecisions,
                )

                selection = variant.compilation.workspace.get(
                    SelectionDecisions
                )
                if selection is not None:
                    stages["selection_decisions"] = selection.stats()
                stages["structure_lookups"] = dict(STRUCTURE_LOOKUPS)
        started = time.perf_counter()
        from .frontend._eager_capture import CAPTURED_RESOURCES

        captured = variant.compilation.workspace.get(CAPTURED_RESOURCES, ())
        resources = (*self._resources, *captured)
        if variant.artifact is not None and not bind_provided_materials(
            variant.compilation.material_bindings,
            variant.artifact.selections,
            resources,
        ):
            stages["artifact"] = "unbound-materials"
            variant = replace(variant, artifact=None)
        if variant.artifact is None and (self._keys or resources):
            prepare_material_bindings(
                variant.compilation, resources=resources, keys=self._keys
            )
        stages["binding_seconds"] = time.perf_counter() - started
        # Fresh entries are encoded before linking consumes the Program in place.
        store_trace = stages.get("trace") == "miss"
        store_cache = stages["cache"] == "miss" and identity is not None
        if store_trace or store_cache:
            self._artifact_registry(backend, stages)
        loaded = self._loaded.pop(signature, None)
        write_key = cast(
            "str | None",
            stages.get("trace_key")
            if store_trace
            or (loaded is not None and stages.get("trace") == "hit")
            else stages.get("cache_key"),
        )
        compilation = variant.compilation
        selections = None
        pending: CacheEntry | None = None
        reload: _CachedProgram | Callable[[], Program] | None = None
        if variant.artifact is None:
            if stages.get("artifact") == "miss":
                selections = provided_material_selections(compilation)
            if isinstance(compilation, DeferredCompilation):
                reload = compilation._load
                # The entry gains the linked executable of this registry.
                pending = loaded if selections is not None else None
            elif (store_trace or store_cache) and write_key is not None:
                started = time.perf_counter()
                pending = (
                    CacheEntry(
                        compilation.program.encode(),
                        {},
                        variant.output_structure,
                        variant.input_names,
                    )
                    if store_trace
                    else self._cached_entry(variant, stages, identity)
                )
                if pending is not None:
                    reload = _CachedProgram(
                        cast(CompileCache, self._cache),
                        write_key,
                        pending.program,
                    )
                stages["encode_seconds"] = time.perf_counter() - started
        plan = variant.compilation.workspace.get(PREPARATION)
        started = time.perf_counter()
        if plan is not None:
            plan.link()
        stages["preparation_link_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        prepared = bind_variant(
            variant, cast("OperationBackend", backend), reload=reload
        )
        variant = prepared._variant
        loader = getattr(variant.compilation, "_load", None)
        if isinstance(loader, _CachedProgram) and (
            pending is None or pending is loaded
        ):
            # Later Program requests read the cache entry this Program came from.
            loader.release()
        physical_arguments = field_values(variant.input_fields, arguments)
        for name, adapter in zip(
            variant.input_names,
            prepared.executable._input_adapters,
            strict=True,
        ):
            adapter(physical_arguments[name])
        stages["link_seconds"] = time.perf_counter() - started
        if stages.get("artifact") == "loaded":
            stages["artifact"] = "hit"
        self._warm(prepared, physical_arguments, stages)
        if pending is not None and write_key is not None:
            self._store(write_key, pending, prepared, selections, stages)
            if pending is not loaded:
                stages["trace" if store_trace else "cache"] = stages["store"]
            if stages["store"] == "stored" and isinstance(
                reload, _CachedProgram
            ):
                reload.release()
        plan = variant.compilation.workspace.get(PREPARATION)
        if plan is not None:
            stages["plaintext_preparation"] = plan.stats()
        if "binding_time_fallback" in variant.compilation.workspace:
            stages["binding_time_fallback"] = variant.compilation.workspace[
                "binding_time_fallback"
            ]
        stages["executions"] = 0
        self._variants[signature] = variant
        self._stages[signature] = stages
        return prepared

    def _prepare(
        self, arguments: Mapping[str, object], signature: CallSignature
    ) -> Specialization:
        prepared = self._prepared.get(signature)
        if prepared is not None:
            self._resident_hits += 1
            return self._refresh_prepared(prepared, arguments)
        prepared = self._build(arguments, signature)
        self._ready_streams[signature] = PreparedStreams.record(
            arguments.values(),
            prepared.compilation.material_bindings.values(),
        )
        self._prepared[signature] = prepared
        return prepared

    def prepare(self, *args: _P.args, **kwargs: _P.kwargs) -> Specialization:
        """Prepare, link, and warm the matching specialization.

        A matching resident specialization is returned directly. This method populates the resident cache under either ``on_miss`` policy.
        """
        with self._execution_lock:
            arguments = self._arguments(args, kwargs)
            signature = replace(
                describe_call(arguments), context=execution_context()
            )
            return self._prepare(arguments, signature)

    def _refresh_prepared(self, prepared, arguments, *, force=False):
        plan = prepared.compilation.workspace.get(PREPARATION)
        if plan is None:
            return prepared
        self._ready_streams[prepared.signature].use()
        bindings = plan.refresh(arguments, force=force)
        current = prepared.compilation.material_bindings
        if bindings is not None and any(
            current.get(symbol) is not value
            for symbol, value in {**plan.leaves, **bindings}.items()
        ):
            # Refreshes that update the bound Tensors in place keep the executable.
            bindings = {**plan.leaves, **bindings}
            started = time.perf_counter()
            executable = prepared.executable.rebind_materials(bindings)
            prepared.compilation.material_bindings.update(bindings)
            source = prepared._variant.source
            if isinstance(source, Compilation):
                source.material_bindings.update(bindings)
            self._graphs.invalidate(prepared)
            prepared = replace(prepared, executable=executable)
            self._prepared[prepared.signature] = prepared
            self._last_prepared = prepared
            self._ready_streams[prepared.signature] = PreparedStreams.record(
                arguments.values(),
                prepared.compilation.material_bindings.values(),
            )
            plan.rebind_seconds += time.perf_counter() - started
        self._stages[prepared.signature]["plaintext_preparation"] = plan.stats()
        return prepared

    def refresh_materials(
        self, *args: _P.args, **kwargs: _P.kwargs
    ) -> Specialization:
        """Reexecute preparation after changes not exposed by Tensor versions."""
        with self._execution_lock:
            arguments = self._arguments(args, kwargs)
            signature = replace(
                describe_call(arguments), context=execution_context()
            )
            prepared = self._prepare(arguments, signature)
            return self._refresh_prepared(prepared, arguments, force=True)

    def _run(self, prepared: Specialization, values: tuple[object, ...]) -> _R:
        arguments = dict(zip(self._argument_names, values, strict=True))
        prepared = self._refresh_prepared(prepared, arguments)
        self._ready_streams[prepared.signature].use()
        stages = self._stages[prepared.signature]
        token = self._timing.start(values)
        plan = prepared.compilation.workspace.get(PREPARATION)
        result = (
            prepared._execute(values)
            if plan is not None and plan.mode == "per_call"
            else self._graphs.execute(prepared, values)
        )
        self._timing.finish(token, steady=cast(int, stages["executions"]) >= 2)
        stages["executions"] = cast(int, stages["executions"]) + 1
        if plan is not None and plan.mode == "per_call":
            released = {symbol: None for symbol in plan.symbols}
            replacement = replace(
                prepared,
                executable=prepared.executable.rebind_materials(released),
            )
            for compilation in (prepared.compilation, prepared._variant.source):
                if isinstance(compilation, Compilation):
                    for symbol in plan.symbols:
                        compilation.material_bindings.pop(symbol, None)
            plan.release()
            self._prepared[prepared.signature] = replacement
            self._last_prepared = replacement
            self._ready_streams[prepared.signature] = PreparedStreams.record(
                arguments.values(),
                replacement.compilation.material_bindings.values(),
            )
            stages["plaintext_preparation"] = plan.stats()
        return cast(_R, result)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run the matching resident specialization, preparing it on a miss when ``on_miss='compile'``."""
        with self._execution_lock:
            return self._call(args, kwargs)

    def _call(self, args: tuple[object, ...], kwargs: dict[str, object]) -> _R:
        values = self._bind_arguments(*args, **kwargs)
        context = execution_context()
        prepared = self._last_prepared
        if (
            prepared is not None
            and prepared.signature.context == context
            and prepared._matches(values)
        ):
            self._resident_hits += 1
            return self._run(prepared, values)
        for candidate in tuple(self._prepared.values()):
            if (
                candidate is not prepared
                and candidate.signature.context == context
                and candidate._matches(values)
            ):
                self._last_prepared = candidate
                self._resident_hits += 1
                return self._run(candidate, values)
        if self.on_miss == "error":
            raise SpecializationMiss(
                "No prepared specialization matches this invocation; "
                "call prepare with these input conditions first"
            )
        arguments = dict(zip(self._argument_names, values, strict=True))
        signature = replace(describe_call(arguments), context=context)
        prepared = self._prepare(arguments, signature)
        self._last_prepared = prepared
        return self._run(prepared, values)

    def stats(self) -> dict[str, object]:
        """Snapshot preparation, cache, execution, and Graph observations.

        Stage durations are host wall times. Execution timing uses host completion on CPU and nonblocking CUDA event queries on the invocation's current stream. Pending device observations are counted separately. Steady execution excludes each specialization's first two compiled calls, including its automatic capture attempt. ``overlap`` sums :meth:`~fhelium.backend.execution.ProgramExecutable.overlap_stats` over the specializations.
        """
        with self._execution_lock:
            records = tuple(
                {"signature": signature, **stages}
                for signature, stages in self._stages.items()
                if isinstance(signature, CallSignature)
            )
            prepare_total = sum(
                float(record.get(stage, 0.0))
                for record in records
                for stage in (
                    "capture_seconds",
                    "pipeline_seconds",
                    "preparation_link_seconds",
                    "binding_seconds",
                    "link_seconds",
                    "cache_load_seconds",
                    "store_seconds",
                    "warm_seconds",
                    "identity_seconds",
                )
            )
            graph_records = self._graphs.stats()
            capture_total = sum(
                float(record["capture_attempt_seconds"])
                for record in graph_records
            )
            timings = self._timing.stats()
            return {
                "specializations": records,
                "resident_cache_hits": self._resident_hits,
                "graphs": graph_records,
                "prepare_seconds": prepare_total,
                "graph_capture_seconds": capture_total,
                "executions": sum(
                    int(record["executions"]) for record in records
                ),
                "execute_seconds": timings["compiled"]["completed_seconds"],
                "timing": timings,
                "cache": dict(self._stages.get("__cache__", {})),
                "overlap": _summed(
                    prepared.executable.overlap_stats()
                    for prepared in self._prepared.values()
                ),
            }

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
            cache=False if self._cache is None else self._cache.directory,
            graph=self._graphs.graph,
        )

        def fork_variant(variant):
            # A persisted executable is restored only for the registry it was linked with.
            variant = replace(variant, artifact=None)
            plan = variant.compilation.workspace.get(PREPARATION)
            if plan is None:
                return variant
            plan = plan.fork()

            def fork_compilation(compilation):
                work = fresh_workspace(compilation.workspace)
                work[PREPARATION] = plan
                if isinstance(compilation, DeferredCompilation):
                    return DeferredCompilation(
                        compilation._load,
                        work,
                        compilation.reports,
                        dict(compilation.material_bindings),
                    )
                return Compilation(
                    compilation.program,
                    work,
                    compilation.reports,
                    dict(compilation.material_bindings),
                )

            source = (
                variant.source() if callable(variant.source) else variant.source
            )
            return replace(
                variant,
                source=fork_compilation(source),
                compilation=fork_compilation(variant.compilation),
            )

        with self._execution_lock:
            rebound._variants = {
                signature: fork_variant(self._variants[signature])
                for signature in self._prepared
            }
            rebound._ready_streams = {
                signature: self._ready_streams[signature].fork()
                for signature in self._prepared
            }
        return rebound

    def clear(self) -> None:
        """Release this callable's specialization cache, retained Graphs, and pools.

        Call this after changing captured Python constants so the next preparation captures their new values. Changes to input Tensor contents reuse the existing specialization while its recorded input conditions hold. Caller resources and compiled results retained by peers made with ``with_backend`` remain live. Persistent cache files are not removed.
        """

        with self._execution_lock:
            self._graphs.clear()
            self._variants.clear()
            self._prepared.clear()
            self._ready_streams.clear()
            self._stages.clear()
            self._last_prepared = None
            self._resident_hits = 0
            self._timing.clear()


def _summed(records) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in records:
        for name, value in record.items():
            totals[name] = totals.get(name, 0) + value
    return totals


__all__ = ["CompiledCallable"]
