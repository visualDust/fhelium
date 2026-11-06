"""Create callable execution interfaces over Compile Programs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

from fhelium.ir import Program

from ._callable import CompiledCallable
from ._compilation import Compilation
from ._pipeline import Pipeline
from ._specialization import CallSignature
from ._workspace import CompileWorkspace
from .frontend._specs import InputSpec

if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext
    from fhelium.values import EvaluationKeySet
    from fhelium.backend import OperationBackend

_P = ParamSpec("_P")
_R = TypeVar("_R")


@overload
def compile(
    source: Callable[_P, _R],
    *,
    backend: OperationBackend | None = None,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
    on_miss: Literal["compile", "error"] = "compile",
    workspace: CompileWorkspace | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    material_names: Mapping[str, object] | None = None,
) -> CompiledCallable[_P, _R]: ...


@overload
def compile(
    source: Program | Compilation,
    *,
    backend: OperationBackend | None = None,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
    on_miss: Literal["compile", "error"] = "compile",
    workspace: CompileWorkspace | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
) -> CompiledCallable[..., Any]: ...


@overload
def compile(
    source: None = None,
    *,
    backend: OperationBackend | None = None,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
    on_miss: Literal["compile", "error"] = "compile",
    workspace: CompileWorkspace | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    material_names: Mapping[str, object] | None = None,
) -> Callable[[Callable[_P, _R]], CompiledCallable[_P, _R]]: ...


def compile(
    source: Program | Compilation | Callable[_P, _R] | None = None,
    *,
    backend: OperationBackend | None = None,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
    on_miss: Literal["compile", "error"] = "compile",
    workspace: CompileWorkspace | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    material_names: Mapping[str, object] | None = None,
) -> (
    CompiledCallable[_P, _R]
    | Callable[[Callable[_P, _R]], CompiledCallable[_P, _R]]
):
    """Create a ``CompiledCallable`` directly or as a decorator.

    ``compile(source, **options)`` returns a callable for a Python callable, Program, or Compilation. For Python sources, ``@compile`` uses default options and ``@compile(**options)`` configures the decorator. Options are passed to ``CompiledCallable`` construction.
    """

    def wrap(
        function: Program | Compilation | Callable[_P, _R],
    ) -> CompiledCallable[_P, _R]:
        return CompiledCallable(
            function,
            backend=backend,
            resources=resources,
            keys=keys,
            pipeline=pipeline,
            on_miss=on_miss,
            workspace=workspace,
            inputs=inputs,
            material_names=material_names,
        )

    return wrap if source is None else wrap(source)


__all__ = ["compile"]
