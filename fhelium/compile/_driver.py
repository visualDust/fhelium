"""Create callable execution interfaces over Compile Programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

from fhelium.ir import Program

from ._callable import CompiledCallable
from ._compilation import Compilation
from ._pipeline import Pipeline
from ._specialization import CallSignature
from ._workspace import CompileWorkspace
from .frontend._specs import InputSpec

if TYPE_CHECKING:
    from fhelium.backend import OperationBackend

_P = ParamSpec("_P")
_R = TypeVar("_R")


@overload
def compile(
    source: Callable[_P, _R],
    *,
    backend: OperationBackend | None = None,
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
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None = None,
    on_miss: Literal["compile", "error"] = "compile",
    workspace: CompileWorkspace | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    material_names: Mapping[str, object] | None = None,
) -> (
    CompiledCallable[_P, _R]
    | Callable[[Callable[_P, _R]], CompiledCallable[_P, _R]]
):
    """Create a lazily prepared callable or decorate a pure Python function.

    Python functions capture ordinary Tensor expressions and Eager calls in
    one Program unless ``inputs`` selects the role-declared FX frontend.
    Ordinary Tensor computations retain their public numerical role; a
    Ciphertext input does not encrypt other inputs or the entire function. Programs and Compilations skip Python capture.
    There is no frontend fallback: unsupported capture or execution raises.
    The original Python reference and standard function metadata are retained.

    An omitted Backend uses the standard operation registry. Capture retains
    actual Tensor materials; it does not bind an Engine owner or generate keys.
    ``material_names`` assigns stable symbols to fixed Tensor/value objects.

    When ``pipeline`` is omitted, preparation uses ``default_lower_and_fuse_pipeline``.
    A supplied Pipeline, or function from ``CallSignature`` to Pipeline, replaces
    that recipe rather than extending it. ``on_miss='error'`` requires ``prepare`` before
    calls with new input conditions or new Backend bindings. Lazy Backend kernel
    compilation may still occur on the first actual execution.

    Use ``capture``, ``capture_eager``, ``Compilation``, and ``Pipeline.run`` when
    a transformed Program rather than a callable executable is the desired
    product. Those lower-level operations do not require a Backend.
    """

    def wrap(
        function: Program | Compilation | Callable[_P, _R],
    ) -> CompiledCallable[_P, _R]:
        return CompiledCallable(
            function,
            backend=backend,
            pipeline=pipeline,
            on_miss=on_miss,
            workspace=workspace,
            inputs=inputs,
            material_names=material_names,
        )

    return wrap if source is None else wrap(source)


__all__ = ["compile"]
