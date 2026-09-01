"""Source-oriented compilation over neutral mixed-level Programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fhelium.ir import Program

from ._compilation import Compilation
from ._pipeline import Pipeline
from ._workspace import CompileWorkspace
from .frontend._capture import capture
from .frontend._specs import InputSpec


def compile(
    source: Program | Compilation | Callable[..., Any],
    *,
    pipeline: Pipeline | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    workspace: CompileWorkspace | None = None,
) -> Compilation:
    """Capture or import ``source`` and optionally run ``pipeline``.

    ``source`` may be a Python callable, a prior ``Compilation``, or any neutral
    ``Program``. ``inputs`` is required only for a callable. Program-external
    materials, frontend state, caller inputs, and pass-produced data travel in
    the Compilation's schema-free ``CompileWorkspace``.

    The result may remain mixed-level or incomplete; this function does not
    select a runtime backend or assert executability.
    """

    if isinstance(source, Program):
        compilation = Compilation(
            source,
            CompileWorkspace() if workspace is None else workspace,
        )
    elif isinstance(source, Compilation):
        if workspace is not None and workspace is not source.workspace:
            raise ValueError(
                "A Compilation already owns its CompileWorkspace; do not "
                "replace it while continuing the same compilation"
            )
        compilation = source
    elif callable(source):
        if inputs is None:
            raise TypeError(
                "compile inputs are required when source is a callable"
            )
        compilation = capture(source, inputs=inputs, workspace=workspace)
    else:
        raise TypeError(
            "compile source must be a Program, Compilation, or callable"
        )

    if pipeline is None:
        return compilation
    if not isinstance(pipeline, Pipeline):
        raise TypeError("compile pipeline must be a Pipeline")
    return pipeline.run(compilation)


__all__ = ["compile"]
