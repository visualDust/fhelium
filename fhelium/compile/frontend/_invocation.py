"""Adapt role-declared Compile FX capture to invocation specialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from fhelium.values import Ciphertext

from .._compilation import Compilation
from .._workspace import CompileWorkspace
from ._capture import capture
from ._eager_capture import capture_eager
from ._specs import InputSpec, static


def capture_invocation(
    function: Callable[..., object],
    arguments: Mapping[str, object],
    *,
    inputs: Mapping[str, InputSpec] | None,
    workspace: CompileWorkspace,
    material_names: Mapping[str, object] | None = None,
) -> Compilation:
    """Select the declared frontend without attempting a fallback.

    Omitted ``inputs`` captures ordinary Tensor expressions and Engine calls
    together, using each actual value's role. Supplied input roles select FX;
    runtime ciphertexts refine their represented state, and static arguments
    use this invocation's values. FX operations still need executable lowering
    or matching Backend implementations.
    """

    if inputs is None:
        return capture_eager(
            function,
            arguments=arguments,
            workspace=workspace,
            material_names=material_names,
        )
    specs: dict[str, InputSpec] = {}
    for name, spec in inputs.items():
        value = arguments[name]
        if spec.role == "static":
            specs[name] = static(value)  # type: ignore[arg-type]
        elif spec.role == "encrypted" and isinstance(value, Ciphertext):
            specs[name] = replace(
                spec,
                depth=value.depth,
                scale=value.scale,
                batch_mode="any" if value.batch_shape else "none",
                polynomial_domain=value.polynomial_domain,
                residue_representation=value.residue_representation,
            )
        else:
            specs[name] = spec
    return capture(
        function,
        inputs=specs,
        workspace=workspace,
        material_names=material_names,
    )


__all__: list[str] = []
