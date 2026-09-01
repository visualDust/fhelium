"""Narrow fused pointwise code generation through optional Triton.

The backend is a vertical slice for JIT source generation. It consumes one
single-block function whose body contains only public-message
``fhelium_semantic.add``, ``fhelium_semantic.multiply``, and
``fhelium_semantic.negate`` operations followed by one ``func.return``. It
does not claim CKKS semantics and performs no native fallback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

import torch
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation, SSAValue

from fhelium.ir import Program, operation_name, value_role
from fhelium.ir.dialects import semantic

from .._contracts import (
    CoverageDiagnostic,
    CoverageReport,
    Executable,
    ProviderPolicy,
)
from .._execution import ExecutionInputs

from .._bindings import RuntimeBindings

_BACKEND_NAME = "triton-pointwise"
_PROVIDER = "triton"
_FAMILY = "pointwise"
_SUPPORTED_OPERATIONS: Mapping[type[Operation], tuple[int, str]] = (
    MappingProxyType(
        {
            semantic.AddOp: (2, "+"),
            semantic.MultiplyOp: (2, "*"),
            semantic.NegateOp: (1, "-"),
        }
    )
)
_KERNEL_CACHE_MAX_ENTRIES = 32
_KERNEL_CACHE: dict[str, tuple[object, object]] = {}
_KERNEL_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class _LoweredPointwise:
    """Carry one internally generated Triton source and its operation names."""

    source: str
    source_digest: str
    operations: frozenset[str]
    input_count: int


def _dependency_available() -> bool:
    """Return whether the optional Triton package can be imported."""

    return importlib.util.find_spec("triton") is not None


def _policy_diagnostics(
    policy: ProviderPolicy | None,
    execution: ExecutionInputs,
) -> list[CoverageDiagnostic]:
    diagnostics: list[CoverageDiagnostic] = []
    if policy is None:
        return diagnostics
    diagnostics.extend(policy.validate(execution))
    if _PROVIDER not in policy.providers:
        diagnostics.append(
            CoverageDiagnostic(
                "provider-not-enabled",
                f"Provider policy does not enable {_PROVIDER!r}.",
                _PROVIDER,
            )
        )
    return diagnostics


def _inspect_program(
    program: Program,
) -> tuple[_LoweredPointwise | None, tuple[CoverageDiagnostic, ...]]:
    """Inspect and lower the backend's fixed single-block operation family."""

    diagnostics: list[CoverageDiagnostic] = []
    operations: set[str] = set()
    try:
        block = program.single_block("main")
    except (KeyError, ValueError) as error:
        return None, (
            CoverageDiagnostic(
                "unsupported-entry-structure",
                f"Triton pointwise backend requires one @main block: {error}",
                "main",
            ),
        )

    values: dict[SSAValue, str] = {
        argument: f"input_{index}" for index, argument in enumerate(block.args)
    }
    for index, argument in enumerate(block.args):
        if value_role(argument) != "message":
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-input-type",
                    "Triton pointwise inputs must use the public message role.",
                    f"argument-{index}",
                )
            )
    expressions: list[tuple[str, str]] = []
    returned: SSAValue | None = None

    body = tuple(block.ops)
    if not body or not isinstance(body[-1], ReturnOp):
        diagnostics.append(
            CoverageDiagnostic(
                "missing-return",
                "Triton pointwise @main requires one final func.return.",
                "main",
            )
        )
        return None, tuple(diagnostics)

    return_op = body[-1]
    if len(return_op.arguments) != 1:
        diagnostics.append(
            CoverageDiagnostic(
                "unsupported-result-arity",
                "Triton pointwise backend supports one returned tensor.",
                "func.return",
            )
        )
    else:
        returned = return_op.arguments[0]
        if value_role(returned) != "message":
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-result-type",
                    "Triton pointwise results must use the public message role.",
                    "func.return",
                )
            )

    for index, operation in enumerate(body[:-1]):
        name = operation_name(operation)
        operations.add(name)
        specification = _SUPPORTED_OPERATIONS.get(type(operation))
        if specification is None:
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-operation",
                    f"Triton pointwise backend does not implement {name!r}.",
                    name,
                )
            )
            continue
        arity, operator = specification
        if operation.regions or operation.successors:
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-operation-structure",
                    f"Operation {name!r} cannot own regions or successors in "
                    "the pointwise backend.",
                    name,
                )
            )
            continue
        semantic_attributes = set(operation.attributes) - {
            "op_name__",
            "fhelium.frontend.target",
        }
        if semantic_attributes or operation.properties:
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-operation-metadata",
                    f"Operation {name!r} carries attributes or properties "
                    "outside the pointwise vertical slice.",
                    name,
                )
            )
            continue
        if len(operation.operands) != arity or len(operation.results) != 1:
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-operation-arity",
                    f"Operation {name!r} requires {arity} operands and one "
                    "result in the pointwise backend.",
                    name,
                )
            )
            continue
        if any(
            value_role(operand) != "message" for operand in operation.operands
        ):
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-operand-type",
                    f"Operation {name!r} requires public message operands.",
                    name,
                )
            )
            continue
        if value_role(operation.results[0]) != "message":
            diagnostics.append(
                CoverageDiagnostic(
                    "unsupported-result-type",
                    f"Operation {name!r} requires a public message result.",
                    name,
                )
            )
            continue
        operand_names: list[str] = []
        unresolved = False
        for operand in operation.operands:
            value_name = values.get(operand)
            if value_name is None:
                diagnostics.append(
                    CoverageDiagnostic(
                        "unresolved-ssa-value",
                        f"Operation {name!r} uses a value not produced earlier "
                        "in the selected block.",
                        name,
                    )
                )
                unresolved = True
                break
            operand_names.append(value_name)
        if unresolved:
            continue
        result_name = f"value_{index}"
        expression = (
            f"{operator}{operand_names[0]}"
            if arity == 1
            else f"{operand_names[0]} {operator} {operand_names[1]}"
        )
        expressions.append((result_name, expression))
        values[operation.results[0]] = result_name

    if returned is not None and returned not in values:
        diagnostics.append(
            CoverageDiagnostic(
                "unresolved-return-value",
                "func.return uses a value unavailable to the pointwise backend.",
                "func.return",
            )
        )

    if diagnostics:
        return None, tuple(diagnostics)
    assert returned is not None
    source = _generate_source(
        input_count=len(block.args),
        expressions=tuple(expressions),
        output_name=values[returned],
    )
    return (
        _LoweredPointwise(
            source=source,
            source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            operations=frozenset(operations),
            input_count=len(block.args),
        ),
        (),
    )


def _generate_source(
    *,
    input_count: int,
    expressions: tuple[tuple[str, str], ...],
    output_name: str,
) -> str:
    """Generate source solely from fixed internal identifiers and templates."""

    parameters = [f"input_{index}" for index in range(input_count)]
    parameters.extend(("output", "n_elements", "BLOCK_SIZE: tl.constexpr"))
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        f"def fhelium_pointwise_kernel({', '.join(parameters)}):",
        "    program_id = tl.program_id(axis=0)",
        "    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n_elements",
    ]
    for index in range(input_count):
        lines.append(
            f"    input_{index} = tl.load(input_{index} + offsets, mask=mask)"
        )
    for result_name, expression in expressions:
        lines.append(f"    {result_name} = {expression}")
    lines.append(f"    tl.store(output + offsets, {output_name}, mask=mask)")
    lines.append("")
    return "\n".join(lines)


def _load_kernel(source: str) -> tuple[object, object]:
    """Load internally generated source without importing Triton at package import."""

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with _KERNEL_CACHE_LOCK:
        cached = _KERNEL_CACHE.get(digest)
        if cached is not None:
            _KERNEL_CACHE.pop(digest)
            _KERNEL_CACHE[digest] = cached
            return cached
        module_name = "fhelium_generated_triton_pointwise_" + digest[:16]
        path = ""
        module: Any | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".py",
                prefix=module_name,
                delete=False,
            ) as generated_file:
                generated_file.write(source)
                path = generated_file.name
            specification = importlib.util.spec_from_file_location(
                module_name, path
            )
            if specification is None or specification.loader is None:
                raise RuntimeError(
                    "Could not create a loader for generated Triton source"
                )
            module = importlib.util.module_from_spec(specification)
            sys.modules[module_name] = module
            specification.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        kernel = getattr(module, "fhelium_pointwise_kernel", None)
        triton_module = getattr(module, "triton", None)
        if kernel is None or triton_module is None:
            raise RuntimeError("Generated Triton module omitted its kernel")
        loaded = (kernel, triton_module)
        if len(_KERNEL_CACHE) >= _KERNEL_CACHE_MAX_ENTRIES:
            _KERNEL_CACHE.pop(next(iter(_KERNEL_CACHE)))
        _KERNEL_CACHE[digest] = loaded
        return loaded


@dataclass
class TritonPointwiseExecutable:
    """Lazily compile and run one fused same-shape CUDA pointwise expression."""

    _program: Program
    _lowered: _LoweredPointwise
    _execution: ExecutionInputs
    _kernel: Any | None = None
    _triton: Any | None = None

    @property
    def backend(self) -> str:
        """Return the backend registry name."""

        return _BACKEND_NAME

    @property
    def manifest(self) -> Mapping[str, object]:
        """Expose generated source, digest, operation names, and provider."""

        return MappingProxyType(
            {
                "backend": self.backend,
                "target": "cuda",
                "provider": _PROVIDER,
                "source": self._lowered.source,
                "source_digest": self._lowered.source_digest,
                "operations": sorted(self._lowered.operations),
                "execution_inputs": self._execution.as_dict(),
            }
        )

    def run(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Run after checking the vertical slice's tensor ABI."""

        if kwargs:
            raise TypeError(
                "Triton pointwise executable accepts positional inputs"
            )
        if len(args) != self._lowered.input_count:
            raise TypeError(
                "Triton pointwise executable expected "
                f"{self._lowered.input_count} inputs, got {len(args)}"
            )
        if not args:
            raise TypeError(
                "Triton pointwise executable requires an input tensor"
            )
        tensors: list[torch.Tensor] = []
        selected_device = torch.device(self._execution.device)
        if selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        for index, value in enumerate(args):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Input {index} must be a torch.Tensor")
            if value.device.type != "cuda":
                raise ValueError(f"Input {index} must reside on CUDA")
            if value.device != selected_device:
                raise ValueError(
                    f"Input {index} must reside on {selected_device}, got "
                    f"{value.device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(
                    f"Input {index} must use torch.float32, got {value.dtype}"
                )
            if not value.is_contiguous():
                raise ValueError(f"Input {index} must be contiguous")
            tensors.append(value)
        reference = tensors[0]
        for index, value in enumerate(tensors[1:], start=1):
            if value.device != reference.device:
                raise ValueError(f"Input {index} uses a different CUDA device")
            if value.shape != reference.shape:
                raise ValueError(f"Input {index} uses a different shape")

        with torch.cuda.device(selected_device):
            output = torch.empty_like(reference)
            element_count = reference.numel()
            if element_count == 0:
                return output
            if self._kernel is None or self._triton is None:
                self._kernel, self._triton = _load_kernel(self._lowered.source)
            kernel = self._kernel
            triton_module = self._triton
            cdiv = getattr(triton_module, "cdiv")
            grid = (cdiv(element_count, 256),)
            kernel[grid](*tensors, output, element_count, BLOCK_SIZE=256)
            return output

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.run(*args, **kwargs)


@dataclass(frozen=True)
class TritonPointwiseBackend:
    """Generate one fused Triton kernel for a fixed pointwise IR family."""

    name: str = _BACKEND_NAME

    def coverage(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> CoverageReport:
        """Report CUDA/provider/dependency and supported-IR coverage."""

        del bindings
        diagnostics = _policy_diagnostics(policy, execution)
        if execution.device.type != "cuda":
            diagnostics.append(
                CoverageDiagnostic(
                    "target-mismatch",
                    "Triton pointwise backend requires a CUDA Session "
                    "execution target.",
                    execution.device.type,
                )
            )
        if not _dependency_available():
            diagnostics.append(
                CoverageDiagnostic(
                    "triton-unavailable",
                    "The optional Triton package is not installed.",
                    _PROVIDER,
                )
            )
        lowered, program_diagnostics = _inspect_program(program)
        diagnostics.extend(program_diagnostics)
        operations = lowered.operations if lowered is not None else frozenset()
        return CoverageReport(self.name, tuple(diagnostics), operations)

    def build(
        self,
        program: Program,
        *,
        execution: ExecutionInputs,
        bindings: RuntimeBindings,
        policy: ProviderPolicy | None = None,
    ) -> Executable:
        """Generate source and return a lazily compiled Triton executable."""

        report = self.coverage(
            program,
            execution=execution,
            bindings=bindings,
            policy=policy,
        )
        if not report.covered:
            detail = "; ".join(item.message for item in report.diagnostics)
            raise RuntimeError(
                "Triton pointwise backend does not cover the supplied Program: "
                + detail
            )
        lowered, diagnostics = _inspect_program(program)
        if lowered is None or diagnostics:
            raise RuntimeError(
                "Triton pointwise lowering changed after coverage"
            )
        return TritonPointwiseExecutable(program.clone(), lowered, execution)


__all__ = ["TritonPointwiseBackend", "TritonPointwiseExecutable"]
