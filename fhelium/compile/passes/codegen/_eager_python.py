"""Emit editable Eager Engine Python from a flat CKKS Program."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation

from fhelium.compile.codegen import (
    EagerPythonSource,
    PythonCodegenError,
)
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core

from ._common import (
    SourceNames,
    append_unique,
    assignment,
    constant_literal,
    entry_point_name,
    floating,
    integer,
    program_block,
    return_statement,
    string,
    unsupported,
)

_PREPARE_OPERATIONS: dict[type[Operation], tuple[str, str]] = {
    ckks.PrepareAddMessageOp: ("add", "message"),
    ckks.PrepareAddPlaintextOp: ("add", "plaintext"),
    ckks.PrepareAddStaticOp: ("add", "static"),
    ckks.PrepareMultiplyMessageOp: ("multiply", "message"),
    ckks.PrepareMultiplyPlaintextOp: ("multiply", "plaintext"),
    ckks.PrepareMultiplyStaticOp: ("multiply", "static"),
}


@dataclass(frozen=True)
class EmitEagerPythonPass:
    """Publish Eager-style Python for the CKKS Program at this pass position."""

    entry_point: str = "generated_eager"
    name: str = "emit-eager-python"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        artifact = emit_eager_python(program, entry_point=self.entry_point)
        shared_data[EagerPythonSource] = artifact
        return PassResult.unchanged(
            program,
            matched=artifact.operation_count,
            diagnostics=(
                f"emitted Eager Python entry point {artifact.entry_point!r}",
            ),
        )


def emit_eager_python(
    program: Program,
    *,
    entry_point: str = "generated_eager",
) -> EagerPythonSource:
    """Emit one Eager function or fail on the first unsupported operation."""

    entry_point = entry_point_name(entry_point)
    block = program_block(program, target="Eager")
    names = SourceNames()
    input_names = tuple(
        names.input(argument, index)
        for index, argument in enumerate(block.args)
    )
    lines = [
        "from collections.abc import Mapping",
        "from fhelium.eager import Engine",
        "",
        f"def {entry_point}(",
        "    engine: Engine,",
        *(f"    {name}: object," for name in input_names),
        "    *,",
        "    materials: Mapping[str, object],",
        "    resources: Mapping[str, object],",
        ") -> object:",
    ]
    material_symbols: list[str] = []
    resource_symbols: list[str] = []
    operation_count = 0
    returned = False

    for index, operation in enumerate(block.ops):
        if operation.regions:
            raise unsupported("Eager", index, operation)
        operation_count += 1

        if isinstance(operation, UnrealizedConversionCastOp):
            if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                raise PythonCodegenError(
                    "Eager Python emission supports only one-to-one boundary casts"
                )
            result = names.results(operation)
            lines.append(assignment(result, names.operand(operation.inputs[0])))
            continue

        if isinstance(operation, core.MaterialRefOp):
            symbol = string(operation.symbol, label="material symbol")
            append_unique(material_symbols, (symbol,))
            lines.append(
                assignment(names.results(operation), f"materials[{symbol!r}]")
            )
            continue

        if isinstance(operation, core.ResourceRefOp):
            symbol = string(operation.symbol, label="resource symbol")
            append_unique(resource_symbols, (symbol,))
            lines.append(
                assignment(names.results(operation), f"resources[{symbol!r}]")
            )
            continue

        if isinstance(operation, core.ConstantOp):
            lines.append(
                assignment(
                    names.results(operation), repr(constant_literal(operation))
                )
            )
            continue

        if isinstance(operation, ReturnOp):
            lines.append(
                return_statement(
                    tuple(names.operand(value) for value in operation.arguments)
                )
            )
            returned = True
            continue

        expression, implicit_resources = _eager_expression(operation, names)
        append_unique(resource_symbols, implicit_resources)
        if expression is None:
            raise unsupported("Eager", index, operation)
        lines.append(assignment(names.results(operation), expression))

    if not returned:
        raise PythonCodegenError("Eager Python emission requires func.return")
    return EagerPythonSource(
        "\n".join(lines) + "\n",
        entry_point,
        input_names,
        tuple(material_symbols),
        tuple(resource_symbols),
        operation_count,
    )


def _eager_expression(
    operation: Operation,
    names: SourceNames,
) -> tuple[str | None, tuple[str, ...]]:
    """Return one Eager call expression and implicit resource symbols."""

    operands = tuple(names.operand(value) for value in operation.operands)
    resources: tuple[str, ...] = ()

    if isinstance(operation, ckks.EncodeOp):
        return (
            "engine.encode("
            f"{operands[0]}, level={integer(operation.level, label='encode level')}, "
            f"scale={floating(operation.scale, label='encode scale')!r})",
            resources,
        )
    if isinstance(operation, ckks.DecodeOp):
        is_real = bool(integer(operation.is_real, label="decode is_real"))
        return f"engine.decode({operands[0]}, is_real={is_real!r})", resources
    if isinstance(operation, ckks.IntegerCoefficientsToRnsOp):
        basis = string(operation.modulus_basis, label="modulus basis")
        return (
            f"engine.integer_coefficients_to_rns({operands[0]}, "
            f"modulus_basis={basis!r})",
            resources,
        )
    if isinstance(operation, ckks.EncryptOp):
        symbol = string(operation.key_symbol, label="public-key symbol")
        return (
            f"engine.encrypt({operands[0]}, resources[{symbol!r}])",
            (symbol,),
        )
    if isinstance(operation, ckks.DecryptOp):
        symbol = string(operation.key_symbol, label="secret-key symbol")
        return (
            f"engine.decrypt({operands[0]}, resources[{symbol!r}])",
            (symbol,),
        )
    if isinstance(operation, ckks.NegateOp):
        return f"engine.negate({operands[0]})", resources
    if isinstance(operation, ckks.AddOp):
        return f"engine.add({operands[0]}, {operands[1]})", resources
    if isinstance(operation, ckks.SubtractOp):
        return f"engine.subtract({operands[0]}, {operands[1]})", resources
    if isinstance(operation, ckks.MultiplyOp):
        return f"engine.multiply({operands[0]}, {operands[1]})", resources
    if isinstance(operation, ckks.RotateOp):
        return (
            f"engine.rotate_with_key({operands[0]}, {operands[1]})",
            resources,
        )
    if isinstance(operation, ckks.RotateManyOp):
        keys = ", ".join(operands[1:])
        return (
            f"engine.rotate_many_with_keys({operands[0]}, ({keys},), "
            "use_hoisting=True)",
            resources,
        )
    if isinstance(operation, ckks.ToNttOp):
        return (
            f"engine.coefficient_domain_to_ntt_domain({operands[0]})",
            resources,
        )
    if isinstance(operation, ckks.FromNttOp):
        return (
            f"engine.ntt_domain_to_coefficient_domain({operands[0]})",
            resources,
        )
    if isinstance(operation, ckks.ToMontgomeryResiduesOp):
        if not isinstance(operation.value.type, ckks.PlaintextType):
            return None, resources
        return (
            f"engine.standard_residues_to_montgomery_residues({operands[0]})",
            resources,
        )
    if isinstance(operation, ckks.ToStandardResiduesOp):
        if not isinstance(operation.value.type, ckks.PlaintextType):
            return None, resources
        return (
            f"engine.montgomery_residues_to_standard_residues({operands[0]})",
            resources,
        )
    if isinstance(operation, ckks.AddScalarOp | ckks.MultiplyScalarOp):
        method = (
            "add_scalar"
            if isinstance(operation, ckks.AddScalarOp)
            else "multiply_scalar"
        )
        scalar = floating(operation.scalar, label="scalar")
        scalar_scale = floating(operation.scalar_scale, label="scalar scale")
        return (
            f"engine.{method}({operands[0]}, {scalar!r}, "
            f"scalar_scale={scalar_scale!r})",
            resources,
        )
    if isinstance(operation, ckks.MultiplyIntegerScalarOp):
        scalar = integer(operation.scalar, label="integer scalar")
        return (
            f"engine.multiply_integer_scalar({operands[0]}, {scalar})",
            resources,
        )
    if isinstance(
        operation, ckks.AddPlaintextOp | ckks.AddCompressedPlaintextOp
    ):
        return f"engine.add_plaintext({operands[0]}, {operands[1]})", resources
    if isinstance(
        operation,
        ckks.MultiplyPlaintextOp | ckks.MultiplyCompressedPlaintextOp,
    ):
        return (
            f"engine.multiply_plaintext({operands[0]}, {operands[1]})",
            resources,
        )
    if isinstance(operation, ckks.RelinearizeOp):
        symbol = "relinearization-key"
        return (
            f"engine.relinearize({operands[0]}, resources[{symbol!r}])",
            (symbol,),
        )
    if isinstance(operation, ckks.SwitchKeyOp):
        symbol = string(operation.key_symbol, label="switch-key symbol")
        return (
            f"engine.switch_key({operands[0]}, resources[{symbol!r}])",
            (symbol,),
        )
    if isinstance(operation, ckks.ConjugateOp):
        symbol = "conjugation-key"
        return (
            f"engine.conjugate({operands[0]}, resources[{symbol!r}])",
            (symbol,),
        )
    if isinstance(operation, ckks.RescaleOp):
        rounding = (
            "nearest"
            if operation.rounding is None
            else string(operation.rounding, label="rescale rounding")
        )
        return (
            f"engine.rescale_to_next_level({operands[0]}, rounding={rounding!r})",
            resources,
        )
    if isinstance(operation, ckks.ModSwitchOp):
        target = integer(
            operation.target_level, label="mod-switch target level"
        )
        return f"engine.mod_switch_to_level({operands[0]}, {target})", resources
    if isinstance(operation, ckks.ReinterpretScaleOp):
        scale = floating(operation.scale, label="reinterpret scale")
        return (
            f"engine.reinterpret_at_scale({operands[0]}, {scale!r})",
            resources,
        )

    prepare = _PREPARE_OPERATIONS.get(type(operation))
    if prepare is not None:
        selected_operation, source_role = prepare
        scale_mode = string(
            operation.attributes.get("scale_mode"),
            label="prepare scale mode",
        )
        return (
            f"engine.prepare_public_operand({operands[0]}, {operands[1]}, "
            f"operation={selected_operation!r}, source_role={source_role!r}, "
            f"scale_mode={scale_mode!r})",
            resources,
        )
    return None, resources


__all__ = ["EmitEagerPythonPass", "emit_eager_python"]
