"""Essential source-oriented compilation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass

import pytest
import torch
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import Preset
from fhelium.config import CkksConfig
from fhelium import compile as fh_compile
from fhelium import ir

_CAPTURE_WEIGHT = torch.tensor([1.25, -0.5], dtype=torch.float64)


def _operation_names(program: ir.Program) -> tuple[str, ...]:
    return tuple(ir.operation_name(operation) for operation in program.walk())


def _semantic_to_ckks_passes() -> tuple[fh_compile.Pass, ...]:
    return (
        fh_compile.EliminateDeadValuesPass(),
        fh_compile.LowerSemanticToLogicalPass(),
        fh_compile.InsertPlaintextPreparationPass(),
        fh_compile.InsertMultiplyNttTransitionsPass(),
        fh_compile.LowerLogicalToCkksPass(),
        fh_compile.InsertRelinearizationPass(),
        fh_compile.InsertRescalePass(),
    )


def test_capture_retains_the_python_reference_and_emits_registered_semantics() -> (
    None
):
    def weighted(
        secret: torch.Tensor,
        bias: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        value = secret * _CAPTURE_WEIGHT + bias
        return -value if enabled else value

    captured = fh_compile.capture(
        weighted,
        inputs={
            "secret": fh_compile.encrypted(),
            "bias": fh_compile.message(),
            "enabled": fh_compile.static(True),
        },
    )

    captured_callable = captured.workspace[fh_compile.CapturedCallable]
    assert isinstance(captured_callable, fh_compile.CapturedCallable)
    assert tuple(captured_callable.runtime_signature.parameters) == (
        "secret",
        "bias",
    )
    assert any(
        isinstance(operation, ir.dialects.semantic.MultiplyOp)
        for operation in captured.program.walk()
    )
    expected = weighted(
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
        True,
    )
    actual = captured_callable.reference(
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(actual, expected)


def test_pipeline_retains_workspace_and_appends_reports() -> None:
    captured = fh_compile.capture(
        lambda value: -value,
        inputs={"value": fh_compile.encrypted()},
    )
    captured_callable = captured.workspace[fh_compile.CapturedCallable]

    first = fh_compile.Pipeline((fh_compile.EliminateDeadValuesPass(),)).run(
        captured
    )
    second = fh_compile.Pipeline(
        (fh_compile.LowerSemanticToLogicalPass(),)
    ).run(first)

    assert first is not captured
    assert first.program is not captured.program
    assert second.workspace is captured.workspace
    assert second.workspace[fh_compile.CapturedCallable] is captured_callable
    assert tuple(report.name for report in second.reports) == (
        "eliminate-dead-values",
        "lower-semantic-to-logical",
    )


def test_compile_partially_lowers_understood_operations() -> None:
    def mixed(secret: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        return torch.sin(secret) + public

    result = fh_compile.Pipeline(_semantic_to_ckks_passes()).run(
        fh_compile.capture(
            mixed,
            inputs={
                "secret": fh_compile.encrypted(),
                "public": fh_compile.message(),
            },
        )
    )

    names = _operation_names(result.program)
    assert "torch.call" in names
    assert "fhelium_ckks.prepare.add.message" in names
    assert "fhelium_ckks.add_plaintext" in names


def test_arithmetic_lowering_preserves_unbound_parameter_placeholders() -> None:
    def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right

    pipeline = fh_compile.Pipeline(
        (
            *_semantic_to_ckks_passes(),
            fh_compile.LowerCkksToRnsNttPass(),
        )
    )
    workspace = fh_compile.CompileWorkspace()

    result = pipeline.run(
        fh_compile.capture(
            add,
            inputs={
                "left": fh_compile.encrypted(),
                "right": fh_compile.encrypted(),
            },
            workspace=workspace,
        )
    )

    assert result.workspace is workspace
    assert "fhelium_rns.add_standard" in _operation_names(result.program)
    assert any(
        isinstance(op, ir.dialects.core.MaterialRefOp)
        for op in result.program.walk()
    )


def test_pass_can_select_ckks_config_after_ckks_ir_exists() -> None:
    def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right

    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    caller_value = object()

    @dataclass(frozen=True)
    class SelectConfigFromCkksProgram:
        name: str = "select-config-from-ckks-program"

        def run(self, compilation: "Compilation") -> fh_compile.PassResult:
            program = compilation.program
            workspace = compilation.workspace
            assert "fhelium_ckks.add" in _operation_names(program)
            assert workspace["caller/value"] is caller_value
            workspace[CkksConfig] = config
            return fh_compile.PassResult.unchanged(program, matched=1)

    workspace = fh_compile.CompileWorkspace({"caller/value": caller_value})
    pipeline = fh_compile.Pipeline(
        (
            *_semantic_to_ckks_passes(),
            SelectConfigFromCkksProgram(),
            fh_compile.LowerCkksToRnsNttPass(),
        )
    )

    result = pipeline.run(
        fh_compile.capture(
            add,
            inputs={
                "left": fh_compile.encrypted(),
                "right": fh_compile.encrypted(),
            },
            workspace=workspace,
        )
    )

    assert result.workspace is workspace
    assert workspace["caller/value"] is caller_value
    assert "fhelium_rns.add_standard" in _operation_names(result.program)


def test_backend_assignment_is_partial_and_persists_in_program_text() -> None:
    source = _rotation_program((1,))
    result = fh_compile.Pipeline(
        (
            fh_compile.AssignImplementationsPass(
                {"fhelium_ckks.rotate": "native-direct-rotate"}
            ),
        )
    ).run(fh_compile.Compilation(source))

    rotation = next(
        operation
        for operation in result.program.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    assignment = rotation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assignment, StringAttr)
    assert assignment.data == "native-direct-rotate"
    reparsed = ir.parse(result.program.to_text())
    reparsed_rotation = next(
        operation
        for operation in reparsed.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    reparsed_assignment = reparsed_rotation.attributes[
        ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE
    ]
    assert isinstance(reparsed_assignment, StringAttr)
    assert reparsed_assignment.data == "native-direct-rotate"


def test_backend_assignment_rejects_conflict_unless_overwrite_is_selected() -> (
    None
):
    source = _rotation_program((1,))
    first = (
        fh_compile.Pipeline(
            (
                fh_compile.AssignImplementationsPass(
                    {"fhelium_ckks.rotate": "first"}
                ),
            )
        )
        .run(fh_compile.Compilation(source))
        .program
    )
    with pytest.raises(ValueError):
        fh_compile.Pipeline(
            (
                fh_compile.AssignImplementationsPass(
                    {"fhelium_ckks.rotate": "second"}
                ),
            )
        ).run(fh_compile.Compilation(first))

    replaced = fh_compile.Pipeline(
        (
            fh_compile.AssignImplementationsPass(
                {"fhelium_ckks.rotate": "second"},
                overwrite=True,
            ),
        )
    ).run(fh_compile.Compilation(first))
    rotation = next(
        operation
        for operation in replaced.program.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    assignment = rotation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assignment, StringAttr)
    assert assignment.data == "second"


def _rotation_program(
    shifts: tuple[int, ...],
    *,
    key_symbol_index: int | None = None,
    ring_dimension: int | None = 32,
) -> ir.Program:
    result_names = ("rotation_one", "rotation_two", "rotation_three")
    ciphertext_type = ir.dialects.ckks.CiphertextType()
    if ring_dimension is not None:
        ciphertext_type = ciphertext_type.with_state(
            {"ring_dimension": IntegerAttr(ring_dimension, 64)}
        )
    block = Block(arg_types=(ciphertext_type,))
    rotations = []
    key_references = []
    for index, shift in enumerate(shifts):
        symbol = (
            "caller-key"
            if index == key_symbol_index
            else f"rotation-key:{shift}"
        )
        key_type = ir.dialects.ckks.EvaluationKeyType().with_state(
            {"rotation_step": IntegerAttr(shift, 64)}
        )
        key_reference = ir.dialects.core.MaterialRefOp(
            key_type,
            symbol=symbol,
        )
        rotation = ir.dialects.ckks.RotateOp(
            block.args[0],
            key_reference.value,
            ciphertext_type,
        )
        rotation.result.name_hint = (
            result_names[index]
            if index < len(result_names)
            else f"rotation_extra_{index}"
        )
        key_references.append(key_reference)
        rotations.append(rotation)
    block.add_ops(
        (
            *(
                operation
                for pair in zip(key_references, rotations, strict=True)
                for operation in pair
            ),
            ReturnOp(*(item.result for item in rotations)),
        )
    )
    return ir.Program.from_function(
        block,
        tuple(ciphertext_type for _ in rotations),
    )


@pytest.mark.parametrize(
    "costs,assigned,lowered",
    [
        ((2.0, 1.0), False, True),
        ((1.0, 2.0), False, False),
        ((1.0, 1.0), False, False),
        (None, False, True),
        ((2.0, 1.0), True, False),
    ],
)
def test_execution_lowering_compares_caller_costs(costs, assigned, lowered):
    from xdsl.dialects.builtin import ArrayAttr
    from fhelium.backend import OperationBackend
    from fhelium.backend.triton._pointwise import TritonFusionImplementation
    from fhelium.compile.passes.lowering import SelectExecutionLoweringsPass
    from fhelium.ir.dialects import ckks, fusion, rns

    def ciphertext(components):
        return ckks.CiphertextType(
            {
                "components": IntegerAttr(components, 64),
                "shape": ArrayAttr(
                    IntegerAttr(x, 64) for x in (components, 2, 256)
                ),
                "polynomial_domain": StringAttr("ntt"),
                "residue_representation": StringAttr("montgomery"),
                "device": StringAttr("cuda:0"),
                "dtype": StringAttr("torch.int64"),
            }
        )

    typ = ciphertext(2)
    block = Block(arg_types=(typ, typ, rns.RnsParametersType()))
    operation = ckks.MultiplyOp(
        block.args[0],
        block.args[1],
        ciphertext(3),
        parameters=(block.args[2],),
    )
    if assigned:
        operation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE] = (
            StringAttr("native-ct2-convolution")
        )
    block.add_ops((operation, ReturnOp(operation)))
    compilation = fh_compile.Compilation(
        ir.Program.from_function(block, (ciphertext(3),))
    )
    uses = tuple(tuple(v.uses) for v in block.args)
    calls = []

    def estimate(whole, candidate):
        assert whole is operation
        assert any(isinstance(op, fusion.FusedOp) for op in candidate)
        assert all(op.parent_block() is not block for op in candidate)
        calls.append(True)
        return costs

    result = SelectExecutionLoweringsPass(
        OperationBackend().registry,
        (TritonFusionImplementation(),),
        execution_cost=estimate,
    ).run(compilation)
    assert bool(result.stats.transformed) == lowered
    assert bool(calls) != assigned
    if not lowered:
        assert tuple(tuple(v.uses) for v in block.args) == uses
    if not assigned and costs is not None:
        assert any(
            "whole_cost=" in line for line in result.decisions[0].details
        )


@pytest.mark.parametrize("lowered", [False, True])
def test_reuse_internal_arithmetic_preserves_materials_rounding_and_returns(
    lowered,
):
    from xdsl.dialects.builtin import UnrealizedConversionCastOp
    from fhelium.ir.dialects import ckks, rns

    ciphertext = ckks.CiphertextType()
    compact = ckks.CompressedPlaintextType()
    parameters = rns.RnsParametersType()
    bundle = rns.RnsBundleType()
    block = Block(
        arg_types=(ciphertext, compact, compact, parameters, parameters)
    )

    def multiply(value, plaintext, table):
        operation = ckks.MultiplyCompressedPlaintextOp.create(
            operands=(value, plaintext, table),
            result_types=(ciphertext,),
            attributes={"compression_layout": StringAttr("contiguous")},
        )
        block.add_op(operation)
        return operation

    def rescale(value, table, rounding="nearest"):
        if lowered:
            cast = UnrealizedConversionCastOp.get((value,), (bundle,))
            operation = rns.RescaleDropLeadingPrimesOp(
                cast.outputs[0],
                (table,),
                bundle,
                rounding=rounding,
                polynomial_domain="ntt",
            )
            result = UnrealizedConversionCastOp.get(
                (operation.result,), (ciphertext,)
            )
            block.add_ops((cast, operation, result))
            return operation, result.outputs[0]
        operation = ckks.RescaleOp(
            value,
            ciphertext,
            parameters=(table,),
            rounding=rounding,
            polynomial_domain="ntt",
        )
        block.add_op(operation)
        return operation, operation.result

    products = []
    quotients = []
    outputs = []
    for plaintext, table in ((1, 3), (1, 3), (2, 3), (1, 4)):
        product = multiply(
            block.args[0], block.args[plaintext], block.args[table]
        )
        quotient, result = rescale(product.result, block.args[table])
        products.append(product)
        quotients.append(quotient)
        outputs.append(multiply(result, block.args[1], block.args[3]).result)
    floor, result = rescale(products[0].result, block.args[3], "floor")
    outputs.append(multiply(result, block.args[1], block.args[3]).result)
    block.add_op(ReturnOp(*outputs))
    program = ir.Program.from_function(
        block, tuple(ciphertext for _ in outputs)
    )

    fh_compile.ReuseIntermediatesPass().run(fh_compile.Compilation(program))

    program.verify_structure()
    assert products[1].parent is None and quotients[1].parent is None
    assert all(
        operation.parent is block
        for operation in (
            products[0],
            *products[2:],
            quotients[0],
            *quotients[2:],
            floor,
        )
    )
    assert tuple(block.last_op.operands) == tuple(outputs)
    assert len({value.owner for value in outputs}) == len(outputs)


@pytest.mark.parametrize(
    "guard",
    [
        "return",
        "view-return",
        "assigned-producer",
        "assigned-reader",
        "assignment",
        "opaque",
        "random",
    ],
)
def test_reuse_preserves_escaping_storage_and_effect_boundaries(guard):
    from xdsl.dialects.builtin import FloatAttr, Float64Type
    from xdsl.dialects.test import TestOp as OpaqueOp
    from fhelium.ir.dialects import ckks

    ciphertext = ckks.CiphertextType()
    block = Block(arg_types=(ciphertext, ciphertext, ckks.MessageType()))
    first = ckks.NegateOp(block.args[0])
    second = ckks.NegateOp(block.args[0])
    leaves = [
        ckks.AddOp(operation, block.args[1]) for operation in (first, second)
    ]
    assignment = StringAttr("caller-selected")
    if guard == "assigned-producer":
        for operation in (first, second):
            operation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE] = (
                assignment
            )
    if guard == "assigned-reader":
        for operation in leaves:
            operation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE] = (
                assignment
            )
    block.add_ops((first, leaves[0]))
    if guard == "assignment":
        barrier = ckks.NegateOp(
            block.args[1],
            attributes={ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE: assignment},
        )
        block.add_op(barrier)
    elif guard == "opaque":
        block.add_op(OpaqueOp())
    elif guard == "random":
        block.add_op(
            ckks.EncodeOp.create(
                operands=(block.args[2],),
                result_types=(ckks.PlaintextType(),),
                attributes={
                    "depth": IntegerAttr(0, 64),
                    "scale": FloatAttr(8.0, Float64Type()),
                },
            )
        )
    block.add_ops((second, leaves[1]))
    if guard == "view-return":
        views = [
            ckks.ReinterpretScaleOp(
                operation, attributes={"scale": FloatAttr(8.0, Float64Type())}
            )
            for operation in (first, second)
        ]
        block.add_ops(views)
        returned = views
    elif guard == "return":
        returned = [first, second]
    else:
        returned = leaves
    block.add_op(ReturnOp(*returned))
    program = ir.Program.from_function(block, (ciphertext, ciphertext))

    fh_compile.ReuseIntermediatesPass().run(fh_compile.Compilation(program))

    program.verify_structure()
    assert first.parent is block and second.parent is block
    assert tuple(block.last_op.operands) == tuple(
        operation.results[0] for operation in returned
    )
