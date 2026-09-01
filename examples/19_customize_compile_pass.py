#!/usr/bin/env python3

"""Define a custom Compile pass and pipeline for an executable CKKS plan.

A caller-defined Compile pass recognizes one constant square matrix multiplied
by an encrypted vector. It precomputes adjusted cyclic diagonals, replaces the
opaque PyTorch call with baby-step/giant-step (BSGS) rotations, pointwise
multiplications, and additions. The source then applies an illustrative
elementwise square activation, which contributes one ciphertext-ciphertext
multiplication. Caller-selected passes place relinearization and rescales,
lower the schedule into RNS/NTT operations, link materials and evaluation
keys, and execute the Program through the Backend.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import cast

import torch
from common import print_table
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter

from fhelium import Preset, compile as fh_compile
from fhelium import ir
from fhelium.backend import OperationBackend
from fhelium.backend.ckks import CkksDeviceResources
from fhelium.backend.rns.chain import RnsChain
from fhelium.backend.rns.decomposition import HybridRnsDecomposition
from fhelium.backend.rns.layout import RnsLayout
from fhelium.config import CkksConfig
from fhelium.eager import Engine
from fhelium.ir.dialects import ckks, core, semantic
from fhelium.ir.dialects import torch as torch_dialect
from fhelium.values import Ciphertext

_MATRIX_SIZE = 8
_BABY_STEP = 3
_INDEX = torch.arange(_MATRIX_SIZE, dtype=torch.float64)
_WEIGHT = (
    0.08 * torch.sin((_INDEX[:, None] + 1.0) * (_INDEX[None, :] + 2.0))
    + 0.04 * torch.cos(_INDEX[:, None] - 2.0 * _INDEX[None, :])
    + 0.35 * torch.eye(_MATRIX_SIZE, dtype=torch.float64)
)


def matrix_vector(x: torch.Tensor) -> torch.Tensor:
    """Apply a fixed matrix and an illustrative elementwise square."""

    linear = torch.matmul(_WEIGHT, x)
    return linear * linear


def _cyclic_diagonal(matrix: torch.Tensor, shift: int) -> torch.Tensor:
    """Return coefficients multiplying ``torch.roll(x, shift)``."""

    row = torch.arange(matrix.shape[0])
    column = (row - shift) % matrix.shape[0]
    return matrix[row, column]


def _bsgs_reference(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    baby_step: int,
) -> torch.Tensor:
    """Evaluate the same BSGS schedule used by the custom pass in cleartext."""

    babies = [torch.roll(vector, shifts=step) for step in range(baby_step)]
    result = torch.zeros_like(vector)
    groups = math.ceil(matrix.shape[0] / baby_step)
    for group in range(groups):
        giant_shift = group * baby_step
        partial = torch.zeros_like(vector)
        for baby in range(baby_step):
            diagonal_index = giant_shift + baby
            if diagonal_index >= matrix.shape[0]:
                break
            diagonal = _cyclic_diagonal(matrix, diagonal_index)
            adjusted = torch.roll(diagonal, shifts=-giant_shift)
            partial = partial + babies[baby] * adjusted
        result = result + torch.roll(partial, shifts=giant_shift)
    return result


@dataclass(frozen=True)
class LowerConstantMatmulToBsgsPass:
    """Replace constant-matrix ``torch.matmul`` with a semantic BSGS schedule."""

    baby_step: int
    slot_count: int
    name: str = "lower-constant-matmul-to-bsgs"

    def run(
        self,
        program: ir.Program,
        shared_data: dict[object, object],
    ) -> fh_compile.PassResult:
        materials = shared_data[fh_compile.ConstantBundle]
        if not isinstance(materials, fh_compile.ConstantBundle):
            raise TypeError("capture workspace has no ConstantBundle")

        matched = transformed = inserted = 0
        plans: list[dict[str, object]] = []
        for operation in tuple(program.walk()):
            if not isinstance(operation, torch_dialect.CallOp):
                continue
            target = operation.attributes.get("fhelium.call.target")
            if (
                not isinstance(target, StringAttr)
                or target.data != "torch.matmul"
            ):
                continue
            matched += 1
            if len(operation.operands) != 2:
                raise ValueError("captured torch.matmul must have two operands")

            matrix_value, encrypted_vector = operation.operands
            matrix_owner = matrix_value.owner
            if not isinstance(matrix_owner, core.MaterialRefOp):
                raise ValueError(
                    "BSGS lowering requires a captured matrix material"
                )
            symbol = matrix_owner.attributes.get("symbol")
            if not isinstance(symbol, StringAttr):
                raise ValueError("captured matrix material has no symbol")
            matrix = materials[symbol.data]
            if not isinstance(matrix, torch.Tensor):
                raise TypeError("captured matrix material must be a Tensor")
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("BSGS lowering requires a square matrix")

            matrix_size = matrix.shape[0]
            if self.slot_count < matrix_size or self.slot_count % matrix_size:
                raise ValueError(
                    "BSGS slot_count must be a positive multiple of the "
                    f"matrix size {matrix_size}, got {self.slot_count}"
                )
            groups = math.ceil(matrix_size / self.baby_step)
            created: list[Operation] = []
            babies = [encrypted_vector]
            for baby in range(1, self.baby_step):
                rotation = semantic.RollOp(
                    encrypted_vector,
                    result_type=operation.result.type,
                    attributes={
                        "shift": IntegerAttr(baby, 64),
                        "dimension": IntegerAttr(-1, 64),
                    },
                )
                rotation.result.name_hint = f"baby_{baby}"
                created.append(rotation)
                babies.append(rotation.result)

            result = None
            diagonal_symbols: list[str] = []
            for group in range(groups):
                giant_shift = group * self.baby_step
                partial = None
                for baby in range(self.baby_step):
                    diagonal_index = giant_shift + baby
                    if diagonal_index >= matrix_size:
                        break
                    diagonal = _cyclic_diagonal(matrix, diagonal_index)
                    adjusted = torch.roll(diagonal, shifts=-giant_shift).clone()
                    adjusted = adjusted.repeat(self.slot_count // matrix_size)
                    diagonal_symbol = f"compile/bsgs/diagonal/{group}/{baby}"
                    materials[diagonal_symbol] = adjusted
                    diagonal_symbols.append(diagonal_symbol)
                    diagonal_type = semantic.PublicType().with_state(
                        {"role": StringAttr("message")}
                    )
                    diagonal_ref = core.MaterialRefOp(
                        diagonal_type,
                        symbol=diagonal_symbol,
                        kind="tensor",
                    )
                    diagonal_ref.value.name_hint = f"diagonal_{diagonal_index}"
                    product = semantic.MultiplyOp(
                        babies[baby],
                        diagonal_ref.value,
                        result_type=operation.result.type,
                    )
                    product.result.name_hint = f"group_{group}_product_{baby}"
                    created.extend((diagonal_ref, product))
                    if partial is None:
                        partial = product.result
                    else:
                        addition = semantic.AddOp(
                            partial,
                            product.result,
                            result_type=operation.result.type,
                        )
                        addition.result.name_hint = f"group_{group}_sum"
                        created.append(addition)
                        partial = addition.result

                if partial is None:
                    raise RuntimeError("BSGS group contained no diagonal")
                if giant_shift:
                    giant = semantic.RollOp(
                        partial,
                        result_type=operation.result.type,
                        attributes={
                            "shift": IntegerAttr(giant_shift, 64),
                            "dimension": IntegerAttr(-1, 64),
                        },
                    )
                    giant.result.name_hint = f"giant_{giant_shift}"
                    created.append(giant)
                    partial = giant.result
                if result is None:
                    result = partial
                else:
                    addition = semantic.AddOp(
                        result,
                        partial,
                        result_type=operation.result.type,
                    )
                    addition.result.name_hint = "bsgs_result"
                    created.append(addition)
                    result = addition.result

            if result is None:
                raise RuntimeError("BSGS lowering produced no result")
            result.name_hint = operation.result.name_hint
            Rewriter.replace_op(
                operation,
                tuple(created),
                new_results=(result,),
            )
            transformed += 1
            inserted += len(created)
            plans.append(
                {
                    "matrix_size": matrix_size,
                    "slot_count": self.slot_count,
                    "baby_step": self.baby_step,
                    "giant_groups": groups,
                    "diagonal_symbols": tuple(diagonal_symbols),
                }
            )

        shared_data["compile/bsgs/plans"] = tuple(plans)
        if transformed:
            shared_data["compile/bsgs/semantic_program"] = program.clone()
        if transformed == 0:
            return fh_compile.PassResult.unchanged(program, matched=matched)
        return fh_compile.PassResult(
            program,
            fh_compile.PassStats(
                matched=matched,
                transformed=transformed,
                inserted=inserted,
                removed=transformed,
            ),
            diagnostics=("lowered torch.matmul to semantic BSGS",),
        )


def _torch_call_targets(program: ir.Program) -> list[str]:
    """Return encoded PyTorch call targets in structural order."""

    targets: list[str] = []
    for operation in program.walk():
        if not isinstance(operation, torch_dialect.CallOp):
            continue
        target = operation.attributes.get("fhelium.call.target")
        if isinstance(target, StringAttr):
            targets.append(target.data)
    return targets


def _program_summary(program: ir.Program) -> tuple[int, int, int]:
    """Return total operation, material-reference, and resource-reference counts."""

    operations = tuple(program.walk())
    return (
        len(operations),
        sum(
            isinstance(operation, core.MaterialRefOp)
            for operation in operations
        ),
        sum(
            isinstance(operation, core.ResourceRefOp)
            for operation in operations
        ),
    )


def _fhelium_operation_counts(program: ir.Program) -> list[tuple[str, int]]:
    """Count non-structural FHElium operations for compact terminal output."""

    counts = Counter(
        operation.name
        for operation in program.walk()
        if operation.name.startswith("fhelium_")
    )
    return sorted(counts.items())


def main() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    device = "cpu"
    capture_inputs = {"x": fh_compile.encrypted(slots=_MATRIX_SIZE)}
    workspace = fh_compile.CompileWorkspace({CkksConfig: config})
    # Capture the source call while retaining the matrix Tensor in the
    # Compilation's ConstantBundle rather than embedding it in textual IR.
    captured = fh_compile.capture(
        matrix_vector,
        inputs=capture_inputs,
        workspace=workspace,
    )

    compiled = fh_compile.Pipeline(
        (
            # Replace torch.matmul with the inspectable BSGS rotation and
            # diagonal-product schedule, then remove the old matrix reference.
            LowerConstantMatmulToBsgsPass(
                _BABY_STEP,
                config.num_slots,
            ),
            fh_compile.EliminateDeadValuesPass(),
            # Classify encrypted/public operands and introduce concrete CKKS
            # plaintext preparation and representation transitions.
            fh_compile.LowerSemanticToLogicalPass(),
            fh_compile.InsertPlaintextPreparationPass(),
            fh_compile.InsertMultiplyNttTransitionsPass(),
            fh_compile.LowerLogicalToCkksPass(),
            # Give each rotation a structured key resource operand. The later
            # Backend link matches keys by rotation step, not symbol spelling.
            fh_compile.ResolveRotationKeyOperandsPass(),
            # The square activation creates one CT×CT product. Select its
            # immediate relinearization independently from rescale placement.
            fh_compile.InsertRelinearizationPass(),
            # Consolidate product rescaling at legal add-tree frontiers before
            # assigning the resulting levels and per-value actual scales.
            fh_compile.LateRescalePass(),
            fh_compile.AssignCkksLevelsPass(entry_level=0),
            fh_compile.AssignCkksScalesPass(
                entry_scale=config.default_scale,
            ),
            # Make message encoding visible in IR, then lower arithmetic while
            # retaining whole-operation rotation implementations.
            fh_compile.LowerMessagePlaintextPreparationPass(),
            fh_compile.LowerCkksToRnsNttPass(
                preserve=frozenset({ckks.RotateOp.name})
            ),
        )
    ).run(captured)

    # Generate only the evaluation keys required by the transformed Program.
    requirements = ir.analyze_evaluation_key_requirements(compiled.program)
    rotation_steps = tuple(sorted(requirements.rotation_steps))
    engine = Engine(config, rng_seed=19)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    rotation_keys = tuple(
        engine.create_rotation_key(step, secret_key, device=device)
        for step in rotation_steps
    )
    relinearization_key = (
        engine.create_relinearization_key(secret_key, device=device)
        if requirements.requires_relinearization
        else None
    )
    evaluation_keys = (
        *rotation_keys,
        *((relinearization_key,) if relinearization_key is not None else ()),
    )
    # The materializer lazily constructs the RNS, NTT, rescale, codec, and
    # key-switch tables requested during this Program link.
    chain = RnsChain(config.num_q_primes, config.num_p_primes)
    layout = RnsLayout(chain, HybridRnsDecomposition(chain))
    device_resources = CkksDeviceResources(
        config=config,
        rns_layout=layout,
        device=device,
        rng_seed=29,
    )
    backend = OperationBackend(
        keys=evaluation_keys,
        materializer=device_resources,
    )
    # link() reads captured/derived materials from the Compilation, combines
    # them with this Backend's live workspace, and returns a fixed executable.
    executable = backend.link(compiled)

    # Check the custom BSGS algebra independently with ordinary float64 data.
    clear_x = 0.03 * torch.sin(0.7 * _INDEX) + 0.01 * torch.cos(1.3 * _INDEX)
    captured_callable = cast(
        fh_compile.CapturedCallable[torch.Tensor],
        captured.workspace[fh_compile.CapturedCallable],
    )
    direct = captured_callable.reference(clear_x)
    bsgs_linear = _bsgs_reference(_WEIGHT, clear_x, _BABY_STEP)
    bsgs = bsgs_linear * bsgs_linear
    torch.testing.assert_close(
        bsgs,
        direct,
        rtol=8 * torch.finfo(torch.float64).eps,
        atol=8 * torch.finfo(torch.float64).eps,
    )
    # Repeat the logical eight-slot vector across the complete CKKS slot ring
    # so full-ring rotations implement the intended small cyclic layout.
    tiled_x = clear_x.repeat(config.num_slots // _MATRIX_SIZE)
    encrypted_x = engine.encrypt_message(
        tiled_x,
        public_key,
        level=0,
        scale=config.default_scale,
        device=device,
    )
    result = executable.run(encrypted_x)
    encrypted_result = cast(Ciphertext, result)
    decoded = engine.decrypt_message(encrypted_result, secret_key)
    expected = direct.repeat(config.num_slots // _MATRIX_SIZE)
    torch.testing.assert_close(
        decoded.real,
        expected,
        rtol=5e-6,
        atol=5e-6,
    )
    torch.testing.assert_close(
        decoded.imag,
        torch.zeros_like(decoded.imag),
        rtol=0.0,
        atol=5e-6,
    )

    captured_targets = _torch_call_targets(captured.program)
    semantic_bsgs = cast(
        ir.Program,
        compiled.workspace["compile/bsgs/semantic_program"],
    )
    semantic_targets = _torch_call_targets(semantic_bsgs)
    compiled_targets = _torch_call_targets(compiled.program)
    captured_summary = _program_summary(captured.program)
    semantic_summary = _program_summary(semantic_bsgs)
    compiled_summary = _program_summary(compiled.program)

    print_table(
        [
            "stage",
            "IR ops",
            "materials",
            "resources",
            "torch.call targets",
            "rotation steps",
        ],
        [
            [
                "captured",
                *captured_summary,
                ", ".join(captured_targets) or "—",
                "—",
            ],
            [
                "semantic BSGS",
                *semantic_summary,
                ", ".join(semantic_targets) or "—",
                "—",
            ],
            [
                "compiled",
                *compiled_summary,
                ", ".join(compiled_targets) or "—",
                ", ".join(str(step) for step in rotation_steps) or "—",
            ],
        ],
    )
    print()
    print_table(
        ["BSGS property", "value"],
        [
            ["matrix size", _MATRIX_SIZE],
            ["CKKS slots", config.num_slots],
            ["baby step", _BABY_STEP],
            ["giant groups", math.ceil(_MATRIX_SIZE / _BABY_STEP)],
            ["derived diagonals", _MATRIX_SIZE],
            [
                "clear maximum error",
                f"{float((bsgs - direct).abs().max()):.3e}",
            ],
            [
                "Backend maximum error",
                f"{float((decoded.real - expected).abs().max()):.3e}",
            ],
            ["output level", encrypted_result.level],
            ["output scale", f"{encrypted_result.scale:.6e}"],
        ],
    )
    print()
    print_table(
        ["pass", "matched", "transformed", "inserted", "removed"],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                report.stats.inserted,
                report.stats.removed,
            ]
            for report in compiled.reports
        ],
    )
    print()
    print_table(
        ["compiled FHElium operation", "count"],
        _fhelium_operation_counts(compiled.program),
    )


if __name__ == "__main__":
    main()
