"""Resolve logical roll steps to caller-bound CKKS rotation-key operands."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass, field

from xdsl.dialects.builtin import IntegerAttr, StringAttr

from fhelium.config import CkksConfig
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core, logical

from .._operation_transforms import (
    cast_before,
    ciphertext_type,
    display_name,
    program_operations,
)
from ._lower_logical_to_ckks import _replace_with_result_cast


def _normalize_step(step: int, slot_count: int) -> int:
    """Normalize one signed roll step to ``[-S/2, S/2)``."""

    return (step + slot_count // 2) % slot_count - slot_count // 2


@dataclass(frozen=True)
class ResolveRotationKeyOperandsPass:
    """Replace logical rolls with CKKS rotations bound to named key resources.

    The pass normalizes each step using the CKKS slot count and names its
    resource ``rotation-key:<step>``. It reads ``CkksConfig`` from the Compile
    workspace and records the resolved coefficient automorphism on the key
    operand. It writes symbolic references only; key creation and live binding
    remain caller responsibilities.
    """

    name: str = field(default="resolve-rotation-key-operands", init=False)

    def _config(self, workspace: dict[object, object]) -> CkksConfig:
        config = workspace.get(CkksConfig)
        if not isinstance(config, CkksConfig):
            raise ValueError(
                "Rotation-key resolution requires CkksConfig in the Compile "
                "workspace"
            )
        return config

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        rotations = tuple(
            operation
            for operation in program_operations(program)
            if isinstance(operation, logical.RollEncryptedOp)
        )
        if not rotations:
            return PassResult.unchanged(program)

        config = self._config(workspace)
        slot_count = config.num_slots
        transformed = inserted = 0
        for operation in rotations:
            if len(operation.operands) != 1:
                raise ValueError(
                    f"{display_name(operation)}: logical roll requires one "
                    "encrypted operand"
                )
            shift = operation.attributes.get("shift")
            if not isinstance(shift, IntegerAttr):
                raise ValueError(
                    f"{display_name(operation)}: logical roll requires an "
                    "integer shift"
                )
            normalized_step = _normalize_step(
                int(shift.value.data),
                slot_count,
            )
            source = operation.operands[0]
            typed = cast_before(
                operation,
                source,
                ciphertext_type(
                    source,
                    domain="coefficient",
                    residues="standard",
                    components=2,
                ),
                name_hint=f"{display_name(operation)}_ciphertext",
            )
            inserted += typed is not source
            if normalized_step == 0:
                _replace_with_result_cast(operation, (), typed)
                transformed += 1
                inserted += 1
                continue
            symbol = f"rotation-key:{normalized_step}"

            key_type = ckks.EvaluationKeyType().with_state(
                {
                    "kind": StringAttr("rotation-key"),
                    "rotation_step": IntegerAttr(normalized_step, 64),
                    "ring_dimension": IntegerAttr(2 * slot_count, 64),
                    "galois_element": IntegerAttr(
                        pow(
                            config.galois_generator,
                            (
                                -normalized_step
                                if config.galois_generator == 5
                                else normalized_step
                            )
                            % (2 * slot_count),
                            4 * slot_count,
                        ),
                        64,
                    ),
                }
            )
            key = core.ResourceRefOp(
                key_type,
                symbol=symbol,
                kind="rotation-key",
            )
            key.value.name_hint = f"rotation_key_{normalized_step}"
            rotation = ckks.RotateOp(typed, key.value, typed.type)
            rotation.result.name_hint = operation.result.name_hint
            _replace_with_result_cast(
                operation,
                (key, rotation),
                rotation.result,
            )
            transformed += 1
            inserted += 3

        return PassResult(
            program,
            PassStats(
                matched=len(rotations),
                transformed=transformed,
                inserted=inserted,
                removed=transformed,
            ),
        )


__all__ = ["ResolveRotationKeyOperandsPass"]
