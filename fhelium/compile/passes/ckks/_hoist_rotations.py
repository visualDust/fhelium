"""Form caller-selected hoisted rotation groups from primitive rotations."""

from __future__ import annotations

from ..._pipeline import (
    DecisionRecord,
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.rewriter import Rewriter

from fhelium.ir import (
    Program,
)
from fhelium.ir.dialects import ckks, core

_RotationCandidate = tuple[ckks.RotateOp, int]
_RotationGroup = tuple[_RotationCandidate, ...]


@dataclass(frozen=True)
class HoistRotationsPass:
    """Group adjacent rotations under a caller-selected maximum group size.

    This pass makes no target, latency, or memory decision. Composing it into a
    pipeline selects hoisting for locally adjacent rotations of the same SSA
    value. ``max_group_size`` limits each emitted group; callers that require a
    resource-aware grouping policy can implement another pass that emits the
    same ``fhelium_ckks.hoisted_rotate_many`` operation.

    Each input rotation has already been resolved to a caller-owned key
    resource. Its key type records the normalized step used for grouping and
    reporting; this pass never selects or creates key material.
    """

    max_group_size: int | None = None
    name: str = "hoist-rotations"

    def __post_init__(self) -> None:
        if self.max_group_size is None:
            return
        if type(self.max_group_size) is not int:
            raise TypeError("max_group_size must be an integer or None")
        if self.max_group_size < 2:
            raise ValueError("max_group_size must be at least 2")

    @staticmethod
    def _normalized_step(
        operation: ckks.RotateOp,
    ) -> tuple[int | None, str | None]:
        state = getattr(operation.key.type, "state", None)
        data = getattr(state, "data", None)
        represented = data.get("rotation_step") if data is not None else None
        if not isinstance(represented, IntegerAttr):
            return (
                None,
                "rotation hoisting requires a represented rotation-key step",
            )
        return int(represented.value.data), None

    @staticmethod
    def _key_symbol(operation: ckks.RotateOp) -> str:
        owner = operation.key.owner
        if isinstance(owner, core.ResourceRefOp) and isinstance(
            owner.symbol, StringAttr
        ):
            return owner.symbol.data
        return "<non-reference-key-operand>"

    def _groups(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> tuple[
        tuple[_RotationGroup, ...],
        int,
        tuple[str, ...],
    ]:
        del workspace
        groups: list[_RotationGroup] = []
        skipped = 0
        diagnostics: list[str] = []
        for function in program.functions:
            for block in function.body.blocks:
                run: list[_RotationCandidate] = []

                def flush() -> None:
                    if len(run) < 2:
                        run.clear()
                        return
                    size = self.max_group_size or len(run)
                    for start in range(0, len(run), size):
                        group = tuple(run[start : start + size])
                        if len(group) >= 2:
                            groups.append(group)
                    run.clear()

                for operation in tuple(block.ops):
                    if not isinstance(operation, ckks.RotateOp):
                        flush()
                        continue
                    normalized, diagnostic = self._normalized_step(operation)
                    if diagnostic is not None:
                        flush()
                        skipped += 1
                        diagnostics.append(diagnostic)
                        continue
                    if normalized is None or normalized == 0:
                        flush()
                        continue
                    if run and operation.value is not run[0][0].value:
                        flush()
                    run.append((operation, normalized))
                flush()
        return tuple(groups), skipped, tuple(dict.fromkeys(diagnostics))

    @staticmethod
    def _rewrite_group(group: _RotationGroup) -> DecisionRecord:
        first = group[0][0]
        block = first.parent_block()
        if block is None:
            raise ValueError("Rotation scheduling requires attached operations")
        steps = tuple(step for _, step in group)
        symbols = tuple(
            HoistRotationsPass._key_symbol(operation) for operation, _ in group
        )
        keys = tuple(operation.key for operation, _ in group)
        result_types = tuple(operation.result.type for operation, _ in group)
        if not all(
            isinstance(result_type, ckks.CiphertextType)
            for result_type in result_types
        ):
            raise TypeError("CKKS rotations require ciphertext result types")
        hoisted = ckks.RotateManyOp(
            first.value,
            keys,
            tuple(result_types),  # type: ignore[arg-type]
        )
        for (source, _), result in zip(group, hoisted.results, strict=True):
            result.name_hint = source.result.name_hint
        block.insert_op_before(hoisted, first)
        for (source, _), result in zip(group, hoisted.results, strict=True):
            source.result.replace_all_uses_with(result)
        for source, _ in group:
            Rewriter.erase_op(source)
        return DecisionRecord(
            subject="rotation-hoisting",
            selected="hoisted",
            candidates=("independent", "hoisted"),
            details=(
                f"offsets={steps}",
                f"key_resources={symbols}",
            ),
        )

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Replace each selected local rotation run and report the schedule."""

        groups, skipped, diagnostics = self._groups(program, workspace)
        if not groups:
            return PassResult.unchanged(
                program,
                matched=skipped,
                skipped=skipped,
                diagnostics=diagnostics,
            )
        decisions = tuple(self._rewrite_group(group) for group in groups)
        return PassResult(
            program,
            PassStats(
                matched=len(groups) + skipped,
                transformed=len(groups),
                inserted=len(groups),
                removed=sum(len(group) for group in groups),
                skipped=skipped,
            ),
            diagnostics=diagnostics,
            decisions=decisions,
        )


__all__ = ["HoistRotationsPass"]
