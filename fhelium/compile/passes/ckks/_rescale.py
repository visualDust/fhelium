"""Place CKKS rescaling from multiplication structure and SSA use."""

from __future__ import annotations

from typing import Literal, cast

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    ArrayAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue, Use

from fhelium.ir import Program
from fhelium.ir.dialects import ckks
from fhelium.config import CkksConfig
from .._operation_transforms import (
    ciphertext_type,
    display_name,
    insert_after_and_replace_uses,
)
from ._transition_state import (
    compatible_states,
    exclusive_to,
    is_ckks_value_bridge,
    is_same_dialect_ckks_cast,
    reconcile_transition_paths,
    representation_pair,
    represented_state,
    retype_path,
    retype_result,
    source_through_casts,
)


@dataclass(frozen=True)
class _RescaleRequest:
    rounding: str = "nearest"

    @property
    def can_coalesce(self) -> bool:
        return self.rounding == "nearest"


_TRANSPARENT_TYPES = (
    ckks.RelinearizeOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
)
_MANUAL_SATISFACTION_PATH_TYPES = (
    *_TRANSPARENT_TYPES,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.NegateOp,
)


def _classify_rescale_use(use: Use) -> tuple[bool, tuple[Use, ...]]:
    """Find manual satisfaction and unsatisfied frontiers below one edge."""

    operation = use.operation
    if isinstance(operation, ckks.RescaleOp):
        return True, ()
    if is_same_dialect_ckks_cast(operation):
        assert isinstance(operation, UnrealizedConversionCastOp)
        if not _cast_preserves_rescale_state(operation):
            raise ValueError(
                "rescale cannot cross a same-dialect cast that changes depth, "
                "prime IDs, or scale"
            )
        output = operation.outputs[0]
        child_uses = tuple(output.uses)
        if not child_uses:
            return False, (use,)
        classified = tuple(_classify_rescale_use(child) for child in child_uses)
        if not any(satisfied for satisfied, _ in classified):
            return False, (use,)
        return True, tuple(
            frontier for _, pending in classified for frontier in pending
        )
    traversable = isinstance(
        operation, _MANUAL_SATISFACTION_PATH_TYPES
    ) or is_ckks_value_bridge(operation)
    if not traversable or len(operation.results) != 1:
        return False, (use,)
    child_uses = tuple(operation.results[0].uses)
    if not child_uses:
        return False, (use,)
    classified = tuple(_classify_rescale_use(child) for child in child_uses)
    if not any(satisfied for satisfied, _ in classified):
        return False, (use,)
    frontiers = tuple(
        frontier for _, pending in classified for frontier in pending
    )
    return True, frontiers


def _cast_preserves_rescale_state(cast: UnrealizedConversionCastOp) -> bool:
    source = represented_state(cast.inputs[0]) or {}
    result = represented_state(cast.outputs[0]) or {}
    for field in ("depth", "prime_ids", "scale"):
        if source.get(field) != result.get(field):
            return False
    return True


def _seed_request(operation: Operation) -> _RescaleRequest | None:
    if isinstance(operation, ckks.MultiplyOp):
        return _RescaleRequest()
    if isinstance(operation, ckks.MultiplyPlaintextOp):
        return _RescaleRequest()
    return None


def _materialize_rescale(
    operation: Operation,
    request: _RescaleRequest,
    *,
    edge: Use | None = None,
    config: CkksConfig | None = None,
) -> tuple[int, SSAValue]:
    opaque_cast = (
        edge.operation
        if edge is not None and is_same_dialect_ckks_cast(edge.operation)
        else None
    )
    if opaque_cast is not None:
        assert isinstance(opaque_cast, UnrealizedConversionCastOp)
        source = opaque_cast.outputs[0]
        if represented_state(
            source
        ) is None or not _cast_preserves_rescale_state(opaque_cast):
            raise ValueError(
                "rescale cannot cross a same-dialect cast that changes depth, "
                "prime IDs, or scale"
            )
    else:
        source = (
            operation.results[0]
            if edge is None
            else edge.operation.operands[edge.index]
        )
    block = operation.parent_block()
    if block is None:
        raise ValueError("rescale carrier is not attached to a block")
    inserted = 0
    if edge is None or opaque_cast is not None:
        insertion_point = operation if opaque_cast is None else opaque_cast

        def insert(created: Operation) -> None:
            nonlocal insertion_point
            insert_after_and_replace_uses(insertion_point, created)
            insertion_point = created

    else:
        consumer = edge.operation
        operand_index = edge.index

        def insert(created: Operation) -> None:
            block.insert_op_before(created, consumer)

    representation = representation_pair(
        source, operation="CKKS rescale placement"
    )
    if representation not in {
        ("coefficient", "standard"),
        ("ntt", "montgomery"),
    }:
        raise ValueError(
            "CKKS rescale placement cannot consume representation "
            f"{representation!r}"
        )

    source_type = ciphertext_type(
        source,
        domain=representation[0],
        residues=representation[1],
    )
    result_state = dict(source_type.state.data)
    depth = result_state.get("depth")
    if isinstance(depth, IntegerAttr):
        source_depth = int(depth.value.data)
        result_state["depth"] = IntegerAttr(depth.value.data + 1, 64)
        prime_ids = result_state.get("prime_ids")
        if isinstance(prime_ids, ArrayAttr):
            drop_count = (
                len(config.q_depth_groups[source_depth])
                if config is not None
                else None
            )
            if drop_count is None:
                result_state.pop("prime_ids")
            else:
                result_state["prime_ids"] = ArrayAttr(
                    prime_ids.data[drop_count:]
                )
        scale = result_state.get("scale")
        if isinstance(scale, FloatAttr) and config is not None:
            result_state["scale"] = FloatAttr(
                float(scale.value.data)
                / float(config.rescale_divisor(source_depth)),
                Float64Type(),
            )
        elif scale is not None:
            result_state.pop("scale")
    elif "scale" in result_state:
        result_state.pop("scale")
    rescaled = ckks.RescaleOp(
        source,
        ckks.CiphertextType().with_state(result_state),
        rounding=request.rounding,
        polynomial_domain=cast(
            Literal["coefficient", "ntt"], representation[0]
        ),
    )
    rescaled.result.name_hint = f"{display_name(operation)}_rescaled"
    insert(rescaled)
    if edge is not None and opaque_cast is None:
        consumer.operands[operand_index] = rescaled.result
    return inserted + 1, rescaled.result


def _place_rescales(
    program: Program,
    *,
    late: bool,
    config: CkksConfig | None,
) -> PassResult:
    if len(program.functions) != 1:
        raise ValueError("rescale placement requires one function")
    block = program.single_block("main")
    if any(
        operation.regions or operation.successors for operation in block.ops
    ):
        raise ValueError(
            "rescale placement supports a flat single-block SSA DAG"
        )

    pending: dict[Operation, _RescaleRequest] = {}
    for operation in block.ops:
        request = _seed_request(operation)
        if request is not None and len(operation.results) == 1:
            pending[operation] = request
    matched = len(pending)
    moved = consumed = 0

    changed = True
    while changed:
        changed = False
        for operation in tuple(block.ops):
            if isinstance(operation, ckks.RescaleOp):
                source, source_value, _ = source_through_casts(operation.value)
                if (
                    source in pending
                    and source is not None
                    and exclusive_to(source_value, operation)
                ):
                    del pending[source]
                    consumed += 1
                    changed = True
                continue

            if isinstance(operation, _TRANSPARENT_TYPES):
                source, source_value, casts = source_through_casts(
                    operation.operands[0]
                )
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                ):
                    continue
                state = represented_state(source_value)
                if state is None:
                    continue
                request = pending.pop(source)
                retype_path(casts, state)
                pending[operation] = request
                moved += 1
                changed = True
                continue

            if not late:
                continue
            if isinstance(operation, ckks.NegateOp):
                source, source_value, casts = source_through_casts(
                    operation.value
                )
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                ):
                    continue
                request = pending[source]
                state = represented_state(source_value)
                if state is None or not request.can_coalesce:
                    continue
                retype_path(casts, state)
                retype_result(operation.result, state)
                del pending[source]
                pending[operation] = request
                moved += 1
                changed = True
                continue
            if not isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
                continue
            sources: list[Operation] = []
            cast_paths = []
            states: list[dict[str, Attribute]] = []
            requests: list[_RescaleRequest] = []
            for operand in operation.operands:
                source, source_value, casts = source_through_casts(operand)
                if (
                    source not in pending
                    or source is None
                    or not exclusive_to(source_value, operation)
                ):
                    break
                request = pending[source]
                state = represented_state(source_value)
                if state is None or not request.can_coalesce:
                    break
                sources.append(source)
                cast_paths.append(casts)
                states.append(state)
                requests.append(request)
            if len(sources) != len(operation.operands) or not states:
                continue
            if any(
                not compatible_states(states[0], state) for state in states[1:]
            ):
                continue
            for casts in cast_paths:
                retype_path(casts, states[0])
            retype_result(operation.result, states[0])
            for source in set(sources):
                del pending[source]
            pending[operation] = requests[0]
            moved += 1
            changed = True

    inserted = 0
    rewritten_roots = []
    for operation in tuple(block.ops):
        request = pending.get(operation)
        if request is not None:
            classified = tuple(
                _classify_rescale_use(use)
                for use in tuple(operation.results[0].uses)
            )
            frontiers = tuple(
                frontier
                for _, pending_uses in classified
                for frontier in pending_uses
            )
            if any(satisfied for satisfied, _ in classified) or any(
                is_same_dialect_ckks_cast(frontier.operation)
                for frontier in frontiers
            ):
                for use in frontiers:
                    created, result = _materialize_rescale(
                        operation, request, edge=use, config=config
                    )
                    inserted += created
                    rewritten_roots.append(result)
                continue
            created, result = _materialize_rescale(
                operation, request, config=config
            )
            inserted += created
            rewritten_roots.append(result)

    if inserted:
        inserted += reconcile_transition_paths(program, tuple(rewritten_roots))
        program.function("main").update_function_type()
    transformed = moved + consumed + inserted
    if transformed == 0:
        return PassResult.unchanged(program, matched=matched)
    return PassResult(
        program,
        PassStats(
            matched=max(matched, transformed),
            transformed=transformed,
            inserted=inserted,
        ),
    )


@dataclass(frozen=True)
class InsertRescalePass:
    """Rescale each multiplication before its first non-transition use."""

    name: str = "insert-rescale"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        config = workspace.get(CkksConfig)
        return _place_rescales(
            program,
            late=False,
            config=config if isinstance(config, CkksConfig) else None,
        )


@dataclass(frozen=True)
class LateRescalePass:
    """Coalesce compatible add/sub/negate regions before rescaling."""

    name: str = "late-rescale"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        config = workspace.get(CkksConfig)
        return _place_rescales(
            program,
            late=True,
            config=config if isinstance(config, CkksConfig) else None,
        )


__all__ = ["InsertRescalePass", "LateRescalePass"]
