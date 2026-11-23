"""Guard and persist builtin lowering decisions and fusion schedules.

Whole-block plans contain positions in a guarded pass-input graph. Local plans
contain port structure and operand-role recipes. Both bind to the current graph
without retaining operations, types, Tensor data, or material descriptions.
The owning source-family cache supplies the compiler/environment identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cache
import hashlib
import json
from math import prod
import struct
from typing import Any

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Data, Operation, ParametrizedAttribute

from fhelium.backend.implementation import OperationImplementationRegistry
from fhelium.ir import (
    DEFAULT_OPERATION_SPECS,
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
)
from fhelium.ir.dialects import ckks, core, rns

from fhelium.ir._structure import AttributeDigests

from ._materials import MATERIAL_REQUEST_ATTRIBUTE


@dataclass
class SelectionPlans:
    """JSON-compatible decisions shared by instances of one source family."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    local_entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "entries": dict(self.entries),
            "local_entries": dict(self.local_entries),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelectionPlans:
        if payload.get("version") != 2:
            return cls()
        return cls(
            dict(payload["entries"]), dict(payload.get("local_entries", {}))
        )


def plan_event(workspace, stage: str, hit: bool, reason: str) -> None:
    stats = workspace.setdefault("selection_plan_stats", {})
    counts = stats.setdefault(stage, {"hits": 0, "misses": 0, "reason": ""})
    counts["hits" if hit else "misses"] += 1
    counts["reason"] = reason


def _literal(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, type) and issubclass(value, Operation):
        return [value.__module__, value.__qualname__]
    if isinstance(value, (tuple, list)):
        return [_literal(item) for item in value]
    raise ValueError("implementation has custom state")


@cache
def _builtin_types():
    from fhelium.backend.assembly import create_builtin_operation_registry
    from fhelium.backend.triton import (
        TritonFusionImplementation,
        TritonRnsImplementation,
        TritonTensorFusionImplementation,
    )

    return {
        type(item)
        for item in create_builtin_operation_registry().implementations
    } | {
        TritonFusionImplementation,
        TritonRnsImplementation,
        TritonTensorFusionImplementation,
    }


def _implementation_identity(implementation):
    from fhelium.backend.triton._weighted_sum import (
        TritonWeightedSumImplementation,
    )

    cls = type(implementation)
    if cls not in _builtin_types():
        raise ValueError("custom implementation or matcher")
    # Subclasses and per-instance callback overrides have no declared contract.
    for name in (
        "match_fusion",
        "new_fusion_matcher",
        "supports_operation",
        "tensor_requirements",
    ):
        method = getattr(implementation, name, None)
        if callable(method) and not getattr(
            method, "__module__", ""
        ).startswith("fhelium.backend."):
            raise ValueError("custom implementation callback")
    return [
        cls.__module__,
        cls.__qualname__,
        _literal(implementation.name),
        _literal(implementation.operation_types),
        _literal(implementation.supports_in_place),
        {
            name: _literal(value)
            for name, value in sorted(vars(implementation).items())
            # This cache is populated only by execution, after selection.
            if not (cls is TritonWeightedSumImplementation and name == "_plans")
        },
    ]


def selection_policy(
    implementations, fusion_implementations, lowerings, execution_cost, partial
):
    from .passes.lowering._driver import DEFAULT_CKKS_LOWERINGS

    if execution_cost is not None:
        raise ValueError("custom execution cost")
    if lowerings is not DEFAULT_CKKS_LOWERINGS:
        raise ValueError("custom lowering registry")
    if (
        type(implementations) is not OperationImplementationRegistry
        or implementations._operation_specs is not DEFAULT_OPERATION_SPECS
    ):
        raise ValueError("custom operation registry")
    return [
        [
            _implementation_identity(item)
            for item in implementations.implementations
        ],
        fusion_policy(fusion_implementations),
        bool(partial),
    ]


def fusion_policy(implementations):
    from fhelium.backend.triton import (
        TritonFusionImplementation,
        TritonTensorFusionImplementation,
    )

    if any(
        type(item)
        not in (TritonFusionImplementation, TritonTensorFusionImplementation)
        for item in implementations
    ):
        raise ValueError("custom fusion matcher")
    return [_implementation_identity(item) for item in implementations]


def _integers(attribute):
    if isinstance(attribute, ArrayAttr) and all(
        isinstance(item, IntegerAttr) for item in attribute
    ):
        return tuple(int(item.value.data) for item in attribute)
    return None


def _attribute_identity(attribute: Attribute):
    cls = type(attribute)
    return (cls.__module__, cls.__qualname__, attribute.name)


def _encode_attribute_data(value, memo, definitions):
    if isinstance(value, Attribute):
        return _encode_attribute(value, memo, definitions)
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float64-bits", struct.pack("!d", value).hex())
    if type(value) is bytes:
        return ("bytes", value.hex())
    if type(value) is tuple:
        return (
            "tuple",
            tuple(
                _encode_attribute_data(item, memo, definitions)
                for item in value
            ),
        )
    if isinstance(value, Enum):
        cls = type(value)
        return (
            "enum",
            cls.__module__,
            cls.__qualname__,
            value.name,
            _encode_attribute_data(value.value, memo, definitions),
        )
    raise ValueError("unsupported attribute payload")


def _encode_attribute(
    attribute: Attribute,
    memo: dict[int, tuple[str, int]],
    definitions: dict[tuple, tuple[str, int]],
) -> tuple[str, int]:
    """Intern immutable IR definitions and return compact content references.

    Identity memoization is request-local: Attribute equality can merge signed
    zero or distinct NaN payloads, whereas these guards preserve their bits.
    Unsupported Python payloads leave selection to the uncached pass.
    """
    identity = id(attribute)
    if identity in memo:
        return memo[identity]

    if isinstance(attribute, ParametrizedAttribute):
        encoded = (
            "parameters",
            _attribute_identity(attribute),
            tuple(
                (
                    name,
                    _encode_attribute(
                        getattr(attribute, name), memo, definitions
                    ),
                )
                for name, _ in attribute.get_irdl_definition().parameters
            ),
        )
    elif isinstance(attribute, DictionaryAttr):
        encoded = (
            "dictionary",
            _attribute_identity(attribute),
            tuple(
                (name, _encode_attribute(value, memo, definitions))
                for name, value in sorted(attribute.data.items())
            ),
        )
    elif isinstance(attribute, Data):
        encoded = (
            "data",
            _attribute_identity(attribute),
            _encode_attribute_data(attribute.data, memo, definitions),
        )
    else:
        raise ValueError("unsupported attribute representation")
    reference = definitions.get(encoded)
    if reference is None:
        reference = ("attribute", len(definitions))
        definitions[encoded] = reference
    memo[identity] = reference
    return reference


class GuardMemo:
    """Context-free guard facts shared by the guards of one pass run.

    Attribute digests and type projections depend only on the immutable object and the projection mode, so one pass reuses them across its operations. Row and prime-set equality classes are numbered per guard.
    """

    def __init__(self) -> None:
        self.digests = AttributeDigests(strict=True)
        self.types: dict[tuple[int, bool, bool], tuple[Any, ...]] = {}
        self.texts: dict[int, tuple[object, str]] = {}

    def text(self, value: object, render) -> str:
        found = self.texts.get(id(value))
        if found is None or found[0] is not value:
            found = self.texts[id(value)] = (value, render(value))
        return found[1]

    def type_facts(self, value_type, parameter: bool, polymorphic: bool):
        key = (id(value_type), parameter, polymorphic)
        found = self.types.get(key)
        if found is None or found[0] is not value_type:
            try:
                facts = self._type_facts(value_type, parameter, polymorphic)
            except ValueError as error:
                facts = ("error", str(error))
            found = self.types[key] = (value_type, facts)
        facts = found[1]
        if facts[0] == "error":
            raise ValueError(facts[1])
        return facts

    def _type_facts(self, value_type, parameter, polymorphic):
        from fhelium.backend.triton._expressions import component_axis

        digest = self.digests
        polynomial = isinstance(
            value_type,
            (
                ckks.CiphertextType,
                ckks.PlaintextType,
                ckks.CompressedPlaintextType,
                rns.RnsBundleType,
            ),
        )
        state = getattr(getattr(value_type, "state", None), "data", None)
        if not polymorphic or state is None:
            return ("fixed", digest(value_type))
        # Parameter/table shape and contents do not participate in the RNS
        # matcher. Keep all ordinary Tensor layouts concrete (Tensor matcher).
        if parameter:
            return (
                "fixed",
                [
                    _attribute_identity(value_type),
                    {
                        key: digest(item)
                        for key, item in sorted(state.items())
                        if key == "device"
                    },
                ],
            )
        if not polynomial:
            return ("fixed", digest(value_type))
        ids = _integers(state.get("prime_ids"))
        contiguous_ids = bool(ids) and ids == tuple(
            range(ids[0], ids[0] + len(ids))
        )
        shape = _integers(state.get("shape"))
        if shape is not None and component_axis(value_type) and len(shape) < 3:
            raise ValueError("ambiguous component and prime axes")
        shaped = shape is not None and len(shape) >= 2
        contiguous_storage = shaped and _integers(
            state.get("strides")
        ) == tuple(prod(shape[index + 1 :]) for index in range(len(shape)))
        projected = {"depth", "scale"}
        if contiguous_ids:
            projected.add("prime_ids")
        if shaped:
            projected.add("shape")
        if contiguous_storage:
            projected.add("strides")
        return (
            "projected",
            _attribute_identity(value_type),
            {
                key: digest(item)
                for key, item in sorted(state.items())
                if key not in projected
            },
            ids if contiguous_ids else None,
            shape if shaped else None,
            contiguous_storage,
        )


def graph_guard(
    block: Block,
    *,
    stage: str,
    policy,
    config=None,
    memo: GuardMemo | None = None,
) -> str:
    """Fingerprint order, edges, escapes, effects and matcher-visible facts.

    Selection passes an isolated operation with its operand alias relations
    and all results escaping, matching its lowering probe. Arithmetic/NTT has a
    row-polymorphic contract; other CKKS lowerings retain concrete state because
    their generated topology can depend on depth groups and digit layouts.
    Fusion passes the complete lowered block, including returns and barriers.
    Its polynomial row dimensions are equality classes, tied to the lengths of
    contiguous prime-id sequences; ring and batch extents stay fixed.
    Attributes enter as :class:`AttributeDigests` content digests; ``memo``
    shares them and type projections between the guards of one pass.
    """
    from .passes.lowering._driver import DEFAULT_CKKS_LOWERINGS
    from .passes.program._purity import known_pure
    from fhelium.backend.triton._expressions import (
        ARITHMETIC_OPS,
        NTT_OPS,
        SUPPORTED_OPERATIONS,
    )
    from fhelium.ir.dialects.torch import TensorCallOp

    if memo is None:
        memo = GuardMemo()
    encode = memo.digests

    operations = tuple(block.ops)
    if any(op.regions or op.successors for op in operations):
        raise ValueError("region or control-flow graph")
    safe_lowerings = (
        ckks.AddOp,
        ckks.SubtractOp,
        ckks.NegateOp,
        ckks.MultiplyOp,
        ckks.AddPlaintextOp,
        ckks.MultiplyPlaintextOp,
        ckks.SumBatchOp,
        ckks.ToNttOp,
        ckks.FromNttOp,
    )
    polymorphic = stage == "fusion" or all(
        not DEFAULT_CKKS_LOWERINGS.supports(op) or type(op) in safe_lowerings
        for op in operations
    )
    positions = {op: index for index, op in enumerate(operations)}
    values = {value: ("arg", index) for index, value in enumerate(block.args)}
    values.update(
        {
            value: ("result", index, result)
            for index, op in enumerate(operations)
            for result, value in enumerate(op.results)
        }
    )
    if any(value not in values for op in operations for value in op.operands):
        raise ValueError("external SSA operand")
    if any(
        use.operation not in positions for value in values for use in value.uses
    ):
        raise ValueError("external SSA use")
    symbols: dict[str, int] = {}
    rows: dict[int, int] = {}
    primes: dict[tuple[int, ...], int] = {}

    def row(value):
        return (
            ["rows", rows.setdefault(value, len(rows))] if value > 0 else value
        )

    def opaque(operation):
        # These classes exhaust both builtin matchers and layout's recursive
        # producer cases. Other operations are rejected before their attrs are
        # read; scheduling uses their edges, order, effects and assignment.
        return stage == "fusion" and not isinstance(
            operation,
            SUPPORTED_OPERATIONS
            + NTT_OPS
            + (
                UnrealizedConversionCastOp,
                TensorCallOp,
                core.MaterialRefOp,
                core.ResourceRefOp,
            ),
        )

    opaque_parameter_ops = (
        ckks.RotateOp,
        ckks.RotateManyOp,
        ckks.SwitchKeyOp,
        ckks.RelinearizeOp,
        ckks.RescaleOp,
        rns.RescaleDropLeadingPrimesOp,
    )

    def parameter_only(value):
        uses = tuple(value.uses)
        for use in uses:
            op = use.operation
            if (
                isinstance(op, NTT_OPS + (ckks.ToNttOp, ckks.FromNttOp))
                and use.index >= 1
            ):
                continue
            if (
                isinstance(op, ARITHMETIC_OPS + (rns.KeySwitchDigitProductOp,))
                and use.index == len(op.operands) - 1
            ):
                continue
            if isinstance(op, safe_lowerings) or (
                opaque(op) and isinstance(op, opaque_parameter_ops)
            ):
                parameters = tuple(getattr(op, "parameters", ()))
                if parameters and use.index >= len(op.operands) - len(
                    parameters
                ):
                    continue
            return False
        return bool(uses) or value.owner is block

    type_memo: dict[tuple[int, bool], object] = {}

    def typ(value):
        value_type = value.type
        parameter = (
            value.owner is block
            or isinstance(value.owner, (core.MaterialRefOp, core.ResourceRefOp))
        ) and parameter_only(value)
        key = (id(value_type), parameter)
        if key not in type_memo:
            type_memo[key] = encode_type(value_type, parameter)
        return type_memo[key]

    def encode_type(value_type, parameter):
        facts = memo.type_facts(value_type, parameter, polymorphic)
        if facts[0] == "fixed":
            return facts[1]
        _, identity, fields, ids, shape, contiguous_storage = facts
        result = dict(fields)
        if ids is not None:
            result["prime_ids"] = [
                "contiguous",
                primes.setdefault(ids, len(primes)),
                row(len(ids)),
            ]
        if shape is not None:
            result["shape"] = [*shape[:-2], row(shape[-2]), shape[-1]]
        if contiguous_storage:
            result["strides"] = "contiguous"
        return [identity, result]

    def attrs(op):
        if opaque(op):
            return {
                key: encode(value)
                for key, value in op.attributes.items()
                if key == EXECUTION_IMPLEMENTATION_ATTRIBUTE
            }
        symbolic = (
            isinstance(op, (core.MaterialRefOp, core.ResourceRefOp))
            and op.symbol is not None
        )
        result = {
            key: encode(value)
            for key, value in sorted(op.attributes.items())
            if key != MATERIAL_REQUEST_ATTRIBUTE
            and not (symbolic and key == "symbol")
            # Lowering and fusion decisions do not read the slot displacement;
            # it selects only the Galois tables of the automorphism.
            and not (isinstance(op, ckks.RotateOp) and key == "rotation_step")
        }
        if symbolic:
            symbol = op.symbol.data
            result["symbol"] = [
                "symbol",
                symbols.setdefault(symbol, len(symbols)),
            ]
        return result

    records = []
    for op in operations:
        spec = DEFAULT_OPERATION_SPECS.get(op.name)
        if not isinstance(op, (ReturnOp, UnrealizedConversionCastOp)) and (
            spec is None or spec.operation_type is not type(op)
        ):
            raise ValueError("unknown operation semantics")
        records.append(
            [
                op.name,
                spec.effect if spec is not None else None,
                known_pure(op),
                [values[value] for value in op.operands],
                [typ(value) for value in op.results],
                attrs(op),
                {
                    key: encode(value)
                    for key, value in sorted(op.properties.items())
                },
            ]
        )
    payload = [
        stage,
        memo.text(
            policy,
            lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        ),
        memo.text(config, repr),
        polymorphic,
        [typ(value) for value in block.args],
        records,
    ]
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=24,
    ).hexdigest()


__all__ = ["SelectionPlans"]
