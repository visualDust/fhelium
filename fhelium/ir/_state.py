"""Shared dataflow analysis for partially represented FHElium value state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from ._dialect import value_role
from ._program import Program
from .dialects import ckks
from .dialects._common import OpenStateType

StateStatus = Literal["known", "symbolic", "dynamic", "conflict"]


@dataclass(frozen=True)
class SymbolicExpression:
    """Represent a state equation whose operands are not all compile-time values."""

    operator: str
    operands: tuple[object, ...]


@dataclass(frozen=True)
class StateFact:
    """Record one known, symbolic, dynamic, or conflicting state property."""

    status: StateStatus
    value: object | None = None
    conflicting_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"known", "symbolic", "dynamic", "conflict"}:
            raise ValueError(f"Unsupported state status {self.status!r}")
        if self.status in {"known", "symbolic"} and self.value is None:
            raise ValueError(f"{self.status} state requires a value")
        if self.status == "conflict" and not self.conflicting_values:
            raise ValueError("conflicting state requires conflicting values")

    @classmethod
    def known(cls, value: object) -> StateFact:
        """Construct one represented or inferred known state property."""

        return cls("known", value)

    @classmethod
    def symbolic(cls, operator: str, *operands: object) -> StateFact:
        """Construct one symbolic state equation."""

        return cls("symbolic", SymbolicExpression(operator, tuple(operands)))

    @classmethod
    def dynamic(cls) -> StateFact:
        """Construct one property that requires later specialization."""

        return cls("dynamic")

    @classmethod
    def conflict(cls, *conflicting_values: str) -> StateFact:
        """Construct one property from incompatible represented values."""

        return cls("conflict", conflicting_values=conflicting_values)


_DYNAMIC = StateFact.dynamic()


@dataclass(frozen=True)
class InferredValueState:
    """Describe inferred state for one SSA value without modifying its IR type."""

    value: SSAValue
    fields: Mapping[str, StateFact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def field(self, name: str) -> StateFact:
        """Return one field or a dynamic placeholder when it is not represented."""

        return self.fields.get(name, _DYNAMIC)

    def with_fields(self, **updates: StateFact) -> InferredValueState:
        """Return the same SSA identity with updated analysis fields."""

        updated_fields = dict(self.fields)
        updated_fields.update(updates)
        return InferredValueState(self.value, updated_fields)


_LAYOUT_FIELDS = (
    "key_space",
    "level",
    "prime_ids",
    "basis",
    "polynomial_domain",
    "residue_representation",
    "components",
    "ring_dimension",
    "dtype",
    "device",
    "shape",
    "batch_shape",
    "layout",
    "scale",
)


def _python_value(attribute: Attribute) -> object:
    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, IntegerAttr):
        return int(attribute.value.data)
    if isinstance(attribute, ArrayAttr):
        return tuple(_python_value(item) for item in attribute)
    return attribute


def _represented(value: SSAValue) -> InferredValueState:
    fields: dict[str, StateFact] = {}
    role = value_role(value)
    if role is not None:
        fields["role"] = StateFact.known(role)
    value_type = value.type
    if isinstance(value_type, OpenStateType):
        fields.update(
            (name, StateFact.known(_python_value(attribute)))
            for name, attribute in value_type.state.data.items()
            if name != "role"
        )
    return InferredValueState(value, fields)


def _equal_field(
    lhs: InferredValueState,
    rhs: InferredValueState,
    name: str,
) -> StateFact:
    left = lhs.field(name)
    right = rhs.field(name)
    if left.status == "conflict":
        return left
    if right.status == "conflict":
        return right
    if left.status == "dynamic" or right.status == "dynamic":
        return _DYNAMIC
    if left == right:
        return left
    return StateFact.conflict(
        f"lhs {name}={left.value!r}", f"rhs {name}={right.value!r}"
    )


def _result_state(
    result: SSAValue,
    inferred: Mapping[str, StateFact],
) -> InferredValueState:
    represented = _represented(result)
    fields = dict(inferred)
    for name, fact in represented.fields.items():
        previous = fields.get(name)
        if previous is None or previous.status == "dynamic":
            fields[name] = fact
        elif fact != previous:
            fields[name] = StateFact.conflict(
                f"inferred {name}={previous.value!r}",
                f"represented {name}={fact.value!r}",
            )
    return InferredValueState(result, fields)


def _unary_state(
    operation: Operation,
    states: Mapping[SSAValue, InferredValueState],
    **updates: StateFact,
) -> InferredValueState:
    source = states[operation.operands[0]]
    fields = dict(source.fields)
    fields.update(updates)
    return _result_state(operation.results[0], fields)


def _binary_state(
    operation: Operation,
    states: Mapping[SSAValue, InferredValueState],
    **updates: StateFact,
) -> InferredValueState:
    lhs = states[operation.operands[0]]
    rhs = states[operation.operands[1]]
    fields = {
        name: _equal_field(lhs, rhs, name)
        for name in _LAYOUT_FIELDS
        if name != "scale"
    }
    fields.update(updates)
    return _result_state(operation.results[0], fields)


def analyze_state_flow(
    program: Program,
    *,
    function: str = "main",
) -> Mapping[SSAValue, InferredValueState]:
    """Infer shared value-state equations through one single-block function.

    Missing properties remain dynamic and incompatible represented properties
    become conflicts. The analysis does not reject or rewrite the Program.
    """

    block = program.single_block(function)
    states: dict[SSAValue, InferredValueState] = {
        argument: _represented(argument) for argument in block.args
    }
    for operation in block.ops:
        if not operation.results:
            continue
        inferred: tuple[InferredValueState, ...] | None = None
        if isinstance(operation, ckks.ToNttOp):
            inferred = (
                _unary_state(
                    operation,
                    states,
                    polynomial_domain=StateFact.known("ntt"),
                    residue_representation=StateFact.known("montgomery"),
                ),
            )
        elif isinstance(operation, ckks.FromNttOp):
            inferred = (
                _unary_state(
                    operation,
                    states,
                    polynomial_domain=StateFact.known("coefficient"),
                    residue_representation=StateFact.known("standard"),
                ),
            )
        elif isinstance(
            operation,
            (ckks.AddScalarOp, ckks.MultiplyIntegerScalarOp),
        ):
            inferred = (_unary_state(operation, states),)
        elif isinstance(operation, ckks.MultiplyScalarOp):
            source = states[operation.ciphertext]
            scalar_scale = operation.scalar_scale
            assert isinstance(scalar_scale, FloatAttr)
            inferred = (
                _unary_state(
                    operation,
                    states,
                    scale=StateFact.symbolic(
                        "multiply",
                        source.field("scale"),
                        StateFact.known(float(scalar_scale.value.data)),
                    ),
                ),
            )
        elif isinstance(operation, ckks.MultiplyOp):
            lhs = states[operation.lhs]
            rhs = states[operation.rhs]
            inferred = (
                _binary_state(
                    operation,
                    states,
                    components=StateFact.known(3),
                    scale=StateFact.symbolic(
                        "multiply", lhs.field("scale"), rhs.field("scale")
                    ),
                ),
            )
        elif isinstance(operation, ckks.RelinearizeOp):
            inferred = (
                _unary_state(
                    operation,
                    states,
                    components=StateFact.known(2),
                    basis=StateFact.known("Q"),
                    polynomial_domain=StateFact.known("coefficient"),
                    residue_representation=StateFact.known("standard"),
                ),
            )
        elif isinstance(
            operation,
            (ckks.SwitchKeyOp, ckks.RotateOp, ckks.ConjugateOp),
        ):
            inferred = (_unary_state(operation, states),)
        elif isinstance(operation, ckks.RotateManyOp):
            source = states[operation.value]
            inferred = tuple(
                _result_state(result, source.fields)
                for result in operation.outputs
            )
        elif isinstance(operation, ckks.RescaleOp):
            source = states[operation.value]
            level = source.field("level")
            prime_ids = source.field("prime_ids")
            next_level = (
                StateFact.known(int(level.value) + 1)
                if level.status == "known" and isinstance(level.value, int)
                else StateFact.symbolic("increment", level)
            )
            next_primes = (
                StateFact.known(tuple(prime_ids.value)[1:])
                if prime_ids.status == "known"
                and isinstance(prime_ids.value, tuple)
                else StateFact.symbolic("drop_leading_prime", prime_ids)
            )
            inferred = (
                _unary_state(
                    operation,
                    states,
                    level=next_level,
                    prime_ids=next_primes,
                    scale=StateFact.symbolic(
                        "divide_by_dropped_prime",
                        source.field("scale"),
                        source.field("level"),
                    ),
                ),
            )
        if inferred is None:
            inferred = tuple(
                _represented(result) for result in operation.results
            )
        for result, state in zip(operation.results, inferred, strict=True):
            states[result] = state
    return MappingProxyType(states)


__all__ = [
    "InferredValueState",
    "StateFact",
    "StateStatus",
    "SymbolicExpression",
    "analyze_state_flow",
]
