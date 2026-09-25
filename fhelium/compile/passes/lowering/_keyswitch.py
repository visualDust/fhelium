"""Expand CKKS key switching into operations over supplied Tensor operands."""

from __future__ import annotations

from typing import cast
import json

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.config import CkksConfig
from fhelium.ir.dialects import ckks, core, ntt, rns
from fhelium.backend.rns._preparation import (
    hybrid_decomposition_facts,
    rns_names,
    table_description,
)

from ._core import CkksLoweringDefinition, LoweredCkksOperation
from ..._materials import (
    declare_material,
    rns_parameter_identity,
    parameter_description,
)
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _component_bundle_type,
    _polynomial_type,
    _rns_type,
)


def _lower_key_operation(
    operation: Operation, config: CkksConfig | None, compilation
) -> LoweredCkksOperation:
    if not isinstance(
        operation,
        (ckks.RelinearizeOp, ckks.SwitchKeyOp, ckks.RotateOp, ckks.ConjugateOp),
    ):
        raise TypeError("Key-switch lowering received another operation")
    if config is None:
        raise ValueError(
            "Key-switch lowering requires CKKS decomposition parameters"
        )
    key = operation.key
    fields = dict(operation.attributes)
    references: list[Operation] = []
    descriptions: dict[str, dict[str, object]] = {}
    if not operation.parameters:
        state = cast(ckks.CiphertextType, operation.value.type).state.data
        represented_depth = state.get("depth")
        if not isinstance(represented_depth, IntegerAttr):
            raise ValueError("Key-switch lowering requires represented depth")
        depth = int(represented_depth.value.data)
        facts = hybrid_decomposition_facts(config, depth)
        digits = cast(tuple[tuple[int, int, int], ...], facts["digits"])
        names = rns_names(digits)
        physical = {
            name: state[name] for name in ("dtype", "device") if name in state
        }
        tensors: dict[str, SSAValue] = {}
        for name in names:
            description = table_description(config, depth, name, facts)
            kind = cast(str, description["kind"])
            typ = (
                rns.RnsParametersType().with_state(physical)
                if kind == "rns_parameters"
                else core.MessageType().with_state(physical)
            )
            declared = parameter_description(
                config,
                kind,
                **{k: v for k, v in description.items() if k != "kind"},
            )
            identity = (
                rns_parameter_identity(
                    config, description.get("prime_ids"), physical
                )
                if kind == "rns_parameters"
                else None
            )
            created, value = declare_material(
                compilation,
                operation,
                f"keyswitch/{depth}/{name}",
                typ,
                declared,
                identity=identity,
            )
            references.extend(created)
            for reference in created:
                descriptions[cast(StringAttr, reference.symbol).data] = declared
            tensors[name] = value
        fields.update(
            parameter_names=ArrayAttr(StringAttr(name) for name in names),
            digits=ArrayAttr(
                ArrayAttr(IntegerAttr(item, 64) for item in span)
                for span in digits
            ),
            **{
                name: IntegerAttr(cast(int, facts[name]), 64)
                for name in ("q_count", "p_count", "key_row_start")
            },
        )
    else:
        names = fields.get("parameter_names")
        spans = fields.get("digits")
        if not isinstance(names, ArrayAttr) or not isinstance(spans, ArrayAttr):
            raise ValueError(
                f"{operation.name} supplied Tensor operands need their names and digit ranges"
            )
        tensors = dict(
            zip(
                (cast(StringAttr, name).data for name in names),
                operation.parameters,
                strict=True,
            )
        )
        digits = tuple(
            tuple(
                int(cast(IntegerAttr, item).value.data)
                for item in cast(ArrayAttr, span)
            )
            for span in spans
        )
    facts = {
        name: value
        for name, value in fields.items()
        if name
        in {
            "parameter_names",
            "digits",
            "p_count",
            "q_count",
            "key_row_start",
            "ntt_backend",
        }
    }
    row_start = cast(IntegerAttr, fields["key_row_start"]).value.data
    input_cast, value = _cast_to_rns(operation.value)
    operations: list[Operation] = [*references, input_cast]
    source_type = cast(ckks.CiphertextType, operation.value.type)
    q_ids = source_type.state.data.get("prime_ids")
    qp_ids = ArrayAttr(
        IntegerAttr(i, 64)
        for i in (
            *[
                int(cast(IntegerAttr, item).value.data)
                for item in cast(ArrayAttr, q_ids)
            ],
            *range(config.num_q_primes, config.total_num_primes),
        )
    )

    def emit(op: Operation) -> SSAValue:
        operations.append(op)
        return op.results[0]

    def polynomial(
        *,
        output_ntt: bool = False,
        qp: bool = False,
        components: int | None = None,
    ) -> Attribute:
        updates: dict[str, Attribute] = {
            "polynomial_domain": StringAttr(
                "ntt" if output_ntt else "coefficient"
            ),
            "residue_representation": StringAttr(
                "montgomery" if output_ntt else "standard"
            ),
            "basis": StringAttr("QP" if qp else "Q"),
            "prime_ids": qp_ids if qp else cast(Attribute, q_ids),
        }
        return (
            _polynomial_type(source_type, **updates)
            if components is None
            else _component_bundle_type(source_type, components, **updates)
        )

    ntt_attributes = {
        name: value
        for name, value in fields.items()
        if name
        in {
            "ntt_backend",
            "ntt_algorithm",
            "ntt_group_width",
            "ntt_radix",
            "ntt_table_layout",
        }
        and value != StringAttr("unknown")
    }
    ntt_attributes["ckks_config"] = StringAttr(json.dumps(config.dumps()))

    def transform(
        value: SSAValue, *, inverse: bool, basis: str, typ: Attribute
    ) -> SSAValue:
        prefix = f"{basis}_{'inverse' if inverse else 'forward'}_"
        supplied = tuple(
            tensors[f"{prefix}{i}"]
            for i in range(sum(name.startswith(prefix) for name in tensors))
        )
        source_state = cast(rns.RnsBundleType, value.type).state.data
        cls = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if inverse
            else (
                ntt.CoefficientMontgomeryToNttMontgomeryOp
                if source_state.get("residue_representation")
                == StringAttr("montgomery")
                else ntt.CoefficientStandardToNttMontgomeryOp
            )
        )
        return emit(
            cls(
                value,
                supplied[0] if supplied else tensors[f"{basis}_parameters"],
                typ,
                tables=supplied[1:],
                attributes=ntt_attributes,
            )
        )

    def local_tables(names):
        return tuple(tensors[name] for name in names), {
            **facts,
            "parameter_names": ArrayAttr(StringAttr(name) for name in names),
        }

    def extract(
        value: SSAValue, component: int, *, output_ntt: bool = False
    ) -> SSAValue:
        return emit(
            rns.ExtractComponentOp(
                value, polynomial(output_ntt=output_ntt), component=component
            )
        )

    def corrections(source: SSAValue, *, output_ntt: bool) -> SSAValue:
        accumulator: SSAValue | None = None
        for index, span in enumerate(digits):
            coefficient_type = cast(
                rns.RnsBundleType, polynomial(qp=True)
            ).with_state({"residue_representation": StringAttr("montgomery")})
            names = tuple(
                name
                for name in tensors
                if name == "qp_parameters" or name.startswith(f"digit{index}_")
            )
            operands, local_facts = local_tables(names)
            lifted = emit(
                rns.HybridModUpDigitOp(
                    source,
                    operands,
                    coefficient_type,
                    digit_index=index,
                    attributes=local_facts,
                )
            )
            transformed = transform(
                lifted,
                inverse=False,
                basis="qp",
                typ=polynomial(qp=True, output_ntt=True),
            )
            product_operation = rns.KeySwitchDigitProductOp(
                transformed,
                key,
                tensors["qp_parameters"],
                polynomial(qp=True, output_ntt=True, components=2),
                key_digit_index=span[2],
                key_row_start=int(row_start),
            )
            product_operation.attributes["key_role"] = StringAttr(
                {
                    ckks.RelinearizeOp: "relinearization",
                    ckks.ConjugateOp: "conjugation",
                    ckks.RotateOp: "rotation",
                    ckks.SwitchKeyOp: "switch",
                }[type(operation)]
            )
            product = emit(product_operation)
            accumulator = (
                product
                if accumulator is None
                else emit(
                    rns.AddMontgomeryLazyOp(
                        accumulator,
                        product,
                        tensors["qp_parameters"],
                        polynomial(qp=True, output_ntt=True, components=2),
                    )
                )
            )
        if accumulator is None:
            raise ValueError("Key switching requires at least one digit")
        names = tuple(
            name
            for name in tensors
            if name
            in {
                "q_parameters",
                "qp_parameters",
                "moddown_inverses",
                "p_inverse",
            }
            or name.startswith(("p_inverse_", "q_forward_"))
        )
        operands, local_facts = local_tables(names)
        if output_ntt:
            return emit(
                rns.ModDownNttQpToQOp(
                    accumulator,
                    operands,
                    polynomial(output_ntt=True, components=2),
                    attributes=local_facts,
                )
            )
        accumulator = transform(
            accumulator,
            inverse=True,
            basis="qp",
            typ=polynomial(qp=True, components=2),
        )
        operands, local_facts = local_tables(
            ("qp_parameters", "moddown_inverses")
        )
        return emit(
            rns.ModDownQpToQOp(
                accumulator,
                operands,
                polynomial(components=2),
                attributes=local_facts,
            )
        )

    requested_ntt = operation.output_domain.data == "ntt"
    if isinstance(operation, ckks.RelinearizeOp):
        if requested_ntt:
            d0, d1, d2 = (extract(value, i, output_ntt=True) for i in range(3))
            d2 = transform(d2, inverse=True, basis="q", typ=polynomial())
        else:
            coefficient = transform(
                value, inverse=True, basis="q", typ=polynomial(components=3)
            )
            d0, d1, d2 = (extract(coefficient, i) for i in range(3))
        correction = corrections(d2, output_ntt=requested_ntt)
        c0, c1 = (
            extract(correction, i, output_ntt=requested_ntt) for i in range(2)
        )
        add = rns.AddMontgomeryLazyOp if requested_ntt else rns.AddStandardOp
        result0 = emit(
            add(
                d0,
                c0,
                tensors["q_parameters"],
                polynomial(output_ntt=requested_ntt),
            )
        )
        result1 = emit(
            add(
                d1,
                c1,
                tensors["q_parameters"],
                polynomial(output_ntt=requested_ntt),
            )
        )
    else:
        if isinstance(operation, ckks.RotateOp):
            if operation.input_domain.data != "coefficient":
                raise ValueError(
                    "Logical rotation lowering requires coefficient input; preserve the whole operation for NTT input"
                )
            step = cast(
                IntegerAttr,
                cast(ckks.EvaluationKeyType, key.type).state.data[
                    "rotation_step"
                ],
            ).value.data
            exponent = -int(step) if config.galois_generator == 5 else int(step)
            value = emit(
                rns.CoefficientAutomorphismOp(
                    value,
                    tensors["q_parameters"],
                    polynomial(components=2),
                    galois_element=pow(
                        config.galois_generator,
                        exponent % config.N,
                        2 * config.N,
                    ),
                )
            )
        elif isinstance(operation, ckks.ConjugateOp):
            value = emit(
                rns.CoefficientAutomorphismOp(
                    value,
                    tensors["q_parameters"],
                    polynomial(components=2),
                    galois_element=2 * config.N - 1,
                )
            )
        d0, d1 = (extract(value, i) for i in range(2))
        # The represented rotation route completes coefficient-domain switching
        # before converting its result, retaining its original rounding schedule.
        retained_ntt = requested_ntt and not isinstance(
            operation, ckks.RotateOp
        )
        correction = corrections(d1, output_ntt=retained_ntt)
        c0, result1 = (
            extract(correction, i, output_ntt=retained_ntt) for i in range(2)
        )
        if retained_ntt:
            d0 = transform(
                d0, inverse=False, basis="q", typ=polynomial(output_ntt=True)
            )
        add = rns.AddMontgomeryLazyOp if retained_ntt else rns.AddStandardOp
        result0 = emit(
            add(
                d0,
                c0,
                tensors["q_parameters"],
                polynomial(output_ntt=retained_ntt),
            )
        )
    packed_type = _rns_type(operation.result.type)
    if isinstance(operation, ckks.RotateOp) and requested_ntt:
        packed = emit(
            rns.PackTwoComponentsOp(result0, result1, polynomial(components=2))
        )
        packed = transform(packed, inverse=False, basis="q", typ=packed_type)
    else:
        packed = emit(rns.PackTwoComponentsOp(result0, result1, packed_type))
    result_cast, result = _cast_to_ckks(packed, operation.result.type)
    operations.append(result_cast)
    return LoweredCkksOperation(tuple(operations), result, descriptions)


KEY_SWITCH_LOWERINGS = tuple(
    CkksLoweringDefinition(
        name, operation, _lower_key_operation, is_default=True
    )
    for name, operation in (
        ("rns-ntt-relinearize", ckks.RelinearizeOp),
        ("rns-ntt-switch-key", ckks.SwitchKeyOp),
        ("rns-ntt-rotate", ckks.RotateOp),
        ("rns-ntt-conjugate", ckks.ConjugateOp),
    )
)
