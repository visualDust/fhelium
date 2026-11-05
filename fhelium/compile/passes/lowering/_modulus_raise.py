"""Lower modulus raising through common key-product and RNS operations."""

from __future__ import annotations

import json
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from fhelium.ir.dialects import ckks, ntt, rns
from fhelium.backend.ckks.tables import encapsulation_configs
from ._core import CkksLoweringDefinition, LoweredCkksOperation
from ._keyswitch import _KeySwitchLowering
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _component_bundle_type,
    _polynomial_type,
    _rns_type,
)


def lower_modulus_raise(operation, config, compilation):
    if config is None or not operation.parameters:
        raise ValueError(
            "Modulus raising requires configuration and numerical operands"
        )
    tensors = dict(
        zip(
            (x.data for x in operation.attributes["parameter_names"]),
            operation.parameters,
            strict=True,
        )
    )
    operations = []
    descriptions = {}

    def emit(op):
        operations.append(op)
        return op.results[0]

    def extension(source, result_type, prefix=""):
        names = (
            tuple(name for name in tensors if name.startswith(prefix))
            if prefix
            else tuple(tensors)
        )
        return emit(
            rns.CenteredBasisExtendOp(
                source,
                tuple(tensors[name] for name in names),
                result_type,
                attributes={
                    "parameter_names": ArrayAttr(
                        StringAttr(name[len(prefix) :]) for name in names
                    )
                },
            )
        )

    if isinstance(operation, ckks.ModulusRaiseOp):
        cast_op, value = _cast_to_rns(operation.value)
        operations.append(cast_op)
        result = extension(value, _rns_type(operation.result.type))
    else:
        depth = int(operation.value.type.state.data["depth"].value.data)
        low_config, return_config = encapsulation_configs(
            config,
            depth,
            tuple(int(x.value.data) for x in operation.low_p_moduli),
            int(operation.return_prime.value.data),
            tuple(int(x.value.data) for x in operation.return_p_moduli),
        )
        low_type = operation.value.type.with_state(
            {
                "depth": IntegerAttr(0, 64),
                "prime_ids": ArrayAttr(
                    IntegerAttr(i, 64) for i in range(low_config.num_q_primes)
                ),
            }
        )
        # Reinterpret the same payload in the low context's Q-row coordinates.
        low_cast, low_value = _cast_to_ckks(operation.value, low_type)
        operations.append(low_cast)
        names = tuple(name for name in tensors if name.startswith("low_"))
        fields = {
            name[4:]: value
            for name, value in operation.attributes.items()
            if name.startswith("low_") and name != "low_p_moduli"
        }
        fields.update(
            parameter_names=ArrayAttr(StringAttr(name[4:]) for name in names),
            ckks_config=StringAttr(json.dumps(low_config.dumps())),
            moddown_rounding=operation.moddown_rounding,
        )
        descriptor = ckks.SwitchKeyOp(
            low_value,
            operation.low_key,
            low_type,
            parameters=tuple(tensors[name] for name in names),
            attributes=fields,
        )
        low = _KeySwitchLowering(descriptor, low_config, compilation).lower()
        operations.extend(low.operations)
        descriptions.update(low.material_descriptions)
        sparse_cast, sparse = _cast_to_rns(low.result)
        operations.append(sparse_cast)
        descriptor.drop_all_references()

        high_ids = ArrayAttr(
            IntegerAttr(i, 64) for i in range(return_config.total_num_primes)
        )
        q_ids = ArrayAttr(
            IntegerAttr(i, 64) for i in range(config.num_q_primes)
        )

        def typ(*, components=None, high=False, ntt_domain=False, target=False):
            fields = {
                "basis": StringAttr("QP" if high else "Q"),
                "prime_ids": high_ids
                if high
                else operation.result.type.state.data["prime_ids"]
                if target
                else q_ids,
                "depth": operation.result.type.state.data.get(
                    "depth", IntegerAttr(0, 64)
                )
                if target
                else IntegerAttr(0, 64),
                "polynomial_domain": StringAttr(
                    "ntt" if ntt_domain else "coefficient"
                ),
                "residue_representation": StringAttr(
                    "montgomery" if ntt_domain else "standard"
                ),
            }
            return (
                _polynomial_type(operation.value.type, **fields)
                if components is None
                else _component_bundle_type(
                    operation.value.type, components, **fields
                )
            )

        raised = extension(sparse, typ(high=True, components=2), "raise_")
        c0 = emit(rns.ExtractComponentOp(raised, typ(high=True), component=0))
        c1 = emit(rns.ExtractComponentOp(raised, typ(high=True), component=1))

        def transform(value, *, inverse, components=None):
            prefix = "return_inverse_" if inverse else "return_forward_"
            tables = tuple(
                tensors[f"{prefix}{i}"]
                for i in range(sum(name.startswith(prefix) for name in tensors))
            )
            cls = (
                ntt.NttMontgomeryToCoefficientStandardOp
                if inverse
                else ntt.CoefficientStandardToNttMontgomeryOp
            )
            return emit(
                cls(
                    value,
                    tables[0],
                    typ(
                        high=True, ntt_domain=not inverse, components=components
                    ),
                    tables=tables[1:],
                    attributes={
                        "ntt_backend": operation.attributes["ntt_backend"],
                        "ckks_config": StringAttr(
                            json.dumps(return_config.dumps())
                        ),
                    },
                )
            )

        c1 = transform(c1, inverse=False)
        products = emit(
            rns.KeySwitchDigitProductOp(
                c1,
                operation.return_key,
                tensors["return_parameters"],
                typ(high=True, ntt_domain=True, components=2),
                key_digit_index=0,
                key_row_start=0,
            )
        )
        products = transform(products, inverse=True, components=2)
        quotient_names = tuple(
            name
            for name in (
                "quotient_parameters",
                "quotient_inverses",
                "quotient_order",
                "quotient_half",
            )
            if name in tensors
        )
        aliases = {
            "quotient_parameters": "parameters",
            "quotient_inverses": "moddown_inverses",
            "quotient_order": "row_order",
            "quotient_half": "half",
        }
        switched = emit(
            rns.RoundedTrailingQuotientOp(
                products,
                tuple(tensors[name] for name in quotient_names),
                typ(components=2),
                drop_count=int(
                    operation.attributes["return_drop_count"].value.data
                ),
                rounding=operation.moddown_rounding.data,
                attributes={
                    "parameter_names": ArrayAttr(
                        StringAttr(aliases[name]) for name in quotient_names
                    )
                },
            )
        )
        start = int(operation.attributes["target_row_start"].value.data)
        count = config.num_q_primes - start
        c0 = emit(
            rns.SliceRowsOp(c0, typ(target=True), start=start + 1, count=count)
        )
        switched = emit(
            rns.SliceRowsOp(
                switched,
                typ(target=True, components=2),
                start=start,
                count=count,
            )
        )
        correction0 = emit(
            rns.ExtractComponentOp(switched, typ(target=True), component=0)
        )
        correction1 = emit(
            rns.ExtractComponentOp(switched, typ(target=True), component=1)
        )
        result0 = emit(
            rns.AddStandardOp(
                c0, correction0, tensors["target_parameters"], typ(target=True)
            )
        )
        result = emit(
            rns.PackTwoComponentsOp(
                result0, correction1, typ(target=True, components=2)
            )
        )
    cast_op, result = _cast_to_ckks(result, operation.result.type)
    operations.append(cast_op)
    return LoweredCkksOperation(tuple(operations), result, descriptions)


MODULUS_RAISE_LOWERINGS = (
    CkksLoweringDefinition(
        "rns-centered-modulus-raise",
        ckks.ModulusRaiseOp,
        lower_modulus_raise,
        is_default=True,
    ),
    CkksLoweringDefinition(
        "rns-ntt-encapsulated-modulus-raise",
        ckks.EncapsulatedModulusRaiseOp,
        lower_modulus_raise,
        is_default=True,
    ),
)
