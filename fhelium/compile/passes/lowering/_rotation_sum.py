"""Expose aggregate key products to ordinary RNS/NTT scheduling and fusion."""

import json
from fhelium.ir.dialects.torch import TensorCallOp
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from fhelium.ir.dialects import ckks, ntt, rns
from ._core import CkksLoweringDefinition, LoweredCkksOperation
from ._keyswitch import _lower_rotation_products
from ._types import (
    _cast_to_ckks,
    _cast_to_rns,
    _component_bundle_type,
    _polynomial_type,
    _rns_type,
    _key_switch_qp_basis,
)


def lower_rotation_sum(operation, config, compilation):
    if config is None or not operation.parameters:
        raise ValueError(
            'Aggregate RNS lowering needs parameters and supplied numerical operands'
        )
    fields = dict(operation.attributes)
    tensors = dict(
        zip(
            (x.data for x in fields['parameter_names']),
            operation.parameters,
            strict=True,
        )
    )
    steps = tuple(int(x.value.data) for x in operation.rotation_steps)
    keys = iter(operation.keys)
    source_type = operation.values[0].type
    qp_ids, qp_rows = _key_switch_qp_basis(
        source_type, tuple(operation.keys), fields
    )
    operations = []
    descriptions = {}

    def emit(op):
        operations.append(op)
        return op.results[0]

    def typ(*, qp=False, ntt_domain=False, components=None):
        state = {
            'basis': StringAttr('QP' if qp else 'Q'),
            'prime_ids': qp_ids
            if qp
            else source_type.state.data.get('prime_ids'),
            'polynomial_domain': StringAttr(
                'ntt' if ntt_domain else 'coefficient'
            ),
            'residue_representation': StringAttr(
                'montgomery' if ntt_domain else 'standard'
            ),
        }
        if qp and qp_ids is None and qp_rows is None:
            state["shape"] = None
        row_count = qp_rows if qp else None
        return (
            _polynomial_type(source_type, row_count=row_count, **state)
            if components is None
            else _component_bundle_type(
                source_type, components, row_count=row_count, **state
            )
        )

    ntt_attrs = {
        k: v
        for k, v in fields.items()
        if k
        in {
            'ckks_config',
            'ntt_backend',
            'ntt_algorithm',
            'ntt_group_width',
            'ntt_radix',
            'ntt_table_layout',
        }
        and v != StringAttr('unknown')
    }

    def transform(value, *, inverse, basis, out_type):
        prefix = f"{basis}_{'inverse' if inverse else 'forward'}_"
        supplied = tuple(
            tensors[prefix + str(i)]
            for i in range(sum(k.startswith(prefix) for k in tensors))
        )
        cls = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if inverse
            else ntt.CoefficientStandardToNttMontgomeryOp
        )
        return emit(
            cls(
                value,
                supplied[0] if supplied else tensors[basis + '_parameters'],
                out_type,
                tables=supplied[1:],
                attributes=ntt_attrs,
            )
        )

    input_ntt = operation.input_domain.data == 'ntt'

    def add(lhs, rhs, *, qp=False):
        if lhs is None:
            return rhs
        cls = rns.AddMontgomeryLazyOp if qp or input_ntt else rns.AddStandardOp
        return emit(
            cls(
                lhs,
                rhs,
                tensors['qp_parameters' if qp else 'q_parameters'],
                typ(qp=True, ntt_domain=True, components=2)
                if qp
                else typ(ntt_domain=input_ntt),
            )
        )

    ordinary0 = ordinary1 = products = None
    names = tuple(k for k in tensors if not k.startswith('sum_'))
    key_fields = {
        k: v
        for k, v in fields.items()
        if k
        not in {
            'operandSegmentSizes',
            'source_count',
            'rotation_steps',
            'rescale',
            'input_domain',
            'output_domain',
            'parameter_names',
        }
        and not k.startswith('sum_')
    }
    key_fields['parameter_names'] = ArrayAttr(StringAttr(k) for k in names)
    for source, step in zip(operation.values, steps, strict=True):
        if step:
            key = next(keys)
            rotation_names = (
                (*names, 'rotation_indices') if input_ntt else names
            )
            rotation_parameters = tuple(tensors[k] for k in names)
            if input_ntt:
                rotation_parameters += (tensors[f'sum_galois_{step}'],)
            rotate = ckks.RotateOp(
                source,
                key,
                source.type,
                parameters=rotation_parameters,
                attributes={
                    **key_fields,
                    'parameter_names': ArrayAttr(
                        StringAttr(k) for k in rotation_names
                    ),
                },
            )
            ops, c0, extended, desc = _lower_rotation_products(
                rotate, config, compilation
            )
            operations.extend(ops)
            descriptions.update(desc)
            ordinary0 = add(ordinary0, c0)
            products = add(products, extended, qp=True)
            # The unattached descriptor's uses must not survive in the Program.
            rotate.drop_all_references()
        else:
            cast_op, value = _cast_to_rns(source)
            operations.append(cast_op)
            c0 = emit(
                rns.ExtractComponentOp(
                    value, typ(ntt_domain=input_ntt), component=0
                )
            )
            c1 = emit(
                rns.ExtractComponentOp(
                    value, typ(ntt_domain=input_ntt), component=1
                )
            )
            ordinary0 = add(ordinary0, c0)
            ordinary1 = add(ordinary1, c1)
    if ordinary0 is None:
        raise ValueError('Rotation sum requires at least one contribution')
    if ordinary1 is None:
        ordinary1 = emit(
            TensorCallOp(
                (ordinary0,),
                typ(ntt_domain=input_ntt),
                kind='function',
                target='torch.zeros_like',
                argument_descriptor=json.dumps(
                    {
                        'args': {
                            'kind': 'tuple',
                            'items': [{'kind': 'ssa', 'operand': 0}],
                        },
                        'kwargs': {'kind': 'mapping', 'entries': []},
                    }
                ),
            )
        )
    ordinary = emit(
        rns.PackTwoComponentsOp(
            ordinary0, ordinary1, typ(ntt_domain=input_ntt, components=2)
        )
    )
    if not input_ntt:
        ordinary = transform(
            ordinary,
            inverse=False,
            basis='q',
            out_type=typ(ntt_domain=True, components=2),
        )
    ordinary = emit(
        rns.ScaleAndPadRowsOp(
            ordinary,
            tensors['sum_p_montgomery'],
            tensors['q_parameters'],
            typ(qp=True, ntt_domain=True, components=2),
            pad_rows=int(fields['p_count'].value.data),
        )
    )
    total = add(products, ordinary, qp=True)
    total = transform(
        total, inverse=True, basis='qp', out_type=typ(qp=True, components=2)
    )
    names = tuple(
        k
        for k in (
            'sum_quotient_parameters',
            'sum_quotient_inverses',
            'sum_quotient_order',
            'sum_quotient_half',
        )
        if k in tensors
    )
    result = emit(
        rns.RoundedTrailingQuotientOp(
            total,
            tuple(tensors[k] for k in names),
            _rns_type(operation.result.type),
            drop_count=int(fields['p_count'].value.data)
            + int(fields['sum_dropped_q_count'].value.data),
            rounding=operation.moddown_rounding.data,
            attributes={
                'parameter_names': ArrayAttr(
                    StringAttr(
                        {
                            'sum_quotient_parameters': 'parameters',
                            'sum_quotient_inverses': 'moddown_inverses',
                            'sum_quotient_order': 'row_order',
                            'sum_quotient_half': 'half',
                        }[k]
                    )
                    for k in names
                )
            },
        )
    )
    cast_op, result = _cast_to_ckks(result, operation.result.type)
    operations.append(cast_op)
    return LoweredCkksOperation(tuple(operations), result, descriptions)


ROTATION_SUM_LOWERINGS = (
    CkksLoweringDefinition(
        'rns-ntt-rotation-sum',
        ckks.RotationSumOp,
        lower_rotation_sum,
        is_default=True,
    ),
)
