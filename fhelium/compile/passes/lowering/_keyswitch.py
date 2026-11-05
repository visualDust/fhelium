"""Expand CKKS key switching into operations over supplied Tensor operands."""

from __future__ import annotations

from typing import Literal, cast
import json

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.config import CkksConfig
from fhelium.backend.ckks._key_switch_preparation import operation_key_facts
from fhelium.ir.dialects import ckks, core, ntt, rns
from fhelium.backend.rns.crt import CRT_OPERANDS
from fhelium.backend.rns._preparation import (
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
    _key_switch_qp_basis,
)


class _KeySwitchLowering:
    """Build shared key products and caller-composed quotient operations."""

    def __init__(
        self, operation: Operation, config: CkksConfig | None, compilation
    ):
        if not isinstance(
            operation,
            (
                ckks.RelinearizeOp,
                ckks.SwitchKeyOp,
                ckks.RotateOp,
                ckks.ConjugateOp,
            ),
        ):
            raise TypeError('Key-switch lowering received another operation')
        if config is None:
            raise ValueError(
                'Key-switch lowering requires CKKS decomposition parameters'
            )
        key = operation.key
        fields = dict(operation.attributes)
        rounding = operation.moddown_rounding.data
        method = operation.modup_method.data
        if method not in {"exact", "crt", "normalized_crt"}:
            raise ValueError(
                "modup_method must be exact, crt, or normalized_crt"
            )
        source_type = cast(ckks.CiphertextType, operation.value.type)
        state = source_type.state.data
        q_ids = state.get("prime_ids")
        qp_ids: ArrayAttr | None = None
        qp_rows: IntegerAttr | None = None
        compound = isinstance(operation, ckks.RelinearizeOp) and bool(
            operation.rescale.value.data
        )
        references: list[Operation] = []
        descriptions: dict[str, dict[str, object]] = {}
        if not operation.parameters:
            depth_value = state.get("depth")
            if not isinstance(depth_value, IntegerAttr):
                raise ValueError(
                    "Key-switch lowering requires represented depth"
                )
            layout_facts = operation_key_facts(
                operation, config, int(depth_value.value.data)
            )
            if layout_facts is None:
                raise ValueError(
                    "Key-switch lowering requires represented key rows and digit supports"
                )
            qp_ids = ArrayAttr(
                IntegerAttr(i, 64)
                for i in cast(tuple[int, ...], layout_facts["qp_prime_ids"])
            )
            depth = int(depth_value.value.data)
            facts = dict(layout_facts)
            digits = cast(tuple[tuple[int, int, int], ...], facts['digits'])
            facts['modup_method'] = method
            facts['moddown_rounding'] = rounding
            if compound:
                group_start = sum(map(len, config.q_depth_groups[:depth]))
                drop_ids = tuple(
                    range(
                        group_start,
                        group_start + len(config.q_depth_groups[depth]),
                    )
                )
                facts.update(
                    rescale=True,
                    joint_drop_prime_ids=drop_ids,
                    joint_drop_count=len(drop_ids),
                    joint_drop_is_prefix=drop_ids
                    == cast(tuple[int, ...], facts["q_prime_ids"])[
                        : len(drop_ids)
                    ],
                    rescale_rounding=cast(
                        ckks.RelinearizeOp, operation
                    ).rescale_rounding.data,
                    output_domain=operation.output_domain.data,
                )
                fields.update(
                    joint_drop_count=IntegerAttr(len(drop_ids), 64),
                    joint_drop_is_prefix=IntegerAttr(
                        int(cast(bool, facts["joint_drop_is_prefix"])), 64
                    ),
                    joint_drop_prime_ids=ArrayAttr(
                        IntegerAttr(i, 64) for i in drop_ids
                    ),
                )
            names = rns_names(
                digits,
                moddown_rounding=rounding,
                modup_method=method,
                rescale=compound,
            )
            physical = {
                name: state[name]
                for name in ('dtype', 'device')
                if name in state
            }
            tensors: dict[str, SSAValue] = {}
            for name in names:
                description = table_description(config, depth, name, facts)
                kind = cast(str, description['kind'])
                typ = (
                    rns.RnsParametersType().with_state(physical)
                    if kind == 'rns_parameters'
                    else core.MessageType().with_state(physical)
                )
                declared = parameter_description(
                    config,
                    kind,
                    **{k: v for k, v in description.items() if k != 'kind'},
                )
                identity = (
                    rns_parameter_identity(
                        config, description.get('prime_ids'), physical
                    )
                    if kind == 'rns_parameters'
                    else None
                )
                created, value = declare_material(
                    compilation,
                    operation,
                    f'keyswitch/{depth}/{name}',
                    typ,
                    declared,
                    identity=identity,
                )
                references.extend(created)
                for reference in created:
                    descriptions[cast(StringAttr, reference.symbol).data] = (
                        declared
                    )
                tensors[name] = value
            fields.update(
                parameter_names=ArrayAttr((StringAttr(name) for name in names)),
                digits=ArrayAttr(
                    (
                        ArrayAttr((IntegerAttr(item, 64) for item in span))
                        for span in digits
                    )
                ),
                **{
                    name: IntegerAttr(cast(int, facts[name]), 64)
                    for name in ('q_count', 'p_count', 'key_row_start')
                },
            )
        else:
            names = fields.get('parameter_names')
            spans = fields.get('digits')
            if not isinstance(names, ArrayAttr) or not isinstance(
                spans, ArrayAttr
            ):
                raise ValueError(
                    f'{operation.name} supplied Tensor operands need their names and digit ranges'
                )
            tensors = dict(
                zip(
                    (cast(StringAttr, name).data for name in names),
                    operation.parameters,
                    strict=True,
                )
            )
            digits = tuple(
                (
                    tuple(
                        (
                            int(cast(IntegerAttr, item).value.data)
                            for item in cast(ArrayAttr, span)
                        )
                    )
                    for span in spans
                )
            )
            qp_ids, qp_rows = _key_switch_qp_basis(source_type, (key,), fields)
        facts = {
            name: value
            for name, value in fields.items()
            if name
            in {
                'parameter_names',
                'digits',
                'p_count',
                'q_count',
                'key_row_start',
                'ntt_backend',
                'joint_drop_count',
                'joint_drop_is_prefix',
                'joint_drop_prime_ids',
            }
        }
        row_start = cast(IntegerAttr, fields['key_row_start']).value.data
        input_cast, value = _cast_to_rns(operation.value)
        operations: list[Operation] = [*references, input_cast]
        ntt_attributes = {
            name: value
            for name, value in fields.items()
            if name
            in {
                'ntt_backend',
                'ntt_algorithm',
                'ntt_group_width',
                'ntt_radix',
                'ntt_table_layout',
            }
            and value != StringAttr('unknown')
        }
        ntt_attributes['ckks_config'] = StringAttr(json.dumps(config.dumps()))
        self.config = config
        self.descriptions = descriptions
        self.digits = digits
        self.facts = facts
        self.key = key
        self.ntt_attributes = ntt_attributes
        self.modup_method = method
        self.operation = operation
        self.operations = operations
        self.q_ids = q_ids
        self.qp_ids = qp_ids
        self.qp_rows = qp_rows
        self.rounding = rounding
        self.row_start = row_start
        self.source_type = source_type
        self.tensors = tensors
        self.value = value

    def emit(self, op: Operation) -> SSAValue:
        self.operations.append(op)
        return op.results[0]

    def polynomial(
        self,
        *,
        output_ntt: bool = False,
        qp: bool = False,
        components: int | None = None,
    ) -> Attribute:
        updates: dict[str, Attribute | None] = {
            'polynomial_domain': StringAttr(
                'ntt' if output_ntt else 'coefficient'
            ),
            'residue_representation': StringAttr(
                'montgomery' if output_ntt else 'standard'
            ),
            'basis': StringAttr('QP' if qp else 'Q'),
            'prime_ids': self.qp_ids if qp else cast(Attribute, self.q_ids),
        }
        if qp and self.qp_ids is None and self.qp_rows is None:
            updates["shape"] = None
        row_count = self.qp_rows if qp else None
        return (
            _polynomial_type(self.source_type, row_count=row_count, **updates)
            if components is None
            else _component_bundle_type(
                self.source_type, components, row_count=row_count, **updates
            )
        )

    def transform(
        self, value: SSAValue, *, inverse: bool, basis: str, typ: Attribute
    ) -> SSAValue:
        prefix = f"{basis}_{('inverse' if inverse else 'forward')}_"
        supplied = tuple(
            (
                self.tensors[f'{prefix}{i}']
                for i in range(
                    sum((name.startswith(prefix) for name in self.tensors))
                )
            )
        )
        source_state = cast(rns.RnsBundleType, value.type).state.data
        cls = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if inverse
            else ntt.CoefficientMontgomeryToNttMontgomeryOp
            if source_state.get('residue_representation')
            == StringAttr('montgomery')
            else ntt.CoefficientStandardToNttMontgomeryOp
        )
        return self.emit(
            cls(
                value,
                supplied[0]
                if supplied
                else self.tensors[f'{basis}_parameters'],
                typ,
                tables=supplied[1:],
                attributes=self.ntt_attributes,
            )
        )

    def local_tables(self, names):
        return (
            tuple((self.tensors[name] for name in names)),
            {
                **self.facts,
                'parameter_names': ArrayAttr(
                    (StringAttr(name) for name in names)
                ),
            },
        )

    def extract(
        self, value: SSAValue, component: int, *, output_ntt: bool = False
    ) -> SSAValue:
        return self.emit(
            rns.ExtractComponentOp(
                value,
                self.polynomial(output_ntt=output_ntt),
                component=component,
            )
        )

    def products(self, source: SSAValue) -> SSAValue:
        """Emit sum_j NTT(ModUp(digit_j(source))) * key_j over QP."""
        accumulator: SSAValue | None = None
        for index, span in enumerate(self.digits):
            coefficient_type = cast(
                rns.RnsBundleType, self.polynomial(qp=True)
            ).with_state({'residue_representation': StringAttr('montgomery')})
            if self.modup_method == "exact":
                names = tuple(
                    (
                        name
                        for name in self.tensors
                        if name == 'qp_parameters'
                        or name.startswith(f'digit{index}_')
                    )
                )
                operands, local_facts = self.local_tables(names)
                lifted = self.emit(
                    rns.HybridModUpDigitOp(
                        source,
                        operands,
                        coefficient_type,
                        digit_index=index,
                        attributes=local_facts,
                    )
                )
            else:
                start, stop, _ = span
                digit_type = _polynomial_type(
                    self.source_type,
                    polynomial_domain=StringAttr('coefficient'),
                    residue_representation=StringAttr('standard'),
                    basis=StringAttr('Q'),
                    prime_ids=ArrayAttr(
                        tuple(cast(ArrayAttr, self.q_ids))[start:stop]
                    ),
                )
                digit = self.emit(
                    rns.SliceRowsOp(
                        source, digit_type, start=start, count=stop - start
                    )
                )
                lifted = self.emit(
                    rns.CrtBasisExtendOp(
                        digit,
                        tuple(
                            self.tensors[f'digit{index}_crt_{name}']
                            for name in CRT_OPERANDS
                        ),
                        coefficient_type,
                    )
                )
            transformed = self.transform(
                lifted,
                inverse=False,
                basis='qp',
                typ=self.polynomial(qp=True, output_ntt=True),
            )
            product_operation = rns.KeySwitchDigitProductOp(
                transformed,
                self.key,
                self.tensors['qp_parameters'],
                self.polynomial(qp=True, output_ntt=True, components=2),
                key_digit_index=span[2],
                key_row_start=int(self.row_start),
            )
            product_operation.attributes['key_role'] = StringAttr(
                {
                    ckks.RelinearizeOp: 'relinearization',
                    ckks.ConjugateOp: 'conjugation',
                    ckks.RotateOp: 'rotation',
                    ckks.SwitchKeyOp: 'switch',
                }[type(self.operation)]
            )
            product = self.emit(product_operation)
            accumulator = (
                product
                if accumulator is None
                else self.emit(
                    rns.AddMontgomeryLazyOp(
                        accumulator,
                        product,
                        self.tensors['qp_parameters'],
                        self.polynomial(qp=True, output_ntt=True, components=2),
                    )
                )
            )
        if accumulator is None:
            raise ValueError('Key switching requires at least one digit')
        return accumulator

    def quotient(
        self,
        accumulator: SSAValue,
        *,
        output_ntt: bool,
        coefficient_c0: SSAValue | None = None,
    ) -> SSAValue:
        """Complete the common rounded P quotient in the requested domain."""
        names = tuple(
            (
                name
                for name in self.tensors
                if name
                in {
                    'q_parameters',
                    'qp_parameters',
                    'moddown_inverses',
                    'p_inverse',
                    'moddown_half',
                }
                or name.startswith(('p_inverse_', 'q_forward_'))
            )
        )
        operands, local_facts = self.local_tables(names)
        if output_ntt:
            return self.emit(
                rns.ModDownNttQpToQOp(
                    accumulator,
                    operands,
                    self.polynomial(output_ntt=True, components=2),
                    coefficient_c0=coefficient_c0,
                    rounding=self.rounding,
                    attributes=local_facts,
                )
            )
        accumulator = self.transform(
            accumulator,
            inverse=True,
            basis='qp',
            typ=self.polynomial(qp=True, components=2),
        )
        operands, local_facts = self.local_tables(
            ('qp_parameters', 'moddown_inverses')
            + (('moddown_half',) if self.rounding == 'nearest' else ())
        )
        return self.emit(
            rns.ModDownQpToQOp(
                accumulator,
                operands,
                self.polynomial(components=2),
                rounding=self.rounding,
                attributes=local_facts,
            )
        )

    def rotation_components(self) -> tuple[SSAValue, SSAValue]:
        value = self.value
        if isinstance(self.operation, ckks.RotateOp):
            if self.operation.input_domain.data == 'ntt':
                indices = self.tensors.get('rotation_indices')
                if indices is None:
                    raise ValueError(
                        'NTT rotation lowering requires supplied permutation indices'
                    )
                value = self.emit(
                    ntt.AutomorphismOp(
                        value,
                        indices,
                        self.polynomial(output_ntt=True, components=2),
                    )
                )
                c0 = self.extract(value, 0, output_ntt=True)
                c1 = self.extract(value, 1, output_ntt=True)
                return c0, self.transform(
                    c1, inverse=True, basis='q', typ=self.polynomial()
                )
            step = cast(
                IntegerAttr,
                cast(ckks.EvaluationKeyType, self.key.type).state.data[
                    'rotation_step'
                ],
            ).value.data
            exponent = (
                -int(step) if self.config.galois_generator == 5 else int(step)
            )
            value = self.emit(
                rns.CoefficientAutomorphismOp(
                    value,
                    self.tensors['q_parameters'],
                    self.polynomial(components=2),
                    galois_element=pow(
                        self.config.galois_generator,
                        exponent % self.config.N,
                        2 * self.config.N,
                    ),
                )
            )
        elif isinstance(self.operation, ckks.ConjugateOp):
            value = self.emit(
                rns.CoefficientAutomorphismOp(
                    value,
                    self.tensors['q_parameters'],
                    self.polynomial(components=2),
                    galois_element=2 * self.config.N - 1,
                )
            )
        d0, d1 = (self.extract(value, i) for i in range(2))
        return (d0, d1)

    def lower(self) -> LoweredCkksOperation:
        if (
            isinstance(self.operation, ckks.RotateOp)
            and self.operation.input_domain.data == 'ntt'
        ):
            raise ValueError(
                'Direct NTT rotation retains its whole-operation implementation'
            )
        requested_ntt = self.operation.output_domain.data == 'ntt'
        if (
            isinstance(self.operation, ckks.RelinearizeOp)
            and self.operation.rescale.value.data
        ):
            d2 = self.transform(
                self.extract(self.value, 2, output_ntt=True),
                inverse=True,
                basis="q",
                typ=self.polynomial(),
            )
            products = self.products(d2)
            names = tuple(
                name
                for name in self.tensors
                if name == "q_parameters" or name.startswith("joint_")
            )
            attributes = {
                **{
                    name: attribute
                    for name, attribute in self.facts.items()
                    if name
                    in {
                        "q_count",
                        "p_count",
                        "ntt_backend",
                        "joint_drop_count",
                        "joint_drop_is_prefix",
                        "joint_drop_prime_ids",
                    }
                },
                **self.ntt_attributes,
                "parameter_names": ArrayAttr(
                    StringAttr(name) for name in names
                ),
            }
            result = self.emit(
                rns.ModDownAddRescaleOp(
                    products,
                    self.value,
                    tuple(self.tensors[name] for name in names),
                    _rns_type(self.operation.result.type),
                    output_domain=cast(
                        Literal["coefficient", "ntt"],
                        self.operation.output_domain.data,
                    ),
                    moddown_rounding=self.rounding,
                    rescale_rounding=self.operation.rescale_rounding.data,
                    attributes=attributes,
                )
            )
            result_cast, result = _cast_to_ckks(
                result, self.operation.result.type
            )
            self.operations.append(result_cast)
            return LoweredCkksOperation(
                tuple(self.operations), result, self.descriptions
            )
        if isinstance(self.operation, ckks.RelinearizeOp):
            if requested_ntt:
                d0, d1, d2 = (
                    self.extract(self.value, i, output_ntt=True)
                    for i in range(3)
                )
                d2 = self.transform(
                    d2, inverse=True, basis='q', typ=self.polynomial()
                )
            else:
                coefficient = self.transform(
                    self.value,
                    inverse=True,
                    basis='q',
                    typ=self.polynomial(components=3),
                )
                d0, d1, d2 = (self.extract(coefficient, i) for i in range(3))
            correction = self.quotient(
                self.products(d2), output_ntt=requested_ntt
            )
            c0, c1 = (
                self.extract(correction, i, output_ntt=requested_ntt)
                for i in range(2)
            )
            add = (
                rns.AddMontgomeryLazyOp if requested_ntt else rns.AddStandardOp
            )
            result0 = self.emit(
                add(
                    d0,
                    c0,
                    self.tensors['q_parameters'],
                    self.polynomial(output_ntt=requested_ntt),
                )
            )
            result1 = self.emit(
                add(
                    d1,
                    c1,
                    self.tensors['q_parameters'],
                    self.polynomial(output_ntt=requested_ntt),
                )
            )
        else:
            d0, d1 = self.rotation_components()
            retained_ntt = requested_ntt and not isinstance(
                self.operation, ckks.RotateOp
            )
            correction = self.quotient(
                self.products(d1),
                output_ntt=retained_ntt,
                coefficient_c0=d0 if retained_ntt else None,
            )
            if retained_ntt:
                result_cast, result = _cast_to_ckks(
                    correction, self.operation.result.type
                )
                self.operations.append(result_cast)
                return LoweredCkksOperation(
                    tuple(self.operations), result, self.descriptions
                )
            c0, result1 = (self.extract(correction, i) for i in range(2))
            result0 = self.emit(
                rns.AddStandardOp(
                    d0,
                    c0,
                    self.tensors['q_parameters'],
                    self.polynomial(),
                )
            )
        packed_type = _rns_type(self.operation.result.type)
        if (
            isinstance(self.operation, ckks.RotateOp)
            and requested_ntt
            and not retained_ntt
        ):
            packed = self.emit(
                rns.PackTwoComponentsOp(
                    result0, result1, self.polynomial(components=2)
                )
            )
            packed = self.transform(
                packed, inverse=False, basis='q', typ=packed_type
            )
        else:
            packed = self.emit(
                rns.PackTwoComponentsOp(result0, result1, packed_type)
            )
        result_cast, result = _cast_to_ckks(packed, self.operation.result.type)
        self.operations.append(result_cast)
        return LoweredCkksOperation(
            tuple(self.operations), result, self.descriptions
        )


def _lower_key_operation(
    operation, config, compilation
) -> LoweredCkksOperation:
    return _KeySwitchLowering(operation, config, compilation).lower()


def _lower_rotation_products(operation, config, compilation):
    """Expose c0 in the source domain and the unrounded NTT/QP key product."""
    lowering = _KeySwitchLowering(operation, config, compilation)
    c0, c1 = lowering.rotation_components()
    products = lowering.products(c1)
    return tuple(lowering.operations), c0, products, lowering.descriptions


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
