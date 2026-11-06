"""Select shared key inventories from actual CKKS operands and group layouts."""

import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium.config import CkksConfig
from fhelium.compile import (
    Compilation,
    CompileWorkspace,
    prepare_material_bindings,
)
from fhelium.compile.frontend._eager_values import state_attributes
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from fhelium.values import (
    ConjugationKey,
    EvaluationKeySet,
    RelinearizationKey,
    RotationKey,
)


CONFIG = CkksConfig(
    logN=4,
    q_depth_groups=((97,), (193,), (257,)),
    p_moduli=(769, 1153),
    default_scale=8.0,
    enforce_security_budget=False,
)


def key(role=RotationKey, step=1, *, compact=False):
    ids = tuple(range(1 if compact else 0, 5))
    digits = ((1, 2),) if compact else ((0, 1), (2,))
    return role(
        data=torch.empty(len(digits), 2, len(ids), 16, dtype=torch.int32),
        prime_ids=ids,
        digit_prime_ids=digits,
        **({'rotation_step': step} if role is RotationKey else {}),
    )


def value_type(depth, *, q_ids=None):
    fields = {
        'dtype': StringAttr('int32'),
        'device': StringAttr('cpu'),
        'ring_dimension': IntegerAttr(16, 64),
    }
    if depth is not None:
        fields['depth'] = IntegerAttr(depth, 64)
    if q_ids is not None:
        fields['prime_ids'] = ArrayAttr([IntegerAttr(i, 64) for i in q_ids])
    return ckks.CiphertextType().with_state(fields)


def compilation(block, results, descriptions):
    block.add_op(ReturnOp(*results))
    program = Program.from_function(
        block, tuple(result.type for result in results)
    )
    for symbol, description in descriptions.items():
        program.set_material_description(symbol, description)
    return Compilation(program, CompileWorkspace({CkksConfig: CONFIG}))


def test_rotation_selection_uses_actual_rows_and_unknowns_not_depth_identity():
    full, compact = key(), key(compact=True)
    inventory = EvaluationKeySet((full, compact))

    def request(depths, *, shared=False, key_operand=True, q_ids=None):
        types = [value_type(depth, q_ids=q_ids) for depth in depths]
        block = Block(arg_types=(*types, ckks.EvaluationKeyType()))
        results, descriptions = [], {}
        for i, typ in enumerate(types):
            symbol = 'key' if shared else f'key{i}'
            reference = core.MaterialRefOp(
                ckks.EvaluationKeyType(), symbol=symbol
            )
            operation = ckks.RotateOp(
                block.args[i],
                reference.value if key_operand else block.args[-1],
                typ,
                parameters=() if key_operand else (reference.value,),
            )
            block.add_ops((reference, operation))
            results.append(operation.result)
            descriptions[symbol] = {'kind': 'RotationKey', 'rotation_step': 1}
        return compilation(block, results, descriptions)

    actual = request((0, 1, 2))
    assert prepare_material_bindings(actual, keys=inventory) == ()
    assert all(
        actual.material_bindings[f'key{i}'] is selected.data
        for i, selected in enumerate((full, compact, compact))
    )
    common = request((0, 1), shared=True)
    assert prepare_material_bindings(common, keys=inventory) == ()
    assert common.material_bindings['key'] is full.data
    partial = request((None, 0), shared=True)
    assert prepare_material_bindings(
        partial, keys=EvaluationKeySet((compact,))
    ) == ('key',)
    actual_rows = request((None,), q_ids=(1, 2))
    assert prepare_material_bindings(actual_rows, keys=inventory) == ()
    assert actual_rows.material_bindings['key0'] is compact.data
    unknown = request((None,))
    assert prepare_material_bindings(unknown, keys=inventory) == ('key0',)
    assert (
        prepare_material_bindings(unknown, keys=EvaluationKeySet((full,))) == ()
    )
    not_key = request((1,), key_operand=False)
    assert prepare_material_bindings(not_key, keys=inventory) == ('key0',)
    assert prepare_material_bindings(not_key, keys={'key0': compact}) == ()
    existing = actual.material_bindings['key1']
    assert prepare_material_bindings(actual, keys={'key1': full}) == ()
    assert actual.material_bindings['key1'] is existing


def test_rotation_groups_select_complete_layout_and_respect_bound_operands():
    full1, full2 = key(step=1), key(step=2)
    compact1, compact2 = key(step=1, compact=True), key(step=2, compact=True)

    def request(*, fixed=None, aggregate=False):
        typ = value_type(1)
        block = Block(arg_types=(typ, typ, typ) if aggregate else (typ,))
        first = core.MaterialRefOp(
            ckks.EvaluationKeyType().with_state(state_attributes(fixed))
            if fixed is not None
            else ckks.EvaluationKeyType(),
            symbol='key1',
        )
        second = core.MaterialRefOp(ckks.EvaluationKeyType(), symbol='key2')
        if aggregate:
            operation = ckks.RotationSumOp(
                tuple(block.args), (first.value, second.value), (0, 1, 2), typ
            )
            results = (operation.result,)
        else:
            operation = ckks.RotateManyOp(
                block.args[0], (first.value, second.value), (typ, typ)
            )
            results = tuple(operation.outputs)
        block.add_ops((first, second, operation))
        current = compilation(
            block,
            results,
            {
                f'key{i}': {'kind': 'RotationKey', 'rotation_step': i}
                for i in (1, 2)
            },
        )
        if fixed is not None:
            current.material_bindings['key1'] = fixed.data
        return current

    incomplete_compact = request()
    assert (
        prepare_material_bindings(
            incomplete_compact, keys=EvaluationKeySet((full1, compact1, full2))
        )
        == ()
    )
    assert incomplete_compact.material_bindings['key1'] is full1.data
    assert incomplete_compact.material_bindings['key2'] is full2.data
    for aggregate in (False, True):
        complete = request(aggregate=aggregate)
        assert (
            prepare_material_bindings(
                complete,
                keys=EvaluationKeySet((full1, compact1, full2, compact2)),
            )
            == ()
        )
        assert complete.material_bindings['key1'] is compact1.data
        assert complete.material_bindings['key2'] is compact2.data
    constrained = request(fixed=full1)
    assert (
        prepare_material_bindings(
            constrained, keys=EvaluationKeySet((compact2, full2))
        )
        == ()
    )
    assert constrained.material_bindings['key2'] is full2.data
    incompatible = request(fixed=full1)
    assert prepare_material_bindings(
        incompatible, keys=EvaluationKeySet((compact2,))
    ) == ('key2',)


def test_relinearization_and_conjugation_inventory_selection():
    for role, operation_type in (
        (RelinearizationKey, ckks.RelinearizeOp),
        (ConjugationKey, ckks.ConjugateOp),
    ):
        full, compact = key(role), key(role, compact=True)
        typ = value_type(1)
        block = Block(arg_types=(typ,))
        reference = core.MaterialRefOp(ckks.EvaluationKeyType(), symbol='key')
        operation = operation_type(block.args[0], reference.value, typ)
        block.add_ops((reference, operation))
        current = compilation(
            block, (operation.result,), {'key': {'kind': role.__name__}}
        )
        assert (
            prepare_material_bindings(
                current, keys=EvaluationKeySet((full, compact))
            )
            == ()
        )
        assert current.material_bindings['key'] is compact.data


def test_declared_layouts_constrain_inventory_and_propagate_to_symbol_aliases():
    import pytest
    from fhelium.backend import OperationBackend
    from fhelium.compile import PrepareOperationOperandsPass

    backend = OperationBackend()
    full, compact = key(), key(compact=True)
    for reverse in (False, True):
        types = (
            ckks.EvaluationKeyType(),
            ckks.EvaluationKeyType().with_state(state_attributes(compact)),
        )
        if reverse:
            types = types[::-1]
        block = Block()
        refs = [core.MaterialRefOp(typ, symbol="key") for typ in types]
        block.add_ops(refs)
        state = compilation(
            block,
            tuple(ref.value for ref in refs),
            {"key": {"kind": "RotationKey", "rotation_step": 1}},
        )
        PrepareOperationOperandsPass(backend.registry).run(state)
        expected = state_attributes(compact)
        assert all(
            ref.value.type.state.data["prime_ids"] == expected["prime_ids"]
            for ref in refs
        )
        assert all(
            ref.value.type.state.data["digit_prime_ids"]
            == expected["digit_prime_ids"]
            for ref in refs
        )
        assert state.material_bindings == {}

    block = Block(arg_types=(value_type(1),))
    ref = core.MaterialRefOp(ckks.EvaluationKeyType(), symbol="key")
    rotation = ckks.RotateOp(block.args[0], ref.value, block.args[0].type)
    block.add_ops((ref, rotation))
    state = compilation(
        block,
        (rotation.result,),
        {"key": {"kind": "RotationKey", "rotation_step": 1}},
    )
    # An explicit full layout is applied before inventory preference for compact keys.
    PrepareOperationOperandsPass(
        backend.registry,
        keys=EvaluationKeySet((full, compact)),
        key_layouts={
            "key": {
                "prime_ids": full.prime_ids,
                "digit_prime_ids": full.digit_prime_ids,
            }
        },
    ).run(state)
    assert state.material_bindings["key"] is full.data
    before = state.program.to_text()
    with pytest.raises(ValueError, match="conflicts with declared"):
        PrepareOperationOperandsPass(
            backend.registry,
            key_layouts={
                "key": {
                    "prime_ids": compact.prime_ids,
                    "digit_prime_ids": compact.digit_prime_ids,
                }
            },
        ).run(state)
    assert state.program.to_text() == before
