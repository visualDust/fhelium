"""Element-dependency queries and their use in generated-kernel selection."""

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Block

from fhelium.ir import OperationDependencies, operation_dependencies
from fhelium.ir.dialects import ckks, fusion, ntt, rns


def _types():
    return rns.RnsBundleType(), rns.RnsParametersType()


def test_row_selection_and_basis_extension_have_different_dependencies():
    value_type, parameters_type = _types()
    block = Block(arg_types=(value_type, parameters_type))
    value, parameters = block.args
    selection = rns.RestrictDepthOp.create(
        operands=(value,),
        result_types=(value_type,),
        attributes={"target_depth": IntegerAttr(1, 64)},
    )
    extension = rns.HybridModUpDigitOp(value, (parameters,), value_type)
    for operation, limb in ((selection, "reindexed"), (extension, "mixing")):
        dependencies = operation_dependencies(operation)
        assert dependencies.kind(0, 0, "coefficient") == "element"
        assert dependencies.kind(0, 0, "limb") == limb
    assert operation_dependencies(extension).kind(0, 1, "limb") == "unknown"
    assert OperationDependencies().kind(0, 0, "coefficient") == "unknown"


def test_rescale_and_whole_key_switch_resolve_their_actual_coordinates():
    value_type, _ = _types()
    block = Block(arg_types=(value_type,))
    for domain, position in (("coefficient", "element"), ("ntt", "mixing")):
        operation = rns.RescaleDropLeadingPrimesOp(
            block.args[0], (), value_type, polynomial_domain=domain
        )
        dependencies = operation_dependencies(operation)
        assert dependencies.kind(0, 0, "coefficient") == position
        assert dependencies.kind(0, 0, "limb") == "mixing"
    switch = ckks.SwitchKeyOp.create(
        operands=(block.args[0], block.args[0]), result_types=(value_type,)
    )
    assert operation_dependencies(switch).kind(0, 0, "limb") == "mixing"
    assert operation_dependencies(switch).kind(0, 0, "coefficient") == "mixing"


def test_region_dependencies_follow_branches_and_retain_unknown_reads():
    value_type, parameters_type = _types()
    outer = Block(arg_types=(value_type, parameters_type))
    body = Block(arg_types=(value_type, parameters_type))
    x, parameters = body.args
    permuted = rns.CoefficientAutomorphismOp(
        x, parameters, value_type, galois_element=3
    )
    summed = rns.AddStandardOp(x, permuted, parameters, value_type)
    body.add_ops((permuted, summed, fusion.YieldOp((summed.result,))))
    fused = fusion.FusedOp(outer.args, (value_type,), body)
    dependencies = operation_dependencies(fused)
    assert dependencies.kind(0, 0, "coefficient") == "mixing"
    assert dependencies.kind(0, 0, "limb") == "element"
    assert dependencies.kind(0, 1, "coefficient") == "unknown"
    clone = fused.clone()
    assert operation_dependencies(clone) == dependencies


def test_triton_matches_only_known_supported_operand_dependencies(monkeypatch):
    from fhelium.backend.triton import _matching

    state = {
        "shape": ArrayAttr(IntegerAttr(n, 64) for n in (2, 256)),
        "device": StringAttr("cuda:0"),
        "dtype": StringAttr("int64"),
    }
    value_type = rns.RnsBundleType(state)
    parameters_type = rns.RnsParametersType()
    block = Block(arg_types=(value_type, value_type, parameters_type))
    x, y, parameters = block.args
    added = rns.AddStandardOp(x, y, parameters, value_type)
    assert _matching.match_region((added,), include_ntt=False) == 1
    monkeypatch.setattr(
        _matching, "operation_dependencies", lambda op: OperationDependencies()
    )
    assert _matching.match_region((added,), include_ntt=False) is None


def test_ntt_and_compact_inputs_have_distinct_position_relations():
    value_type, parameters_type = _types()
    block = Block(arg_types=(value_type, value_type, parameters_type))
    x, weight, parameters = block.args
    transform = ntt.CoefficientStandardToNttMontgomeryOp.create(
        operands=(x, parameters), result_types=(value_type,)
    )
    compact = ckks.MultiplyCompressedPlaintextOp.create(
        operands=(x, weight, parameters), result_types=(value_type,)
    )
    assert (
        operation_dependencies(transform).kind(0, 0, "coefficient") == "mixing"
    )
    dependencies = operation_dependencies(compact)
    assert dependencies.kind(0, 0, "coefficient") == "element"
    assert dependencies.kind(0, 1, "coefficient") == "reindexed"
    assert dependencies.kind(0, 2, "coefficient") == "unknown"
