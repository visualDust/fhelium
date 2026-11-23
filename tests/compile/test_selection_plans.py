"""Guarded selection reuses decisions while rebuilding current operations."""

import json
import struct

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    BytesAttr,
    DenseArrayBase,
    DictionaryAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f64,
    i32,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block

from fhelium import Ciphertext, CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend
from fhelium.backend.ckks.materialization import CkksDeviceResources
from fhelium.backend.triton import TritonFusionImplementation
from fhelium.compile._selection_plan import SelectionPlans, graph_guard
from fhelium.compile.passes.fusion import FuseOperationsPass
from fhelium.compile.passes.lowering._select_execution import (
    SelectExecutionLoweringsPass,
)
from fhelium.eager import Engine
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, fusion, rns


def _integers(values):
    return ArrayAttr(IntegerAttr(value, 64) for value in values)


def _product(rows, plans, *, escape=False, parameter_alias=True):
    typ = ckks.CiphertextType(
        {
            "depth": IntegerAttr(4 - rows, 64),
            "scale": FloatAttr(float(2 ** (20 + rows)), f64),
            "prime_ids": _integers(range(4 - rows, 4)),
            "components": IntegerAttr(2, 64),
            "batch_shape": _integers(()),
            "ring_dimension": IntegerAttr(256, 64),
            "shape": _integers((2, rows, 256)),
            "strides": _integers((rows * 256, 256, 1)),
            "basis": StringAttr("Q"),
            "dtype": StringAttr("torch.int64"),
            "device": StringAttr("cuda:0"),
            "polynomial_domain": StringAttr("ntt"),
            "residue_representation": StringAttr("montgomery"),
        }
    )
    result_type = typ.with_state(
        components=IntegerAttr(3, 64),
        shape=_integers((3, rows, 256)),
        scale=FloatAttr(float(2 ** (40 + 2 * rows)), f64),
    )
    parameters = rns.RnsParametersType(
        {
            "shape": _integers((rows, 4)),
            "device": StringAttr("cuda:0"),
        }
    )
    block = Block(arg_types=(typ, typ, parameters, parameters))
    product = ckks.MultiplyOp(
        block.args[0], block.args[1], result_type, parameters=(block.args[2],)
    )
    negative = ckks.NegateOp(
        product,
        result_type,
        parameters=(block.args[2 if parameter_alias else 3],),
    )
    results = (
        (negative.result, product.result) if escape else (negative.result,)
    )
    block.add_ops((product, negative, ReturnOp(*results)))
    return fc.Compilation(
        Program.from_function(block, tuple(value.type for value in results)),
        fc.CompileWorkspace({SelectionPlans: plans}),
    )


def _passes():
    generated = TritonFusionImplementation()
    return (
        SelectExecutionLoweringsPass(OperationBackend().registry, (generated,)),
        FuseOperationsPass((generated,)),
    )


def test_cross_row_plan_hit_skips_probes_and_search_and_uses_current_types(
    monkeypatch,
):
    from fhelium.compile.passes import fusion as fusion_pass
    from fhelium.compile.passes.lowering import (
        _select_execution as selection_pass,
    )

    plans = SelectionPlans()
    select, fuse = _passes()
    first = _product(3, plans)
    select.run(first)
    fuse.run(first)
    assert any(
        isinstance(op, fusion.FusedOp)
        for op in first.program.single_block("main").ops
    )
    plans = SelectionPlans.from_dict(json.loads(json.dumps(plans.to_dict())))
    current = _product(2, plans)
    reference = _product(2, SelectionPlans())
    select.run(reference)
    fuse.run(reference)
    lower = selection_pass.apply_lowering
    actual_lowerings = []

    def only_current(operation, config, compilation, **kwargs):
        assert compilation is current, "a plan hit must not lower a probe"
        actual_lowerings.append(operation.name)
        return lower(operation, config, compilation, **kwargs)

    def no_search(*args, **kwargs):
        raise AssertionError("a plan hit must not search fusion candidates")

    monkeypatch.setattr(selection_pass, "apply_lowering", only_current)
    monkeypatch.setattr(fusion_pass, "_fusion_schedule", no_search)
    select.run(current)
    fuse.run(current)
    assert actual_lowerings == [ckks.MultiplyOp.name, ckks.NegateOp.name]
    assert current.workspace["selection_plan_stats"]["selection"]["hits"] == 2
    assert current.workspace["selection_plan_stats"]["fusion"]["hits"] == 1
    assert current.program.to_text() == reference.program.to_text()


def test_fusion_guard_preserves_escaping_outputs_and_parameter_identity():
    plans = SelectionPlans()
    select, fuse = _passes()
    initial = _product(3, plans)
    select.run(initial)
    fuse.run(initial)
    for changes in ({"escape": True}, {"parameter_alias": False}):
        current = _product(2, plans, **changes)
        reference = _product(2, SelectionPlans(), **changes)
        select.run(current)
        select.run(reference)
        fuse.run(current)
        fuse.run(reference)
        assert (
            current.workspace["selection_plan_stats"]["fusion"]["misses"] == 1
        )
        assert current.program.to_text() == reference.program.to_text()


def test_custom_selection_policy_bypasses_plans():
    plans = SelectionPlans()
    select, _ = _passes()
    select.run(_product(3, plans))
    current = _product(3, plans)
    calls = []

    def cost(operation, expression):
        calls.append(operation.name)
        return (1, 2)

    custom = SelectExecutionLoweringsPass(
        select.implementations,
        select.fusion_implementations,
        execution_cost=cost,
    )
    custom.run(current)
    assert calls
    assert current.workspace["selection_plan_stats"]["selection"]["hits"] == 0
    assert (
        current.workspace["selection_plan_stats"]["selection"]["reason"]
        == "custom execution cost"
    )

    class Matcher:
        name = "custom"

        def match_fusion(self, operations):
            calls.append("match")
            return None

    current = _product(3, plans)
    FuseOperationsPass((Matcher(),)).run(current)
    assert "match" in calls
    assert (
        current.workspace["selection_plan_stats"]["fusion"]["reason"]
        == "custom fusion matcher"
    )


def test_cpu_weighted_product_reuses_decisions_across_depth_and_scale():
    base = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=base.q_depth_groups[:3],
        p_moduli=base.p_moduli,
        default_scale=base.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=617)
    backend = OperationBackend()
    plans = SelectionPlans()
    resources = CkksDeviceResources(config=config, device="cpu")

    def weighted_product(lhs, rhs, weight):
        a = engine.coefficient_domain_to_ntt_domain(lhs)
        b = engine.coefficient_domain_to_ntt_domain(rhs)
        product = engine.multiply(a, b)
        weighted = engine.multiply_plaintext(product, weight)
        doubled = engine.subtract(weighted, engine.negate(weighted))
        return engine.ntt_domain_to_coefficient_domain(doubled)

    for depth in (0, 1):
        ids = tuple(range(depth, 3))
        x = Ciphertext(
            torch.randint(0, 1024, (2, len(ids), config.N), dtype=torch.int64),
            depth,
            float(2 ** (20 + depth)),
            ids,
        )
        y = x.clone()
        weight = engine.prepare_plaintext_for_multiplication(
            engine.encode(0.125, depth=depth, device="cpu")
        )
        compilation = fc.capture_eager(
            weighted_product,
            arguments={"lhs": x, "rhs": y, "weight": weight},
            workspace=fc.CompileWorkspace(
                {CkksConfig: config, SelectionPlans: plans}
            ),
        )
        compilation = fc.default_lower_and_fuse_pipeline(backend).run(
            compilation
        )
        assert (
            fc.prepare_material_bindings(compilation, resources=resources) == ()
        )
        actual = backend.link(compilation).run(x, y, weight)
        expected = weighted_product(x, y, weight)
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
        assert (actual.depth, actual.scale, actual.prime_ids) == (
            expected.depth,
            expected.scale,
            expected.prime_ids,
        )
        if depth:
            stats = compilation.workspace["selection_plan_stats"]
            assert stats["selection"]["hits"] > 0
            assert stats["selection"]["misses"] == 0
            if "fusion" in stats:
                assert stats["fusion"]["hits"] == 1
        plans = SelectionPlans.from_dict(
            json.loads(json.dumps(plans.to_dict()))
        )


def test_guard_attribute_roundtrip_preserves_types_strings_and_float_bits(
    monkeypatch,
):
    compilation = _product(3, SelectionPlans())
    program = compilation.program
    operation = program.single_block("main").first_op
    assert operation is not None
    fields = {
        "number": IntegerAttr(7, 64),
        "float": FloatAttr(0.0, f64),
        "text": StringAttr('quoted "value"\nnext'),
        "bytes": BytesAttr(b"\x00\xff"),
        "segments": DenseArrayBase.from_list(i32, (1, 2)),
        "nested": ArrayAttr((StringAttr("leaf"), IntegerAttr(3, 32))),
    }
    operation.attributes["guard_probe"] = DictionaryAttr(fields)
    restored = Program.parse(program.to_text())

    def no_printer(attribute):
        raise AssertionError("guard encoding must not print attributes")

    monkeypatch.setattr(Attribute, "__str__", no_printer)

    def guard():
        return graph_guard(
            program.single_block("main"), stage="selection", policy=()
        )

    original = guard()
    assert original == graph_guard(
        restored.single_block("main"), stage="selection", policy=()
    )
    for field, replacement in (
        ("number", IntegerAttr(7, 32)),
        ("text", StringAttr('quoted "value"\nother')),
        ("float", FloatAttr(-0.0, f64)),
    ):
        operation.attributes["guard_probe"] = DictionaryAttr(
            {**fields, field: replacement}
        )
        assert guard() != original
    # FloatData equality deliberately treats NaNs as equal; guard identity
    # must still distinguish payload bits in the same Attribute collection.
    nan_a = FloatAttr(
        struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0], f64
    )
    nan_b = FloatAttr(
        struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0], f64
    )
    operation.attributes["guard_probe"] = ArrayAttr((nan_a, nan_b))
    distinct = guard()
    operation.attributes["guard_probe"] = ArrayAttr((nan_a, nan_a))
    assert guard() != distinct
