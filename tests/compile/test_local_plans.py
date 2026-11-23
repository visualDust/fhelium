"""Local recipes preserve current SSA relationships and global fusion decisions."""

import json

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import compile as fc
from fhelium.backend.triton import TritonFusionImplementation
from fhelium.backend.triton._matching import OperationAnalysis, RegionMatcher
from fhelium.compile._local_plans import Occurrence, local_plan_scope
from fhelium.compile._selection_plan import SelectionPlans, fusion_policy
from fhelium.compile.passes.fusion import FuseOperationsPass, _fusion_schedule
from fhelium.ir import Program
from fhelium.ir.dialects import ntt, rns


def _integers(values):
    return ArrayAttr(IntegerAttr(value, 64) for value in values)


def _program(rows, branches, plans, *, alternate_parameter=False, escape=False):
    typ = rns.RnsBundleType(
        {
            "shape": _integers((rows, 256)),
            "prime_ids": _integers(range(5 - rows, 5)),
            "ring_dimension": IntegerAttr(256, 64),
            "batch_shape": _integers(()),
            "basis": StringAttr("Q"),
            "dtype": StringAttr("torch.int64"),
            "device": StringAttr("cuda:0"),
        }
    )
    parameters = rns.RnsParametersType({"device": StringAttr("cuda:0")})
    block = Block(arg_types=(typ, typ, parameters, parameters))
    transformed = ntt.CoefficientStandardToNttMontgomeryOp(
        block.args[0], block.args[2], typ
    )
    block.add_op(transformed)
    outputs = []
    for index in range(branches):
        parameter = block.args[3 if alternate_parameter and index % 2 else 2]
        doubled = rns.AddMontgomeryLazyOp(
            transformed, transformed, parameter, typ
        )
        total = rns.AddMontgomeryLazyOp(doubled, block.args[1], parameter, typ)
        restored = ntt.NttMontgomeryToCoefficientStandardOp(
            total, parameter, typ
        )
        block.add_ops((doubled, total, restored))
        outputs.append(restored.result)
    if escape:
        outputs.append(transformed.result)
    block.add_op(ReturnOp(*outputs))
    return fc.Compilation(
        Program.from_function(block, tuple(value.type for value in outputs)),
        fc.CompileWorkspace({SelectionPlans: plans}),
    )


def _implementations():
    return (
        TritonFusionImplementation(name="ntt-expressions", include_ntt=True),
        TritonFusionImplementation(),
    )


def test_port_graph_preserves_operand_multiplicity_and_current_materials():
    compilation = _program(3, 2, SelectionPlans())
    block = compilation.program.single_block("main")
    operations = tuple(block.ops)
    first = Occurrence.from_operations(
        (operations[1], operations[2]), origin={"source": "capture"}
    )
    second = Occurrence.from_operations(
        (operations[4], operations[5]), origin={"source": "lowering"}
    )
    assert first.port_graph == second.port_graph
    assert first.nodes != second.nodes
    assert first.escaping_outputs() == (operations[2].result,)

    typ = block.args[0].type
    other = rns.AddMontgomeryLazyOp(
        block.args[0], block.args[1], block.args[3], typ
    )
    repeated = rns.AddMontgomeryLazyOp(
        block.args[0], block.args[0], block.args[2], typ
    )
    equivalent = rns.AddMontgomeryLazyOp(
        block.args[1], block.args[1], block.args[3], typ
    )
    left = Occurrence.from_operations((repeated,))
    right = Occurrence.from_operations((equivalent,))
    assert left.port_graph == right.port_graph
    assert left.inputs[-1] is block.args[2]
    assert right.inputs[-1] is block.args[3]
    assert left.port_graph != Occurrence.from_operations((other,)).port_graph
    for operation in (other, repeated, equivalent):
        operation.drop_all_references()


def test_local_recipes_preserve_schedules_and_final_ir_across_rows_and_topology():
    plans = SelectionPlans()
    implementations = _implementations()
    for rows, branches in ((3, 3), (2, 2), (1, 1)):
        current = _program(rows, branches, plans)
        reference = _program(
            rows,
            branches,
            SelectionPlans.from_dict(json.loads(json.dumps(plans.to_dict()))),
        )
        reference.workspace["local_plan_reuse"] = False
        operations = tuple(current.program.single_block("main").ops)
        scope = local_plan_scope(
            current.workspace, fusion_policy(implementations)
        )
        assert scope is not None
        assert _fusion_schedule(
            operations, implementations, local_plans=scope
        ) == _fusion_schedule(operations, implementations)
        FuseOperationsPass(implementations).run(current)
        FuseOperationsPass(implementations).run(reference)
        assert current.program.to_text() == reference.program.to_text()
        assert current.workspace["local_plan_stats"]["facts_hits"] > 0
        assert current.workspace["local_plan_stats"]["recipe_hits"] > 0
        plans = SelectionPlans.from_dict(
            json.loads(json.dumps(plans.to_dict()))
        )
        assert plans.local_entries


def test_reused_recipes_recheck_parameter_identity_and_live_escaping_uses():
    implementations = _implementations()
    plans = SelectionPlans()
    FuseOperationsPass(implementations).run(_program(3, 3, plans))
    current = _program(2, 2, plans, alternate_parameter=True, escape=True)
    reference = _program(
        2,
        2,
        SelectionPlans.from_dict(json.loads(json.dumps(plans.to_dict()))),
        alternate_parameter=True,
        escape=True,
    )
    reference.workspace["local_plan_reuse"] = False
    operations = tuple(current.program.single_block("main").ops)
    scope = local_plan_scope(current.workspace, fusion_policy(implementations))
    assert scope is not None
    matcher = RegionMatcher(include_ntt=True)
    matcher.use_analysis(OperationAnalysis(scope.matcher_recipe, scope.stats))
    matcher.append(operations[0])
    # Two branches each read the transform twice, and the return reads it once.
    assert matcher.remaining[operations[0].result] == 5
    matcher.append(operations[1])
    assert matcher.remaining[operations[0].result] == 3
    assert matcher.probe(operations[4]) is None
    assert matcher.remaining[operations[0].result] == 3
    FuseOperationsPass(implementations).run(current)
    FuseOperationsPass(implementations).run(reference)
    assert current.program.to_text() == reference.program.to_text()


def test_region_rejections_reduce_checks_without_changing_groups_and_expire_on_reset(
    monkeypatch,
):
    from fhelium.compile.passes import fusion as fusion_pass

    # Exercise rejection memory independently of the scheduler's static filter.
    monkeypatch.setattr(
        fusion_pass._CandidateSet, "candidates", lambda self: self.all
    )
    implementations = _implementations()
    current = _program(2, 4, SelectionPlans(), alternate_parameter=True)
    reference = _program(2, 4, SelectionPlans(), alternate_parameter=True)
    checks = []
    check = RegionMatcher._check_operation
    probe = RegionMatcher.probe

    def count_check(self, operation):
        checks.append(operation)
        return check(self, operation)

    def without_memory(self, operation):
        self._rejected.clear()
        return probe(self, operation)

    monkeypatch.setattr(RegionMatcher, "_check_operation", count_check)
    FuseOperationsPass(implementations).run(current)
    remembered_checks = len(checks)
    checks.clear()
    with monkeypatch.context() as context:
        context.setattr(RegionMatcher, "probe", without_memory)
        FuseOperationsPass(implementations).run(reference)
    assert remembered_checks < len(checks)
    assert current.program.to_text() == reference.program.to_text()

    fresh = _program(2, 2, SelectionPlans(), alternate_parameter=True)
    operations = tuple(fresh.program.single_block("main").ops)
    matcher = RegionMatcher(include_ntt=True)
    matcher.use_analysis(OperationAnalysis())
    matcher.append(operations[0])
    candidate = operations[4]
    assert matcher.probe(candidate) is None
    before = len(checks)
    assert matcher.probe(candidate) is None
    assert len(checks) == before
    matcher.reset()
    assert matcher.probe(candidate) is not None


def test_tensor_incremental_matching_preserves_mixed_schedules_and_resets(
    monkeypatch,
):
    from fhelium.backend.triton._tensor import TritonTensorFusionImplementation
    from fhelium.ir.dialects import core
    from fhelium.ir.dialects.torch import TensorCallOp

    class WholeCandidate:
        name = "triton-tensor-fused"

        def match_fusion(self, operations):
            return tensor.match_fusion(operations)

    tensor = TritonTensorFusionImplementation()
    current = _program(2, 3, SelectionPlans(), alternate_parameter=True)
    block = current.program.single_block("main")
    end = block.last_op
    outputs = []
    for dtype, device in (
        ("torch.float32", "cuda:0"),
        ("torch.float64", "cuda:0"),
        ("torch.float32", "cuda:1"),
    ):
        typ = core.MessageType(
            {
                "shape": _integers((16,)),
                "strides": _integers((1,)),
                "dtype": StringAttr(dtype),
                "device": StringAttr(device),
            }
        )
        value = block.insert_arg(typ, len(block.args))
        for target in ("torch.add", "torch.mul"):
            operation = TensorCallOp(
                (value, value),
                typ,
                kind="function",
                target=target,
                argument_descriptor=json.dumps(
                    {
                        "args": {
                            "kind": "tuple",
                            "items": [
                                {"kind": "ssa", "operand": 0},
                                {"kind": "ssa", "operand": 1},
                            ],
                        },
                        "kwargs": {"kind": "mapping", "entries": []},
                    }
                ),
            )
            block.insert_op_before(operation, end)
            value = operation.result
        outputs.append(value)
    end.operands = (*end.operands, *outputs)
    operations = tuple(block.ops)

    def schedule(implementation):
        ordered, groups = _fusion_schedule(
            operations, (*_implementations(), implementation)
        )
        return ordered, [(ops, impl.name, count) for ops, impl, count in groups]

    assert schedule(tensor) == schedule(WholeCandidate())
    from fhelium.compile.passes import fusion as fusion_pass

    implementations = (*_implementations(), tensor)
    scope = local_plan_scope(current.workspace, fusion_policy(implementations))
    filtered = _fusion_schedule(operations, implementations, local_plans=scope)
    with monkeypatch.context() as patch:
        patch.setattr(
            fusion_pass._CandidateSet, "candidates", lambda self: self.all
        )
        unfiltered = _fusion_schedule(
            operations, implementations, local_plans=scope
        )
    assert filtered == unfiltered
    groups = schedule(tensor)[1]
    assert [count for _, name, count in groups if name == tensor.name] == [
        2,
        2,
        2,
    ]
