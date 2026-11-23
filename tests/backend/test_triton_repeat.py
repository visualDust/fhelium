"""A planned repeat preserves the unrolled endpoint's integer residues."""

from __future__ import annotations

_INTERPRETER = r'''
import ast
import sys
import torch
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntegerAttr, StringAttr
from xdsl.ir import Block
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__]
sys.path.insert(0, sys.argv[1])
from fhelium.ir.dialects import ckks, fusion, ntt, rns
from fhelium.backend.ntt._stages import NttStages
from fhelium.backend.triton._expressions import ExpressionGraph, InputLayout
from fhelium.backend.triton._ntt import _Endpoint, _Step, _source, _plan_steps
from fhelium.backend.triton._pointwise import PointwisePlan
from fhelium.backend.triton._budget import DEFAULT_KERNEL_BUDGET, kernel_cost
from fhelium.backend.triton._ntt_codegen import compile_endpoint
from fhelium.backend.triton._repeat import REPEAT_ATTRIBUTE, RepeatPlans

n, groups, terms = 16, 8, 24

def bundle(components=None, domain="ntt"):
    shape = (*((components,) if components is not None else ()), 1, n)
    state = {
        "shape": ArrayAttr(IntegerAttr(v, 64) for v in shape),
        "ring_dimension": IntegerAttr(n, 64),
        "dtype": StringAttr("torch.int64"),
        "device": StringAttr("cuda:0"),
        "polynomial_domain": StringAttr(domain),
        "residue_representation": StringAttr("montgomery" if domain == "ntt" else "standard"),
    }
    if components is not None:
        state["components"] = IntegerAttr(components, 64)
    return rns.RnsBundleType(state)

ct, plain, output = bundle(2), bundle(), bundle(2, "coefficient")
block = Block(arg_types=(ct, *((ct,) * groups), *((plain,) * (groups * terms)),
                        rns.RnsParametersType(), plain))
shared = block.args[0]
parameters, twiddle = block.args[-2:]
results = []
for g in range(groups):
    value = block.args[1 + g]
    for j in range(terms):
        weight = block.args[1 + groups + g * terms + j]
        product = rns.MultiplyPlaintextOp(shared, weight, parameters, ct)
        total = rns.AddStandardOp(value, product, parameters, ct)
        block.add_ops((product, total))
        value = total.result
    inverse = ntt.NttMontgomeryToCoefficientStandardOp(value, parameters, output, tables=(twiddle,))
    block.add_op(inverse)
    results.append(inverse.result)
block.add_op(fusion.YieldOp(results))
outer = Block(arg_types=tuple(v.type for v in block.args))
region = fusion.FusedOp(outer.args, (output,) * groups, block)
pointwise_region = region.clone()
plans = RepeatPlans()
identified = plans.identify(region)
assert identified == (tuple(range(groups)),), identified
assert plans.identify(region) == identified

# A represented independent body uses the same anchor plan and endpoint source.
compact = region.clone()
compact_block = compact.body.block
original = tuple(compact_block.ops)[:-1]
body = Block(arg_types=(IndexType(), ct, rns.RnsParametersType(), plain, ct, *((plain,) * terms)))
mapper = dict(zip((compact_block.args[0], *compact_block.args[-2:], *compact_block.args[1:2], *compact_block.args[1+groups:1+groups+terms]), tuple(body.args)[1:], strict=True))
for op in original[:2*terms+1]:
    body.add_op(op.clone(value_mapper=mapper))
body.add_op(ckks.RepeatYieldOp((body.last_op.results[0],)))
loop = ckks.RepeatOp((compact_block.args[0], *compact_block.args[-2:]), tuple(v for g in range(groups) for v in (compact_block.args[1+g], *compact_block.args[1+groups+g*terms:1+groups+(g+1)*terms])), (), (output,)*groups, body, count=groups, yield_count=1)
compact_block.insert_op_before(loop, original[0])
compact_block.last_op.operands = loop.outputs
for op in reversed(original):
    compact_block.erase_op(op)
assert plans.identify(compact) == identified

q, radix = 97, 1 << 62
k = (-pow(q, -1, radix)) % radix
parameters_data = torch.tensor((2*q, q, 0, k & 0x7fffffff, k >> 31,
                                radix*radix % q, 0, pow(n, -1, q)*radix % q), dtype=torch.int64).reshape(8, 1)
root = pow(5, (q-1)//(2*n), q)
bit_reverse = lambda i: int(f"{i:04b}"[::-1], 2)
twiddle_data = torch.tensor([pow(root, -bit_reverse(i), q)*radix % q for i in range(n)], dtype=torch.int64).reshape(1, n)
generator = torch.Generator().manual_seed(7)
operands = [torch.randint(0, q, (2, 1, n), dtype=torch.int64, generator=generator) for _ in range(1+groups)]
operands += [torch.randint(0, q, (1, n), dtype=torch.int64, generator=generator) for _ in range(groups*terms)]
operands += [parameters_data, twiddle_data]

sources, answers = [], []
for enabled in (False, True, "structured"):
    if enabled == "structured":
        region = compact
    if enabled:
        region.attributes[REPEAT_ATTRIBUTE] = ArrayAttr(ArrayAttr(IntegerAttr(i, 64) for i in group) for group in identified)
    else:
        region.attributes.pop(REPEAT_ATTRIBUTE, None)
    expression = ExpressionGraph(region)
    data_indices, _ = expression.input_indices
    data = tuple(operands[i] for i in data_indices)
    layouts = tuple(InputLayout(tuple(t.shape), tuple(t.stride()), i <= groups) for i,t in zip(data_indices,data,strict=True))
    graph = expression.bind(layouts)
    step = _Step(tuple(_Endpoint(i, 0, 4, True, True, True) for i in range(groups)))
    schedules = (NttStages(None, 2, ((0, 4, True),)),) * groups
    stores = tuple((value,) for value in graph.outputs)
    kernel = _source(graph, schedules, step, stores, ())
    sources.append(kernel.source)
    outputs = tuple(torch.empty((2,1,n),dtype=torch.int64) for _ in range(groups))
    arguments = [*(data[i] for i in kernel.inputs), *outputs]
    for _ in range(groups):
        arguments.extend((twiddle_data, *twiddle_data.stride()))
    arguments.extend((parameters_data, *parameters_data.stride(), 62, n, 128, *kernel.scalars))
    compile_endpoint(kernel.source)[(1,1,1)](*arguments, num_warps=4)
    answers.append(outputs)
    unchanged = _plan_steps(graph, schedules, (step,), stores, graph.outputs, DEFAULT_KERNEL_BUDGET)
    assert unchanged[0] == (step,) and unchanged[4][0].source == kernel.source
    if enabled is False:
        steps, writes, required, prior, kernels = _plan_steps(graph, schedules, (step,), stores, graph.outputs, (64, 1536, 12000))
        assert len(steps) > 1 and not any(writes) and not any(prior)
        bounded = tuple(torch.empty_like(t) for t in outputs)
        for part, emitted in zip(steps, kernels, strict=True):
            arguments = [*(data[i] for i in emitted.inputs), *(bounded[e.anchor] for e in part.endpoints)]
            for _ in part.endpoints:
                arguments.extend((twiddle_data, *twiddle_data.stride()))
            arguments.extend((parameters_data, *parameters_data.stride(), 62, n, 128, *emitted.scalars))
            compile_endpoint(emitted.source)[(1,1,1)](*arguments, num_warps=4)
        assert all(torch.equal(a,b) for a,b in zip(outputs,bounded,strict=True))
assert "tl.range(" not in sources[0]
assert sources[1].count("tl.range(") == 1
assert len(ast.parse(sources[1]).body[0].args.args) == len(ast.parse(sources[0]).body[0].args.args) + 1
assert sources[2] == sources[1]
assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(*answers,strict=True))
# Pointwise partitions retain complete ordered reductions and every output.
for op in tuple(pointwise_region.body.block.ops):
    if isinstance(op, ntt.NttMontgomeryToCoefficientStandardOp):
        op.result.replace_all_uses_with(op.operands[0])
        pointwise_region.body.block.erase_op(op)
expression = ExpressionGraph(pointwise_region)
data_indices, _ = expression.input_indices
data = tuple(operands[i] for i in data_indices)
layouts = tuple(InputLayout(tuple(t.shape), tuple(t.stride()), i <= groups) for i,t in zip(data_indices,data,strict=True))
graph = expression.bind(layouts)
plans = [PointwisePlan.build(graph, kernel_budget=budget) for budget in (None, (64,1536,12000))]
assert plans[0].kernel_count == 1 and plans[1].kernel_count > 1
default_pointwise = PointwisePlan.build(graph)
assert default_pointwise.kernel_count > 1
from fhelium.backend.triton._budget import DEFAULT_KERNEL_BUDGET, DEFAULT_POINTWISE_BUDGET
assert DEFAULT_KERNEL_BUDGET[0] == 768 and DEFAULT_POINTWISE_BUDGET[0] == 128

def leaves(plan):
    if plan.parts:
        for part in plan.parts:
            yield from leaves(part)
    else:
        yield plan

results = []
for plan in plans:
    storage = {}
    for part in leaves(plan):
        outputs = tuple(torch.empty(v.shape, dtype=torch.int64) for v in part.materialized)
        arguments = [*(data[i] for i in part.inputs), *outputs, parameters_data, *parameters_data.stride(), 62, 256, *part.scalars]
        part.kernel.function[(1,1,1)](*arguments, num_warps=4)
        storage.update((v.root,t) for v,t in zip(part.materialized,outputs,strict=True))
    results.append(tuple(storage[v.root] for v in graph.outputs))
assert all(torch.equal(a,b) for a,b in zip(*results,strict=True))
# A carried MAC emits a runtime loop and loads distinct ciphertexts in place.
from fhelium.backend.triton._repeat import graph_nodes
carried_block = Block(arg_types=(ct, *((ct,) * terms), *((plain,) * terms), rns.RnsParametersType()))
body = Block(arg_types=(IndexType(), rns.RnsParametersType(), ct, plain, ct))
product = rns.MultiplyPlaintextOp(body.args[2], body.args[3], body.args[1], ct)
total = rns.AddStandardOp(body.args[4], product, body.args[1], ct)
body.add_ops((product, total, ckks.RepeatYieldOp((total.result,))))
loop = ckks.RepeatOp((carried_block.args[-1],), tuple(v for j in range(terms) for v in
    (carried_block.args[1+j], carried_block.args[1+terms+j])), (carried_block.args[0],),
    (ct,), body, count=terms, yield_count=0)
carried_block.add_ops((loop, fusion.YieldOp(loop.outputs)))
outer = Block(arg_types=tuple(v.type for v in carried_block.args))
carried_region = fusion.FusedOp(outer.args, (ct,), carried_block)
owner = torch.randint(0,q,(terms,1,n),dtype=torch.int64,generator=generator)
carried_data = tuple(torch.randint(0,q,(2,1,n),dtype=torch.int64,generator=generator) for _ in range(1+terms)) + tuple(owner[j] for j in range(terms))
carried_layouts = tuple(InputLayout(tuple(t.shape),tuple(t.stride()),i<=terms,
    owner=1+terms if i>terms else None, offset=(i-1-terms)*n if i>terms else 0)
    for i,t in enumerate(carried_data))
carried_answers = []
for compact_mac in (False, True):
    expression = ExpressionGraph(carried_region)
    if not compact_mac:
        expression.__dict__["nodes"] = tuple(graph_nodes(carried_block.ops))
    graph = expression.bind(carried_layouts)
    plan = PointwisePlan.build(graph,kernel_budget=None)
    import inspect
    source = inspect.getsource(plan.kernel.function.fn)
    if compact_mac:
        assert source.count("tl.range(") == 2
        assert kernel_cost(source)[0] == 2
        assert len([i for i in plan.inputs if i>terms]) == 1
    target = torch.empty((2,1,n),dtype=torch.int64)
    args = [*(carried_data[i] for i in plan.inputs),target,parameters_data,*parameters_data.stride(),62,256,*plan.scalars]
    plan.kernel.function[(1,1,1)](*args,num_warps=4)
    carried_answers.append(target)
assert torch.equal(*carried_answers)
# The inverse transform consumes the completed carry, never a partial sum.
from fhelium.backend.triton._pointwise import TritonFusionImplementation
ntt_block = Block(arg_types=tuple(v.type for v in carried_block.args) + (plain,))
mapper = dict(zip(carried_block.args,ntt_block.args[:-1],strict=True))
ntt_loop = loop.clone(value_mapper=mapper)
inverse = ntt.NttMontgomeryToCoefficientStandardOp(ntt_loop.outputs[0],ntt_block.args[-2],output,tables=(ntt_block.args[-1],))
ntt_block.add_ops((ntt_loop,inverse,fusion.YieldOp((inverse.result,))))
assert TritonFusionImplementation().match_fusion((ntt_loop,)) is not None
ntt_outer = Block(arg_types=tuple(v.type for v in ntt_block.args))
ntt_region = fusion.FusedOp(ntt_outer.args,(output,),ntt_block)
ntt_answers = []
for compact_mac in (False,True):
    expression = ExpressionGraph(ntt_region)
    if not compact_mac:
        expression.__dict__["nodes"] = tuple(graph_nodes(ntt_block.ops))
    graph = expression.bind(carried_layouts)
    step = _Step((_Endpoint(0,0,4,True,True,True),))
    schedule = (NttStages(None,2,((0,4,True),)),)
    kernel = _source(graph,schedule,step,((graph.outputs[0],),),())
    target = torch.empty((2,1,n),dtype=torch.int64)
    args = [*(carried_data[i] for i in kernel.inputs),target,twiddle_data,*twiddle_data.stride(),parameters_data,*parameters_data.stride(),62,n,128,*kernel.scalars]
    compile_endpoint(kernel.source)[(1,1,1)](*args,num_warps=4)
    ntt_answers.append(target)
assert torch.equal(*ntt_answers)
print("repeat and bounded-kernel residues equal")
'''


def test_repeat_preserves_unrolled_endpoint_residues():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES=""
    )
    result = subprocess.run(
        [sys.executable, "-c", _INTERPRETER, str(root)],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "repeat and bounded-kernel residues equal" in result.stdout
