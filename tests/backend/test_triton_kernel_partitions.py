"""Bounded Triton kernels and weighted-sum loops preserve unrolled integer residues."""

from __future__ import annotations

_INTERPRETER = r'''
import sys
import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Block
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__]
sys.path.insert(0, sys.argv[1])
from fhelium.ir.dialects import fusion, ntt, rns
from fhelium.backend.ntt._stages import NttStages
from fhelium.backend.triton._expressions import ExpressionGraph, InputLayout
from fhelium.backend.triton._ntt import _Endpoint, _Step, _source, _plan_steps
from fhelium.backend.triton._pointwise import PointwisePlan
from fhelium.backend.triton._budget import DEFAULT_KERNEL_BUDGET, kernel_cost
from fhelium.backend.triton._ntt_codegen import compile_endpoint

n, groups, terms = 16, 8, 24

def bundle(components=None, domain="ntt", rows=None):
    shape = (*((components,) if components is not None else ()), *((rows,) if rows is not None else ()), 1, n)
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

# Splitting endpoints under a smaller kernel budget preserves every output.
expression = ExpressionGraph(region)
data_indices, _ = expression.input_indices
data = tuple(operands[i] for i in data_indices)
layouts = tuple(InputLayout(tuple(t.shape), tuple(t.stride()), i <= groups) for i,t in zip(data_indices,data,strict=True))
graph = expression.bind(layouts)
step = _Step(tuple(_Endpoint(i, 0, 4, True, True, True) for i in range(groups)))
schedules = (NttStages(None, ((0, 4, True),)),) * groups
stores = tuple((value,) for value in graph.outputs)
kernel = _source(graph, schedules, step, stores, (), 2)
outputs = tuple(torch.empty((2,1,n),dtype=torch.int64) for _ in range(groups))
arguments = [*(data[i] for i in kernel.inputs), *outputs]
for _ in range(groups):
    arguments.extend((twiddle_data, *twiddle_data.stride()))
arguments.extend((parameters_data, *parameters_data.stride(), 62, n, 128, *kernel.scalars))
compile_endpoint(kernel.source)[(1,1,1)](*arguments, num_warps=4)
unchanged = _plan_steps(graph, schedules, (step,), stores, graph.outputs, kernel_cost(kernel.source), 2)
assert unchanged[0] == (step,) and unchanged[4][0].source == kernel.source
steps, writes, required, prior, kernels = _plan_steps(graph, schedules, (step,), stores, graph.outputs, (256, 1536, 12000), 2)
assert len(steps) > 1 and not any(writes) and not any(prior)
bounded = tuple(torch.empty_like(t) for t in outputs)
for part, emitted in zip(steps, kernels, strict=True):
    arguments = [*(data[i] for i in emitted.inputs), *(bounded[e.anchor] for e in part.endpoints)]
    for _ in part.endpoints:
        arguments.extend((twiddle_data, *twiddle_data.stride()))
    arguments.extend((parameters_data, *parameters_data.stride(), 62, n, 128, *emitted.scalars))
    compile_endpoint(emitted.source)[(1,1,1)](*arguments, num_warps=4)
assert all(torch.equal(a,b) for a,b in zip(outputs,bounded,strict=True))
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
# A weighted sum emits a runtime loop over its terms and loads distinct ciphertexts in place.
import inspect
def weighted(loop, inverse=False):
    plains = (bundle(rows=terms),) if loop == "stacked" else (plain,) * terms
    arguments = (*((ct,) * terms), *plains, rns.RnsParametersType(), *((plain,) if inverse else ()))
    block = Block(arg_types=arguments)
    params = block.args[terms + len(plains)]
    if loop == "stacked":
        value = rns.MontgomeryWeightedSumOp(params, block.args[:terms], (block.args[terms],), ct, plaintext_term_axis=0)
        block.add_op(value)
        value = value.result
    elif loop:
        value = rns.MontgomeryWeightedSumOp(params, block.args[:terms], block.args[terms:2 * terms], ct)
        block.add_op(value)
        value = value.result
    else:
        value = None
        for j in range(terms):
            product = rns.MultiplyPlaintextOp(block.args[j], block.args[terms + j], params, ct)
            block.add_op(product)
            if value is None:
                value = product.result
            else:
                total = rns.AddStandardOp(value, product, params, ct)
                block.add_op(total)
                value = total.result
    if inverse:
        transform = ntt.NttMontgomeryToCoefficientStandardOp(value, params, output, tables=(block.args[-1],))
        block.add_op(transform)
        value = transform.result
    block.add_op(fusion.YieldOp((value,)))
    outer = Block(arg_types=arguments)
    return fusion.FusedOp(outer.args, (output if inverse else ct,), block)
owner = torch.randint(0,q,(terms,1,n),dtype=torch.int64,generator=generator)
weighted_data = tuple(torch.randint(0,q,(2,1,n),dtype=torch.int64,generator=generator) for _ in range(terms)) + tuple(owner[j] for j in range(terms))
weighted_layouts = tuple(InputLayout(tuple(t.shape),tuple(t.stride()),i<terms,
    owner=terms if i>=terms else None, offset=(i-terms)*n if i>=terms else 0)
    for i,t in enumerate(weighted_data))
stacked_data = weighted_data[:terms] + (owner,)
stacked_layouts = weighted_layouts[:terms] + (InputLayout(tuple(owner.shape), tuple(owner.stride()), False),)
weighted_answers = []
for loop in (False, True, "stacked"):
    data, layouts = (stacked_data, stacked_layouts) if loop == "stacked" else (weighted_data, weighted_layouts)
    graph = ExpressionGraph(weighted(loop)).bind(layouts)
    plan = PointwisePlan.build(graph,kernel_budget=None)
    source = inspect.getsource(plan.kernel.function.fn)
    if loop:
        assert source.count("tl.range(") == 2
        # Each component multiplies its first term once and the loop body once.
        assert kernel_cost(source)[0] == 4
        assert len([i for i in plan.inputs if i>=terms]) == 1
    target = torch.empty((2,1,n),dtype=torch.int64)
    args = [*(data[i] for i in plan.inputs),target,parameters_data,*parameters_data.stride(),62,256,*plan.scalars]
    plan.kernel.function[(1,1,1)](*args,num_warps=4)
    weighted_answers.append(target)
assert all(torch.equal(weighted_answers[0], a) for a in weighted_answers[1:])
# The inverse transform consumes the completed sum, never a partial sum.
ntt_answers = []
for loop in (False,True,"stacked"):
    data, layouts = (stacked_data, stacked_layouts) if loop == "stacked" else (weighted_data, weighted_layouts)
    graph = ExpressionGraph(weighted(loop, True)).bind(layouts)
    step = _Step((_Endpoint(0,0,4,True,True,True),))
    schedule = (NttStages(None, ((0, 4, True),)),)
    kernel = _source(graph,schedule,step,((graph.outputs[0],),),(),2)
    target = torch.empty((2,1,n),dtype=torch.int64)
    args = [*(data[i] for i in kernel.inputs),target,twiddle_data,*twiddle_data.stride(),parameters_data,*parameters_data.stride(),62,n,128,*kernel.scalars]
    compile_endpoint(kernel.source)[(1,1,1)](*args,num_warps=4)
    ntt_answers.append(target)
assert all(torch.equal(ntt_answers[0], a) for a in ntt_answers[1:])
print("bounded-kernel and weighted-sum residues equal")
'''


def test_bounded_kernels_and_weighted_sums_preserve_unrolled_residues():
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
    assert "bounded-kernel and weighted-sum residues equal" in result.stdout
