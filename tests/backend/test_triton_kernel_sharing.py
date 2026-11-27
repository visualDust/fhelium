"""Generated Triton kernels share source across depths and physical layouts."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Block

from fhelium.backend.triton._expressions import (
    ExpressionGraph,
    InputLayout,
)
from fhelium.backend.triton._pointwise import PointwisePlan
from fhelium.ir.dialects import fusion, rns

N = 256


def _bundle(shape: tuple[int, ...]) -> rns.RnsBundleType:
    return rns.RnsBundleType(
        {
            "shape": ArrayAttr(IntegerAttr(x, 64) for x in shape),
            "device": StringAttr("cuda:0"),
            "dtype": StringAttr("torch.int64"),
        }
    )


def _region(rows: int) -> fusion.FusedOp:
    value = _bundle((rows, N))
    block = Block(arg_types=(value, value, rns.RnsParametersType()))
    x, y, parameters = block.args
    total = rns.AddStandardOp(x, y, parameters, value)
    product = rns.MontgomeryMultiplyOp(total, x, parameters, value)
    block.add_ops((total, product, fusion.YieldOp((product,))))
    outer = Block(arg_types=tuple(arg.type for arg in block.args))
    return fusion.FusedOp(outer.args, (value,), block)


def _plan(rows: int, strides: tuple[int, int]) -> PointwisePlan:
    layouts = (InputLayout((rows, N), strides), InputLayout((rows, N), strides))
    return PointwisePlan.build(ExpressionGraph(_region(rows)).bind(layouts))


def test_kernel_shares_source_across_rows_and_strides():
    depth_a = _plan(4, (N, 1))
    depth_b = _plan(6, (2 * N, 2))
    assert depth_a.kernel.function is depth_b.kernel.function
    assert depth_a.scalars != depth_b.scalars
    assert depth_a.inputs == depth_b.inputs == (0, 1)


_INTERPRETER_CHECK = """
import sys
import torch
# Import this checkout even when an editable install of another checkout is active.
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__]
sys.path.insert(0, sys.argv[1])
from tests.backend.test_triton_kernel_sharing import N, _plan

moduli = (1099510054913, 1099515691009)
radix = 1 << 62
rows = []
for q in moduli:
    k = (-pow(q, -1, radix)) % radix
    rows.append((2 * q, q & 0xFFFFFFFF, q >> 32, k & 0x7FFFFFFF, k >> 31, radix * radix % q, 0, 0))
parameters = torch.tensor(rows, dtype=torch.int64).T.contiguous()
generator = torch.Generator().manual_seed(0)
x = torch.stack([torch.randint(0, q, (N,), dtype=torch.int64, generator=generator) for q in moduli])
y = torch.stack([torch.randint(0, q, (N,), dtype=torch.int64, generator=generator) for q in moduli])
wide = torch.zeros((2, 2, 2 * N), dtype=torch.int64)
wide[0, :, ::2] = x
wide[1, :, ::2] = y
strided = wide[0, :, ::2], wide[1, :, ::2]
plan = _plan(2, tuple(strided[0].stride()))
out = torch.empty((2, N), dtype=torch.int64)
plan.kernel.function[plan.kernel.grid](
    *(strided[i] for i in plan.inputs), out, parameters,
    parameters.stride(0), parameters.stride(1), 62, 256, *plan.scalars,
    num_warps=plan.kernel.num_warps, **plan.kernel.compile_options,
)
for i, q in enumerate(moduli):
    expected = torch.tensor(
        [(int(x[i, j]) + int(y[i, j])) % q * int(x[i, j]) * pow(radix, -1, q) % q for j in range(N)],
        dtype=torch.int64,
    )
    assert torch.equal(out[i] % q, expected), i
print("ok")
"""


def test_interpreter_matches_modular_reference():
    """Check strided inputs against modular arithmetic on CPU.

    The interpreter is selected when ``triton.jit`` decorates the arithmetic
    helpers, so the check runs in a fresh process importing this checkout.
    """
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    completed = subprocess.run(
        [sys.executable, "-P", "-c", _INTERPRETER_CHECK, root],
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_compact_implicit_load_broadcasts_scalar_pointer():
    """Reconstruct strided sparse values with uniform row loads and a lane mask."""
    import pytest
    import torch
    from fhelium.backend.triton._expressions import (
        Emitter,
        Expr,
        KernelArguments,
    )
    from fhelium.backend.triton._pointwise import compile_source

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    explicit = torch.tensor(
        [[3, 0, 5, 0, 7, 0, 11, 0], [13, 0, 17, 0, 19, 0, 23, 0]],
        dtype=torch.int64,
        device="cuda:0",
    )[:, ::2]
    implicit = torch.tensor(
        [[101, 0], [211, 0]], dtype=torch.int64, device="cuda:0"
    )[:, :1]
    inputs = (explicit, implicit)
    layouts = tuple(
        InputLayout(tuple(x.shape), tuple(x.stride())) for x in inputs
    )
    arguments = KernelArguments(layouts)
    emitter = Emitter(layouts, arguments)
    expression = Expr(
        "compact",
        (Expr("input", data=(0, 0)), Expr("input", data=(1, 0))),
        (2, 4, 4),
    )
    result = emitter.emit(expression, "lane", "row", "0")
    parameters = (
        *arguments.pointer_parameters,
        "out",
        "BLOCK: tl.constexpr",
        *arguments.scalar_parameters,
    )
    lines = (
        "row = tl.program_id(1).to(tl.int64)",
        "lane = tl.arange(0, BLOCK)",
        "mask = lane < 16",
        *emitter.lines,
        f"tl.store(out + row * 16 + lane, {result}, mask=mask)",
    )
    source = f"def fused({', '.join(parameters)}):\n" + "".join(
        f"    {line}\n" for line in lines
    )
    output = torch.empty((2, 16), dtype=torch.int64, device="cuda:0")
    compile_source(source)[(1, 2)](
        *(inputs[i] for i in arguments.inputs),
        output,
        32,
        *arguments.scalar_values,
    )
    expected = implicit.expand(2, 16).clone()
    expected[:, ::4] = explicit
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
