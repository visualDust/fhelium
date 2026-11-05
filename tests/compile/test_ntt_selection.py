"""NTT selection with partial choices and caller-supplied table data."""

from typing import cast

import pytest
import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import Preset
from fhelium.backend import OperationBackend
from fhelium.backend.ntt import NttContext
from fhelium.backend.rns.context import RnsContext
from fhelium.compile import Compilation, Pipeline, prepare_material_bindings
from fhelium.compile.passes.backend import (
    AssignNttImplementationPass,
    SelectNttImplementationsPass,
)
from fhelium.config import CkksConfig
from fhelium.ir import Program
from fhelium.ir.dialects import core, ntt, rns


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_ntt_selection_preserves_supplied_tables_and_open_device(
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    context = NttContext(RnsContext(config, device=device))
    ids = (2, 3)
    supplied = context.tensor_operands(ids, inverse=False)
    source = torch.randint(
        0, 65536, (2, len(ids), config.N), dtype=torch.int64, device=device
    )

    def integers(values):
        return ArrayAttr([IntegerAttr(x, 64) for x in values])

    typ = rns.RnsBundleType(
        {
            "shape": integers(source.shape),
            "strides": integers(source.stride()),
            "ring_dimension": IntegerAttr(config.N, 64),
            "prime_ids": integers(ids),
            "dtype": StringAttr("torch.int64"),
            "polynomial_domain": StringAttr("coefficient"),
            "residue_representation": StringAttr("standard"),
        }
    )
    transformed = typ.with_state(
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    block = Block(arg_types=(typ,))
    parameters = core.MaterialRefOp(
        rns.RnsParametersType(), symbol="parameters"
    )
    twiddles = core.MaterialRefOp(
        core.MessageType(), symbol="supplied-forward-table"
    )
    forward = ntt.CoefficientStandardToNttMontgomeryOp(
        block.args[0], parameters, transformed, tables=(twiddles,)
    )
    inverse = ntt.NttMontgomeryToCoefficientStandardOp(forward, parameters, typ)
    block.add_ops(
        (parameters, twiddles, forward, inverse, ReturnOp(forward, inverse))
    )
    compilation = Compilation(
        Program.from_function(block, (transformed, typ)),
        material_bindings={
            "parameters": supplied[0],
            "supplied-forward-table": supplied[1],
        },
    )
    compilation.workspace[CkksConfig] = config
    backend = OperationBackend()
    selection = SelectNttImplementationsPass(backend.registry)
    assert not selection.run(compilation).changed
    compilation = Pipeline(
        (
            AssignNttImplementationPass(
                algorithm="radix2_indexed"
                if device == "cpu"
                else "radix2_compact",
                group_width=None if device == "cpu" else 16,
            ),
            selection,
        )
    ).run(compilation)
    assert prepare_material_bindings(compilation, resources=context) == ()
    assert (
        compilation.material_bindings["supplied-forward-table"] is supplied[1]
    )
    assert (
        "device"
        not in compilation.program.single_block("main").args[0].type.state.data
    )
    executable = backend.link(compilation)
    for _ in range(2):
        result, restored = executable.run(source)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)
        assert result.shape == source.shape
        source.add_(1)

    if device == "cpu":
        linked_forward = next(
            operation
            for operation in compilation.program.single_block("main").walk()
            if isinstance(operation, ntt.CoefficientStandardToNttMontgomeryOp)
        )
        index_ref = linked_forward.tables[1].owner
        assert isinstance(index_ref, core.MaterialRefOp)
        assert index_ref.symbol is not None
        symbol = index_ref.symbol.data
        original = compilation.material_bindings[symbol]
        invalid = original.clone()
        invalid[0, 0] = config.N
        compilation.material_bindings[symbol] = invalid
        with pytest.raises(
            RuntimeError, match="out-of-range coefficient index"
        ):
            backend.link(compilation)
        # A new dictionary binding does not change the already linked Tensor.
        torch.testing.assert_close(
            cast(tuple[torch.Tensor, ...], executable.run(source))[1],
            source,
            rtol=0,
            atol=0,
        )
        compilation.material_bindings[symbol] = original.clone()
        relinked = backend.link(compilation)
        torch.testing.assert_close(
            cast(tuple[torch.Tensor, ...], relinked.run(source))[1],
            source,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "policy",
    ["radix2_compact_group16_smem8", "radix2_compact_group16_shoup32_smem8"],
)
def test_ntt_fusion_preserves_surrounding_arithmetic_and_escaping_results(
    policy,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fhelium import compile as fc
    from fhelium.eager import Engine
    from fhelium.ir.dialects import fusion

    base = CkksConfig.parse(Preset.slots8192_scale25_depth14_int32)
    config = CkksConfig(
        logN=8,
        q_depth_groups=base.q_depth_groups[-3:],
        p_moduli=base.p_moduli,
        default_scale=base.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, ntt_backend=policy, rng_seed=913, rng_nonce=41)
    secret = engine.create_secret_key(device="cuda")
    public = engine.create_public_key(secret)
    message = torch.linspace(
        -0.02, 0.03, config.num_slots, device="cuda", dtype=torch.float64
    )
    value = engine.encrypt_message(message, public, device="cuda")
    original = value.data.clone()

    def source(x):
        added = engine.add(x, x)
        transformed = engine.coefficient_domain_to_ntt_domain(added)
        doubled = engine.add(transformed, transformed)
        restored = engine.ntt_domain_to_coefficient_domain(doubled)
        return engine.add(restored, x), transformed

    compiled = fc.compile(source)
    prepared = compiled.prepare(value)
    regions = [
        op
        for op in prepared.compilation.program.module.walk()
        if isinstance(op, fusion.FusedOp)
    ]
    assert any(
        isinstance(
            op,
            (
                ntt.CoefficientStandardToNttMontgomeryOp,
                ntt.NttMontgomeryToCoefficientStandardOp,
            ),
        )
        for region in regions
        for op in region.body.block.walk()
    )
    moduli = torch.tensor(
        [config.moduli[i] for i in value.prime_ids],
        device="cuda",
        dtype=torch.int64,
    ).reshape(1, -1, 1)
    for argument in (value, engine.negate(value)):
        actual = compiled(argument)
        expected = source(argument)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                result.data.to(torch.int64) % moduli,
                reference.data.to(torch.int64) % moduli,
                rtol=0,
                atol=0,
            )
            assert (
                result.depth,
                result.scale,
                result.prime_ids,
                result.polynomial_domain,
                result.residue_representation,
            ) == (
                reference.depth,
                reference.scale,
                reference.prime_ids,
                reference.polynomial_domain,
                reference.residue_representation,
            )
    torch.testing.assert_close(value.data, original, rtol=0, atol=0)
