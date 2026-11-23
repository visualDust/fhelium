"""Captured loop bodies preserve expanded SSA and eager numerical results."""

import json

import pytest
import torch
from xdsl.dialects.builtin import IntegerAttr

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend, OperationImplementationRegistry
from fhelium.compile._template import StructuralTemplate, build_template
from fhelium.eager import Engine
from fhelium.ir import Program
from fhelium.ir.dialects import ckks


@pytest.mark.parametrize("with_rotation", (False, True))
def test_handwritten_repeat_expands_to_the_same_program_and_residues(
    with_rotation,
):
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=tuple((prime,) for prime in preset.q_moduli[:3]),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=417)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    value = engine.encrypt_message(clear, public, device="cpu")
    weights = [
        [
            engine.prepare_plaintext_for_multiplication(
                engine.encode(
                    torch.full_like(clear, 0.125 * (1 + group + term)),
                    device="cpu",
                )
            )
            for term in range(3)
        ]
        for group in range(3)
    ]

    keys = (
        [
            engine.create_rotation_key(step, secret, device="cpu")
            for step in (1, 2)
        ]
        if with_rotation
        else []
    )

    def weighted(current, weight):
        return engine.multiply_plaintext(current, weight)

    def evaluator(current):
        transformed = engine.coefficient_domain_to_ntt_domain(current)
        groups = []
        for index, row in enumerate(weights):
            partial = weighted(transformed, row[0])
            for weight in row[1:]:
                partial = engine.add(partial, weighted(transformed, weight))
            partial = engine.ntt_domain_to_coefficient_domain(partial)
            if with_rotation and index:
                partial = engine.rotate_with_key(partial, keys[index - 1])
            groups.append(partial)
        total = groups[0]
        for partial in groups[1:]:
            total = engine.add(total, partial)
        return total

    backend = OperationBackend()
    expected = evaluator(value)
    texts = []
    for enabled in (False, True):
        workspace = fc.CompileWorkspace(
            {
                "structural_template_recording": True,
                "ckks_repeat": enabled,
                OperationImplementationRegistry: backend.registry,
            }
        )
        source = fc.capture_eager(
            evaluator, arguments={"current": value}, workspace=workspace
        )
        assert (
            any(
                isinstance(operation, ckks.RepeatOp)
                for operation in source.program.walk()
            )
            is enabled
        )
        if enabled and with_rotation:
            rotation_loops = [
                op
                for op in source.program.walk()
                if isinstance(op, ckks.RepeatOp)
                and "rotation_steps" in op.attributes
            ]
            assert len(rotation_loops) == 1
            assert tuple(
                rotation_loops[0].attributes["rotation_steps"].data
            ) == tuple(
                IntegerAttr(
                    source.program.material_description(key.owner.symbol.data)[
                        "rotation_step"
                    ],
                    64,
                )
                for key in (rotation_loops[0].row(i)[1] for i in range(2))
            )
        template = build_template(source)
        assert template is not None
        template = StructuralTemplate(
            Program.parse(template.program.to_text()),
            json.loads(json.dumps(template.metadata)),
        )
        owners = {id(weight.data): weight for row in weights for weight in row}
        leaves = {
            symbol: owners[id(tensor)]
            for symbol, tensor in source.material_bindings.items()
            if id(tensor) in owners
        }
        instance = template.instantiate(
            {"current": value},
            material_bindings=source.material_bindings,
            leaf_states=leaves,
            workspace=workspace,
        )
        assert instance is not None
        optimized = fc.default_lower_and_fuse_pipeline(backend).run(instance)
        if enabled and with_rotation:
            assert any(
                isinstance(op, ckks.RepeatOp)
                and "rotation_steps" in op.attributes
                for op in optimized.program.walk()
            )
        assert (
            fc.prepare_material_bindings(
                optimized,
                resources=engine._dispatcher_for(
                    torch.device("cpu")
                ).device_resources,
            )
            == ()
        )
        executable = backend.link(optimized)
        assert not any(
            isinstance(op, ckks.RepeatOp) for op in executable.program.walk()
        )
        texts.append(executable.program.to_text())
        actual = executable.run(value)
        moduli = torch.tensor(
            [config.moduli[i] for i in expected.prime_ids], dtype=torch.int64
        ).view(-1, 1)
        assert torch.equal(actual.data % moduli, expected.data % moduli)
    if not with_rotation:
        assert texts[0] == texts[1]


def test_batched_weighted_sum_captures_one_carried_loop():
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=tuple((prime,) for prime in preset.q_moduli[:3]),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=523)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    value = engine.encrypt_message(clear, public, device="cpu")
    diagonals = torch.tensor(
        [[0.5, -0.25, 0.125, 0.75], [0.25, 0.5, -0.5, 0.125]] * 2,
        dtype=torch.float64,
    )
    factors = (1.0, -1.0, 2.0, 1.0)

    def terms(current):
        transformed = engine.coefficient_domain_to_ntt_domain(current)
        negated = engine.negate(transformed)
        doubled = engine.add(transformed, transformed)
        return [transformed, negated, doubled, transformed]

    def source(current, weights):
        plains = engine.prepare_compressed_plaintext(
            weights, depth=current.depth, scale=current.scale, device="cpu"
        )
        total = engine.sum_plaintext_products(terms(current), plains)
        return engine.ntt_domain_to_coefficient_domain(total)

    plains = engine.prepare_compressed_plaintext(
        diagonals, depth=value.depth, scale=value.scale, device="cpu"
    )
    batched = engine.sum_plaintext_products(terms(value), plains)
    listed = engine.sum_plaintext_products(terms(value), plains.unbind_batch())
    assert torch.equal(batched.data, listed.data)

    captured = fc.capture_eager(
        lambda current: engine.sum_plaintext_products(terms(current), plains),
        arguments={"current": value},
    )
    loops = [
        op for op in captured.program.walk() if isinstance(op, ckks.RepeatOp)
    ]
    assert len(loops) == 1
    assert loops[0].count.value.data == 3
    assert len(loops[0].carried) == 1 and len(loops[0].outputs) == 1
    # Later plaintext rows are read from the one batched operand by round index.
    rows = [
        op
        for op in loops[0].body.block.ops
        if isinstance(op, ckks.SelectBatchOp)
    ]
    assert len(rows) == 1 and rows[0].index is loops[0].body.block.args[0]
    assert loops[0].table_width.value.data == 1
    assert not any(
        isinstance(op, ckks.RepeatIterationOp) for op in captured.program.walk()
    )

    compiled = fc.compile(source, cache=False, graph=False)
    weighted = sum(
        factor * row.repeat(config.num_slots // row.numel())
        for factor, row in zip(factors, diagonals, strict=True)
    )
    for weights in (diagonals, diagonals * 0.5):
        actual = engine.decrypt_message(compiled(value, weights), secret)
        scale = 1.0 if weights is diagonals else 0.5
        torch.testing.assert_close(
            actual.real, clear * weighted * scale, rtol=0, atol=1e-8
        )
