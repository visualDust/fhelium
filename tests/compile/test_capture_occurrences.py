"""Python helper occurrences share structural eligibility with current materials."""

import json

import torch

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.backend import OperationBackend
from fhelium.compile._local_plans import Occurrence
from fhelium.compile._origins import get_origin
from fhelium.compile._preparation import _input_specialization
from fhelium.compile._template import StructuralTemplate, build_template
from fhelium.eager import Engine
from fhelium.ir import Program
from fhelium.ir.dialects import ckks


def test_handwritten_loop_helper_keeps_independent_current_material_ports():
    preset = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    config = CkksConfig(
        logN=8,
        q_depth_groups=tuple((prime,) for prime in preset.q_moduli[:3]),
        p_moduli=preset.p_moduli[:1],
        default_scale=preset.default_scale,
        enforce_security_budget=False,
    )
    engine = Engine(config, rng_seed=389)
    secret = engine.create_secret_key(device="cpu")
    public = engine.create_public_key(secret, device="cpu")
    clear = torch.linspace(-0.02, 0.03, config.num_slots, dtype=torch.float64)
    value = engine.coefficient_domain_to_ntt_domain(
        engine.encrypt_message(clear, public, device="cpu")
    )
    weights = [
        engine.prepare_plaintext_for_multiplication(
            engine.encode(torch.full_like(clear, scalar), device="cpu")
        )
        for scalar in (0.125, 0.25, 0.5)
    ]

    def term(current, weight):
        return engine.multiply_plaintext(current, weight)

    def evaluator(current):
        total = term(current, weights[0])
        for weight in weights[1:]:
            total = engine.add(total, term(current, weight))
        return total

    captured = fc.capture_eager(
        evaluator,
        arguments={"current": value},
        workspace=fc.CompileWorkspace(
            {"structural_template_recording": True, "ckks_repeat": False}
        ),
        material_names={
            f"weight{i}": weight.data for i, weight in enumerate(weights)
        },
    )
    disabled = fc.capture_eager(
        evaluator,
        arguments={"current": value},
        workspace=fc.CompileWorkspace(
            {"structural_template_recording": True, "local_plan_reuse": False}
        ),
    )
    assert build_template(disabled) is not None
    assert all(
        "origin" not in record
        for record in disabled.workspace["structural_template_metadata"][
            "operations"
        ]
    )
    template = build_template(captured)
    assert template is not None
    template = StructuralTemplate(
        Program.parse(template.program.to_text()),
        json.loads(json.dumps(template.metadata)),
    )
    instance = template.instantiate(
        {"current": value},
        material_bindings={
            f"weight{i}": weight.data for i, weight in enumerate(weights)
        },
        leaf_states={f"weight{i}": weight for i, weight in enumerate(weights)},
    )
    assert instance is not None
    specialized = _input_specialization(instance, {"current": value})
    products = [
        op
        for op in specialized.program.walk()
        if isinstance(op, ckks.MultiplyPlaintextOp)
    ]
    origins = [get_origin(op, specialized.workspace) for op in products]
    assert len(products) == 3
    assert all(
        origin is not None and origin["source"] == "capture"
        for origin in origins
    )
    assert origins[0]["site"] == origins[1]["site"] == origins[2]["site"]
    assert len({origin["occurrence"] for origin in origins}) == 3
    occurrences = [
        Occurrence.from_operations((op,), origin=origin)
        for op, origin in zip(products, origins, strict=True)
    ]
    assert (
        occurrences[0].port_graph
        == occurrences[1].port_graph
        == occurrences[2].port_graph
    )
    assert len({op.plaintext for op in products}) == 3
    for occurrence, product in zip(occurrences, products, strict=True):
        assert product.plaintext in occurrence.inputs
    # The shared eligibility mechanism also accepts a location without Python provenance.
    assert (
        Occurrence.from_operations((products[0],), origin=None).port_graph
        == occurrences[0].port_graph
    )

    backend = OperationBackend()
    optimized = fc.default_lower_and_fuse_pipeline(backend).run(specialized)
    resources = engine._dispatcher_for(torch.device("cpu")).device_resources
    assert fc.prepare_material_bindings(optimized, resources=resources) == ()
    result = backend.link(optimized).run(value)
    expected = evaluator(value)
    moduli = torch.tensor(
        [config.moduli[i] for i in value.prime_ids], dtype=torch.int64
    ).view(-1, 1)
    assert torch.equal(result.data % moduli, expected.data % moduli)
