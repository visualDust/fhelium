"""Grouped RNS products preserve component, group, and batch coordinates."""

import math

from typing import Any

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium.backend.rns.context import RnsContext
from fhelium.native.wrapper import rns_ops


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_grouped_products_match_independent_groups(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preset = (
        Preset.slots8192_scale25_depth14_int32
        if dtype == torch.int32
        else Preset.slots32768_scale40_depth34_int64
    )
    base = CkksConfig.parse(preset)
    config = CkksConfig.parse(
        base.dumps(),
        logN=8,
        q_depth_groups=base.q_depth_groups[:2],
        enforce_security_budget=False,
    )
    context = RnsContext(config, device=device)
    terms, groups, rows, coefficients = 8, 8, 2, 256
    for batch_shape in ((), (2, 3)):
        shape = (2, *batch_shape, rows, coefficients)
        values = torch.arange(
            math.prod(shape), dtype=dtype, device=device
        ).reshape(shape)
        ciphertexts = [values + 3 * term for term in range(terms)]
        parameters = context.rns_parameters_for(ciphertexts[0])
        for shared_plaintext in (True, False):
            plaintext_shape = (
                (1,) * len(batch_shape) + (rows, coefficients)
                if shared_plaintext
                else (*batch_shape, rows, coefficients)
            )
            weights = torch.arange(
                math.prod(plaintext_shape), dtype=dtype, device=device
            ).reshape(plaintext_shape)
            plaintexts = [
                weights + 11 * group + term + 1
                for group in range(groups)
                for term in range(terms)
            ]
            expected = torch.stack(
                [
                    rns_ops.montgomery_weighted_sum(
                        ciphertexts,
                        plaintexts[group * terms : (group + 1) * terms],
                        parameters,
                    )
                    for group in range(groups)
                ],
                dim=1,
            )
            actual = rns_ops.montgomery_weighted_sums(
                ciphertexts, plaintexts, groups, parameters
            )
            assert actual.shape == (2, groups, *batch_shape, rows, coefficients)
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_represented_weighted_sums_preserve_mixed_layouts_and_strides(
    device, dtype
):
    from fhelium.eager import Engine
    from fhelium.runtime.signature import ValueSignature
    from fhelium.values import Ciphertext, CompressedPlaintext, Plaintext

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preset = (
        Preset.slots8192_scale25_depth14_int32
        if dtype == torch.int32
        else Preset.slots8192_scale40_depth7_int64
    )
    base = CkksConfig.parse(preset)
    config = CkksConfig.parse(
        base.dumps(),
        logN=8,
        q_depth_groups=base.q_depth_groups[:2],
        enforce_security_budget=False,
    )
    engine = Engine(config)
    state: dict[str, Any] = dict(
        depth=0,
        scale=16.0,
        prime_ids=(0, 1),
        polynomial_domain="ntt",
        residue_representation="montgomery",
        modulus_basis="Q",
    )
    storage = torch.arange(2 * 2 * 2 * 512, dtype=dtype, device=device).reshape(
        2, 2, 2, 512
    )
    ciphertexts = [
        Ciphertext(data=(storage + i)[..., ::2], **state) for i in range(4)
    ]
    rows = []
    for group in range(2):
        compact = (
            torch.arange(2 * 2 * 16, dtype=dtype, device=device).reshape(
                2, 2, 16
            )
            + 7
            + group
        )[..., ::2]
        implicit = (
            torch.arange(8, dtype=dtype, device=device).reshape(2, 4)
            + 3
            + group
        )[:, ::2]
        rows.append(
            [
                Plaintext(
                    message=None,
                    data=torch.ones(2, 2, 256, dtype=dtype, device=device)
                    * (group + 11),
                    representation="rns",
                    **state,
                ),
                CompressedPlaintext(
                    data=compact,
                    ring_dimension=256,
                    compression_layout="cyclic",
                    **state,
                ),
                CompressedPlaintext(
                    data=compact,
                    ring_dimension=256,
                    compression_layout="contiguous",
                    **state,
                ),
                CompressedPlaintext(
                    data=compact,
                    implicit_data=implicit,
                    ring_dimension=256,
                    compression_layout="strided_sparse",
                    **state,
                ),
            ]
        )
    expected_rows = [
        engine.sum_ciphertexts(
            [
                engine.multiply_plaintext(c, p)
                for c, p in zip(ciphertexts, row, strict=True)
            ]
        )
        for row in rows
    ]
    expected = Ciphertext.stack_batch(expected_rows)
    selected = engine.plaintext_product_implementation(
        device, compressed=True, implementation="auto"
    )
    implementations = ["native-montgomery-weighted-sum", selected]
    for implementation in dict.fromkeys(implementations):
        actual = engine.sum_plaintext_product_groups(
            ciphertexts, rows, implementation=implementation, term_chunk_size=3
        )
        assert torch.equal(actual.data, expected.data)
        assert ValueSignature.from_value(actual) == ValueSignature.from_value(
            expected
        )
        single = engine.sum_plaintext_products(
            ciphertexts,
            rows[0],
            implementation=implementation,
            term_chunk_size=3,
        )
        assert torch.equal(single.data, expected_rows[0].data)
        # Numerical materials are live operands, not frozen inside the plan.
        rows[0][1].data.add_(1)
        changed = engine.sum_plaintext_products(
            ciphertexts,
            rows[0],
            implementation=implementation,
            term_chunk_size=3,
        )
        oracle = engine.sum_ciphertexts(
            [
                engine.multiply_plaintext(c, p)
                for c, p in zip(ciphertexts, rows[0], strict=True)
            ]
        )
        assert torch.equal(changed.data, oracle.data)
        rows[0][1].data.sub_(1)

        # A new parameter-table row stride changes the launch ABI, not its values.
        from fhelium.backend.implementation import OperationInvocation
        from fhelium.ir.dialects import rns

        dispatcher = engine._dispatcher_for(device)
        parameters = dispatcher.rns_context.rns_parameters_for_prime_ids((0, 1))
        padded = torch.zeros(
            parameters.size(0),
            parameters.stride(0) + 1,
            dtype=parameters.dtype,
            device=parameters.device,
        )
        rebound = padded[:, : parameters.size(1)]
        rebound.copy_(parameters)
        plaintexts = [p for row in rows for p in row]
        layouts = tuple(
            p.compression_layout
            if isinstance(p, CompressedPlaintext)
            else "dense"
            for p in plaintexts
        )
        implicit_rows = [
            p.implicit_data
            for p in plaintexts
            if isinstance(p, CompressedPlaintext)
            and p.compression_layout == "strided_sparse"
        ]
        operands = tuple(
            [c.data for c in ciphertexts]
            + [p.data for p in plaintexts]
            + implicit_rows
            + [rebound]
        )
        invocation = OperationInvocation._from_eager(
            rns.MontgomeryWeightedSumsOp,
            len(operands),
            1,
            {
                "term_count": 4,
                "group_count": 2,
                "plaintext_layouts": layouts,
                "term_chunk_size": 3,
            },
        )
        consumer = dispatcher.backend.registry.resolve_type(
            rns.MontgomeryWeightedSumsOp,
            requested=implementation,
            in_place=False,
        )
        rebound_result = consumer.execute(
            invocation, operands, (), in_place=False
        )[0]
        assert torch.equal(rebound_result, expected.data)
