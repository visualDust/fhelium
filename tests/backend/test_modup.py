"""Hybrid digit extension preserves the represented integer across widths."""

import math

import pytest
import torch

from fhelium import CkksConfig, Preset
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.modup import modup_digit


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_modup_digit_matches_integer_reference_across_workspace_sizes(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    base = CkksConfig.parse(Preset.slots32768_scale40_depth34_int64)
    for width in (16, 17, 18):
        config = CkksConfig.parse(
            base.dumps(),
            q_depth_groups=base.q_depth_groups[:width],
            p_moduli=(*base.q_moduli[width:], *base.p_moduli),
        )
        context = RnsContext(config, device=device)
        ids = tuple(range(width))
        assert context.rns_layout.digit_rows(0) == (ids,)
        parameters = context.row_parameters(ids)
        qp_ids = context.rns_layout.prime_ids(0, include_p=True)
        modulus = math.prod(config.q_moduli)
        # Include negative centered integers and lazy input representatives.
        integers = [0, 1, modulus - 1, modulus // 2, 10**100]
        integers += [modulus - i**17 for i in range(1, 28)]
        source = torch.tensor(
            [
                [
                    value % q + (q if j % 2 else 0)
                    for j, value in enumerate(integers)
                ]
                for q in config.q_moduli
            ],
            dtype=context.dtype,
            device=device,
        ).unsqueeze(0)
        original = source.clone()
        tables = {
            "qp_parameters": context.rns_parameters_for_prime_ids(qp_ids),
            "digit0_normalizers": parameters.mixed_radix_normalizers,
            "digit0_propagation": parameters.mixed_radix_propagation_coefficients,
            "digit0_extension": parameters.basis_extension_coefficients,
            **{
                f"digit0_reduction{i}": tensor
                for i, tensor in enumerate(
                    parameters.montgomery_reduction_parameters
                )
            },
        }
        result = modup_digit(source, tables, 0, (0, width, 0))
        radix = context.montgomery_parameters.R
        expected = torch.tensor(
            [
                [(value * radix) % config.moduli[i] for value in integers]
                for i in qp_ids
            ],
            dtype=context.dtype,
            device=device,
        ).unsqueeze(0)
        primes = tables["qp_parameters"][0] // 2
        assert torch.equal(result % primes[:, None], expected)
        assert torch.equal(source, original)
