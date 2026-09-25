"""Prepare periodic CKKS messages using their sparse polynomial embedding."""

from __future__ import annotations

import torch

from fhelium.backend.ntt.plans.twiddles import _primitive_2nth_root
from fhelium.backend.rns.context import lift_integer_coefficients_exact
from fhelium.native.wrapper import ntt_ops, rns_ops
from fhelium.rng.csprng import stochastic_round_

from ._embedding import _twister, inverse_embed_slots


def periodic_extent(period: int, ring_dimension: int, generator: int) -> int:
    """Return the coefficient count in the embedded negacyclic subring."""
    if period < 1 or period & (period - 1) or ring_dimension // 2 % period:
        raise ValueError(
            "Periodic slot count must be a power of two dividing N/2"
        )
    # Generator 3 alternates the two conjugacy classes modulo four.
    extent = max(4 if generator == 3 else 2, 2 * period)
    if extent >= ring_dimension:
        raise ValueError(
            "The periodic message must admit a shorter encoded axis"
        )
    return extent


def periodic_materials(
    config, context, period: int, depth: int, basis: str, state
):
    """Build compact embedding and indexed NTT tables with the full-ring roots."""
    n = config.N
    u = periodic_extent(period, n, config.galois_generator)
    device = context.device
    ids = context.rns_layout.prime_ids(depth, include_p=basis == "QP")
    moduli = tuple(config.moduli[i] for i in ids)
    parameters = context.rns_parameters_for_prime_ids(ids)
    step = 5 if config.galois_generator == 5 else pow(3, -1, 2 * u)
    pre = torch.tensor(
        [(pow(step, i, 2 * u) - 1) // 2 for i in range(u // 2)],
        dtype=torch.int64,
        device=device,
    )
    even, odd, twists = [], [], []
    for stage in range(u.bit_length() - 1):
        groups = 1 << stage
        span = u // (2 * groups)
        indices = [
            2 * group * span + lane
            for group in range(groups)
            for lane in range(span)
        ]
        even.append(indices)
        odd.append([index + span for index in indices])
        twists.append(
            [groups + group for group in range(groups) for _ in range(span)]
        )
    log_u = u.bit_length() - 1
    reversed_indices = [int(f"{i:0{log_u}b}"[::-1], 2) for i in range(u)]
    rows = []
    for q in moduli:
        root = pow(_primitive_2nth_root(q, n), n // u, q)
        rows.append([pow(root, exponent, q) for exponent in reversed_indices])
    compact = torch.tensor(rows, dtype=context.dtype, device=device)
    rns_ops.to_montgomery_(compact, parameters)
    twiddle_indices = torch.tensor(twists, dtype=torch.long, device=device)
    twiddles = compact[:, twiddle_indices].contiguous()
    return (
        pre,
        _twister(u, str(device)),
        state,
        parameters,
        torch.tensor(
            [2 * q for q in moduli], dtype=context.dtype, device=device
        ),
        torch.tensor(even, dtype=torch.int32, device=device),
        torch.tensor(odd, dtype=torch.int32, device=device),
        twiddles,
    )


def prepare_periodic_tensor(
    message,
    pre,
    twister,
    state,
    parameters,
    twice_moduli,
    even,
    odd,
    twiddles,
    *,
    ring_dimension,
    scale,
    min_modulus,
    polynomial_domain="ntt",
):
    """Compute compact NTT rows for p(X)=a(X**(N/U)), with a of degree below U.

    The U-point inverse embedding uses the original generator's slot order.
    Rounding selects word j*N/U of the full coefficient stream and advances
    by N words per batch item. NTT roots are psi_N**(N/U), so each returned
    bit-reversed NTT entry represents N/U contiguous entries of the full NTT.
    FFT roundoff can differ from a full-size FFT near rounding thresholds.
    """
    slots = (
        message
        if message.size(-1) == pre.numel()
        else message.repeat_interleave(pre.numel(), dim=-1)
    )
    coefficients = inverse_embed_slots(slots, pre=pre, twister=twister)
    rounded = stochastic_round_(
        coefficients * float(scale),
        state,
        word_stride=ring_dimension // coefficients.size(-1),
    )
    # The int64 bound selects general integer remainder without reading a
    # device reduction back to the host; every rounded value is covered.
    residues = lift_integer_coefficients_exact(
        rounded, twice_moduli, min_modulus=min_modulus, max_abs=1 << 63
    )
    if polynomial_domain == "coefficient":
        rns_ops.to_montgomery_(residues, parameters)
        implicit = torch.zeros(
            (*residues.shape[:-1], 1),
            dtype=residues.dtype,
            device=residues.device,
        )
        return residues, implicit
    if residues.size(-1) < 256:
        from fhelium.backend.ntt.executors.indexed_radix2 import (
            forward_small_to_montgomery,
        )

        return forward_small_to_montgomery(
            residues, even, odd, twiddles, parameters
        )
    return ntt_ops.forward_ntt_to_montgomery_indexed(
        residues, even, odd, twiddles, parameters
    )


class NativePrepareCompressedPlaintextImplementation:
    """Prepare compact NTT plaintext rows from one period and supplied tables."""

    name = "native-prepare-compressed-plaintext"
    supports_in_place = False

    def __init__(self):
        from fhelium.ir.dialects.ckks import PrepareCompressedPlaintextOp

        self.operation_types = (PrepareCompressedPlaintextOp,)

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        result = prepare_periodic_tensor(
            *inputs,
            ring_dimension=invocation.attributes["ring_dimension"],
            scale=invocation.attributes["scale"],
            min_modulus=invocation.attributes["min_modulus"],
            polynomial_domain=invocation.attributes.get(
                "polynomial_domain", "ntt"
            ),
        )
        return result if isinstance(result, tuple) else (result,)
