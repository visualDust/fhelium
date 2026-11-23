"""Prepare periodic CKKS messages using their sparse polynomial embedding."""

from __future__ import annotations

import torch

from fhelium.native.wrapper import ckks_ops, ntt_ops, rns_ops
from fhelium.backend.random.stream import advance_

from ._embedding import _twister


def periodic_extent(period: int, ring_dimension: int, generator: int) -> int:
    """Return the coefficient count in the embedded negacyclic subring."""
    return ckks_ops.periodic_extent(
        period, ring_dimension.bit_length() - 1, generator
    )


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
    pre = ckks_ops.periodic_slot_positions(u, config.galois_generator).to(
        device
    )
    even_indices, odd_indices, twiddles = ntt_ops.subring_indexed_radix2_plan(
        list(moduli), config.logN, u.bit_length() - 1, context.dtype
    )
    ntt_ops.validate_indexed_schedule(even_indices, odd_indices, u)
    twiddles = twiddles.to(device)
    rns_ops.to_montgomery_(twiddles.view(twiddles.size(0), -1), parameters)
    return (
        pre,
        _twister(u, str(device)),
        state,
        parameters,
        torch.tensor(
            [2 * q for q in moduli], dtype=context.dtype, device=device
        ),
        even_indices.to(device=device),
        odd_indices.to(device=device),
        twiddles,
    )


class NativePrepareCompressedPlaintextImplementation:
    """Prepare the compact plaintext of a periodic message $a(X^{N/U})$.

    The libfhelium operation ``prepare_periodic_plaintext`` applies the
    $U$-point inverse embedding in the slot order of the generator, rounds
    coefficient $j$ with stream word $jN/U$, and evaluates the $U$-point
    indexed NTT whose roots are $\\psi_N^{N/U}$, so each bit-reversed entry
    represents $N/U$ contiguous entries of the full NTT. Coefficient output
    returns Montgomery residues and a zero implicit row. The rounding state
    advances once.
    """

    name = "native-prepare-compressed-plaintext"
    supports_in_place = False

    def __init__(self):
        from fhelium.ir.dialects.ckks import PrepareCompressedPlaintextOp

        self.operation_types = (PrepareCompressedPlaintextOp,)

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        del resources, in_place
        attributes = invocation.attributes
        result = ckks_ops.prepare_periodic_plaintext(
            *inputs,
            int(attributes["ring_dimension"]),
            float(attributes["scale"]),
            str(attributes.get("polynomial_domain", "ntt")),
        )
        advance_(inputs[3])
        return tuple(result)
