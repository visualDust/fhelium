"""Prepare periodic CKKS messages using their sparse polynomial embedding."""

from __future__ import annotations

import torch

from fhelium.native.wrapper import ckks_ops

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.lowering import ExecBuilder
from fhelium.backend.resources import BoundResource


def periodic_extent(period: int, ring_dimension: int, generator: int) -> int:
    """Return the coefficient count in the embedded negacyclic subring."""
    return ckks_ops.periodic_extent(
        period, ring_dimension.bit_length() - 1, generator
    )


def periodic_materials(
    config, context, period: int, depth: int, basis: str, state
):
    """Return the compact embedding and indexed NTT tables of ``fhelium_ckks_ops::periodic_table_set`` with the parameters and live rounding state, in the operand order of ``prepare_periodic_plaintext``."""
    ids = context.rns_layout.prime_ids(depth, include_p=basis == "QP")
    parameters = context.rns_parameters_for_prime_ids(ids)
    names, tensors = ckks_ops.periodic_table_set(
        parameters,
        [config.moduli[i] for i in ids],
        config.logN,
        period,
        config.galois_generator,
    )
    tables = dict(zip(names, tensors, strict=True))
    return (
        tables["pre"],
        tables["twister"],
        state,
        parameters,
        tables["twice_moduli"],
        tables["even_indices"],
        tables["odd_indices"],
        tables["twiddles"],
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

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        attributes = invocation.attributes
        (
            message,
            pre,
            twister,
            seed,
            parameters,
            twice_moduli,
            even,
            odd,
            twiddles,
        ) = inputs
        result = ckks_ops.prepare_periodic_plaintext(
            message,
            pre,
            twister,
            seed,
            parameters,
            twice_moduli,
            even,
            odd,
            twiddles,
            int(attributes["ring_dimension"]),
            float(attributes["scale"]),
            str(attributes.get("polynomial_domain", "ntt")),
        )
        return tuple(result)
