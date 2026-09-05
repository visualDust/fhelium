r"""Remove the auxiliary P basis while retaining NTT/Montgomery Q rows."""

from __future__ import annotations

import torch

from fhelium.backend.ckks.resources import KeySwitchExecutionResource
from fhelium.backend.ntt.executors.compact_radix2 import CompactRadix2NttBackend
from fhelium.native.wrapper import ckks_ops


def moddown_ntt_qp_to_q(
    source: torch.Tensor,
    plan: KeySwitchExecutionResource,
    level: int,
    *,
    coefficient_c0: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Return $\widehat{x}_Q/P+\operatorname{NTT}_Q(-r/P)$.

    ``source`` holds NTT/Montgomery residues over active QP. Inverting only
    the P rows determines the coefficient representative $r\in[0,P)$ used
    by coefficient-domain ModDown. Applying that same ModDown to zero Q
    rows produces $-r/P\bmod q_i$. Its forward NTT is added to the retained
    Q evaluations multiplied by $P^{-1}$. The result is a non-aliasing
    active-Q NTT/Montgomery tensor; source storage is unchanged.
    When ``coefficient_c0`` is supplied, it is added to component zero's
    coefficient correction before the forward NTT, adding NTT(c0) without
    another transform. The input c0 remains unchanged.
    """

    rns_context = plan.rns_context
    ntt_context = plan.ntt_context
    p_count = rns_context.config.num_p_primes
    p_rows = source[..., -p_count:, :].clone()
    ntt_context.inverse_to_standard_(
        p_rows, parameter_row_start=rns_context.config.num_q_primes
    )
    q_rows = source[..., :-p_count, :]
    parameters = rns_context.basis_parameters(level, include_p=True)
    correction = ckks_ops.keyswitch_moddown_qp_to_q(
        torch.zeros_like(q_rows),
        p_rows,
        plan.moddown_tables[level],
        parameters.native_parameters,
    )
    if coefficient_c0 is not None:
        rns_context.add_standard_(correction[0], coefficient_c0)
    backend = ntt_context.ntt_backend
    if isinstance(backend, CompactRadix2NttBackend):
        backend.forward_to_montgomery_add_scaled_(
            correction, q_rows, plan.p_inverse_montgomery[level:], level
        )
        return correction
    ntt_context.forward_to_montgomery_(correction, parameter_row_start=level)
    q_rows = q_rows.clone()
    rns_context.montgomery_mul_row_scalars_(
        q_rows, plan.p_inverse_montgomery[level:]
    )
    return rns_context.add_lazy(q_rows, correction)
