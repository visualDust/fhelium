"""RNS quotient tables and depth-indexed rescaling data."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.backend.ntt.context import NttContext
import torch
from fhelium.native.wrapper import rns_ops

from .context import RnsContext
from .crt import CRT_OPERANDS, centered_quotient_tables


def centered_basis_extension_tables(
    source: RnsContext,
    target: RnsContext,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    r"""Prepare Garner and centered-extension operands for two residue bases.

    Source and destination use the same word format and device. For source
    product Q, subtraction of Q mod p recenters the upper CRT half in each
    target row p. All row parameters and half digits are ordinary tensors;
    execution does not retain either preparation context.
    """
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("Basis-extension contexts must share dtype and device")
    source_moduli = [source.config.moduli[i] for i in source_prime_ids]
    target_moduli = [target.config.moduli[i] for i in target_prime_ids]
    if not source_moduli or not target_moduli:
        raise ValueError("Basis extension requires nonempty bases")
    dtype, device = source.dtype, source.device
    normalizers, propagation, extension = rns_ops.mixed_radix_tables(
        source_moduli, target_moduli, dtype
    )
    source_index = torch.tensor(
        source_prime_ids, dtype=torch.int64, device=device
    )
    target_index = torch.tensor(
        target_prime_ids, dtype=torch.int64, device=device
    )
    source_parameters = source.rns_parameter_tensor.index_select(
        1, source_index
    )
    lo, hi, neg_lo, neg_hi = source_parameters[1:5].unbind(0)
    return {
        "source_parameters": source_parameters,
        "target_parameters": target.rns_parameter_tensor.index_select(
            1, target_index
        ),
        "source_lo": lo,
        "source_hi": hi,
        "source_neg_lo": neg_lo,
        "source_neg_hi": neg_hi,
        "normalizers": normalizers.to(device),
        "propagation": propagation.to(device),
        "extension": extension.to(device),
        "half_digits": rns_ops.half_product_digits(source_moduli, dtype).to(
            device
        ),
        "source_product": rns_ops.product_residues(
            source_moduli, target_moduli, 1, False, dtype
        ).to(device),
        "target_moduli": torch.tensor(
            target_moduli, dtype=dtype, device=device
        ),
    }


def trailing_quotient_inverses(
    moduli: tuple[int, ...],
    drop_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    r"""Build $p_j^{-1}R\bmod q_i$ for successive trailing-row removal.

    Row j removes the j-th prime from the end. Only columns before that
    prime participate; unused columns are zero.
    """
    return rns_ops.trailing_quotient_inverses(
        list(moduli), drop_count, dtype
    ).to(device)


def moddown_inverse_tables(
    rns_context: RnsContext,
) -> tuple[torch.Tensor, ...]:
    """Return the trailing-quotient P inverses over each depth's QP basis."""

    moduli = rns_context.montgomery_parameters.moduli
    return tuple(
        trailing_quotient_inverses(
            tuple(
                moduli[i]
                for i in rns_context.rns_layout.prime_ids(depth, include_p=True)
            ),
            rns_context.config.num_p_primes,
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
        for depth in range(rns_context.config.max_depth + 1)
    )


RESCALE_TABLE_NAMES = (
    "source_parameters",
    "target_parameters",
    "dropped_parameters",
    "ordered_parameters",
    "drop_positions",
    "keep_positions",
    "inverses",
    "dropped_offset",
    "target_offset",
)
"""Rational rescale tables; ``restore`` and ``target_keep_positions`` are
added when primes enter the basis."""


def rescale_tables(
    context: RnsContext,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
    *,
    rounding: str = "nearest",
    ntt_context: NttContext | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    r"""Prepare the tables of one rational rescale from basis $B$ to $B'$.

    ``source_prime_ids`` and ``target_prime_ids`` list the rows of $B$ and
    $B'$ in their stored order. With $D=B\setminus B'$ and $R=B'\setminus B$,
    the tables describe $\lfloor (xR+H)/D \rfloor$ over $B'$, where $H=(D-1)/2$
    for nearest rounding and $H=0$ for floor rounding. ``restore`` and
    ``target_keep_positions`` are present only when $R$ is nonempty; otherwise
    the retained rows are $B'$ in order. Supplying ``ntt_context`` adds the dropped
    inverse and target forward NTT schedules and $D^{-1}\bmod b'$ for NTT
    input.
    """

    if rounding not in {"nearest", "floor"}:
        raise ValueError("rescale rounding must be nearest or floor")
    source, target = tuple(source_prime_ids), tuple(target_prime_ids)
    source_set, target_set = set(source), set(target)
    dropped = tuple(i for i in source if i not in target_set)
    restored = tuple(i for i in target if i not in source_set)
    kept = tuple(i for i in source if i in target_set)
    if not dropped:
        raise ValueError("Rescaling must drop at least one prime")
    moduli = context.config.moduli
    D = [moduli[i] for i in dropped]
    R = [moduli[i] for i in restored]
    dtype, device = context.dtype, context.device
    ordered = (*target, *dropped)

    def positions(ids: tuple[int, ...], basis: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(
            tuple(basis.index(i) for i in ids), dtype=torch.long, device=device
        )

    def offset(ids: tuple[int, ...]) -> torch.Tensor:
        return (
            rns_ops.quotient_offset_residues(
                [], D, False, rounding == "nearest", [moduli[i] for i in ids], dtype
            )
            .to(device)
            .view(-1, 1)
            .expand(-1, context.config.N)
        )

    tensors: dict[str, torch.Tensor] = {
        "source_parameters": context.rns_parameters_for_prime_ids(source),
        "target_parameters": context.rns_parameters_for_prime_ids(target),
        "dropped_parameters": context.rns_parameters_for_prime_ids(dropped),
        "ordered_parameters": context.rns_parameters_for_prime_ids(ordered),
        "drop_positions": positions(dropped, source),
        "keep_positions": positions(kept, source),
        "inverses": trailing_quotient_inverses(
            tuple(moduli[i] for i in ordered),
            len(dropped),
            dtype=dtype,
            device=device,
        ),
        "dropped_offset": offset(dropped),
        "target_offset": offset(target),
    }
    if restored:
        tensors["target_keep_positions"] = positions(kept, target)
        tensors["restore"] = rns_ops.product_residues(
            R, [moduli[i] for i in source], 1, True, dtype
        ).to(device)
    facts: dict[str, object] = {}
    if ntt_context is not None:
        for basis, ids, inverse in (
            ("dropped", dropped, True),
            ("target", target, False),
        ):
            direction = "inverse" if inverse else "forward"
            for index, tensor in enumerate(
                ntt_context.tensor_operands(ids, inverse=inverse)
            ):
                tensors[f"{basis}_{direction}_{index}"] = tensor
        tensors["product_inverse"] = rns_ops.product_residues(
            D, [moduli[i] for i in target], -1, True, dtype
        ).to(device)
        facts["ntt_backend"] = ntt_context.ntt_backend_name
    return tensors, facts


MODDOWN_RESCALE_TABLE_NAMES = (
    "joint_remaining_parameters",
    "joint_dropped_parameters",
    "joint_ordered_parameters",
    "joint_p_times_base",
    "joint_inverses",
    "joint_product_inverse",
    "joint_dropped_offset",
    "joint_remaining_offset",
    "joint_drop_positions",
    "joint_remaining_positions",
)
"""Fused ModDown/rescale tables; a restored prime set adds ``joint_restore``,
``joint_product_parameters``, and ``joint_target_keep_positions``."""

MODDOWN_RESTORE_TABLE_NAMES = (
    "joint_restore",
    "joint_product_parameters",
    "joint_target_keep_positions",
)


def centered_moddown_rescale_names(rescale_rounding: str) -> tuple[str, ...]:
    """Name the tables of the centered P quotient followed by exact rescaling."""
    return (
        *(f"joint_crt_{part}_{name}" for part in ("d", "t") for name in CRT_OPERANDS),
        "joint_crt_p_inverse",
        "joint_crt_inverses",
        "joint_crt_ordered_parameters",
        *(("joint_crt_half",) if rescale_rounding == "nearest" else ()),
    )


def moddown_rescale_tables(
    context: RnsContext,
    q_prime_ids: tuple[int, ...],
    p_prime_ids: tuple[int, ...],
    drop_prime_ids: tuple[int, ...],
    *,
    target_prime_ids: tuple[int, ...] | None = None,
    moddown_rounding: str = "nearest",
    rescale_rounding: str = "nearest",
    basis_conversion: str = "exact",
) -> dict[str, torch.Tensor]:
    r"""Prepare the quotient of $R(v+Pa)+H_P+PH_D$ by $PD$.

    Q and P identify actual ordered input rows; D identifies the complete Q set consumed by rescaling. ``target_prime_ids`` orders the output basis $B'$, which defaults to the remaining Q rows in their input order; its rows outside Q form the restored set $R$, and $R=1$ when that set is empty. ``basis_conversion="normalized_crt"`` adds ``joint_crt_`` tables for a centered quotient by P followed by the exact quotient by D with ``rescale_rounding``; ``moddown_rounding`` then does not apply. All returned values are ordinary numerical Tensors on this RNS context's device.
    """
    if moddown_rounding not in {"nearest", "floor"} or rescale_rounding not in {
        "nearest",
        "floor",
    }:
        raise ValueError("quotient rounding must be nearest or floor")
    q, p, d = tuple(q_prime_ids), tuple(p_prime_ids), tuple(drop_prime_ids)
    if not d or len(set(d)) != len(d) or not set(d) <= set(q):
        raise ValueError(
            "The rescale group must be a nonempty subset of Q rows"
        )
    remaining = tuple(i for i in q if i not in d)
    target = remaining if target_prime_ids is None else tuple(target_prime_ids)
    if not target or not set(remaining) <= set(target):
        raise ValueError("The rescale target must contain the remaining Q rows")
    restored = tuple(i for i in target if i not in q)
    if not restored and target != remaining:
        raise ValueError("A rescale target without restored primes keeps the Q order")
    dropped, ordered = (*d, *p), (*target, *d, *p)
    moduli = context.config.moduli
    P = [moduli[i] for i in p]
    D = [moduli[i] for i in d]
    R = [moduli[i] for i in restored]
    dtype, device = context.dtype, context.device

    def tensor(values, dtype=None):
        return torch.tensor(
            values,
            dtype=context.dtype if dtype is None else dtype,
            device=device,
        )

    def offset(ids):
        return (
            rns_ops.quotient_offset_residues(
                P,
                D,
                moddown_rounding == "nearest",
                rescale_rounding == "nearest",
                [moduli[i] for i in ids],
                dtype,
            )
            .to(device)
            .view(-1, 1)
            .expand(-1, context.config.N)
        )

    tables = {
        "joint_remaining_parameters": context.rns_parameters_for_prime_ids(
            target
        ),
        "joint_dropped_parameters": context.rns_parameters_for_prime_ids(
            dropped
        ),
        "joint_ordered_parameters": context.rns_parameters_for_prime_ids(
            ordered
        ),
        "joint_p_times_base": rns_ops.product_residues(
            P + R, [moduli[i] for i in q], 1, True, dtype
        ).to(device),
        "joint_inverses": trailing_quotient_inverses(
            tuple(moduli[i] for i in ordered),
            len(dropped),
            dtype=dtype,
            device=device,
        ),
        "joint_product_inverse": rns_ops.product_residues(
            P + D, [moduli[i] for i in target], -1, True, dtype
        ).to(device),
        "joint_dropped_offset": offset(dropped),
        "joint_remaining_offset": offset(target),
        "joint_drop_positions": tensor(
            tuple(q.index(i) for i in d), torch.long
        ),
        "joint_remaining_positions": tensor(
            tuple(q.index(i) for i in remaining), torch.long
        ),
    }
    if basis_conversion == "normalized_crt":
        target_d = (*target, *d)
        tables.update(
            centered_quotient_tables(context, p, d, "joint_crt_d_"),
            **centered_quotient_tables(
                context, p, target, "joint_crt_t_", divisor_prime_ids=d
            ),
            joint_crt_p_inverse=rns_ops.product_residues(
                P, D, -1, True, dtype
            ).to(device),
            joint_crt_inverses=trailing_quotient_inverses(
                tuple(moduli[i] for i in target_d),
                len(d),
                dtype=dtype,
                device=device,
            ),
            joint_crt_ordered_parameters=context.rns_parameters_for_prime_ids(
                target_d
            ),
        )
        if rescale_rounding == "nearest":
            tables["joint_crt_half"] = rns_ops.half_product_residues(
                D, [moduli[i] for i in target_d], dtype
            ).to(device)
    if restored:
        tables.update(
            joint_restore=rns_ops.product_residues(
                R, [moduli[i] for i in (*q, *p)], 1, True, dtype
            ).to(device),
            joint_product_parameters=context.rns_parameters_for_prime_ids(
                (*q, *p)
            ),
            joint_target_keep_positions=tensor(
                tuple(target.index(i) for i in remaining), torch.long
            ),
        )
    return tables
