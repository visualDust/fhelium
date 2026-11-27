"""RNS quotient tables and depth-indexed rescaling data."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.backend.ntt.context import NttContext
import torch
from fhelium.native.wrapper import rns_ops

from .context import RnsContext
from .crt import CRT_OPERANDS


def shared_parameter_views(
    context: RnsContext, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Replace prepared ``*parameters`` tables by the context's shared views.

    Native table builders gather RNS parameter columns into new Tensors.
    Programs identify numerical materials by Tensor, and fusion groups
    operations that read one parameter Tensor, so each table whose columns are
    rows of ``context`` is replaced by
    ``context.rns_parameters_for_prime_ids`` of those rows.
    """
    catalog = context.rns_parameter_tensor
    columns: dict[tuple[int, ...], int] | None = None
    for name, value in tensors.items():
        if (
            not name.endswith("parameters")
            or value.dim() != 2
            or value.size(0) != catalog.size(0)
            or value.dtype != catalog.dtype
            or value.device != catalog.device
        ):
            continue
        if columns is None:
            columns = {
                tuple(column): index
                for index, column in enumerate(catalog.t().tolist())
            }
        ids = tuple(
            columns.get(tuple(column), -1) for column in value.t().tolist()
        )
        if -1 not in ids:
            tensors[name] = context.rns_parameters_for_prime_ids(ids)
    return tensors


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
    names, tensors = rns_ops.basis_extension_table_set(
        source.rns_parameter_tensor,
        list(source.config.moduli),
        target.rns_parameter_tensor,
        list(target.config.moduli),
        source.config.logN,
        list(source_prime_ids),
        list(target_prime_ids),
    )
    return shared_parameter_views(
        target,
        shared_parameter_views(source, dict(zip(names, tensors, strict=True))),
    )


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

    names, values = rns_ops.rescale_table_set(
        context.rns_parameter_tensor,
        list(context.config.moduli),
        context.config.logN,
        list(source_prime_ids),
        list(target_prime_ids),
        rounding,
        ntt_context is not None,
    )
    tensors = shared_parameter_views(
        context, dict(zip(names, values, strict=True))
    )
    facts: dict[str, object] = {}
    if ntt_context is not None:
        target = tuple(target_prime_ids)
        dropped = tuple(i for i in source_prime_ids if i not in set(target))
        for basis, ids, inverse in (
            ("dropped", dropped, True),
            ("target", target, False),
        ):
            direction = "inverse" if inverse else "forward"
            for index, tensor in enumerate(
                ntt_context.tensor_operands(ids, inverse=inverse)
            ):
                tensors[f"{basis}_{direction}_{index}"] = tensor
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
        *(
            f"joint_crt_{part}_{name}"
            for part in ("d", "t")
            for name in CRT_OPERANDS
        ),
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
    names, tensors = rns_ops.joint_rescale_table_set(
        context.rns_parameter_tensor,
        list(context.config.moduli),
        context.config.logN,
        list(q_prime_ids),
        list(p_prime_ids),
        list(drop_prime_ids),
        [] if target_prime_ids is None else list(target_prime_ids),
        moddown_rounding,
        rescale_rounding,
        basis_conversion,
    )
    return shared_parameter_views(
        context, dict(zip(names, tensors, strict=True))
    )
