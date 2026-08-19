"""Semantic interface shared by NTT algorithm families.

This module separates *what* representation transition the RNS runtime needs
from *how* a particular backend factors and launches the transform. Concrete
backends own family-specific device tables and kernels; callers depend only on
the :class:`NttBackend` protocol below.

All operands are dense ``[*batch, prime, coefficient]`` tensors. An unbatched
operand keeps the original ``[prime, coefficient]`` rank. The prime rows are a
contiguous interval in the engine's canonical QP parameter order, and callers
supply that interval with ``parameter_row_start`` with ``parameter_row_start``. This prevents
a partial basis or key-switch digit from being matched to parameters merely
because it happens to have a particular row count.
"""

from __future__ import annotations

from typing import Protocol

import torch


class NttBackend(Protocol):
    r"""Representation-explicit NTT operations consumed by ``RnsRuntime``.

    Method names state both the input and output representation. ``forward``
    consumes coefficient-domain data and produces NTT-domain data; ``inverse``
    performs the reverse transition. ``montgomery`` and ``standard`` identify
    the residue representation. A trailing underscore means that the operand
    is mutated in place.

    ``parameter_row_start`` is the zero-based start of the operand's prime-row
    interval in the backend's complete canonical QP tables. Implementations
    use it to select the exact per-prime twiddles, roots, and RNS parameters;
    they must not infer prime identity from ``operand.size(-2)``.

    The protocol intentionally contains no radix, grouping, or table-layout
    fields. Those are construction-time properties of a concrete backend, not
    semantic differences visible to ``RnsRuntime``.

    Every operand is a dense integral tensor on one execution device
    ``[*batch, limb, coefficient_or_ntt_index]`` with final extent
    $N=2^{\mathtt{logN}}$. A concrete backend is usable only on devices for
    which its native schemas have dispatcher implementations. CPU engines use
    indexed radix-2 execution. There is no broadcasting. Limb ``j`` is
    aligned with canonical parameter row ``parameter_row_start + j`` and
    therefore with that row's exact prime $q_i$. All tables use the same
    dtype/device.

    Let $\psi_i$ be the prepared primitive $2N$-th root modulo $q_i$ and
    $\operatorname{br}(k)$ the ``logN``-bit reversal. The stored NTT order is

    $$
    A_i[k]=\sum_{n=0}^{N-1}a_i[n]
      \psi_i^{(2\operatorname{br}(k)+1)n}\pmod{q_i}.
    $$

    The inverse implements the corresponding normalized inverse with
    $N^{-1}\bmod q_i$. Transform kernels accept and return lazy representatives
    in $[0,2q_i)$ except ``inverse_to_standard_`` (canonical $[0,q_i)$) and
    ``inverse_to_centered_`` (centered representatives). A trailing underscore
    preserves and mutates operand storage; the functional forward allocates an
    output that does not alias any input.
    """

    name: str

    def forward_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map coefficient/Montgomery to NTT/Montgomery in place."""

        ...

    def forward_to_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map coefficient/standard to NTT/Montgomery in place."""

        ...

    def forward_to_montgomery(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> torch.Tensor:
        """Return NTT/Montgomery output without mutating or aliasing input."""

        ...

    def inverse_montgomery_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map NTT/Montgomery to coefficient/Montgomery in place."""

        ...

    def inverse_to_standard_lazy_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map in place to lazy coefficient/standard residues in $[0,2q_i)$."""

        ...

    def inverse_to_standard_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map in place to coefficient/standard residues in $[0,q_i)$."""

        ...

    def inverse_to_centered_(
        self, operand: torch.Tensor, parameter_row_start: int
    ) -> None:
        """Map in place to centered coefficient/standard residues."""

        ...


def slice_ntt_parameter_rows(
    operand: torch.Tensor,
    twiddles: torch.Tensor,
    rns_params: torch.Tensor,
    parameter_row_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align complete backend tables with the prime rows of one operand.

    Backends retain tables for the engine's complete canonical QP row order,
    while an individual operation may contain only a contiguous subset. For
    example, a lower-level Q or QP basis omits a prefix of Q primes, and an
    internal key-switch digit may select another specified contiguous interval.
    ``parameter_row_start`` supplies the missing global-row identity.

    Args:
        operand: Dense ``[*batch, prime, coefficient]`` transform operand. Its
            penultimate dimension determines the number of active prime rows.
        twiddles: Complete backend table whose first dimension follows the
            engine's canonical QP prime-row order.
        rns_params: Complete RNS parameter tensor with shape
            ``[parameter, prime]`` in the same canonical prime-row order.
        parameter_row_start: Zero-based canonical row corresponding to
            ``operand[..., 0, :]``.

    Returns:
        A pair of zero-copy basic-slice views: the active twiddle rows and the
        matching RNS parameter columns. Both have the same prime-row count and
        order as ``operand``.

    Raises:
        ValueError: If ``operand`` has rank less than two, the start is
            negative, or the requested row interval exceeds either complete
            table.

    Notes:
        The helper deliberately does not infer ``parameter_row_start`` from
        the operand row count. Multiple RNS subsets can have the same number
        of rows while referring to different primes.
    """

    if operand.ndim < 2:
        raise ValueError(
            "NTT operands must have shape [*batch, prime, coefficient], "
            f"got {operand.shape}"
        )
    if parameter_row_start < 0:
        raise ValueError("parameter_row_start must be non-negative")
    row_count = operand.size(-2)
    row_stop = parameter_row_start + row_count
    if row_stop > twiddles.size(0) or row_stop > rns_params.size(1):
        raise ValueError(
            "NTT operand rows exceed the prepared canonical parameter range: "
            f"start={parameter_row_start}, rows={row_count}, "
            f"twiddle_rows={twiddles.size(0)}, parameter_rows={rns_params.size(1)}"
        )
    return (
        twiddles[parameter_row_start:row_stop],
        rns_params[:, parameter_row_start:row_stop],
    )
