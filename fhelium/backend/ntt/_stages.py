"""Stage partitions used by generated NTT endpoints and native interiors."""

from __future__ import annotations

from dataclasses import dataclass

from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    CompactRadix2Policy,
    NttBackendPolicy,
    resolve_ntt_backend_policy,
)


@dataclass(frozen=True)
class NttStages:
    """Describe synchronized intervals and the algorithm's numerical table ABI.

    Each interval ``(start, end, tiled)`` covers transform stages
    ``[start, end)``. A tiled interval acts on contiguous coefficient tiles; an
    untiled interval acts on coefficients spaced by the stride of its stages.
    """

    radix: int | None
    intervals: tuple[tuple[int, int, bool], ...]
    twiddle_encoding: str = "montgomery"
    whole: bool = False


def endpoint_stages(
    policy: NttBackendPolicy | str, log_n: int | None, inverse: bool
) -> NttStages | None:
    if isinstance(policy, str):
        policy = resolve_ntt_backend_policy(policy)
    if isinstance(policy, CompactRadix2Policy):
        if log_n is None:
            return NttStages(None, (), policy.twiddle_encoding)
        # Two passes: the spaced pass holds floor(logN/2) stages, as in the
        # native compact transform.
        spaced = log_n // 2
        intervals = (
            ((0, log_n - spaced, True), (log_n - spaced, log_n, False))
            if inverse
            else ((0, spaced, False), (spaced, log_n, True))
        )
        return NttStages(None, intervals, policy.twiddle_encoding)
    if not isinstance(policy, CompactFixedRadixPolicy):
        # A complete executor owns the transform and its representation changes.
        # Zero-length ends materialize only the surrounding pointwise expressions.
        return NttStages(
            None,
            ()
            if log_n is None
            else ((0, 0, False), (0, log_n, False), (log_n, log_n, False)),
            whole=True,
        )
    if log_n is not None and log_n % policy.radix_bits:
        return None
    radix, group = policy.radix, policy.radix_bits
    tile = 8 // group * group
    if log_n is None:
        return NttStages(radix, ())
    if log_n <= tile:
        intervals = ((0, log_n, True),)
    elif inverse:
        last = max(tile, log_n - group)
        intervals = ((0, tile, True), (tile, last, False), (last, log_n, False))
    else:
        last = log_n - tile
        first = min(group, last)
        intervals = (
            (0, first, False),
            (first, last, False),
            (last, log_n, True),
        )
    return NttStages(
        radix, tuple(part for part in intervals if part[0] != part[1])
    )
