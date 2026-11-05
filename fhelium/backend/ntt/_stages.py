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
    """Describe synchronized intervals and the algorithm's numerical table ABI."""

    radix: int | None
    group: int
    intervals: tuple[tuple[int, int, bool], ...]
    twiddle_encoding: str = "montgomery"
    whole: bool = False


def endpoint_stages(
    policy: NttBackendPolicy | str, log_n: int | None, inverse: bool
) -> NttStages | None:
    if isinstance(policy, str):
        policy = resolve_ntt_backend_policy(policy)
    if isinstance(policy, CompactRadix2Policy):
        radix, group, tile = None, policy.grouped_radix2_stage_count, 8
    elif isinstance(policy, CompactFixedRadixPolicy):
        if log_n is not None and log_n % policy.radix_bits:
            return None
        radix, group = policy.radix, policy.radix_bits
        tile = 8 // group * group
    else:
        # A complete executor owns the transform and its representation changes.
        # Zero-length ends materialize only the surrounding pointwise expressions.
        return NttStages(
            None,
            0,
            ()
            if log_n is None
            else ((0, 0, False), (0, log_n, False), (log_n, log_n, False)),
            whole=True,
        )
    encoding = (
        policy.twiddle_encoding
        if isinstance(policy, CompactRadix2Policy)
        else "montgomery"
    )
    if log_n is None:
        return NttStages(radix, group, (), encoding)
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
        radix,
        group,
        tuple(part for part in intervals if part[0] != part[1]),
        encoding,
    )
