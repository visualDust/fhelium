"""Execution adapters that consume one direct key-switch digit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from fhelium.engine.ntt.backends.compact_radix2 import CompactRadix2NttBackend
from fhelium.native.wrapper import ckks_ops, ntt_ops

if TYPE_CHECKING:
    from fhelium.engine.rns.runtime import RnsRuntime


class DirectKeySwitchDigitConsumer(Protocol):
    r"""Consume one coefficient-domain QP digit into two accumulators.

    The hybrid key-switch executor owns digit production and lifetime. A
    consumer owns only the representation transition and multiply-accumulate,
    which is the exact operation that an ordinary two-kernel implementation
    and a fused native implementation may differ.

    ``digit_qp`` and both accumulators are integral
    ``[*batch, limb, coefficient_or_ntt_index]`` tensors whose limb order is
    the same exact active $Q_\ell P$ interval. The digit enters in
    coefficient/Montgomery lazy form. Consumers mutate it to NTT/Montgomery
    and add products with ``key_digit[key_component, limb, ntt_index]`` into
    the two NTT/Montgomery accumulators. ``parameter_row_start`` maps local
    digit rows to parameter prime ids; ``key_row_start`` independently maps
    them to key storage. No input tensors may alias one another.
    """

    def consume_(
        self,
        digit_qp: torch.Tensor,
        key_digit: torch.Tensor,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        *,
        parameter_row_start: int,
        key_row_start: int,
    ) -> None:
        """Transform and consume ``digit_qp`` without retaining it.

        Mutates ``digit_qp``, ``accumulator0_qp``, and ``accumulator1_qp``;
        key material and parameter/twiddle tables remain read-only.
        """

        ...


@dataclass(frozen=True)
class StandardDirectKeySwitchDigitConsumer:
    """Use an arbitrary NTT backend followed by the shared KSK accumulator."""

    rns_runtime: RnsRuntime

    def consume_(
        self,
        digit_qp: torch.Tensor,
        key_digit: torch.Tensor,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        *,
        parameter_row_start: int,
        key_row_start: int,
    ) -> None:
        active_params = self.rns_runtime._forward_montgomery_with_parameters_(
            digit_qp,
            include_p=True,
            parameter_row_start=parameter_row_start,
        )
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator0_qp,
            accumulator1_qp,
            digit_qp,
            key_digit,
            active_params,
            key_row_start,
        )


@dataclass(frozen=True)
class CompactTailFusedDirectKeySwitchDigitConsumer:
    """Fuse the validated compact final-eight-stage NTT into KSK consume."""

    backend: CompactRadix2NttBackend

    def __post_init__(self) -> None:
        if self.backend.grouped_radix2_stage_count != 4:
            raise ValueError(
                "compact tail/KSK fusion requires the group16 radix-2 backend"
            )

    def consume_(
        self,
        digit_qp: torch.Tensor,
        key_digit: torch.Tensor,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        *,
        parameter_row_start: int,
        key_row_start: int,
    ) -> None:
        twiddles, params = self.backend._active_native_inputs(
            digit_qp,
            self.backend.forward_twiddles,
            parameter_row_start,
        )
        ntt_ops.forward_ntt_montgomery_compact_keyswitch_accumulate_(
            digit_qp,
            twiddles,
            params,
            key_digit,
            accumulator0_qp,
            accumulator1_qp,
            key_row_start,
        )


def create_direct_keyswitch_digit_consumer(
    rns_runtime: RnsRuntime,
) -> DirectKeySwitchDigitConsumer:
    """Bind one direct digit consumer when the engine is constructed.

    Concrete-backend inspection is confined to this construction function.
    The key-switch hot path receives one stable protocol implementation and
    performs no capability discovery or backend-name dispatch.
    """

    backend = rns_runtime.ntt_backend
    if (
        isinstance(backend, CompactRadix2NttBackend)
        and backend.grouped_radix2_stage_count == 4
    ):
        return CompactTailFusedDirectKeySwitchDigitConsumer(backend)
    return StandardDirectKeySwitchDigitConsumer(rns_runtime)
