"""Tensor views used by complete hybrid CKKS key-switch implementations."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING
import torch
from fhelium.backend.rns.context import RnsContext

if TYPE_CHECKING:
    from fhelium.backend.ntt.context import NttContext


@dataclass(frozen=True)
class KeySwitchTables:
    """Own the tables used by hybrid CKKS key switching."""

    rns_context: RnsContext
    ntt_context: NttContext | None
    moddown_p_drop_inverses_montgomery_by_depth: tuple[torch.Tensor, ...]
    galois_generator: int = 3
    p_inverse_montgomery: torch.Tensor = field(
        init=False, repr=False, compare=False
    )
    _tensor_views: dict[
        int, tuple[dict[str, torch.Tensor], dict[str, object]]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.ntt_context is not None
            and self.ntt_context.rns_context is not self.rns_context
        ):
            raise ValueError(
                "Key-switch NTT context composes another RNS context"
            )
        if not self.moddown_p_drop_inverses_montgomery_by_depth:
            raise ValueError("Key-switch resource requires ModDown tables")
        if self.galois_generator not in {3, 5}:
            raise ValueError(
                "Key-switch resource galois generator must be 3 or 5"
            )
        config = self.rns_context.config
        p_product = prod(config.p_moduli)
        radix = self.rns_context.montgomery_parameters.R
        object.__setattr__(
            self,
            "p_inverse_montgomery",
            torch.tensor(
                [pow(p_product, -1, q) * radix % q for q in config.q_moduli],
                dtype=self.rns_context.dtype,
                device=self.rns_context.device,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.rns_context.device

    @property
    def moddown_tables(self) -> tuple[torch.Tensor, ...]:
        """Return the depth-indexed P-drop inverse tables read by ModDown."""

        return self.moddown_p_drop_inverses_montgomery_by_depth

    def tensor_operands(
        self, depth: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Expose the existing key-switch tables and static digit ranges.

        Returned data can populate ordinary Program Tensor placeholders. No
        execution resource object is needed to consume these table views.
        """
        if depth in self._tensor_views:
            return self._tensor_views[depth]
        rns = self.rns_context
        q_ids = rns.rns_layout.prime_ids(depth)
        qp_ids = rns.rns_layout.prime_ids(depth, include_p=True)
        tensors = {
            "q_parameters": rns.rns_parameters_for_prime_ids(q_ids),
            "qp_parameters": rns.rns_parameters_for_prime_ids(qp_ids),
            "moddown_inverses": self.moddown_tables[depth],
            "p_inverse": self.p_inverse_montgomery[q_ids[0] :],
        }
        if self.ntt_context is not None:
            for basis, prime_ids in (
                ("q", q_ids),
                ("qp", qp_ids),
                ("p", rns.rns_chain.p_prime_ids),
            ):
                for direction, inverse in (
                    ("forward", False),
                    ("inverse", True),
                ):
                    for index, tensor in enumerate(
                        self.ntt_context.tensor_operands(
                            prime_ids, inverse=inverse
                        )
                    ):
                        tensors[f"{basis}_{direction}_{index}"] = tensor
        digits = []
        for index, spec in enumerate(rns.rns_layout.digit_specs(depth)):
            parameters = rns.row_parameters(spec.prime_ids)
            start, stop = (
                spec.component_row_ids[0],
                spec.component_row_ids[-1] + 1,
            )
            digits.append((start, stop, spec.key_digit_index))
            tensors[f"digit{index}_parameters"] = parameters.native_parameters
            if stop - start > 1:
                assert parameters.mixed_radix_normalizers is not None
                assert (
                    parameters.mixed_radix_propagation_coefficients is not None
                )
                tensors[f"digit{index}_normalizers"] = (
                    parameters.mixed_radix_normalizers.contiguous()
                )
                tensors[f"digit{index}_propagation"] = (
                    parameters.mixed_radix_propagation_coefficients.contiguous()
                )
                for part, tensor in enumerate(
                    parameters.montgomery_reduction_parameters
                ):
                    tensors[f"digit{index}_reduction{part}"] = (
                        tensor.contiguous()
                    )
            coefficients = parameters.basis_extension_coefficients
            if coefficients is None:
                coefficients = torch.empty(
                    0, 0, dtype=rns.dtype, device=rns.device
                )
            tensors[f"digit{index}_extension"] = coefficients[
                :, qp_ids[0] : qp_ids[-1] + 1
            ]
        facts: dict[str, object] = {
            "digits": tuple(digits),
            "q_count": len(q_ids),
            "p_count": rns.config.num_p_primes,
            "key_row_start": qp_ids[0],
            "galois_generator": self.galois_generator,
        }
        if self.ntt_context is not None:
            facts["ntt_backend"] = self.ntt_context.ntt_backend_name
        self._tensor_views[depth] = (tensors, facts)
        return tensors, facts
