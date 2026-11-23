"""Numerical reconstruction tables consumed by CKKS decryption."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from fhelium.backend.rns.context import RnsContext
from fhelium.native.wrapper import rns_ops


@dataclass(frozen=True)
class DecryptReconstructionTables:
    """Hold device tables for centered reconstruction of the complete active Q basis."""

    source_prime_ids_by_depth: tuple[tuple[int, ...], ...]
    normalizers_by_depth: tuple[torch.Tensor, ...]
    propagation_by_depth: tuple[torch.Tensor, ...]
    device: torch.device
    half_digits_by_depth: tuple[torch.Tensor, ...]

    def tensor_operands(
        self, context: RnsContext, depth: int
    ) -> tuple[torch.Tensor, ...]:
        """Return the Q parameters and mixed-radix reconstruction tables."""
        return (
            context.rns_parameters_for_prime_ids(
                self.source_prime_ids_by_depth[depth]
            ),
            self.normalizers_by_depth[depth],
            self.propagation_by_depth[depth],
            self.half_digits_by_depth[depth],
        )

    @classmethod
    def create(cls, context: RnsContext) -> DecryptReconstructionTables:
        """Build reconstruction tables for one context-local RNS context."""

        half_digits_by_depth = []
        source_prime_ids_by_depth = []
        normalizers_by_depth = []
        propagation_by_depth = []
        for depth in range(context.config.max_depth + 1):
            q_prime_ids = context.rns_layout.prime_ids(depth)
            source_prime_ids = tuple(q_prime_ids)
            source_prime_ids_by_depth.append(source_prime_ids)
            source_moduli = [
                int(context.montgomery_parameters.moduli[index])
                for index in source_prime_ids
            ]
            normalizers, propagation, _ = rns_ops.mixed_radix_tables(
                source_moduli, [], context.dtype
            )
            half_digits_by_depth.append(
                rns_ops.half_product_digits(source_moduli, context.dtype).to(
                    context.device
                )
            )
            normalizers_by_depth.append(normalizers.to(context.device))
            propagation_by_depth.append(propagation.to(context.device))
        return cls(
            tuple(source_prime_ids_by_depth),
            tuple(normalizers_by_depth),
            tuple(propagation_by_depth),
            context.device,
            tuple(half_digits_by_depth),
        )
