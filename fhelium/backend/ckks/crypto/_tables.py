"""Numerical reconstruction tables consumed by CKKS decryption."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from fhelium.backend.rns.context import RnsContext
from fhelium.native.wrapper import ckks_ops


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
            _, (_, normalizers, propagation, half_digits) = (
                ckks_ops.reconstruction_table_set(
                    context.rns_parameter_tensor,
                    list(context.config.moduli),
                    context.config.logN,
                    list(source_prime_ids),
                )
            )
            half_digits_by_depth.append(half_digits)
            normalizers_by_depth.append(normalizers)
            propagation_by_depth.append(propagation)
        return cls(
            tuple(source_prime_ids_by_depth),
            tuple(normalizers_by_depth),
            tuple(propagation_by_depth),
            context.device,
            tuple(half_digits_by_depth),
        )
