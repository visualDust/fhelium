"""Streaming rotation sums with one extended-basis quotient."""

from __future__ import annotations

import torch

from fhelium.backend._native_operands import (
    key_switch_options,
    mapping_tables,
    rotation_galois,
)
from fhelium.backend.ckks.tables import RotationSumTables
from fhelium.native.wrapper import ckks_ops


class RotationSumAccumulator:
    r"""Accumulate rotations under one destination relation before division.

    Each nonzero-step key product contributes in NTT/Montgomery QP. The
    ordinary components contribute P times their Q residues and zero P
    residues. The final coefficient quotient is
    floor((sum_g z_g + h) / D), D=P or P*M, h=0 or (D-1)/2
    (``rotation_sum_add_`` and ``rotation_sum_finish_``). Inputs, keys and
    tables are unchanged; this object owns its accumulators.
    """

    def __init__(self, tables: RotationSumTables, *, input_domain: str = "ntt"):
        if input_domain not in {"ntt", "coefficient"}:
            raise ValueError(
                "Rotation-sum input domain must be ntt or coefficient"
            )
        self.tables = tables
        self.input_domain = input_domain
        self.ordinary: torch.Tensor | None = None
        self.products: torch.Tensor | None = None

    def _arguments(self):
        tensors = self.tables.tensor_operands()
        options = key_switch_options(self.tables.attributes)
        options["input_domain"] = self.input_domain
        return tensors, mapping_tables("key_switch", tensors), options

    def _products(self, tensors) -> torch.Tensor:
        if self.products is None:
            assert self.ordinary is not None
            self.products = self.ordinary.new_zeros(
                (
                    *self.ordinary.shape[:-2],
                    tensors["qp_parameters"].size(1),
                    self.ordinary.size(-1),
                )
            )
        return self.products

    def add(
        self,
        source: torch.Tensor,
        key: torch.Tensor | None,
        step: int,
        *,
        seed: torch.Tensor | None = None,
        key_rounds: int = 0,
    ) -> None:
        """Consume CT2 with the common Q layout, scale and polynomial domain.

        A seeded key supplies its seed rows and ChaCha round count.
        """
        tensors, tables, options = self._arguments()
        if seed is not None:
            options.update(key_seeds=[seed], key_rounds=key_rounds)
        if self.ordinary is None:
            self.ordinary = torch.zeros_like(source)
        # Step zero reads no key products, and the QP layout of a later key
        # selects the product rows.
        products = self._products(tensors) if step else source.new_empty(0)
        ckks_ops.rotation_sum_add_(
            self.ordinary,
            products,
            source,
            key,
            rotation_galois(self.tables.attributes, (int(step),), source)
            if step
            else [],
            tables,
            **options,
        )

    def finish(self) -> torch.Tensor:
        """Return surviving Q coefficient/standard residues after one quotient."""
        if self.ordinary is None:
            raise ValueError("Rotation sum requires at least one contribution")
        tensors, tables, options = self._arguments()
        result = ckks_ops.rotation_sum_finish_(
            self.ordinary,
            self._products(tensors),
            tables,
            **options,
            remaining_q_count=self.tables.remaining_q_count,
        )
        self.ordinary = self.products = None
        return result
