"""Streaming rotation sums with one extended-basis quotient."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.backend.ckks.key_switch import accumulate_key_switch_products
from fhelium.backend.ckks.tables import RotationSumTables
from fhelium.backend.ntt.automorphism import ntt_galois_indices
from fhelium.backend.ntt.operations import execute_named_transform
from fhelium.backend.rns.automorphism import coefficient_galois_gather_indices
from fhelium.backend.rns.moddown import divide_trailing_rows
from fhelium.native.wrapper import ckks_ops, rns_ops
from ._galois import rotation_galois_element


class RotationSumAccumulator:
    r"""Accumulate rotations under one destination relation before division.

    Each nonzero-step key product contributes in NTT/Montgomery QP. The
    ordinary components contribute P times their Q residues and zero P
    residues. The final coefficient quotient is
    floor((sum_g z_g + h) / D), D=P or P*M, h=0 or (D-1)/2.
    Inputs, keys and tables are unchanged; this object owns its accumulators.
    """

    def __init__(self, tables: RotationSumTables, *, input_domain: str = "ntt"):
        if input_domain not in {"ntt", "coefficient"}:
            raise ValueError(
                "Rotation-sum input domain must be ntt or coefficient"
            )
        self.tables = tables
        self.input_domain = input_domain
        self.products: torch.Tensor | None = None
        self.component0: torch.Tensor | None = None
        self.component1: torch.Tensor | None = None

    def add(
        self,
        source: torch.Tensor,
        key: torch.Tensor | None,
        step: int,
        *,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Consume CT2 with the common Q layout, scale and polynomial domain."""
        tensors, facts = self.tables.tensors, self.tables.attributes
        if step:
            if key is None:
                raise ValueError(
                    "A nonzero rotation requires its evaluation key"
                )
            element = rotation_galois_element(
                source.size(-1), step, int(cast(int, facts["galois_generator"]))
            )
            if self.input_domain == "ntt":
                if indices is None:
                    indices = ntt_galois_indices(
                        source.size(-1), element, source.device
                    )
                rotated = ckks_ops.apply_ntt_galois_automorphism(
                    source, indices
                )
                component = rotated[1].clone()
                execute_named_transform(
                    component, tensors, facts, "q", "inverse_to_standard_"
                )
            else:
                indices, signs = coefficient_galois_gather_indices(
                    source.size(-1), element, source.device
                )
                rotated = ckks_ops.apply_coefficient_galois_automorphism(
                    source, indices, signs, tensors["q_parameters"][0]
                )
                component = rotated[1]
            self.products = accumulate_key_switch_products(
                component,
                tensors=tensors,
                attributes=facts,
                key=key,
                accumulator=self.products,
            )
        else:
            rotated = source
        add = (
            rns_ops.add_lazy
            if self.input_domain == "ntt"
            else rns_ops.add_standard
        )
        self.component0 = (
            rotated[0].clone()
            if self.component0 is None
            else add(self.component0, rotated[0], tensors["q_parameters"])
        )
        if step == 0:
            self.component1 = (
                rotated[1].clone()
                if self.component1 is None
                else add(self.component1, rotated[1], tensors["q_parameters"])
            )

    def finish(self) -> torch.Tensor:
        """Return surviving Q coefficient/standard residues after one quotient."""
        if self.component0 is None:
            raise ValueError("Rotation sum requires at least one contribution")
        ordinary = torch.stack(
            (
                self.component0,
                torch.zeros_like(self.component0)
                if self.component1 is None
                else self.component1,
            )
        )
        tables = self.tables
        tensors, facts = tables.tensors, tables.attributes
        if self.input_domain == "coefficient":
            execute_named_transform(
                ordinary, tensors, facts, "q", "forward_to_montgomery_"
            )
        rns_ops.montgomery_mul_row_scalars_(
            ordinary, tables.p_montgomery, tensors["q_parameters"]
        )
        products = self.products
        q_count = int(cast(int, facts["q_count"]))
        if products is None:
            products = torch.zeros(
                (
                    *ordinary.shape[:-2],
                    q_count + int(cast(int, facts["p_count"])),
                    ordinary.size(-1),
                ),
                dtype=ordinary.dtype,
                device=ordinary.device,
            )
        products[..., :q_count, :].copy_(
            rns_ops.add_lazy(
                products[..., :q_count, :], ordinary, tensors["q_parameters"]
            )
        )
        execute_named_transform(
            products, tensors, facts, "qp", "inverse_to_standard_"
        )
        result = divide_trailing_rows(
            products,
            tables.quotient_inverses,
            tables.quotient_parameters,
            products.size(-2) - tables.remaining_q_count,
            half=tables.quotient_half,
            row_order=tables.quotient_row_order,
            rounding=str(facts.get("moddown_rounding", "nearest")),
        )
        self.products = self.component0 = self.component1 = None
        return result
