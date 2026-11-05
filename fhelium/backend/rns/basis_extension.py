"""Centered mixed-radix extension between ordered residue bases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fhelium.ir.dialects import rns

import torch

from fhelium.native.wrapper import rns_ops


def centered_basis_extend(
    source: torch.Tensor, tensors: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    r"""Return residues of centered CRT(source) in the supplied target basis.

    Input layout is [*batch, source row, coefficient]. Numerical operands bind
    the source and target moduli, Garner normalizers/propagation, extension
    coefficients, centered threshold and source-product correction. The output
    uses [*batch, target row, coefficient], standard residues and owned storage.
    """
    parameters = tensors["source_parameters"]
    digits = source.clone()
    rns_ops.reduce_to_standard_(digits, parameters)
    count = source.size(-2)
    if count > 1:
        digits = rns_ops.mixed_radix_decompose(
            digits,
            tensors["normalizers"],
            tensors["propagation"],
            tensors["source_lo"],
            tensors["source_hi"],
            tensors["source_neg_lo"],
            tensors["source_neg_hi"],
        )
        rns_ops.reduce_to_standard_(digits, parameters)
    target_parameters = tensors["target_parameters"]
    output = rns_ops.mixed_radix_basis_extend_to_montgomery(
        digits,
        tensors["extension"],
        target_parameters,
        target_parameters.size(-1),
    )
    rns_ops.from_montgomery_(output, target_parameters)
    rns_ops.reduce_to_standard_(output, target_parameters)
    equal = torch.ones_like(digits[..., 0, :], dtype=torch.bool)
    negative = torch.zeros_like(equal)
    half = tensors["half_digits"]
    for row in range(count - 1, -1, -1):
        negative |= equal & (digits[..., row, :] > half[row])
        equal &= digits[..., row, :] == half[row]
    shape = (*([1] * (output.ndim - 2)), -1, 1)
    corrected = torch.where(
        negative.unsqueeze(-2),
        output - tensors["source_product"].view(shape),
        output,
    )
    return torch.remainder(corrected, tensors["target_moduli"].view(shape))


@dataclass(frozen=True)
class NativeCenteredBasisExtensionImplementation:
    """Execute centered CRT extension from supplied numerical Tensor operands."""

    name: str = "native-centered-basis-extension"
    operation_types: tuple = (rns.CenteredBasisExtendOp,)
    supports_in_place: bool = False

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        del resources, in_place
        tensors = dict(
            zip(
                invocation.attributes["parameter_names"],
                inputs[1:],
                strict=True,
            )
        )
        return (centered_basis_extend(inputs[0], tensors),)
