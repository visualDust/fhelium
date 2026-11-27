"""Centered mixed-radix extension between ordered residue bases."""

from __future__ import annotations

from dataclasses import dataclass

from fhelium.backend._native_operands import named_tables
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops
import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.lowering import ExecBuilder
from fhelium.backend.resources import BoundResource


@dataclass(frozen=True)
class NativeCenteredBasisExtensionImplementation:
    """Return residues of the centered CRT value in the target basis.

    The libfhelium operation ``centered_basis_extend`` reads the source and
    target moduli, Garner normalizers and propagation, extension coefficients,
    centered threshold, and source-product correction. Output rows hold
    standard residues.
    """

    name: str = "native-centered-basis-extension"
    operation_types: tuple = (rns.CenteredBasisExtendOp,)
    supports_in_place: bool = False

    def resource_requirements(self, invocation):
        return ()

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        return (
            rns_ops.centered_basis_extend(
                inputs[0],
                named_tables(
                    "basis_extension",
                    invocation.attributes["parameter_names"],
                    inputs[1:],
                ),
            ),
        )
