"""Whole ciphertext convolution and compressed-plaintext arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops


@dataclass(frozen=True)
class NativeCiphertextMultiplyImplementation:
    """Execute whole two-component CKKS convolution in one native call."""

    supports_in_place: bool = False
    name: str = "native-ct2-convolution"
    operation_types: tuple[type[Operation], ...] = (ckks.MultiplyOp,)

    @staticmethod
    def tensor_requirements(operation, config):
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

        if operation.parameters:
            return {}, {}
        ids = operation.lhs.type.state.data.get("prime_ids")
        if not isinstance(ids, ArrayAttr) or not all(
            isinstance(x, IntegerAttr) for x in ids
        ):
            return None
        return {}, {
            "rns_parameters": {
                "kind": "rns_parameters",
                "prime_ids": [int(x.value.data) for x in ids],
            }
        }

    def supports_operation(self, operation: Operation) -> bool:
        """Require the numerical operands consumed by this whole implementation."""
        return bool(cast(ckks.MultiplyOp, operation).parameters)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, in_place
        lhs, rhs, parameters = inputs
        return (
            ckks_ops.multiply_two_component_ntt_montgomery(
                lhs,
                rhs,
                parameters,
            ),
        )


@dataclass(frozen=True)
class NativeCompressedPlaintextImplementation:
    """Evaluate compact plaintext layouts without expanding plaintext storage.

    Coefficient addition consumes pR. NTT addition first represents the compact
    plaintext as pR², then reuses the coefficient primitive's REDC to compute
    cR+pR. Sparse multiplication fills implicit positions using a one-value
    cyclic row and replaces the strided explicit positions with their products.
    """

    supports_in_place = True
    name = "native-compressed-plaintext"
    operation_types = (
        ckks.AddCompressedPlaintextOp,
        ckks.MultiplyCompressedPlaintextOp,
    )

    @staticmethod
    def tensor_requirements(operation, config):
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

        layout = operation.attributes.get("compression_layout")
        implicit_count = int(getattr(layout, "data", None) == "strided_sparse")
        if len(operation.parameters) > implicit_count:
            return {}, {}
        ids = operation.ciphertext.type.state.data.get("prime_ids")
        if not isinstance(ids, ArrayAttr) or not all(
            isinstance(x, IntegerAttr) for x in ids
        ):
            return None
        return {}, {
            "rns_parameters": {
                "kind": "rns_parameters",
                "prime_ids": [int(x.value.data) for x in ids],
            }
        }

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        del resources
        parameters = inputs[-1]
        ciphertext, compressed = inputs[:2]
        layout = str(invocation.attributes["compression_layout"])
        implicit = inputs[2] if layout == "strided_sparse" else None
        if invocation.operation_type is ckks.AddCompressedPlaintextOp:
            domain = str(
                invocation.attributes.get("polynomial_domain", "coefficient")
            )
            add = (
                ckks_ops.add_compressed_plaintext_
                if in_place
                else ckks_ops.add_compressed_plaintext
            )
            return (
                add(
                    ciphertext, compressed, implicit, parameters, layout, domain
                ),
            )
        multiply = (
            ckks_ops.multiply_compressed_plaintext_
            if in_place
            else ckks_ops.multiply_compressed_plaintext
        )
        return (multiply(ciphertext, compressed, implicit, parameters, layout),)
