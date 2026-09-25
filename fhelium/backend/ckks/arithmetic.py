"""Whole ciphertext convolution and compressed-plaintext arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops


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

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        parameters = inputs[-1]
        ciphertext, compressed = inputs[:2]
        layout = invocation.attributes["compression_layout"]
        implicit = inputs[2] if layout == "strided_sparse" else None
        if invocation.operation_type is ckks.AddCompressedPlaintextOp:
            if (
                invocation.attributes.get("polynomial_domain", "coefficient")
                == "ntt"
            ):
                compressed = compressed.clone()
                rns_ops.to_montgomery_(compressed, parameters)
                if implicit is not None:
                    implicit = implicit.clone()
                    rns_ops.to_montgomery_(implicit, parameters)
            output = ciphertext if in_place else ciphertext.clone()
            if layout == "cyclic":
                ckks_ops.add_cyclic_compressed_plaintext_component_(
                    output[0], compressed, parameters
                )
            elif layout == "contiguous":
                ckks_ops.add_contiguous_compressed_plaintext_component_(
                    output[0], compressed, parameters
                )
            elif layout == "strided_sparse" and implicit is not None:
                implicit = implicit.squeeze(-1)
                if implicit.device.type == "cpu":
                    implicit = implicit.reshape(
                        -1, implicit.size(-1)
                    ).contiguous()
                ckks_ops.add_strided_plaintext_component_(
                    output[0], compressed, implicit, parameters
                )
            else:
                raise ValueError(
                    "Compressed addition requires a supported layout and its implicit rows"
                )
            return (output,)

        def multiply(component):
            if layout == "cyclic":
                return rns_ops.montgomery_mul_cyclic_compressed(
                    component, compressed, parameters
                )
            if layout == "contiguous":
                return rns_ops.montgomery_mul_contiguous_compressed(
                    component, compressed, parameters
                )
            if layout == "strided_sparse" and implicit is not None:
                result = rns_ops.montgomery_mul_cyclic_compressed(
                    component, implicit, parameters
                )
                repeat = component.size(-1) // compressed.size(-1)
                result[..., ::repeat].copy_(
                    rns_ops.montgomery_mul(
                        component[..., ::repeat], compressed, parameters
                    )
                )
                return result
            raise ValueError(
                "Compressed multiplication requires a supported layout and its implicit rows"
            )

        if compressed.shape[:-2].numel() == 1:
            product = multiply(ciphertext)
        else:
            product = torch.stack(
                tuple(multiply(component) for component in ciphertext)
            )
        if in_place:
            ciphertext.copy_(product)
            return (ciphertext,)
        return (product,)
