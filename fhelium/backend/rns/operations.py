"""Native implementations of logical RNS operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops, rns_ops

from fhelium.backend.rns._operand_state import (
    _active_level,
    _operand_basis,
)
from fhelium.backend.rns.context import RnsContext


class _OperandResourceImplementation:
    """Declare that resources arrive as operation operands in IR order."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


@dataclass(frozen=True)
class NativeRnsLinearImplementation(_OperandResourceImplementation):
    """Execute standard-range RNS add, subtract, and negate operations."""

    name: str = "native-rns-linear"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.AddStandardOp,
        rns.SubtractStandardOp,
        rns.NegateStandardOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        lhs = values[0]
        resource = cast(RnsContext, resources[0].value)
        include_p = _operand_basis(invocation) == "QP"
        if invocation.operation_type is rns.NegateStandardOp:
            output_data = lhs if in_place else lhs.clone()
            output_data.neg_()
            level = _active_level(
                lhs,
                resource,
                include_p=include_p,
            )
            stop = resource.qp_row_stop if include_p else resource.q_row_stop
            active_moduli = resource.moduli[level:stop]
            output_data.remainder_(
                active_moduli.view(
                    *([1] * (output_data.ndim - 2)),
                    active_moduli.numel(),
                    1,
                )
            )
            return (output_data,)
        rhs = values[1]
        try:
            lhs.view(
                -1,
                lhs.size(-2),
                lhs.size(-1),
            )
            rhs.view(
                -1,
                rhs.size(-2),
                rhs.size(-1),
            )
        except RuntimeError:
            if in_place:
                method = (
                    resource.add_standard_
                    if invocation.operation_type is rns.AddStandardOp
                    else resource.sub_standard_
                )
                for component in range(lhs.size(0)):
                    method(
                        lhs[component],
                        rhs[component],
                        include_p=include_p,
                    )
                return (lhs,)
            method = (
                resource.add_standard
                if invocation.operation_type is rns.AddStandardOp
                else resource.sub_standard
            )
            return (
                torch.stack(
                    tuple(
                        method(
                            lhs[component],
                            rhs[component],
                            include_p=include_p,
                        )
                        for component in range(lhs.size(0))
                    ),
                    dim=0,
                ),
            )
        else:
            if not in_place:
                method = (
                    resource.add_standard
                    if invocation.operation_type is rns.AddStandardOp
                    else resource.sub_standard
                )
                return (method(lhs, rhs, include_p=include_p),)
            method = (
                resource.add_standard_
                if invocation.operation_type is rns.AddStandardOp
                else resource.sub_standard_
            )
            method(lhs, rhs, include_p=include_p)
        return (lhs,)


@dataclass(frozen=True)
class NativeRnsTransitionImplementation(_OperandResourceImplementation):
    """Execute residue conversion, level restriction, and scale metadata ops."""

    name: str = "native-rns-transition"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.StandardToMontgomeryOp,
        rns.MontgomeryToStandardOp,
        rns.RestrictLevelOp,
        rns.ReinterpretScaleOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        source = values[0]
        if invocation.operation_type in {
            rns.StandardToMontgomeryOp,
            rns.MontgomeryToStandardOp,
        }:
            resource = cast(RnsContext, resources[0].value)
            data = source if in_place else source.clone()
            include_p = _operand_basis(invocation) == "QP"
            if invocation.operation_type is rns.StandardToMontgomeryOp:
                resource.to_montgomery_(
                    data,
                    include_p=include_p,
                )
            else:
                resource.from_montgomery_(
                    data,
                    include_p=include_p,
                )
            return (data,)
        if resources:
            expected_prime_ids = invocation.result_prime_ids[0]
            if expected_prime_ids is None:
                raise ValueError(
                    "Level restriction requires a physical result row count"
                )
            result_rows = len(expected_prime_ids)
            source_rows = source.size(-2)
            if not 0 < result_rows <= source_rows:
                raise ValueError(
                    "Level restriction result rows must be within the source "
                    "Tensor extent"
                )
            selected = source.narrow(
                -2,
                source_rows - result_rows,
                result_rows,
            )
            data = selected if in_place else selected.clone()
            return (data,)
        return (source if in_place else source.clone(),)


@dataclass(frozen=True)
class NativePlaintextArithmeticImplementation(_OperandResourceImplementation):
    """Execute prepared plaintext addition and multiplication."""

    name: str = "native-plaintext-arithmetic"
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = (
        rns.AddPlaintextOp,
        rns.MultiplyPlaintextOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        ciphertext, plaintext = values
        resource = cast(RnsContext, resources[0].value)
        include_p = _operand_basis(invocation) == "QP"
        parameters = resource.rns_parameters_for(
            ciphertext,
            include_p=include_p,
        )
        if invocation.operation_type is rns.AddPlaintextOp:
            if (
                invocation.attributes.get("polynomial_domain", "coefficient")
                == "ntt"
            ):
                component0 = resource.add_lazy(
                    ciphertext[0],
                    plaintext,
                    include_p=include_p,
                )
                if in_place:
                    ciphertext[0].copy_(component0)
                    return (ciphertext,)
                return (
                    torch.cat((component0.unsqueeze(0), ciphertext[1:]), dim=0),
                )
            if in_place:
                ckks_ops.add_prepared_plaintext_component_(
                    ciphertext[0],
                    plaintext,
                    parameters,
                )
                return (ciphertext,)
            component0 = ckks_ops.add_prepared_plaintext_component(
                ciphertext[0],
                plaintext,
                parameters,
            )
            return (
                torch.cat((component0.unsqueeze(0), ciphertext[1:]), dim=0),
            )
        if plaintext.shape[:-2].numel() == 1:
            return (
                resource.montgomery_mul(
                    ciphertext,
                    plaintext,
                    include_p=include_p,
                ),
            )
        products = [
            resource.montgomery_mul(
                ciphertext[component],
                plaintext,
                include_p=include_p,
            )
            for component in range(ciphertext.size(0))
        ]
        return (torch.stack(products, dim=0),)


@dataclass(frozen=True)
class NativeMontgomeryMultiplyImplementation(_OperandResourceImplementation):
    """Multiply two logical NTT/Montgomery polynomial bundles."""

    name: str = "native-montgomery-multiply"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.MontgomeryMultiplyOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        lhs, rhs = values
        resource = cast(RnsContext, resources[0].value)
        return (
            resource.montgomery_mul(
                lhs,
                rhs,
                include_p=_operand_basis(invocation) == "QP",
            ),
        )


@dataclass(frozen=True)
class NativeRnsStructureImplementation(_OperandResourceImplementation):
    """Extract and assemble logical RNS component bundles."""

    name: str = "native-rns-structure"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.ExtractComponentOp,
        rns.PackTwoComponentsOp,
        rns.PackThreeComponentsOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if resources:
            raise ValueError("RNS structural operations accept no resources")
        if invocation.operation_type is rns.ExtractComponentOp:
            if len(values) != 1:
                raise ValueError("RNS component extraction requires one value")
            source = values[0]
            component = int(invocation.attributes["component"])  # type: ignore[arg-type]
            if component >= source.size(0):
                raise IndexError(
                    f"RNS component {component} is outside "
                    f"[0, {source.size(0)})"
                )
            return (source[component],)
        expected_count = (
            3 if invocation.operation_type is rns.PackThreeComponentsOp else 2
        )
        if len(values) != expected_count:
            raise ValueError(
                f"RNS component packing requires {expected_count} values"
            )
        return (torch.stack(values, dim=0),)


def _rns_resource(
    resources: tuple[BoundResource, ...],
) -> RnsContext:
    return cast(RnsContext, resources[0].value)


@dataclass(frozen=True)
class NativeHybridModUpImplementation(_OperandResourceImplementation):
    """Execute one represented hybrid-digit ModUp using concrete RNS tables."""

    name: str = "native-hybrid-modup"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.HybridModUpDigitOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if len(values) != 1:
            raise ValueError("Hybrid ModUp requires one source polynomial")
        source = values[0]
        resource = _rns_resource(resources)
        level = _active_level(source, resource, include_p=False)
        digit_specs = resource.rns_layout.digit_specs(level)
        digit_index = int(invocation.attributes["digit_index"])  # type: ignore[arg-type]
        if digit_index >= len(digit_specs):
            raise IndexError(
                "Hybrid ModUp digit index is outside the active layout"
            )
        digit_spec = digit_specs[digit_index]
        source_rows = digit_spec.component_row_ids
        mixed = source[..., source_rows[0] : source_rows[-1] + 1, :].clone()
        digit_width = len(source_rows)
        if digit_width > 1:
            if digit_width > 8:
                raise ValueError(
                    "native-hybrid-modup supports digit widths through eight"
                )
            row_parameters = resource.row_parameters(digit_spec.prime_ids)
            normalizers = row_parameters.mixed_radix_normalizers
            propagation = row_parameters.mixed_radix_propagation_coefficients
            if normalizers is None or propagation is None:
                raise RuntimeError(
                    "Hybrid ModUp is missing mixed-radix parameter tables"
                )
            modulus_lo, modulus_hi, neg_inv_lo, neg_inv_hi = (
                row_parameters.montgomery_reduction_parameters
            )
            mixed = rns_ops.mixed_radix_decompose(
                mixed,
                normalizers.contiguous(),
                propagation.contiguous(),
                modulus_lo.contiguous(),
                modulus_hi.contiguous(),
                neg_inv_lo.contiguous(),
                neg_inv_hi.contiguous(),
            )
        active_basis = resource.basis_parameters(level, include_p=True)
        basis_extension = resource.row_parameters(
            digit_spec.prime_ids
        ).basis_extension_coefficients
        if basis_extension is None:
            basis_extension = torch.empty(
                0,
                0,
                dtype=source.dtype,
                device=source.device,
            )
        start = active_basis.parameter_row_start
        stop = start + len(active_basis.prime_ids)
        return (
            rns_ops.mixed_radix_basis_extend_to_montgomery(
                mixed,
                basis_extension[:, start:stop],
                active_basis.native_parameters,
                len(active_basis.prime_ids),
            ),
        )


@dataclass(frozen=True)
class NativeMontgomeryAccumulateImplementation(_OperandResourceImplementation):
    """Accumulate equal-layout Montgomery bundles modulo twice each prime."""

    name: str = "native-montgomery-accumulate"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.AddMontgomeryLazyOp,)

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if len(values) != 2:
            raise ValueError("Montgomery accumulation requires two values")
        lhs, rhs = values
        resource = _rns_resource(resources)
        return (
            resource.add_lazy(
                lhs,
                rhs,
                include_p=_operand_basis(invocation) == "QP",
            ),
        )


@dataclass(frozen=True)
class NativeCoefficientAutomorphismImplementation(
    _OperandResourceImplementation
):
    """Apply one coefficient-domain Galois automorphism through the native op."""

    name: str = "native-coefficient-automorphism"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.CoefficientAutomorphismOp,
    )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        if len(values) != 1:
            raise ValueError("Coefficient automorphism requires one value")
        source = values[0]
        resource = _rns_resource(resources)
        ring_dimension = source.size(-1)
        modulus = 2 * ring_dimension
        galois_element = int(invocation.attributes["galois_element"]) % modulus  # type: ignore[arg-type]
        source_index = torch.arange(
            ring_dimension, device=source.device, dtype=torch.int64
        )
        mapped = (galois_element * source_index) % modulus
        destination = mapped % ring_dimension
        sign = 1 - 2 * (((mapped // ring_dimension) & 1) != 0).to(torch.int8)
        gather = torch.empty(
            ring_dimension, dtype=torch.int32, device=source.device
        )
        gather[destination] = source_index.to(torch.int32)
        source_sign = torch.empty(
            ring_dimension, dtype=torch.int8, device=source.device
        )
        source_sign[destination] = sign
        twice_modulus = resource.twice_modulus_for_basis(
            _active_level(
                source,
                resource,
                include_p=_operand_basis(invocation) == "QP",
            ),
            include_p=_operand_basis(invocation) == "QP",
        )
        return (
            ckks_ops.apply_coefficient_galois_automorphism(
                source,
                gather,
                source_sign,
                twice_modulus,
            ),
        )


__all__ = [
    "NativeCoefficientAutomorphismImplementation",
    "NativeHybridModUpImplementation",
    "NativeMontgomeryAccumulateImplementation",
    "NativeMontgomeryMultiplyImplementation",
    "NativePlaintextArithmeticImplementation",
    "NativeRnsLinearImplementation",
    "NativeRnsStructureImplementation",
    "NativeRnsTransitionImplementation",
]
