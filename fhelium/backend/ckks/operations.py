"""Native direct implementations of whole CKKS tensor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.ckks.rotation._hoisted import _RotationHoistExecutor
from fhelium.backend.ckks.crypto._galois import (
    coefficient_galois_gather_indices,
)
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.executors.compact_radix2 import CompactRadix2NttBackend
from fhelium.backend.ntt.resources import NTT_RESOURCE_KIND
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend.rns._operand_state import _active_level, _operand_basis
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import RNS_RESOURCE_KIND
from fhelium.values import KeySwitchKey, RelinearizationKey, RotationKey
from fhelium.ir.dialects import ckks, rns
from fhelium.native.wrapper import ckks_ops, rns_ops

from .resources import (
    CONJUGATION_KEY_RESOURCE_KIND,
    KEY_SWITCH_KEY_RESOURCE_KIND,
    KEY_SWITCH_PLAN_RESOURCE_KIND,
    RELINEARIZATION_KEY_RESOURCE_KIND,
    KeySwitchExecutionResource,
)
from ._moddown import moddown_ntt_qp_to_q


def _key_switch_contexts(
    resources: tuple[BoundResource, ...],
    *,
    offset: int = 0,
) -> tuple[RnsContext, NttContext, KeySwitchExecutionResource]:
    """Resolve the RNS, NTT, and key-switch contexts for a direct call."""

    rns_context = cast(RnsContext, resources[offset].value)
    ntt_context = cast(NttContext, resources[offset + 1].value)
    plan = cast(KeySwitchExecutionResource, resources[offset + 2].value)
    if (
        ntt_context.rns_context is not rns_context
        or plan.rns_context is not rns_context
        or plan.ntt_context is not ntt_context
    ):
        raise ValueError("Key-switch resources bind different contexts")
    return rns_context, ntt_context, plan


def _hybrid_modup_digit(
    source: torch.Tensor,
    *,
    rns_context: RnsContext,
    level: int,
    digit_index: int,
) -> torch.Tensor:
    """Extend one coefficient-domain Q digit into active QP rows."""

    digit_spec = rns_context.rns_layout.digit_specs(level)[digit_index]
    source_rows = digit_spec.component_row_ids
    mixed = source[..., source_rows[0] : source_rows[-1] + 1, :]
    digit_width = len(source_rows)
    row_parameters = rns_context.row_parameters(digit_spec.prime_ids)
    if digit_width > 1:
        if digit_width > 8:
            raise ValueError(
                "native streaming key switching supports digit widths "
                "through eight"
            )
        normalizers = row_parameters.mixed_radix_normalizers
        propagation = row_parameters.mixed_radix_propagation_coefficients
        if normalizers is None or propagation is None:
            raise RuntimeError(
                "Streaming relinearization is missing mixed-radix tables"
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
    active_basis = rns_context.basis_parameters(level, include_p=True)
    basis_extension = row_parameters.basis_extension_coefficients
    if basis_extension is None:
        basis_extension = torch.empty(
            0,
            0,
            dtype=source.dtype,
            device=source.device,
        )
    start = active_basis.parameter_row_start
    stop = start + len(active_basis.prime_ids)
    return rns_ops.mixed_radix_basis_extend_to_montgomery(
        mixed,
        basis_extension[:, start:stop],
        active_basis.native_parameters,
        len(active_basis.prime_ids),
    )


def _stream_key_switch_corrections(
    source: torch.Tensor,
    *,
    rns_context: RnsContext,
    ntt_context: NttContext,
    plan: KeySwitchExecutionResource,
    key: KeySwitchKey,
    level: int,
    output_ntt: bool = False,
    coefficient_c0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stream a coefficient Q component and return Q key-switch corrections.

    Compact radix-2 resources consume each digit's NTT evaluations in the
    key-product tail kernel. Other NTT resources use a transform followed by
    key multiplication. Each digit scratch is released before preparing the
    next one. NTT output may combine coefficient c0 into the correction first.
    """

    qp_parameters = rns_context.basis_parameters(level, include_p=True)
    accumulator: torch.Tensor | None = None
    for digit_index, digit_spec in enumerate(
        rns_context.rns_layout.digit_specs(level)
    ):
        lifted = _hybrid_modup_digit(
            source,
            rns_context=rns_context,
            level=level,
            digit_index=digit_index,
        )
        if accumulator is None:
            accumulator = torch.zeros(
                (2, *lifted.shape),
                dtype=lifted.dtype,
                device=lifted.device,
            )
        backend = ntt_context.ntt_backend
        if isinstance(backend, CompactRadix2NttBackend):
            backend.forward_montgomery_accumulate_key_(
                lifted,
                key.digit(digit_spec.key_digit_index),
                accumulator,
                qp_parameters.parameter_row_start,
            )
        else:
            ntt_context.forward_montgomery_(
                lifted,
                parameter_row_start=qp_parameters.parameter_row_start,
            )
            ckks_ops.keyswitch_accumulate_digit_products_(
                accumulator[0],
                accumulator[1],
                lifted,
                key.digit(digit_spec.key_digit_index),
                qp_parameters.native_parameters,
                qp_parameters.parameter_row_start,
            )
        del lifted
    if accumulator is None:
        raise ValueError("Key switching requires at least one RNS digit")

    if output_ntt:
        return moddown_ntt_qp_to_q(
            accumulator, plan, level, coefficient_c0=coefficient_c0
        )

    ntt_context.inverse_to_standard_(
        accumulator,
        parameter_row_start=qp_parameters.parameter_row_start,
    )
    p_count = rns_context.config.num_p_primes
    inverses = plan.moddown_tables[level]
    return ckks_ops.keyswitch_moddown_qp_to_q(
        accumulator[..., :-p_count, :],
        accumulator[..., -p_count:, :],
        inverses,
        qp_parameters.native_parameters,
    )


@dataclass(frozen=True)
class NativeKeySwitchImplementation:
    r"""Switch CT2 secret relations with a shared streaming QP accumulator.

    For input phase $c_0+c_1s_{src}$, the key produces corrections
    $(d_0,d_1)$ under $s_{dst}$ and the result is $(c_0+d_0,d_1)$.
    Conjugation first applies $X\mapsto X^{-1}$ to both components.
    Inputs and outputs use coefficient-domain standard Q residues.
    """

    supports_in_place: bool = False
    name: str = "native-key-switch-streaming"
    operation_types: tuple[type[Operation], ...] = (
        ckks.SwitchKeyOp,
        ckks.ConjugateOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        key_requirement = (
            ResourceRequirement(
                "conjugation-key", CONJUGATION_KEY_RESOURCE_KIND
            )
            if invocation.operation_type is ckks.ConjugateOp
            else ResourceRequirement(
                str(invocation.attributes["key_symbol"]),
                KEY_SWITCH_KEY_RESOURCE_KIND,
            )
        )
        return (
            ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
            ResourceRequirement("active-ntt-plan", NTT_RESOURCE_KIND),
            ResourceRequirement(
                "active-key-switch-plan", KEY_SWITCH_PLAN_RESOURCE_KIND
            ),
            key_requirement,
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        source = inputs[0]
        if source.size(0) != 2:
            raise ValueError("Key switching requires ciphertext components=2")
        rns_context, ntt_context, plan = _key_switch_contexts(resources)
        key = cast(KeySwitchKey, resources[3].value)
        level = _active_level(source, rns_context, include_p=False)
        if invocation.operation_type is ckks.ConjugateOp:
            indices, signs = coefficient_galois_gather_indices(
                rns_context.config.N,
                2 * rns_context.config.N - 1,
                source.device,
            )
            source = ckks_ops.apply_coefficient_galois_automorphism(
                source,
                indices,
                signs,
                rns_context.twice_modulus_for_basis(level, include_p=False),
            )
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        corrections = _stream_key_switch_corrections(
            source[1],
            rns_context=rns_context,
            ntt_context=ntt_context,
            plan=plan,
            key=key,
            level=level,
            output_ntt=output_ntt,
            coefficient_c0=source[0],
        )
        if output_ntt:
            return (corrections,)
        result0 = rns_context.add_standard(
            source[0], corrections[0], include_p=False
        )
        return (torch.stack((result0, corrections[1]), dim=0),)


@dataclass(frozen=True)
class NativeCiphertextMultiplyImplementation:
    """Execute whole two-component CKKS convolution in one native call."""

    supports_in_place: bool = False
    name: str = "native-ct2-convolution"
    operation_types: tuple[type[Operation], ...] = (ckks.MultiplyOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        lhs, rhs = inputs
        resource = cast(RnsContext, resources[0].value)
        parameters = resource.rns_parameters_for(
            lhs,
            include_p=_operand_basis(invocation) == "QP",
        )
        return (
            ckks_ops.multiply_two_component_ntt_montgomery(
                lhs,
                rhs,
                parameters,
            ),
        )


@dataclass(frozen=True)
class NativeRelinearizeImplementation:
    """Relinearize CT3 through one streaming hybrid key-switch schedule."""

    supports_in_place: bool = False
    name: str = "native-relinearize-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RelinearizeOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "relinearization-key",
                RELINEARIZATION_KEY_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        source = inputs[0]
        if source.size(0) != 3:
            raise ValueError(
                "Relinearization requires ciphertext components=3, "
                f"got {source.size(0)}"
            )
        rns_context, ntt_context, plan = _key_switch_contexts(resources)
        key = cast(RelinearizationKey, resources[3].value)

        level = _active_level(source, rns_context, include_p=False)
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        switched_component = source[2].clone()
        ntt_context.inverse_to_standard_(switched_component)
        corrections = _stream_key_switch_corrections(
            switched_component,
            rns_context=rns_context,
            ntt_context=ntt_context,
            plan=plan,
            key=key,
            level=level,
            output_ntt=output_ntt,
        )
        if output_ntt:
            return (
                rns_context.add_lazy(source[:2], corrections, include_p=False),
            )
        coefficient = source[:2].clone()
        ntt_context.inverse_to_standard_(coefficient)
        return (
            rns_context.add_standard(
                coefficient[:2],
                corrections,
                include_p=False,
            ),
        )


@dataclass(frozen=True)
class NativeRotateImplementation:
    """Rotate CT2 through one automorphism and streaming hybrid key switch."""

    supports_in_place: bool = False
    name: str = "native-rotate-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateOp,)
    _ntt_galois_source_index_cache: dict[
        tuple[int, int, int, str], torch.Tensor
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _ntt_galois_source_indices(
        self,
        n: int,
        generator: int,
        rotation_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (n, generator, rotation_step % n, str(device))
        cached = self._ntt_galois_source_index_cache.get(cache_key)
        if cached is not None:
            return cached
        source = torch.arange(n, dtype=torch.int64, device=device)
        remaining = source.clone()
        bit_reversed = torch.zeros_like(source)
        for _ in range(n.bit_length() - 1):
            bit_reversed = (bit_reversed << 1) | (remaining & 1)
            remaining >>= 1
        destination_exponents = 2 * bit_reversed + 1
        exponent = -rotation_step if generator == 5 else rotation_step
        galois_element = pow(generator, exponent % n, 2 * n)
        source_bit_reversed = (
            (destination_exponents * galois_element) % (2 * n) - 1
        ) // 2
        indices = bit_reversed.index_select(
            0, source_bit_reversed.to(torch.long)
        ).to(torch.int32)
        self._ntt_galois_source_index_cache[cache_key] = indices
        return indices

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        source = inputs[0]
        if source.size(0) != 2:
            raise ValueError(
                "Rotation requires ciphertext components=2, "
                f"got {source.size(0)}"
            )
        key = cast(RotationKey, resources[0].value)
        rns_context, ntt_context, plan = _key_switch_contexts(
            resources,
            offset=1,
        )
        shift = key.rotation_step
        exponent = -shift if plan.galois_generator == 5 else shift
        galois_element = pow(
            plan.galois_generator,
            exponent % rns_context.config.N,
            2 * rns_context.config.N,
        )
        level = _active_level(source, rns_context, include_p=False)
        input_ntt = (
            invocation.attributes.get("input_domain", "coefficient") == "ntt"
        )
        if input_ntt:
            source_indices = self._ntt_galois_source_indices(
                rns_context.config.N,
                plan.galois_generator,
                key.rotation_step,
                source.device,
            )
            rotated = ckks_ops.apply_ntt_galois_automorphism(
                source, source_indices
            )
            switched_component = rotated[1].clone()
            ntt_context.inverse_to_standard_(switched_component)
        else:
            source_indices, source_sign = coefficient_galois_gather_indices(
                rns_context.config.N,
                galois_element,
                source.device,
            )
            rotated = ckks_ops.apply_coefficient_galois_automorphism(
                source,
                source_indices,
                source_sign,
                rns_context.twice_modulus_for_basis(level, include_p=False),
            )
            switched_component = rotated[1]
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        corrections = _stream_key_switch_corrections(
            switched_component,
            rns_context=rns_context,
            ntt_context=ntt_context,
            plan=plan,
            key=key,
            level=level,
            output_ntt=output_ntt,
            coefficient_c0=rotated[0] if output_ntt and not input_ntt else None,
        )
        if output_ntt:
            return (
                torch.stack(
                    (
                        rns_context.add_lazy(
                            rotated[0], corrections[0], include_p=False
                        ),
                        corrections[1],
                    )
                )
                if input_ntt
                else corrections,
            )
        component0 = rotated[0]
        if input_ntt:
            component0 = component0.clone()
            ntt_context.inverse_to_standard_(component0)
        result0 = rns_context.add_standard(
            component0,
            corrections[0],
            include_p=False,
        )
        return (torch.stack((result0, corrections[1]), dim=0),)


@dataclass(frozen=True)
class NativeCompressedPlaintextImplementation:
    """Execute compressed plaintext arithmetic on Tensor payloads."""

    supports_in_place: bool = False
    name: str = "native-compressed-plaintext"
    operation_types: tuple[type[Operation], ...] = (
        ckks.AddCompressedPlaintextOp,
        ckks.MultiplyCompressedPlaintextOp,
    )

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        resource = cast(RnsContext, resources[0].value)
        ciphertext, compressed = inputs[:2]
        layout = cast(str, invocation.attributes["compression_layout"])
        include_p = _operand_basis(invocation) == "QP"
        parameters = resource.rns_parameters_for(
            ciphertext,
            include_p=include_p,
        )
        if invocation.operation_type is ckks.AddCompressedPlaintextOp:
            component0 = ciphertext[0]
            if layout == "cyclic" and len(inputs) == 2:
                output0 = ckks_ops.add_cyclic_compressed_plaintext_component(
                    component0,
                    compressed,
                    parameters,
                )
            elif layout == "contiguous" and len(inputs) == 2:
                output0 = (
                    ckks_ops.add_contiguous_compressed_plaintext_component(
                        component0,
                        compressed,
                        parameters,
                    )
                )
            elif layout == "strided_sparse" and len(inputs) == 3:
                output0 = ckks_ops.add_strided_plaintext_component(
                    component0,
                    compressed,
                    inputs[2],
                    parameters,
                )
            else:
                raise ValueError(
                    "Compressed addition inputs differ from its layout"
                )
            output = ciphertext.clone()
            output[0].copy_(output0)
            return (output,)
        if len(inputs) != 2 or layout not in {"cyclic", "contiguous"}:
            raise ValueError(
                "Compressed multiplication requires cyclic or contiguous data"
            )
        if compressed.shape[:-2].numel() == 1:
            if layout == "cyclic":
                return (
                    resource.montgomery_mul_cyclic_compressed(
                        ciphertext,
                        compressed,
                        include_p=include_p,
                    ),
                )
            return (
                resource.montgomery_mul_contiguous_compressed(
                    ciphertext,
                    compressed,
                    include_p=include_p,
                ),
            )
        products = []
        for component_index in range(ciphertext.size(0)):
            component = ciphertext[component_index]
            if layout == "cyclic":
                product = resource.montgomery_mul_cyclic_compressed(
                    component,
                    compressed,
                    include_p=include_p,
                )
            else:
                product = resource.montgomery_mul_contiguous_compressed(
                    component,
                    compressed,
                    include_p=include_p,
                )
            products.append(product)
        return (torch.stack(products, dim=0),)


@dataclass(frozen=True)
class NativeKeySwitchDigitProductImplementation:
    """Multiply one NTT QP digit by an evaluation-key digit."""

    name: str = "native-key-switch-digit-product"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.KeySwitchDigitProductOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

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
            raise ValueError("Key-switch digit product requires one digit")
        digit = values[0]
        key = cast(KeySwitchKey, resources[0].value)
        rns_context = cast(RnsContext, resources[1].value)
        key_digit_index = int(invocation.attributes["key_digit_index"])  # type: ignore[arg-type]
        if key_digit_index >= key.digit_count:
            raise IndexError(
                "Key-switch key digit index is outside key storage"
            )
        accumulator0 = torch.zeros_like(digit)
        accumulator1 = torch.zeros_like(digit)
        level = _active_level(digit, rns_context, include_p=True)
        active = rns_context.basis_parameters(level, include_p=True)
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator0,
            accumulator1,
            digit,
            key.digit(key_digit_index),
            active.native_parameters,
            active.parameter_row_start,
        )
        return (torch.stack((accumulator0, accumulator1), dim=0),)


@dataclass(frozen=True)
class NativeKeySwitchModDownImplementation:
    """Divide QP accumulators by P in coefficient or NTT representation."""

    name: str = "native-key-switch-moddown"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.ModDownQpToQOp,
        rns.ModDownNttQpToQOp,
    )

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

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
            raise ValueError("Key-switch ModDown requires one accumulator")
        source = values[0]
        if source.size(0) != 2:
            raise ValueError(
                "Key-switch ModDown requires two correction components"
            )
        if invocation.operation_type is rns.ModDownNttQpToQOp:
            rns_context, _, plan = _key_switch_contexts(resources)
            level = _active_level(source, rns_context, include_p=True)
            return (moddown_ntt_qp_to_q(source, plan, level),)
        rns_context = cast(RnsContext, resources[0].value)
        plan = cast(KeySwitchExecutionResource, resources[1].value)
        level = _active_level(source, rns_context, include_p=True)
        p_count = rns_context.config.num_p_primes
        inverses = plan.moddown_tables[level]
        parameters = rns_context.basis_parameters(
            level, include_p=True
        ).native_parameters
        corrections = tuple(
            ckks_ops.keyswitch_moddown_qp_to_q(
                source[component, ..., :-p_count, :],
                source[component, ..., -p_count:, :],
                inverses,
                parameters,
            )
            for component in range(2)
        )
        return (torch.stack(corrections, dim=0),)


@dataclass(frozen=True)
class NativeHoistedRotateManyImplementation:
    """Execute one scheduled rotation group without changing its membership."""

    supports_in_place: bool = False
    name: str = "native-rotate-many-hoisted"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateManyOp,)
    _executor_entry: (
        tuple[
            KeySwitchExecutionResource,
            RnsContext,
            NttContext,
            _RotationHoistExecutor,
        ]
        | None
    ) = field(default=None, init=False, repr=False, compare=False)

    def _executor(
        self,
        plan: KeySwitchExecutionResource,
        rns_context: RnsContext,
        ntt_context: NttContext,
    ) -> _RotationHoistExecutor:
        current = self._executor_entry
        if (
            current is not None
            and current[0] is plan
            and current[1] is rns_context
            and current[2] is ntt_context
        ):
            return current[3]
        executor = _RotationHoistExecutor(
            config=rns_context.config,
            rns_context=rns_context,
            ntt_context=ntt_context,
            moddown_p_drop_inverses_montgomery_by_level=plan.moddown_tables,
            galois_generator=plan.galois_generator,
        )
        object.__setattr__(
            self,
            "_executor_entry",
            (plan, rns_context, ntt_context, executor),
        )
        return executor

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return (
            ResourceRequirement(
                "active-rns-parameters",
                RNS_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-ntt-plan",
                NTT_RESOURCE_KIND,
            ),
            ResourceRequirement(
                "active-key-switch-plan",
                KEY_SWITCH_PLAN_RESOURCE_KIND,
            ),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del in_place
        key_count = invocation.result_count
        if len(inputs) != 1 or len(resources) != key_count + 3:
            raise ValueError("Rotate-many input or resource count differs")
        source = inputs[0]
        if source.size(0) != 2:
            raise ValueError(
                "operand 0 requires ciphertext components=2, "
                f"got {source.size(0)}"
            )
        rns_context, ntt_context, plan = _key_switch_contexts(
            resources,
            offset=key_count,
        )
        level = _active_level(source, rns_context, include_p=False)
        executor = self._executor(plan, rns_context, ntt_context)
        prepared = executor.prepare(source[1], level)
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        c0_ntt = (
            ntt_context.forward_to_montgomery(
                source[0], parameter_row_start=level
            )
            if output_ntt
            else None
        )
        outputs: list[torch.Tensor] = []
        for key_binding in resources[:key_count]:
            key = cast(RotationKey, key_binding.value)
            if c0_ntt is not None:
                output0, output1 = executor.apply_ntt(
                    c0_ntt, prepared, key, plan
                )
            else:
                rotated_c0 = executor.rotate_component(
                    source[0],
                    level=level,
                    key=key,
                )
                output0, output1 = executor.apply(rotated_c0, prepared, key)
            outputs.append(torch.stack((output0, output1), dim=0))
        return tuple(outputs)


__all__ = [
    "NativeCiphertextMultiplyImplementation",
    "NativeHoistedRotateManyImplementation",
    "NativeKeySwitchDigitProductImplementation",
    "NativeKeySwitchImplementation",
    "NativeKeySwitchModDownImplementation",
    "NativeRelinearizeImplementation",
    "NativeRotateImplementation",
]
