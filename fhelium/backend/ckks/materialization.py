"""Materialize one device's reusable CKKS Backend resources."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.config import CkksConfig
from fhelium.rng import Csprng

from ..ntt.context import NttContext
from ..ntt.resources import NTT_RESOURCE_KIND
from ..resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from ..rns.context import RnsContext
from ..rns.layout import RnsLayout
from ..rns.resources import RNS_RESOURCE_KIND
from .codec import (
    CKKS_CONFIG_RESOURCE_KIND,
    CKKS_CONFIG_RESOURCE_SYMBOL,
    RANDOM_STREAM_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_SYMBOL,
)
from .crypto import (
    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
    DecryptReconstructionResource,
    KeyGenerationResource,
)
from .resources import (
    KEY_SWITCH_PLAN_RESOURCE_KIND,
    RESCALE_RESOURCE_KIND,
    KeySwitchExecutionResource,
    RescaleExecutionResource,
)


def _rescale_inverse_matrix(rns_context: RnsContext) -> torch.Tensor:
    montgomery = rns_context.montgomery_parameters
    rows = []
    for dropped_id, dropped_prime in enumerate(montgomery.moduli):
        rows.append(
            [
                0
                if remaining_id == dropped_id
                else (pow(dropped_prime, -1, remaining_prime) * montgomery.R)
                % remaining_prime
                for remaining_id, remaining_prime in enumerate(
                    montgomery.moduli
                )
            ]
        )
    return torch.tensor(
        rows,
        dtype=rns_context.dtype,
        device=rns_context.device,
    )


def _keyswitch_moddown_tables(
    rns_context: RnsContext,
) -> tuple[torch.Tensor, ...]:
    config = rns_context.config
    montgomery = rns_context.montgomery_parameters
    p_moduli = montgomery.moduli[-config.num_p_primes :][::-1]
    inverse_rows = [
        [
            (pow(dropped, -1, modulus) * montgomery.R) % modulus
            for modulus in montgomery.moduli[: -drop_step - 1]
        ]
        for drop_step, dropped in enumerate(p_moduli)
    ]
    depth0_prime_ids = rns_context.rns_layout.prime_ids(0, include_p=True)
    base_rows = [
        torch.tensor(
            [
                inverse_rows[drop_step][prime_id]
                for prime_id in depth0_prime_ids[: -drop_step - 1]
            ],
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
        for drop_step in range(config.num_p_primes)
    ]
    tables = []
    for depth in range(config.max_depth + 1):
        start = rns_context.basis_parameters(depth).parameter_row_start
        rows = [row[start:] for row in base_rows]
        packed = torch.empty(
            (len(rows), max(row.numel() for row in rows)),
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
        for row_index, row in enumerate(rows):
            packed[row_index, : row.numel()] = row
        tables.append(packed)
    return tuple(tables)


class CkksDeviceResources:
    """Own lazily constructed CKKS resources for one concrete device."""

    def __init__(
        self,
        *,
        config: CkksConfig,
        rns_layout: RnsLayout,
        device: str | torch.device,
        ntt_backend: str | None = None,
        rng_seed: int | None = None,
        rng_nonce: int | None = None,
        rns_dtype: torch.dtype | None = None,
    ) -> None:
        self.config = config
        self.rns_layout = rns_layout
        self.device = torch.device(device)
        self.ntt_backend = ntt_backend
        self.rng_seed = rng_seed
        self.rng_nonce = rng_nonce
        self.rns_dtype = rns_dtype
        self._bindings: dict[str, BoundResource] = {}
        self._rns_context: RnsContext | None = None
        self._ntt_context: NttContext | None = None
        self._rng: Csprng | None = None
        self._key_generation: KeyGenerationResource | None = None
        self._rescale: RescaleExecutionResource | None = None
        self._key_switch: KeySwitchExecutionResource | None = None
        self._decrypt_reconstruction: DecryptReconstructionResource | None = (
            None
        )

    @property
    def rns_context(self) -> RnsContext:
        """Return the device's RNS arithmetic context."""

        context = self._rns_context
        if context is not None:
            return context
        from fhelium.native import require_native_backend

        require_native_backend(self.device.type)
        context = RnsContext(
            self.config,
            device=self.device,
            rns_layout=self.rns_layout,
            dtype=self.rns_dtype,
        )
        self._rns_context = context
        return context

    @property
    def ntt_context(self) -> NttContext:
        """Return the NTT context composed with this device's RNS context."""

        context = self._ntt_context
        if context is None:
            context = NttContext(
                self.rns_context,
                ntt_backend=self.ntt_backend,
            )
            self._ntt_context = context
        return context

    @property
    def rng(self) -> Csprng:
        """Return the stateful random stream assigned to this device."""

        rng = self._rng
        if rng is not None:
            return rng
        stream_id = (
            0
            if self.device.type == "cpu"
            else 1 + int(cast(int, self.device.index))
        )
        stream_nonce = (
            None
            if self.rng_nonce is None
            else (self.rng_nonce + stream_id) & ((1 << 64) - 1)
        )
        rng = Csprng(
            num_coefs=self.config.N,
            num_channels=[len(self.rns_layout.prime_ids(0))],
            num_repeating_channels=max(self.config.num_p_primes, 2),
            sigma=self.config.sigma,
            devices=[str(self.device)],
            torch_dtype=self.rns_context.dtype,
            seed=self.rng_seed,
            nonce=stream_nonce,
        )
        self._rng = rng
        return rng

    @property
    def key_generation(self) -> KeyGenerationResource:
        """Return the resources used to generate keys on this device."""

        resource = self._key_generation
        if resource is None:
            resource = KeyGenerationResource.create(
                config=self.config,
                rng=self.rng,
                rns_context=self.rns_context,
                ntt_context=self.ntt_context,
            )
            self._key_generation = resource
        return resource

    def _value(self, kind: str) -> object:
        if kind == RNS_RESOURCE_KIND:
            return self.rns_context
        if kind == NTT_RESOURCE_KIND:
            return self.ntt_context
        if kind == CKKS_CONFIG_RESOURCE_KIND:
            return self.config
        if kind == RANDOM_STREAM_RESOURCE_KIND:
            return self.rng
        if kind == RESCALE_RESOURCE_KIND:
            resource = self._rescale
            if resource is None:
                resource = RescaleExecutionResource(
                    self.rns_context,
                    _rescale_inverse_matrix(self.rns_context),
                )
                self._rescale = resource
            return resource
        if kind == KEY_SWITCH_PLAN_RESOURCE_KIND:
            resource = self._key_switch
            if resource is None:
                resource = KeySwitchExecutionResource(
                    self.rns_context,
                    self.ntt_context,
                    _keyswitch_moddown_tables(self.rns_context),
                    self.config.galois_generator,
                )
                self._key_switch = resource
            return resource
        if kind == DECRYPT_RECONSTRUCTION_RESOURCE_KIND:
            resource = self._decrypt_reconstruction
            if resource is None:
                resource = DecryptReconstructionResource.create(
                    self.rns_context
                )
                self._decrypt_reconstruction = resource
            return resource
        raise KeyError(f"CKKS cannot construct resource kind {kind!r}")

    @property
    def bindings(self) -> ResourceBindings:
        """Return the complete built-in CKKS execution binding set."""

        return self.materialize(
            (
                ResourceRequirement("active-rns-parameters", RNS_RESOURCE_KIND),
                ResourceRequirement("active-ntt-plan", NTT_RESOURCE_KIND),
                ResourceRequirement(
                    "active-rescale-plan", RESCALE_RESOURCE_KIND
                ),
                ResourceRequirement(
                    "active-key-switch-plan",
                    KEY_SWITCH_PLAN_RESOURCE_KIND,
                ),
                ResourceRequirement(
                    CKKS_CONFIG_RESOURCE_SYMBOL,
                    CKKS_CONFIG_RESOURCE_KIND,
                ),
                ResourceRequirement(
                    RANDOM_STREAM_RESOURCE_SYMBOL,
                    RANDOM_STREAM_RESOURCE_KIND,
                ),
                ResourceRequirement(
                    "ckks-decrypt-reconstruction",
                    DECRYPT_RECONSTRUCTION_RESOURCE_KIND,
                ),
            )
        )

    def materialize(
        self,
        requirements: tuple[ResourceRequirement, ...],
        /,
    ) -> ResourceBindings:
        """Return all constructible missing resources in one batch."""

        selected: list[BoundResource] = []
        for requirement in requirements:
            resource = self._bindings.get(requirement.symbol)
            if resource is None:
                try:
                    value = self._value(requirement.kind)
                except KeyError:
                    raise KeyError(
                        "Caller must supply non-materializable Backend "
                        f"resource {requirement.symbol!r}/{requirement.kind}"
                    ) from None
                resource = BoundResource(
                    requirement.symbol,
                    requirement.kind,
                    value,
                )
                self._bindings[requirement.symbol] = resource
            elif resource.kind != requirement.kind:
                raise ValueError(
                    f"Resource symbol {requirement.symbol!r} was materialized "
                    f"as {resource.kind!r}, not {requirement.kind!r}"
                )
            selected.append(resource)
        return ResourceBindings(tuple(selected))


__all__ = [
    "CkksDeviceResources",
]
