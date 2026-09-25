"""Prepare one device's CKKS numerical tables and sampling state."""

from __future__ import annotations

from typing import cast

import torch

from fhelium.config import CkksConfig
from fhelium.rng import Csprng

from ..ntt.context import NttContext
from ..resources import (
    BoundResource,
    ResourceBindings,
    ResourceRequirement,
)
from ..rns.context import RnsContext
from ..rns.layout import RnsLayout
from .crypto import (
    RANDOM_STREAM_RESOURCE_KIND,
    RANDOM_STREAM_RESOURCE_SYMBOL,
)
from .crypto import (
    DecryptReconstructionTables,
    KeyGenerationResource,
)
from .tables import KeySwitchTables
from ..rns.tables import (
    RescaleTables,
    rescale_inverse_matrix,
    moddown_inverse_tables,
)


class CkksDeviceResources:
    """Own lazily constructed CKKS resources for one concrete device.

    The default RNS layout follows the configuration's Q depth groups and
    hybrid decomposition. Callers can supply an existing layout for reuse.
    """

    def __init__(
        self,
        *,
        config: CkksConfig,
        device: str | torch.device,
        rns_layout: RnsLayout | None = None,
        ntt_backend: str | None = None,
        rng_seed: int | None = None,
        rng_nonce: int | None = None,
        rns_dtype: torch.dtype | None = None,
    ) -> None:
        self.config = config
        self.rns_layout = (
            RnsLayout.from_config(config) if rns_layout is None else rns_layout
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.ntt_backend = ntt_backend
        self.rng_seed = rng_seed
        self.rng_nonce = rng_nonce
        self.rns_dtype = rns_dtype
        self._bindings: dict[str, BoundResource] = {}
        self._periodic_tables: dict[
            tuple[int, int, str], tuple[torch.Tensor, ...]
        ] = {}
        self._rns_context: RnsContext | None = None
        self._ntt_context: NttContext | None = None
        self._rng: Csprng | None = None
        self._key_generation: KeyGenerationResource | None = None
        self._rescale: RescaleTables | None = None
        self._key_switch: KeySwitchTables | None = None
        self._decrypt_reconstruction: DecryptReconstructionTables | None = None

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

    def periodic_encode_operands(self, period: int, depth: int, basis: str):
        """Return compact tables and the existing live rounding stream."""
        from .codec._periodic import periodic_materials

        cache = self._periodic_tables
        identity = (period, depth, basis)
        if identity not in cache:
            cache[identity] = periodic_materials(
                self.config,
                self.rns_context,
                period,
                depth,
                basis,
                self.rng.rounding_state,
            )
        return cache[identity]

    def encode_operands(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the inverse-embedding permutation, twister, and live rounding state."""
        from .codec import _embedding

        pre, _ = _embedding._permutations(
            self.config.N, str(self.device), self.config.galois_generator
        )
        return (
            pre,
            _embedding._twister(self.config.N, str(self.device)),
            self.rng.rounding_state,
        )

    def decode_operands(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the coefficient-embedding permutation and skewer."""
        from .codec import _embedding

        _, post = _embedding._permutations(
            self.config.N, str(self.device), self.config.galois_generator
        )
        return post, _embedding._skewer(self.config.N, str(self.device))

    def key_switch_operands(
        self, depth: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Prepare Tensor views and digit ranges for a hybrid key switch."""
        tables = self._key_switch
        if tables is None:
            tables = KeySwitchTables(
                self.rns_context,
                self.ntt_context,
                moddown_inverse_tables(self.rns_context),
                self.config.galois_generator,
            )
            self._key_switch = tables
        return tables.tensor_operands(depth)

    def rescale_operands(
        self,
        row_count: int,
        drop_count: int,
        *,
        include_p: bool = False,
        input_domain: str = "coefficient",
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Prepare quotient and transform tables for one rescale group."""
        tables = self._rescale
        if tables is None:
            tables = RescaleTables(
                self.rns_context, rescale_inverse_matrix(self.rns_context)
            )
            self._rescale = tables
        return tables.tensor_operands(
            row_count,
            drop_count,
            include_p=include_p,
            ntt_context=self.ntt_context if input_domain == "ntt" else None,
        )

    def reconstruction_operands(self, depth: int) -> tuple[torch.Tensor, ...]:
        """Prepare Q parameters and centered mixed-radix reconstruction tables."""
        tables = self._decrypt_reconstruction
        if tables is None:
            tables = DecryptReconstructionTables.create(self.rns_context)
            self._decrypt_reconstruction = tables
        return tables.tensor_operands(self.rns_context, depth)

    @property
    def bindings(self) -> ResourceBindings:
        """Return this device's bindable random-stream execution resource."""
        return self.materialize(
            (
                ResourceRequirement(
                    RANDOM_STREAM_RESOURCE_SYMBOL, RANDOM_STREAM_RESOURCE_KIND
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
                if requirement.kind != RANDOM_STREAM_RESOURCE_KIND:
                    raise KeyError(
                        f"CkksDeviceResources cannot construct execution resource {requirement.kind!r}; numerical parameters are Tensor operands"
                    )
                value = self.rng
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
