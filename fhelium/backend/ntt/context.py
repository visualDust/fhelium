"""Build one NTT executor over a device-local RNS parameter context."""

from __future__ import annotations

import torch

from fhelium.backend.ntt.factory import create_ntt_backend
from fhelium.backend.ntt.tables import prepare_ntt_tables
from fhelium.backend.rns.context import RnsContext
from fhelium.config.ntt import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    resolve_ntt_backend_policy,
    validate_ntt_backend_for_log_n,
)


class NttContext:
    r"""Own one NTT policy, its tables, and its configured executor.

    The context composes an :class:`RnsContext` whose parameter tensor is
    passed unchanged to native NTT executors. Transform methods preserve prime
    rows and operand storage unless the method is the functional
    :meth:`forward_to_montgomery` variant.
    """

    def __init__(
        self,
        rns_context: RnsContext,
        ntt_backend: str | None = None,
    ) -> None:
        if not isinstance(rns_context, RnsContext):
            raise TypeError("NttContext requires an RnsContext")
        self.rns_context = rns_context
        self.device = rns_context.device
        self.config = rns_context.config
        if ntt_backend is None:
            ntt_backend = (
                DEFAULT_CPU_NTT_BACKEND
                if self.device.type == "cpu"
                else DEFAULT_NTT_BACKEND
            )
        self.ntt_policy = resolve_ntt_backend_policy(ntt_backend)
        validate_ntt_backend_for_log_n(self.ntt_policy, self.config.logN)
        if (
            self.device.type == "cpu"
            and self.ntt_policy.name != DEFAULT_CPU_NTT_BACKEND
        ):
            raise ValueError(
                "CPU execution currently requires ntt_backend="
                f"{DEFAULT_CPU_NTT_BACKEND!r}"
            )
        self.ntt_backend_name = self.ntt_policy.name
        self.ntt_tables = prepare_ntt_tables(
            self.ntt_policy,
            self.config,
            materialize_parameter_rows=rns_context.materialize_parameter_rows,
            device=self.device,
            dtype=rns_context.dtype,
        )
        self.ntt_tables.convert_twiddles_to_montgomery_(
            rns_context.rns_parameter_tensor
        )
        self.ntt_backend = create_ntt_backend(
            self.ntt_policy,
            ntt_tables=self.ntt_tables,
            rns_params=rns_context.rns_parameter_tensor,
        )

    def _active_row_start(
        self,
        a: torch.Tensor,
        *,
        include_p: bool,
        parameter_row_start: int | None,
    ) -> int:
        return self.rns_context.parameter_row_start_for(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )

    def forward_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Apply the negacyclic forward NTT to Montgomery residues in place."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.forward_montgomery_(a, row_start)

    def forward_to_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Apply forward NTT and standard-to-Montgomery conversion in place."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.forward_to_montgomery_(a, row_start)

    def forward_to_montgomery(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> torch.Tensor:
        r"""Return the non-aliasing forward NTT/Montgomery transition."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        return self.ntt_backend.forward_to_montgomery(a, row_start)

    def inverse_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Apply normalized inverse NTT and keep Montgomery form."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_montgomery_(a, row_start)

    def inverse_to_standard_lazy_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Inverse NTT to coefficient/standard lazy residues in place."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_to_standard_lazy_(a, row_start)

    def inverse_to_standard_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Inverse NTT to coefficient/standard reduced residues in place."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_to_standard_(a, row_start)

    def inverse_to_centered_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Inverse NTT to centered coefficient/standard residues in place."""

        row_start = self._active_row_start(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_to_centered_(a, row_start)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"NttContext(backend={self.ntt_backend_name!r}, "
            f"backend_impl={self.ntt_backend}, logN={self.config.logN}, "
            f"device={self.device}, ntt_policy={self.ntt_policy})"
        )
