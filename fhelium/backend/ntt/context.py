"""Provide NTT tables and transforms over a device-local RNS context."""

from __future__ import annotations

import torch

from fhelium.backend.ntt.operations import prepare_transition
from fhelium.backend.ntt.tables import prepare_ntt_tables
from fhelium.backend.rns.context import RnsContext
from fhelium.config.ntt import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    resolve_ntt_backend_policy,
    validate_ntt_backend_for_log_n,
)


class NttContext:
    r"""Own one NTT policy and its tables, and use shared prepared transforms.

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
        self._tensor_operands: dict[
            tuple[tuple[int, ...], bool], tuple[torch.Tensor, ...]
        ] = {}
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

    def tensor_operands(
        self, prime_ids: tuple[int, ...], *, inverse: bool
    ) -> tuple[torch.Tensor, ...]:
        """Return native table views for the selected rows and transform direction.

        Parameter generation belongs to this context. The returned Tensors can
        be supplied directly or placed in a Program's external material table.
        """
        cache = self._tensor_operands
        identity = (prime_ids, inverse)
        cached = cache.get(identity)
        if cached is not None:
            return cached
        executor = self.ntt_tables
        direction = "inverse" if inverse else "forward"
        rows = slice(prime_ids[0], prime_ids[-1] + 1)
        parameters = self.rns_context.rns_parameters_for_prime_ids(prime_ids)
        if self.ntt_backend_name == "radix2_indexed":
            result = (
                parameters,
                getattr(executor, f"{direction}_twiddles")[rows].contiguous(),
                getattr(executor, f"{direction}_even_indices"),
                getattr(executor, f"{direction}_odd_indices"),
            )
        elif self.ntt_backend_name.startswith("radix2_compact"):
            result = (
                parameters,
                getattr(executor, f"{direction}_twiddles")[rows].contiguous(),
            )
        else:
            result = (
                parameters,
                getattr(executor, f"{direction}_outer_twiddles")[
                    rows
                ].contiguous(),
                getattr(executor, f"{direction}_radix_root_powers")[rows],
            )
        cache[identity] = result
        return result

    def _transform(
        self,
        a: torch.Tensor,
        row_start: int,
        transition: str,
        *,
        in_place: bool = True,
    ) -> torch.Tensor:
        ids = tuple(range(row_start, row_start + a.size(-2)))
        operands = (
            a,
            *self.tensor_operands(
                ids, inverse=transition.startswith("inverse")
            ),
        )
        return prepare_transition(
            transition, self.ntt_backend_name, in_place=in_place
        )(operands)

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
        self._transform(a, row_start, "forward_montgomery_")

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
        self._transform(a, row_start, "forward_to_montgomery_")

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
        return self._transform(
            a, row_start, "forward_to_montgomery_", in_place=False
        )

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
        self._transform(a, row_start, "inverse_montgomery_")

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
        self._transform(a, row_start, "inverse_to_standard_lazy_")

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
        self._transform(a, row_start, "inverse_to_standard_")

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
        self._transform(a, row_start, "inverse_to_centered_")

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"NttContext(backend={self.ntt_backend_name!r}, "
            f"logN={self.config.logN}, "
            f"device={self.device}, ntt_policy={self.ntt_policy})"
        )
