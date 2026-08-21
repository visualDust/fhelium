from __future__ import annotations

import torch

from fhelium.config.ckks import CkksConfig
from fhelium.config.ntt import (
    DEFAULT_CPU_NTT_BACKEND,
    DEFAULT_NTT_BACKEND,
    resolve_ntt_backend_policy,
    validate_ntt_backend_for_log_n,
)
from fhelium.engine.rns.decomposition import HybridRnsDecomposition
from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.rns.chain import RnsChain
from fhelium.engine.rns.layout import RnsLayout
from fhelium.engine.rns.parameters import (
    RnsParameterStore,
    RnsRowParameters,
)
from fhelium.engine.ntt import create_ntt_backend, prepare_ntt_tables
from fhelium.native.wrapper import rns_ops


class RnsRuntime:
    r"""Own one engine's dense RNS parameters and representation transitions.

    Unless a method states otherwise, an RNS operand is an integral tensor
    with shape ``[*batch, limb, coefficient_or_ntt_index]`` and final extent
    $N$.  Limb row ``j`` represents the modulus whose canonical parameter id is
    ``prime_ids[j]``.  Public full-basis calls derive those ids from ``level``
    and internal calls supply ``parameter_row_start`` for a contiguous
    row interval.  ``include_p`` is only this internal row selector; it is not
    public ``modulus_basis`` metadata.

    Functional methods allocate non-aliasing output.  A trailing underscore
    mutates the operand and preserves its storage.  Standard and Montgomery
    conversions do not change polynomial domain.  Forward NTT maps coefficient
    index to NTT index and inverse NTT maps it back, independently for every
    batch item and prime row.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        device: str | torch.device | None = None,
        ntt_backend: str | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)

        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("RnsRuntime requires a CPU or CUDA device")
        from fhelium.native import require_native_backend

        require_native_backend(self.device.type)
        if ntt_backend is None:
            ntt_backend = (
                DEFAULT_CPU_NTT_BACKEND
                if self.device.type == "cpu"
                else DEFAULT_NTT_BACKEND
            )
        self.ntt_policy = resolve_ntt_backend_policy(ntt_backend)
        validate_ntt_backend_for_log_n(self.ntt_policy, ckks_config.logN)
        if (
            self.device.type == "cpu"
            and self.ntt_policy.name != DEFAULT_CPU_NTT_BACKEND
        ):
            raise ValueError(
                "CPU execution currently requires ntt_backend="
                f"{DEFAULT_CPU_NTT_BACKEND!r}"
            )
        self.ntt_backend_name = self.ntt_policy.name

        self.config: CkksConfig = ckks_config
        self.montgomery_parameters = MontgomeryParameters(ckks_config)

        self.rns_chain = RnsChain(
            num_q_primes=self.config.num_q_primes,
            num_p_primes=self.config.num_p_primes,
        )
        self.hybrid_decomposition = HybridRnsDecomposition(self.rns_chain)
        self.rns_layout = RnsLayout(self.rns_chain, self.hybrid_decomposition)

        # =============================================

        self._prepare_runtime_parameters()

        self.modulus_values = [
            self.montgomery_parameters.moduli[index]
            for index in self.rns_layout.prime_ids(0, include_p=True)
        ]

        self.level_row_starts = [
            self.rns_layout.start_row(level)
            for level in range(self.rns_basis_level_count)
        ]
        self.qp_row_stop = len(self.rns_layout.prime_ids(0, include_p=True))
        self.q_row_stop = len(self.rns_layout.prime_ids(0))

        self._build_parameter_store()

    @property
    def rns_basis_level_count(self) -> int:
        """Number of RNS row-suffix levels, including private structural levels."""

        return self.rns_chain.rns_basis_level_count

    # -------------------------------------------------------------------------------------------------
    # Arrange according to partitioning scheme input variables, and copy to GPUs for fast access.
    # -------------------------------------------------------------------------------------------------

    def _materialize_parameter_rows(self, variable) -> torch.Tensor:
        """Select this rank's canonical QP row order into one tensor."""

        source = torch.as_tensor(variable, dtype=self.config.torch_dtype)
        row_ids = torch.tensor(
            self.rns_layout.prime_ids(0, include_p=True),
            dtype=torch.long,
        )
        return source.index_select(0, row_ids).to(self.device)

    def _prepare_inverse_ntt_scale_montgomery(self) -> torch.Tensor:
        r"""Materialize $N^{-1}R \bmod q_i$ in canonical QP row order."""

        values = [
            (inverse_ntt_scale * self.montgomery_parameters.R) % modulus
            for inverse_ntt_scale, modulus in zip(
                self.config.inverse_ntt_scale,
                self.config.moduli,
                strict=True,
            )
        ]
        return self._materialize_parameter_rows(values)

    def _prepare_runtime_parameters(self):
        scale = 2**self.config.scale_bits
        self.scaled_montgomery_r2 = self._materialize_parameter_rows(
            [
                (montgomery_r2 * scale) % q
                for montgomery_r2, q in zip(
                    self.montgomery_parameters.montgomery_r2, self.config.moduli
                )
            ]
        )

        self.montgomery_r2 = self._materialize_parameter_rows(
            self.montgomery_parameters.montgomery_r2
        )

        self.moduli = self._materialize_parameter_rows(
            self.montgomery_parameters.moduli
        )
        self.twice_modulus = self._materialize_parameter_rows(
            self.montgomery_parameters.twice_modulus
        )
        self.modulus_lo = self._materialize_parameter_rows(
            self.montgomery_parameters.modulus_lower_bits
        )
        self.modulus_hi = self._materialize_parameter_rows(
            self.montgomery_parameters.modulus_higher_bits
        )
        self.neg_inv_modulus_lo = self._materialize_parameter_rows(
            self.montgomery_parameters.neg_inv_modulus_lower_bits
        )
        self.neg_inv_modulus_hi = self._materialize_parameter_rows(
            self.montgomery_parameters.neg_inv_modulus_higher_bits
        )

        self.ntt_tables = prepare_ntt_tables(
            self.ntt_policy,
            self.config,
            materialize_parameter_rows=self._materialize_parameter_rows,
            device=self.device,
        )

        self.inverse_ntt_scale_montgomery = (
            self._prepare_inverse_ntt_scale_montgomery()
        )

        # Integral tensor [parameter, limb] in canonical level-zero QP
        # prime-id order. Parameter rows are [2q_i, q_i low, q_i high,
        # -q_i^{-1} low, -q_i^{-1} high, R^2 mod q_i,
        # Delta_0 R^2 mod q_i, N^{-1} R mod q_i]. The scaled-R2 row remains
        # stored for parameter-table compatibility; no public conversion op
        # implicitly applies Delta_0. Native operands receive the corresponding
        # limb slice, so parameter column j always describes operand limb j.
        self.rns_parameter_tensor = torch.stack(
            [
                self.twice_modulus,
                self.modulus_lo,
                self.modulus_hi,
                self.neg_inv_modulus_lo,
                self.neg_inv_modulus_hi,
                self.montgomery_r2,
                self.scaled_montgomery_r2,
                self.inverse_ntt_scale_montgomery,
            ]
        ).contiguous()

        self.ntt_tables.convert_twiddles_to_montgomery_(
            self.rns_parameter_tensor
        )

        self.montgomery_reduction_parameter_tables = (
            self.modulus_lo,
            self.modulus_hi,
            self.neg_inv_modulus_lo,
            self.neg_inv_modulus_hi,
        )

        self.ntt_backend = create_ntt_backend(
            self.ntt_policy,
            ntt_tables=self.ntt_tables,
            rns_params=self.rns_parameter_tensor,
        )

    def _build_parameter_store(self) -> None:
        self.parameter_store = RnsParameterStore(
            rns_layout=self.rns_layout,
            montgomery_parameters=self.montgomery_parameters,
            device=self.device,
            torch_dtype=self.config.torch_dtype,
            rns_basis_level_count=self.rns_basis_level_count,
            level_row_starts=self.level_row_starts,
            basis_row_stops=(self.qp_row_stop, self.q_row_stop),
            montgomery_reduction_parameter_tables=self.montgomery_reduction_parameter_tables,
            native_parameter_tensor=self.rns_parameter_tensor,
            montgomery_r2=self.montgomery_r2,
            scaled_montgomery_r2=self.scaled_montgomery_r2,
            twice_modulus=self.twice_modulus,
            moduli=self.modulus_values,
        )
        self._native_parameter_views: dict[tuple[int, int], torch.Tensor] = {}
        self._native_parameter_views_by_prime_ids: dict[
            tuple[int, ...], torch.Tensor
        ] = {}
        for row_id in range(self.qp_row_stop):
            self._cache_row_parameters(
                self.parameter_store.row_parameters((row_id,))
            )
        self._cache_row_parameters(
            self.parameter_store.row_parameters(self.rns_chain.p_prime_ids)
        )
        for level in range(self.rns_basis_level_count):
            for include_p in (False, True):
                self._cache_row_parameters(
                    self.parameter_store.basis_parameters(
                        level, include_p=include_p
                    )
                )
            for digit_spec in self.rns_layout.digit_specs(level):
                self._cache_row_parameters(
                    self.parameter_store.row_parameters(digit_spec.prime_ids)
                )

    def _cache_row_parameters(self, rows: RnsRowParameters) -> None:
        row_stop = rows.parameter_row_start + len(rows.prime_ids)
        self._native_parameter_views[(rows.parameter_row_start, row_stop)] = (
            rows.native_parameters
        )
        self._native_parameter_views_by_prime_ids[rows.prime_ids] = (
            rows.native_parameters
        )

    def _native_parameter_view(
        self, parameter_row_start: int, parameter_row_stop: int
    ) -> torch.Tensor:
        cache_key = (parameter_row_start, parameter_row_stop)
        cached = self._native_parameter_views.get(cache_key)
        if cached is None:
            cached = self.rns_parameter_tensor[
                :, parameter_row_start:parameter_row_stop
            ]
            self._native_parameter_views[cache_key] = cached
        return cached

    def _active_rns_parameters(
        self,
        tensor: torch.Tensor,
        *,
        include_p: bool,
        rns_dimension: int = -2,
        parameter_row_start: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Return parameters whose rows exactly match ``tensor``'s RNS rows.

        The engine stores one complete ``[parameter, Q | P]`` tensor.  Native
        primitives should not know that storage convention: they receive a
        view with the same row count and order as their operand and therefore
        index parameter row ``i`` for operand row ``i``.

        Full-basis operands infer their dropped-Q prefix from the row count and
        whether the basis is ``Q`` or ``QP``. Internal digit operations pass an
        specified ``parameter_row_start`` because a key-switch digit is not a
        complete level basis.
        """

        row_count = tensor.size(rns_dimension)
        if parameter_row_start is None:
            basis_stop = self.qp_row_stop if include_p else self.q_row_stop
            parameter_row_start = basis_stop - row_count
        parameter_row_stop = parameter_row_start + row_count
        if (
            not 0
            <= parameter_row_start
            < parameter_row_stop
            <= self.qp_row_stop
        ):
            raise ValueError(
                "RNS operand rows do not map to the engine parameter tensor: "
                f"rows={row_count}, start={parameter_row_start}, "
                f"stop={parameter_row_stop}, include_p={include_p}"
            )
        return (
            self._native_parameter_view(
                parameter_row_start, parameter_row_stop
            ),
            parameter_row_start,
        )

    def rns_parameters_for(
        self,
        tensor: torch.Tensor,
        *,
        include_p: bool = False,
        rns_dimension: int = -2,
        parameter_row_start: int | None = None,
    ) -> torch.Tensor:
        """Return ``[parameter, limb]`` parameters aligned with an RNS operand.

        The result is a zero-copy CUDA view with the engine integral dtype.
        Column ``j`` describes ``tensor[..., j, :]`` exactly; this method does
        not mutate or alias the operand itself.
        """

        params, _ = self._active_rns_parameters(
            tensor,
            include_p=include_p,
            rns_dimension=rns_dimension,
            parameter_row_start=parameter_row_start,
        )
        return params

    def rns_parameters_for_prime_ids(
        self, prime_ids: tuple[int, ...]
    ) -> torch.Tensor:
        """Return a zero-copy parameter view for canonical local RNS rows."""

        cached = self._native_parameter_views_by_prime_ids.get(prime_ids)
        if cached is not None:
            return cached
        if not prime_ids:
            raise ValueError("RNS prime_ids cannot be empty")
        start = prime_ids[0]
        stop = start + len(prime_ids)
        if prime_ids != tuple(range(start, stop)):
            raise ValueError(
                "RNS prime_ids must be a contiguous canonical interval: "
                f"{prime_ids}"
            )
        if not 0 <= start < stop <= self.qp_row_stop:
            raise ValueError(
                f"RNS prime_ids are outside [0, {self.qp_row_stop}): {prime_ids}"
            )
        cached = self._native_parameter_view(start, stop)
        self._native_parameter_views_by_prime_ids[prime_ids] = cached
        return cached

    def _operand_rns_parameters(
        self,
        tensor: torch.Tensor,
        *,
        include_p: bool,
        prime_ids: tuple[int, ...] | None,
    ) -> torch.Tensor:
        if prime_ids is None:
            params, _ = self._active_rns_parameters(tensor, include_p=include_p)
            return params
        if tensor.size(-2) != len(prime_ids):
            raise ValueError(
                "RNS operand row count does not match prime_ids: "
                f"rows={tensor.size(-2)}, prime_ids={prime_ids}"
            )
        return self.rns_parameters_for_prime_ids(prime_ids)

    def row_parameters(self, key) -> RnsRowParameters:
        """Return immutable host/device parameters for ``prime_ids``."""

        return self.parameter_store.row_parameters(key)

    def basis_parameters(
        self, level: int, *, include_p: bool = False
    ) -> RnsRowParameters:
        """Return parameters for the Q or QP rows active at ``level``."""

        return self.parameter_store.basis_parameters(level, include_p=include_p)

    def twice_modulus_for_basis(
        self, level: int, *, include_p: bool = False
    ) -> torch.Tensor:
        r"""Return integral row vector $[2q_i]$ for the active basis."""

        return self.parameter_store.twice_modulus_for_basis(
            level, include_p=include_p
        )

    def moduli_for_basis(
        self, level: int, *, include_p: bool = False
    ) -> list[int]:
        """Return host integers in the active basis's ``prime_ids`` order."""

        return self.parameter_store.moduli_for_basis(level, include_p=include_p)

    # -------------------------------------------------------------------------------------------------
    # Helper functions to do the Montgomery and NTT operations.
    # -------------------------------------------------------------------------------------------------

    def to_montgomery_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Replace each standard residue $x_i$ by $x_iR \bmod q_i$.

        ``a`` is coefficient- or NTT-domain ``[*batch, limb, N]`` in the lazy
        interval $[0,2q_i)$. The integral tensor is mutated in place and
        remains lazy Montgomery RNS with unchanged axes and prime rows.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.to_montgomery_(
            a,
            params,
        )

    def montgomery_mul_row_scalars_(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
    ) -> None:
        r"""Mutate ``a[..., i, :]`` to $a_i b_i R^{-1}\bmod q_i$.

        ``b`` has shape ``[limb]`` and is aligned with the same prime
        rows. Both inputs use one integral dtype and Montgomery form;
        polynomial domain is preserved and the result is lazy in $[0,2q_i)$.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.montgomery_mul_row_scalars_(
            a,
            b,
            params,
        )

    def montgomery_mul_row_scalars_canonical(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
    ) -> torch.Tensor:
        r"""Return canonical $a_i b_iR^{-1}\bmod q_i$ row products.

        ``a`` is integral ``[*batch, limb, index]`` and ``b`` is the
        aligned Montgomery ``[limb]`` scalar vector for the same prime
        rows. Polynomial domain is preserved. Output has canonical
        Montgomery residues in $[0,q_i)$ and aliases neither input.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        return rns_ops.montgomery_mul_row_scalars_canonical(
            a,
            b,
            params,
        )

    def montgomery_mul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $a_i b_iR^{-1}\bmod q_i$ without mutating either input.

        Operands have equal ``[*batch, limb, N]`` layouts, integral dtype,
        prime-row mapping, polynomial domain, and Montgomery form. The
        output has the same state, is lazy in $[0,2q_i)$, and aliases neither
        input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.montgomery_mul(
            a,
            b,
            params,
        )

    def montgomery_mul_cyclic_compressed(
        self,
        a: torch.Tensor,
        compressed_b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Multiply by a cyclically repeated compact Montgomery operand.

        ``a`` is integral ``[*batch, limb, N]`` and ``compressed_b`` is
        ``[*compressed_batch, limb, unique_index]`` on the same device, domain,
        prime rows, and Montgomery form. The compact final axis repeats
        cyclically to extent $N$. After leading axes are collapsed, the compact
        batch count must equal the dense batch count or be one; no other
        broadcasting occurs. Output is lazy Montgomery RNS and aliases neither
        input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.montgomery_mul_cyclic_compressed(
            a,
            compressed_b,
            params,
        )

    def montgomery_mul_contiguous_compressed(
        self,
        a: torch.Tensor,
        compressed_b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Multiply by a block-repeated compact Montgomery operand.

        Tensor state and broadcasting match
        :meth:`montgomery_mul_cyclic_compressed`, but each compact value is
        repeated in one contiguous block along the expanded extent-$N$ axis.
        Output is lazy Montgomery RNS and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.montgomery_mul_contiguous_compressed(
            a,
            compressed_b,
            params,
        )

    def from_montgomery_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Replace $x_iR\bmod q_i$ by standard $x_i\bmod q_i$ in place.

        Shape, integral dtype, prime rows, polynomial domain, and
        storage are preserved. The output is lazy in $[0,2q_i)$.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.from_montgomery_(
            a,
            params,
        )

    def canonicalize_residues_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Reduce lazy $[0,2q_i)$ residues to canonical $[0,q_i)$ in place.

        Axes, integral dtype, prime rows, polynomial domain, residue
        representation, and storage are unchanged.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.canonicalize_residues_(
            a,
            params,
        )

    def center_residues_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Map canonical residues to centered representatives in place.

        Per prime row, $x_i>\lfloor q_i/2\rfloor$ maps to $x_i-q_i$; other
        residues remain unchanged. Input must be canonical standard residues.
        Shape, integral dtype, prime rows, polynomial domain, and storage
        are preserved.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.center_residues_(
            a,
            params,
        )

    def shift_residues_positive_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Add $q_i$ to every centered representative in place.

        This storage-preserving operation consumes the centered interval
        $[-\lfloor q_i/2\rfloor,\lfloor q_i/2\rfloor]$ and returns congruent
        positive lazy representatives in approximately $[q_i/2,3q_i/2]$;
        axes, integral dtype, prime rows, and polynomial domain remain.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.shift_residues_positive_(
            a,
            params,
        )

    def add_lazy(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $(a_i+b_i)\bmod 2q_i$ as lazy residues.

        Equal-shape integral operands use the same domain, representation,
        and prime rows and lie in $[0,2q_i)$. Output has the same shape
        and state, lies in $[0,2q_i)$, and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.add_lazy(
            a,
            b,
            params,
        )

    def sub_lazy(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
    ) -> torch.Tensor:
        r"""Return $(a_i-b_i)\bmod 2q_i$ as non-aliasing lazy residues."""

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        return rns_ops.sub_lazy(
            a,
            b,
            params,
        )

    def lift_centered_coefficients(
        self,
        a: torch.Tensor,
        level: int = 0,
        *,
        include_p: bool = False,
    ) -> torch.Tensor:
        r"""Lift ``[*batch, coefficient]`` integers into coefficient RNS.

        The input is an integral tensor of final extent $N$ with
        $-q_i<x<q_i$ for every selected prime. Output is
        ``[*batch, limb, coefficient]`` in standard representation with limb
        order ``rns_layout.prime_ids(level, include_p=include_p)`` and lazy
        range $[0,2q_i)$. The functional result does not alias ``a``.
        """

        return rns_ops.lift_centered_coefficients(
            a,
            self.twice_modulus_for_basis(level, include_p=include_p),
        )

    def lift_integer_coefficients_exact(
        self,
        coefficients: torch.Tensor,
        level: int = 0,
        *,
        include_p: bool = False,
        max_abs: int | None = None,
    ) -> torch.Tensor:
        r"""Lift signed compact coefficients without a one-prime assumption.

        For row $i$, the native centered lift stores $x+q_i$ as a lazy
        residue, so it is valid only while $-q_i<x<q_i$ for every active
        modulus. Wider machine integers require an actual
        remainder for each RNS row.
        """

        prime_ids = self.rns_layout.prime_ids(level, include_p=include_p)
        moduli = tuple(
            int(self.montgomery_parameters.moduli[index]) for index in prime_ids
        )
        if max_abs is None:
            max_abs = int(torch.max(torch.abs(coefficients)).item())
        if max_abs < min(moduli):
            return self.lift_centered_coefficients(
                coefficients,
                level,
                include_p=include_p,
            )
        modulus_tensor = torch.tensor(
            moduli,
            dtype=coefficients.dtype,
            device=coefficients.device,
        ).view(*([1] * (coefficients.ndim - 1)), -1, 1)
        return torch.remainder(coefficients.unsqueeze(-2), modulus_tensor)

    def add_canonical(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $(a_i+b_i)\bmod q_i$ in canonical $[0,q_i)$ form.

        Equal-shape integral operands have layout
        ``[*batch, limb, coefficient_or_ntt_index]``, identical domain and
        residue representation, and the same prime rows. Output
        preserves that state and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.add_canonical(
            a,
            b,
            params,
        )

    def add_canonical_(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace ``a`` by $(a_i+b_i)\bmod q_i$ in canonical form.

        The tensor layout and state requirements match
        :meth:`add_canonical`; ``a`` storage is mutated in place, while ``b``
        is read-only.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        rns_ops.add_canonical_(
            a,
            b,
            params,
        )

    def sub_canonical(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $(a_i-b_i)\bmod q_i$ in canonical $[0,q_i)$ form.

        Equal-shape integral operands have layout
        ``[*batch, limb, coefficient_or_ntt_index]``, identical domain and
        residue representation, and the same prime rows. Output
        preserves that state and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.sub_canonical(
            a,
            b,
            params,
        )

    def sub_canonical_(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace ``a`` by $(a_i-b_i)\bmod q_i$ in canonical form.

        The tensor layout and state requirements match
        :meth:`sub_canonical`; ``a`` storage is mutated in place, while ``b``
        is read-only.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        rns_ops.sub_canonical_(
            a,
            b,
            params,
        )

    def forward_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Apply the negacyclic forward NTT to Montgomery residues in place.

        ``a`` maps from ``[*batch, limb, coefficient]`` to the same storage
        viewed as ``[*batch, limb, ntt_index]``. For row $i$ it evaluates the
        polynomial modulo $X^N+1$ and $q_i$ at the backend's canonical NTT
        points. Integral CUDA dtype, prime rows, Montgomery form, and
        lazy $[0,2q_i)$ range are preserved.
        """

        self._forward_montgomery_with_parameters_(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )

    def _forward_montgomery_with_parameters_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool,
        parameter_row_start: int | None = None,
    ) -> torch.Tensor:
        """Transform ``a`` and return its already-resolved parameter view."""

        params, row_start = self._active_rns_parameters(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.forward_montgomery_(a, row_start)
        return params

    def forward_to_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Apply forward NTT and standard-to-Montgomery conversion in place.

        The state transition is ``coefficient`` + ``standard`` to ``ntt`` +
        ``montgomery`` on unchanged ``[*batch, limb, N]`` integral
        storage and prime rows; output is lazy in $[0,2q_i)$.
        """

        _, row_start = self._active_rns_parameters(
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

        _, row_start = self._active_rns_parameters(
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
        r"""Apply normalized inverse NTT while retaining Montgomery form.

        Per prime row normalization multiplies by
        $N^{-1}R\bmod q_i$ under Montgomery reduction. The integral
        tensor changes from NTT indices to coefficients in place; prime rows,
        storage, Montgomery representation, and lazy $[0,2q_i)$ range remain.
        """

        _, row_start = self._active_rns_parameters(
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
        r"""Inverse NTT to coefficient/standard lazy $[0,2q_i)$ in place."""

        _, row_start = self._active_rns_parameters(
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
        r"""Inverse NTT to canonical coefficient/standard $[0,q_i)$ in place."""

        _, row_start = self._active_rns_parameters(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_to_standard_(a, row_start)

    def inverse_to_centered_(
        self,
        a,
        *,
        include_p: bool = False,
        parameter_row_start: int | None = None,
    ) -> None:
        r"""Inverse NTT to centered coefficient/standard residues in place."""

        _, row_start = self._active_rns_parameters(
            a,
            include_p=include_p,
            parameter_row_start=parameter_row_start,
        )
        self.ntt_backend.inverse_to_centered_(a, row_start)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"RnsRuntime(backend='{self.ntt_backend_name}', "
            f"backend_impl={self.ntt_backend}, logN={self.config.logN}, "
            f"N={self.config.N}, levels={self.rns_basis_level_count}, "
            f"q_prime_count={self.config.num_q_primes}, "
            f"p_prime_count={self.config.num_p_primes}, "
            f"device={self.device}, "
            f"row_count_level0={len(self.rns_layout.prime_ids(0))}, "
            f"row_count_qp_level0="
            f"{len(self.rns_layout.prime_ids(0, include_p=True))}, "
            f"ntt_policy={self.ntt_policy})"
        )
