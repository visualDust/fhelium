"""Build device-local RNS parameter views and execute residue arithmetic."""

from __future__ import annotations

import torch

from fhelium.config.ckks import CkksConfig
from fhelium.backend.rns.montgomery import MontgomeryParameters
from fhelium.backend.rns.format import RnsExecutionFormat
from fhelium.backend.rns.layout import RnsLayout
from fhelium.backend.rns.parameters import (
    RnsParameterStore,
    RnsRowParameters,
)
from fhelium.native.wrapper import rns_ops


class RnsContext:
    r"""Own dense RNS parameters and residue-representation arithmetic.

    Unless a method states otherwise, an RNS operand is an integral tensor
    with shape ``[*batch, limb, coefficient_or_ntt_index]`` and final extent
    $N$. Limb row ``j`` represents the modulus whose parameter id is
    ``prime_ids[j]``.  Public full-basis calls derive those ids from ``depth``
    and internal calls supply ``parameter_row_start`` for a contiguous
    row interval.  ``include_p`` is only this internal row selector; it is not
    public ``modulus_basis`` metadata.

    Functional methods allocate non-aliasing output.  A trailing underscore
    mutates the operand and preserves its storage.  Standard and Montgomery
    conversions do not change polynomial domain.
    """

    def __init__(
        self,
        ckks_config: CkksConfig,
        device: str | torch.device | None = None,
        *,
        rns_layout: RnsLayout | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)

        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("RnsContext requires a CPU or CUDA device")
        from fhelium.native import require_native_backend

        require_native_backend(self.device.type)
        self.config: CkksConfig = ckks_config
        self.execution_format = RnsExecutionFormat.select(
            ckks_config.moduli, dtype
        )
        self.dtype = self.execution_format.dtype
        self.montgomery_parameters = MontgomeryParameters(
            ckks_config, self.execution_format
        )

        if rns_layout is None:
            rns_layout = RnsLayout.from_config(self.config)
        elif (
            rns_layout.chain.num_q_primes != self.config.num_q_primes
            or rns_layout.chain.num_p_primes != self.config.num_p_primes
        ):
            raise ValueError(
                "RNS layout Q/P rows differ from the CKKS configuration"
            )
        self.rns_layout = rns_layout
        self.rns_chain = rns_layout.chain
        self.hybrid_decomposition = rns_layout.hybrid_decomposition

        # =============================================

        self._prepare_context_parameters()

        self.modulus_values = [
            self.montgomery_parameters.moduli[index]
            for index in self.rns_layout.prime_ids(0, include_p=True)
        ]

        self.qp_row_stop = len(self.rns_layout.prime_ids(0, include_p=True))
        self.q_row_stop = len(self.rns_layout.prime_ids(0))

        self._build_parameter_store()
        self._integer_scalar_parameters: dict[int, torch.Tensor] = {}

    @property
    def basis_count(self) -> int:
        """Number of public CKKS basis positions, excluding the structural basis."""

        return self.rns_chain.basis_count

    # -------------------------------------------------------------------------------------------------
    # Arrange according to partitioning scheme input variables, and copy to GPUs for fast access.
    # -------------------------------------------------------------------------------------------------

    def integer_scalar_parameters(self, scalar: int) -> torch.Tensor:
        r"""Return the reusable QP row vector $kR \bmod q_i$ for integer $k$."""

        parameters = self._integer_scalar_parameters.get(scalar)
        if parameters is None:
            radix = self.montgomery_parameters.R
            parameters = torch.tensor(
                tuple((scalar * radix) % q for q in self.config.moduli),
                dtype=self.dtype,
                device=self.device,
            )
            self._integer_scalar_parameters[scalar] = parameters
        return parameters

    def _prepare_context_parameters(self):
        # Integral tensor [parameter, limb] in depth-zero QP prime-id order,
        # built by libfhelium. Parameter rows are [2q_i, q_i low, q_i high,
        # -q_i^{-1} low, -q_i^{-1} high, R^2 mod q_i, Delta_0 R^2 mod q_i,
        # N^{-1} R mod q_i]. The scaled-R2 row remains stored for
        # parameter-table compatibility; no public conversion op implicitly
        # applies Delta_0. Native operands receive the corresponding limb
        # slice, so parameter column j always describes operand limb j.
        moduli = [
            self.config.moduli[index]
            for index in self.rns_layout.prime_ids(0, include_p=True)
        ]
        self.rns_parameter_tensor = rns_ops.parameter_table(
            moduli, self.config.logN, self.config.default_scale, self.dtype
        ).to(self.device)
        (
            self.twice_modulus,
            self.modulus_lo,
            self.modulus_hi,
            self.neg_inv_modulus_lo,
            self.neg_inv_modulus_hi,
            self.montgomery_r2,
            self.scaled_montgomery_r2,
            self.inverse_ntt_scale_montgomery,
        ) = self.rns_parameter_tensor.unbind(0)
        self.moduli = self.twice_modulus >> 1
        self.montgomery_reduction_parameter_tables = (
            self.modulus_lo,
            self.modulus_hi,
            self.neg_inv_modulus_lo,
            self.neg_inv_modulus_hi,
        )

    def _build_parameter_store(self) -> None:
        self.parameter_store = RnsParameterStore(
            rns_layout=self.rns_layout,
            montgomery_parameters=self.montgomery_parameters,
            device=self.device,
            torch_dtype=self.dtype,
            montgomery_reduction_parameter_tables=self.montgomery_reduction_parameter_tables,
            native_parameter_tensor=self.rns_parameter_tensor,
            modulus_tensor=self.moduli,
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
        for depth in range(self.rns_chain.basis_count):
            for include_p in (False, True):
                self._cache_row_parameters(
                    self.parameter_store.basis_parameters(
                        depth, include_p=include_p
                    )
                )
            for digit_spec in self.rns_layout.digit_specs(depth):
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
        """Return parameter rows in the operand's RNS prime order.

        The context stores one complete ``[parameter, Q | P]`` tensor. Native primitives receive a view with the same row count and order as their operand, so parameter row ``i`` describes operand row ``i``.

        Full-basis operands infer their dropped-Q prefix from the row count and
        whether the basis is ``Q`` or ``QP``. Internal digit operations pass a ``parameter_row_start`` because a key-switch digit is not a
        complete depth basis.
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
                "RNS operand rows do not map to the context parameter tensor: "
                f"rows={row_count}, start={parameter_row_start}, "
                f"stop={parameter_row_stop}, include_p={include_p}"
            )
        return (
            self._native_parameter_view(
                parameter_row_start, parameter_row_stop
            ),
            parameter_row_start,
        )

    def parameter_row_start_for(
        self,
        tensor: torch.Tensor,
        *,
        include_p: bool,
        rns_dimension: int = -2,
        parameter_row_start: int | None = None,
    ) -> int:
        """Resolve the context row interval used by a physical RNS operand."""

        _, row_start = self._active_rns_parameters(
            tensor,
            include_p=include_p,
            rns_dimension=rns_dimension,
            parameter_row_start=parameter_row_start,
        )
        return row_start

    def rns_parameters_for(
        self,
        tensor: torch.Tensor,
        *,
        include_p: bool = False,
        rns_dimension: int = -2,
        parameter_row_start: int | None = None,
    ) -> torch.Tensor:
        """Return ``[parameter, limb]`` parameters aligned with an RNS operand.

        The result is a zero-copy view of the context parameter table, on its device and with its integral dtype. Column ``j`` describes ``tensor[..., j, :]``. The operand remains unchanged and shares no storage with this parameter view.
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
        """Return parameters in requested catalog order, viewing contiguous intervals."""

        cached = self._native_parameter_views_by_prime_ids.get(prime_ids)
        if cached is not None:
            return cached
        if (
            not prime_ids
            or len(set(prime_ids)) != len(prime_ids)
            or any(
                type(i) is not int or not 0 <= i < self.qp_row_stop
                for i in prime_ids
            )
        ):
            raise ValueError("RNS prime_ids must be distinct catalog row IDs")
        start = prime_ids[0]
        stop = start + len(prime_ids)
        if prime_ids == tuple(range(start, stop)):
            cached = self._native_parameter_view(start, stop)
        else:
            indices = torch.tensor(
                prime_ids, dtype=torch.long, device=self.device
            )
            cached = self._native_parameter_view(
                0, self.qp_row_stop
            ).index_select(1, indices)
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
        self, depth: int, *, include_p: bool = False
    ) -> RnsRowParameters:
        """Return parameters for the Q or QP rows active at ``depth``."""

        return self.parameter_store.basis_parameters(depth, include_p=include_p)

    def twice_modulus_for_basis(
        self, depth: int, *, include_p: bool = False
    ) -> torch.Tensor:
        r"""Return integral row vector $[2q_i]$ for the active basis."""

        return self.parameter_store.twice_modulus_for_basis(
            depth, include_p=include_p
        )

    def moduli_for_basis(
        self, depth: int, *, include_p: bool = False
    ) -> list[int]:
        """Return host integers in the active basis's ``prime_ids`` order."""

        return self.parameter_store.moduli_for_basis(depth, include_p=include_p)

    # -------------------------------------------------------------------------------------------------
    # Helper functions to do the Montgomery and NTT operations.
    # -------------------------------------------------------------------------------------------------

    def to_montgomery_(
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace each standard residue $x_i$ by $x_iR \bmod q_i$.

        ``a`` is coefficient- or NTT-domain ``[*batch, limb, N]`` in the lazy
        interval $[0,2q_i)$. The integral tensor is mutated in place and
        remains lazy Montgomery RNS with unchanged axes and prime rows.
        """

        params = self._operand_rns_parameters(
            a,
            include_p=include_p,
            prime_ids=prime_ids,
        )
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

        ``b`` has shape ``[limb]`` and stores Montgomery scalars aligned with
        the same prime rows. The operation preserves whether ``a`` uses
        standard or Montgomery residues, preserves polynomial domain, and
        returns a lazy representative in $[0,2q_i)$.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.montgomery_mul_row_scalars_(
            a,
            b,
            params,
        )

    def montgomery_mul_row_scalars_standard(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
    ) -> torch.Tensor:
        r"""Return least nonnegative $a_i b_iR^{-1}\bmod q_i$ row products.

        ``a`` is integral ``[*batch, limb, index]`` and ``b`` is the aligned
        Montgomery ``[limb]`` scalar vector for the same prime rows. The
        operation preserves ``a``'s standard or Montgomery representation and
        polynomial domain. Output lies in $[0,q_i)$ and aliases neither input.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        return rns_ops.montgomery_mul_row_scalars_standard(
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
        self,
        a: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace $x_iR\bmod q_i$ by standard $x_i\bmod q_i$ in place.

        Shape, integral dtype, prime rows, polynomial domain, and
        storage are preserved. The output is lazy in $[0,2q_i)$.
        """

        params = self._operand_rns_parameters(
            a,
            include_p=include_p,
            prime_ids=prime_ids,
        )
        rns_ops.from_montgomery_(
            a,
            params,
        )

    def reduce_to_standard_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Reduce lazy $[0,2q_i)$ residues to $[0,q_i)$ in place.

        Axes, integral dtype, prime rows, polynomial domain, residue
        representation, and storage are unchanged.
        """

        params, _ = self._active_rns_parameters(a, include_p=include_p)
        rns_ops.reduce_to_standard_(
            a,
            params,
        )

    def center_residues_(
        self, a: torch.Tensor, *, include_p: bool = False
    ) -> None:
        r"""Map $[0,q_i)$ residues to centered representatives in place.

        Per prime row, $x_i>\lfloor q_i/2\rfloor$ maps to $x_i-q_i$; other
        residues remain unchanged. Input must be standard residues in $[0,q_i)$.
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

    def lift_integer_coefficients(
        self,
        coefficients: torch.Tensor,
        depth: int = 0,
        *,
        include_p: bool = False,
    ) -> torch.Tensor:
        r"""Lift signed ``[*batch, coefficient]`` integers into coefficient RNS.

        ``coefficients`` is an int32 or int64 tensor of final extent $N$.
        Output is ``[*batch, limb, coefficient]`` in standard representation
        with limb order ``rns_layout.prime_ids(depth, include_p=include_p)``;
        row $i$ holds $x+q_i$ for $-q_i\le x<q_i$ and $x \bmod q_i$
        otherwise, both in the lazy range $[0,2q_i)$. The functional result
        does not alias ``coefficients``.
        """

        return rns_ops.lift_integer_coefficients(
            coefficients,
            self.twice_modulus_for_basis(depth, include_p=include_p),
        )

    def add_standard(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $(a_i+b_i)\bmod q_i$ in $[0,q_i)$.

        Equal-shape integral operands have layout
        ``[*batch, limb, coefficient_or_ntt_index]``, identical domain and
        residue representation, and the same prime rows. Output
        preserves that state and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.add_standard(
            a,
            b,
            params,
        )

    def add_standard_(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace ``a`` by $(a_i+b_i)\bmod q_i$ in $[0,q_i)$.

        The tensor layout and state requirements match
        :meth:`add_standard`; ``a`` storage is mutated in place, while ``b``
        is read-only.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        rns_ops.add_standard_(
            a,
            b,
            params,
        )

    def sub_standard(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        r"""Return $(a_i-b_i)\bmod q_i$ in $[0,q_i)$.

        Equal-shape integral operands have layout
        ``[*batch, limb, coefficient_or_ntt_index]``, identical domain and
        residue representation, and the same prime rows. Output
        preserves that state and aliases neither input.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        return rns_ops.sub_standard(
            a,
            b,
            params,
        )

    def sub_standard_(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        include_p: bool = False,
        prime_ids: tuple[int, ...] | None = None,
    ) -> None:
        r"""Replace ``a`` by $(a_i-b_i)\bmod q_i$ in $[0,q_i)$.

        The tensor layout and state requirements match
        :meth:`sub_standard`; ``a`` storage is mutated in place, while ``b``
        is read-only.
        """

        params = self._operand_rns_parameters(
            a, include_p=include_p, prime_ids=prime_ids
        )
        rns_ops.sub_standard_(
            a,
            b,
            params,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"RnsContext(logN={self.config.logN}, N={self.config.N}, "
            f"depths={self.basis_count}, "
            f"q_prime_count={self.config.num_q_primes}, "
            f"p_prime_count={self.config.num_p_primes}, "
            f"device={self.device}, "
            f"row_count_depth0={len(self.rns_layout.prime_ids(0))}, "
            f"row_count_qp_depth0="
            f"{len(self.rns_layout.prime_ids(0, include_p=True))})"
        )
