"""Tensor views used by complete hybrid CKKS key-switch implementations."""

from __future__ import annotations

from fhelium.config import CkksConfig
from fhelium.backend.rns.tables import (
    centered_basis_extension_tables,
    trailing_quotient_inverses,
    moddown_inverse_tables,
)
from dataclasses import dataclass, field
from collections.abc import Mapping
from math import prod
from typing import TYPE_CHECKING, cast
import torch
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.decomposition import key_decomposition_facts
from fhelium.backend.rns.crt import crt_basis_extension_tables
from ._key_switch_preparation import key_switch_runtime_attributes

if TYPE_CHECKING:
    from fhelium.backend.ntt.context import NttContext


def encapsulation_configs(
    config: CkksConfig,
    source_depth: int,
    low_p_moduli: tuple[int, ...],
    return_prime: int,
    return_p_moduli: tuple[int, ...],
) -> tuple[CkksConfig, CkksConfig]:
    """Describe the low switch basis and guard-prime return basis."""
    source_q = tuple(
        q for group in config.q_depth_groups[source_depth:] for q in group
    )

    def auxiliary(groups, p_moduli):
        return CkksConfig(
            logN=config.logN,
            default_scale=config.default_scale,
            sigma=config.sigma,
            galois_generator=config.galois_generator,
            security_bits=config.security_bits,
            enforce_security_budget=False,
            q_depth_groups=groups,
            p_moduli=p_moduli,
        )

    return (
        auxiliary((source_q,), low_p_moduli),
        auxiliary(((return_prime,), *config.q_depth_groups), return_p_moduli),
    )


def prepare_encapsulated_raise_operands(
    rns_context,
    ntt_context,
    *,
    source_depth: int,
    target_depth: int,
    low_p_moduli: tuple[int, ...],
    return_prime: int,
    return_p_moduli: tuple[int, ...],
    rounding: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Prepare low key-switch, centered extension and return-quotient tables.

    The return hint uses [guard, Q, P] rows and one digit. Its relation is
    H0+H1*s = guard*P*t+e; quotient row order is [Q, guard, P]. Neither
    preparation context is retained by the returned numerical operands.
    """

    from fhelium.backend.ntt.context import NttContext

    low_config, return_config = encapsulation_configs(
        rns_context.config,
        source_depth,
        low_p_moduli,
        return_prime,
        return_p_moduli,
    )
    policy = ntt_context.ntt_backend_name
    low_rns = RnsContext(
        low_config, device=rns_context.device, dtype=rns_context.dtype
    )
    high_rns = RnsContext(
        return_config, device=rns_context.device, dtype=rns_context.dtype
    )
    low_ntt = NttContext(low_rns, ntt_backend=policy)
    high_ntt = NttContext(high_rns, ntt_backend=policy)
    low_tensors, low_facts = KeySwitchTables(
        low_rns,
        low_ntt,
        moddown_inverse_tables(low_rns),
        low_config.galois_generator,
    ).tensor_operands(0, moddown_rounding=rounding, modup_method="exact")
    tensors = {"low_" + name: value for name, value in low_tensors.items()}
    facts = {
        "low_" + name: value
        for name, value in key_switch_runtime_attributes(low_facts).items()
    }
    low_ids = low_rns.rns_layout.prime_ids(0)
    high_ids = high_rns.rns_layout.prime_ids(0, include_p=True)
    extension = centered_basis_extension_tables(
        low_rns, high_rns, low_ids, high_ids
    )
    tensors.update(
        {"raise_" + name: value for name, value in extension.items()}
    )
    tensors["return_parameters"] = high_rns.rns_parameter_tensor
    for inverse, direction in ((False, "forward"), (True, "inverse")):
        for index, value in enumerate(
            high_ntt.tensor_operands(high_ids, inverse=inverse)
        ):
            tensors[f"return_{direction}_{index}"] = value
    q_count = rns_context.config.num_q_primes
    order = (
        *range(1, q_count + 1),
        0,
        *range(q_count + 1, return_config.total_num_primes),
    )
    tensors["quotient_order"] = torch.tensor(
        order, dtype=torch.int64, device=rns_context.device
    )
    tensors["quotient_parameters"] = tensors["return_parameters"].index_select(
        1, tensors["quotient_order"]
    )
    moduli = (*rns_context.config.q_moduli, return_prime, *return_p_moduli)
    drop_count = len(return_p_moduli) + 1
    tensors["quotient_inverses"] = trailing_quotient_inverses(
        moduli,
        drop_count,
        high_rns.montgomery_parameters.R,
        dtype=rns_context.dtype,
        device=rns_context.device,
    )
    if rounding == "nearest":
        half = (return_prime * prod(return_p_moduli) - 1) // 2
        tensors["quotient_half"] = torch.tensor(
            [half % q for q in moduli],
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
    target_ids = rns_context.rns_layout.prime_ids(target_depth)
    tensors["target_parameters"] = rns_context.rns_parameters_for_prime_ids(
        target_ids
    )
    facts.update(
        ntt_backend=policy,
        q_count=q_count,
        target_row_start=target_ids[0],
        return_drop_count=drop_count,
        moddown_rounding=rounding,
    )
    return tensors, facts


@dataclass(frozen=True)
class KeySwitchTables:
    """Own the tables used by hybrid CKKS key switching."""

    rns_context: RnsContext
    ntt_context: NttContext | None
    moddown_p_drop_inverses_montgomery_by_depth: tuple[torch.Tensor, ...] = ()
    galois_generator: int = 3
    _p_inverse_montgomery: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _tensor_views: dict[
        tuple[object, ...], tuple[dict[str, torch.Tensor], dict[str, object]]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.ntt_context is not None
            and self.ntt_context.rns_context is not self.rns_context
        ):
            raise ValueError(
                "Key-switch NTT context composes another RNS context"
            )
        if self.galois_generator not in {3, 5}:
            raise ValueError(
                "Key-switch resource galois generator must be 3 or 5"
            )

    @property
    def p_inverse_montgomery(self) -> torch.Tensor:
        """Prepare the P-only inverse vector when that quotient is selected."""
        if self._p_inverse_montgomery is None:
            config = self.rns_context.config
            p_product = prod(config.p_moduli)
            radix = self.rns_context.montgomery_parameters.R
            object.__setattr__(
                self,
                "_p_inverse_montgomery",
                torch.tensor(
                    [
                        pow(p_product, -1, q) * radix % q
                        for q in config.q_moduli
                    ],
                    dtype=self.rns_context.dtype,
                    device=self.rns_context.device,
                ),
            )
        return cast(torch.Tensor, self._p_inverse_montgomery)

    @property
    def device(self) -> torch.device:
        return self.rns_context.device

    @property
    def moddown_tables(self) -> tuple[torch.Tensor, ...]:
        """Return the depth-indexed P-drop inverse tables read by ModDown."""

        if not self.moddown_p_drop_inverses_montgomery_by_depth:
            object.__setattr__(
                self,
                "moddown_p_drop_inverses_montgomery_by_depth",
                moddown_inverse_tables(self.rns_context),
            )
        return self.moddown_p_drop_inverses_montgomery_by_depth

    def moddown_tensor_operands(
        self,
        q_ids: tuple[int, ...],
        p_ids: tuple[int, ...],
        *,
        rounding: str = "nearest",
        depth: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare quotient data for the represented ordered Q and P rows."""
        token = ("moddown", q_ids, p_ids, rounding)
        if token in self._tensor_views:
            return self._tensor_views[token][0]
        rns = self.rns_context
        qp_ids = (*q_ids, *p_ids)
        moduli = tuple(rns.config.moduli[i] for i in qp_ids)
        product = prod(rns.config.moduli[i] for i in p_ids)
        radix = rns.montgomery_parameters.R
        default_p = (
            depth is not None
            and p_ids == rns.rns_chain.p_prime_ids
            and q_ids == rns.rns_layout.prime_ids(depth)
        )
        tensors = {
            "q_parameters": rns.rns_parameters_for_prime_ids(q_ids),
            "qp_parameters": rns.rns_parameters_for_prime_ids(qp_ids),
            "moddown_inverses": (
                self.moddown_tables[cast(int, depth)]
                if default_p
                else trailing_quotient_inverses(
                    moduli,
                    len(p_ids),
                    radix,
                    dtype=rns.dtype,
                    device=rns.device,
                )
            ),
            "p_inverse": (
                self.p_inverse_montgomery[q_ids[0] :]
                if default_p
                else torch.tensor(
                    [
                        pow(product, -1, rns.config.moduli[i])
                        * radix
                        % rns.config.moduli[i]
                        for i in q_ids
                    ],
                    dtype=rns.dtype,
                    device=rns.device,
                )
            ),
        }
        if rounding == "nearest":
            half = (product - 1) // 2
            tensors["moddown_half"] = torch.tensor(
                [half % q for q in moduli], dtype=rns.dtype, device=rns.device
            )
        self._tensor_views[token] = (tensors, {})
        return tensors

    def tensor_operands(
        self,
        depth: int,
        *,
        moddown_rounding: str = "nearest",
        modup_method: str = "exact",
        rescale: bool = False,
        rescale_rounding: str = "nearest",
        output_domain: str = "coefficient",
        key_prime_ids: tuple[int, ...] | None = None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Expose the selected digit-lift tables and active key digit ranges.

        Returned data can populate ordinary Program Tensor placeholders. No
        execution resource object is needed to consume these table views.
        """
        if modup_method not in {"exact", "crt", "normalized_crt"}:
            raise ValueError(
                "modup_method must be exact, crt, or normalized_crt"
            )
        cache_key = (
            depth,
            moddown_rounding,
            modup_method,
            rescale,
            rescale_rounding if rescale else None,
            output_domain if rescale else None,
            key_prime_ids,
            digit_prime_ids,
        )
        if cache_key in self._tensor_views:
            return self._tensor_views[cache_key]
        rns = self.rns_context
        if digit_prime_ids is None:
            if key_prime_ids is not None:
                raise ValueError(
                    "Key digit supports are required with key prime rows"
                )
            digit_prime_ids = tuple(
                spec.prime_ids for spec in rns.rns_layout.digit_specs(0)
            )
        if key_prime_ids is None:
            key_prime_ids = rns.rns_layout.prime_ids(0, include_p=True)
        facts = key_decomposition_facts(
            rns.config, depth, key_prime_ids, digit_prime_ids
        )
        facts["moddown_rounding"] = moddown_rounding
        q_ids = cast(tuple[int, ...], facts["q_prime_ids"])
        p_ids = cast(tuple[int, ...], facts["p_prime_ids"])
        qp_ids = cast(tuple[int, ...], facts["qp_prime_ids"])
        if rescale:
            from fhelium.backend.rns.tables import moddown_rescale_tables

            start = sum(map(len, rns.config.q_depth_groups[:depth]))
            drop_ids = tuple(
                range(start, start + len(rns.config.q_depth_groups[depth]))
            )
            tensors = {
                "q_parameters": rns.rns_parameters_for_prime_ids(q_ids),
                "qp_parameters": rns.rns_parameters_for_prime_ids(qp_ids),
            }
            tensors.update(
                moddown_rescale_tables(
                    rns,
                    q_ids,
                    p_ids,
                    drop_ids,
                    moddown_rounding=moddown_rounding,
                    rescale_rounding=rescale_rounding,
                )
            )
            facts.update(
                rescale=True,
                rescale_rounding=rescale_rounding,
                output_domain=output_domain,
                joint_drop_prime_ids=drop_ids,
                joint_drop_count=len(drop_ids),
                joint_drop_is_prefix=drop_ids == q_ids[: len(drop_ids)],
            )
        else:
            tensors = dict(
                self.moddown_tensor_operands(
                    q_ids, p_ids, rounding=moddown_rounding, depth=depth
                )
            )
        if self.ntt_context is not None:
            transforms = [
                (basis, ids, inverse)
                for basis, ids in (("q", q_ids), ("qp", qp_ids), ("p", p_ids))
                for inverse in (False, True)
            ]
            if rescale:
                remaining = tuple(i for i in q_ids if i not in drop_ids)
                transforms = [
                    ("q", q_ids, True),
                    ("qp", qp_ids, False),
                    ("joint_dropped", (*drop_ids, *p_ids), True),
                    (
                        "joint_remaining",
                        remaining,
                        output_domain == "coefficient",
                    ),
                ]
            for basis, ids, inverse in transforms:
                direction = "inverse" if inverse else "forward"
                for index, tensor in enumerate(
                    self.ntt_context.tensor_operands(ids, inverse=inverse)
                ):
                    tensors[f"{basis}_{direction}_{index}"] = tensor
        digits = cast(tuple[tuple[int, int, int], ...], facts["digits"])
        for index, (start, stop, _) in enumerate(digits):
            rows = q_ids[start:stop]
            if modup_method != "exact":
                crt = crt_basis_extension_tables(
                    rns,
                    rows,
                    qp_ids,
                    method="raw" if modup_method == "crt" else "normalized",
                )
                tensors.update(
                    (f"digit{index}_crt_{name}", value)
                    for name, value in crt.items()
                )
                continue
            parameters = rns.row_parameters(rows)
            tensors[f"digit{index}_parameters"] = parameters.native_parameters
            if stop - start > 1:
                assert parameters.mixed_radix_normalizers is not None
                assert (
                    parameters.mixed_radix_propagation_coefficients is not None
                )
                tensors[f"digit{index}_normalizers"] = (
                    parameters.mixed_radix_normalizers.contiguous()
                )
                tensors[f"digit{index}_propagation"] = (
                    parameters.mixed_radix_propagation_coefficients.contiguous()
                )
                for part, tensor in enumerate(
                    parameters.montgomery_reduction_parameters
                ):
                    tensors[f"digit{index}_reduction{part}"] = (
                        tensor.contiguous()
                    )
            coefficients = parameters.basis_extension_coefficients
            if coefficients is None:
                coefficients = torch.empty(
                    0, 0, dtype=rns.dtype, device=rns.device
                )
            if not coefficients.numel():
                extension = coefficients
            elif qp_ids == tuple(range(qp_ids[0], qp_ids[-1] + 1)):
                extension = coefficients[:, qp_ids[0] : qp_ids[-1] + 1]
            else:
                extension = coefficients.index_select(
                    1, torch.tensor(qp_ids, dtype=torch.long, device=rns.device)
                )
            tensors[f"digit{index}_extension"] = extension
        facts.update(
            modup_method=modup_method,
            galois_generator=self.galois_generator,
        )
        if self.ntt_context is not None:
            facts["ntt_backend"] = self.ntt_context.ntt_backend_name
        self._tensor_views[cache_key] = (tensors, facts)
        return tensors, facts


@dataclass(frozen=True)
class RotationSumTables:
    """Tensor operands for a shared QP key-product sum and its quotient.

    ``divisor`` is P, or P times the complete leading Q depth group. The
    quotient row order is surviving Q, dropped Q, P. Its pairwise inverses
    and optional nearest-rounding offset are prepared before execution.
    """

    tensors: Mapping[str, torch.Tensor]
    attributes: Mapping[str, object]
    p_montgomery: torch.Tensor
    quotient_row_order: torch.Tensor | None
    quotient_parameters: torch.Tensor
    quotient_inverses: torch.Tensor
    quotient_half: torch.Tensor | None
    remaining_q_count: int
    dropped_q_count: int
    divisor: int
    rescale_divisor: int

    def tensor_operands(self) -> dict[str, torch.Tensor]:
        """Return numerical operands without retaining a preparation owner."""
        values = dict(self.tensors)
        values.update(
            {
                "sum_p_montgomery": self.p_montgomery,
                "sum_quotient_parameters": self.quotient_parameters,
                "sum_quotient_inverses": self.quotient_inverses,
            }
        )
        if self.quotient_row_order is not None:
            values["sum_quotient_order"] = self.quotient_row_order
        if self.quotient_half is not None:
            values["sum_quotient_half"] = self.quotient_half
        return values

    @classmethod
    def from_operands(cls, values, attributes):
        """Bind supplied numerical operands; never construct missing tables."""
        return cls(
            tensors={
                k: v for k, v in values.items() if not k.startswith("sum_")
            },
            attributes=attributes,
            p_montgomery=values["sum_p_montgomery"],
            quotient_row_order=values.get("sum_quotient_order"),
            quotient_parameters=values["sum_quotient_parameters"],
            quotient_inverses=values["sum_quotient_inverses"],
            quotient_half=values.get("sum_quotient_half"),
            remaining_q_count=int(attributes["sum_remaining_q_count"]),
            dropped_q_count=int(attributes["sum_dropped_q_count"]),
            divisor=int(attributes["sum_divisor"]),
            rescale_divisor=int(attributes["sum_rescale_divisor"]),
        )


def prepare_rotation_sum_tables(
    tables: KeySwitchTables,
    depth: int,
    *,
    rescale: bool = False,
    rounding: str = "nearest",
    modup_method: str = "exact",
    key_prime_ids: tuple[int, ...] | None = None,
    digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
) -> RotationSumTables:
    """Prepare ordinary numerical data; key material remains caller supplied."""
    rns = tables.rns_context
    config = rns.config
    tensors, facts = tables.tensor_operands(
        depth,
        moddown_rounding=rounding,
        modup_method=modup_method,
        key_prime_ids=key_prime_ids,
        digit_prime_ids=digit_prime_ids,
    )
    q_ids = cast(tuple[int, ...], facts["q_prime_ids"])
    p_ids = cast(tuple[int, ...], facts["p_prime_ids"])
    next_ids = tuple(rns.rns_layout.prime_ids(depth + 1)) if rescale else q_ids
    drop = len(q_ids) - len(next_ids)
    if rescale and (drop <= 0 or not next_ids or q_ids[drop:] != next_ids):
        raise ValueError(
            "Rotation-sum rescale requires a leading complete Q group"
        )
    if rounding not in {"floor", "nearest"}:
        raise ValueError(
            "Rotation-sum quotient rounding must be floor or nearest"
        )
    p_product = prod(config.moduli[i] for i in p_ids)
    m_product = prod(config.moduli[i] for i in q_ids[:drop])
    radix = rns.montgomery_parameters.R
    p_montgomery = torch.tensor(
        [p_product * radix % config.moduli[i] for i in q_ids],
        dtype=rns.dtype,
        device=rns.device,
    )
    quotient_ids = (*next_ids, *q_ids[:drop], *p_ids)
    moduli = tuple(config.moduli[i] for i in quotient_ids)
    aux_count = len(p_ids) + drop
    rows = trailing_quotient_inverses(
        moduli,
        aux_count,
        radix,
        dtype=rns.dtype,
        device=rns.device,
    )
    divisor = p_product * m_product
    half = (
        torch.tensor(
            [((divisor - 1) // 2) % q for q in moduli],
            dtype=rns.dtype,
            device=rns.device,
        )
        if rounding == "nearest"
        else None
    )
    order = None
    if drop:
        q_count = len(q_ids)
        order = torch.tensor(
            [
                *range(drop, q_count),
                *range(drop),
                *range(q_count, q_count + len(p_ids)),
            ],
            dtype=torch.int64,
            device=rns.device,
        )
    return RotationSumTables(
        tensors=dict(tensors),
        attributes={
            **key_switch_runtime_attributes(facts),
            "moddown_rounding": rounding,
        },
        p_montgomery=p_montgomery,
        quotient_row_order=order,
        quotient_parameters=(
            tensors["qp_parameters"]
            if order is None
            else tensors["qp_parameters"].index_select(1, order)
        ),
        quotient_inverses=rows,
        quotient_half=half,
        remaining_q_count=len(next_ids),
        dropped_q_count=drop,
        divisor=divisor,
        rescale_divisor=m_product,
    )
