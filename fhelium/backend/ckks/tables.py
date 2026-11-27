"""Tensor views used by complete hybrid CKKS key-switch implementations."""

from __future__ import annotations

from fhelium.config import CkksConfig
from dataclasses import dataclass, field
from collections.abc import Mapping
from math import prod
from typing import TYPE_CHECKING, cast
import torch
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.decomposition import key_decomposition_facts
from fhelium.backend.rns.layout import RnsLayout
from fhelium.backend.rns.tables import shared_parameter_views
from fhelium.native.wrapper import ckks_ops
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
    """Describe the low switch basis and guard-prime return basis.

    The low basis is the source-depth Q basis with ``low_p_moduli``; the return
    basis is [guard, full Q] with ``return_p_moduli``.
    """
    source_q = config.active_q_moduli(source_depth)

    def auxiliary(groups, p_moduli):
        return CkksConfig.suffix_chain(
            logN=config.logN,
            scale=config.scales[0],
            sigma=config.sigma,
            galois_generator=config.galois_generator,
            security_bits=config.security_bits,
            enforce_security_budget=False,
            q_depth_groups=groups,
            p_moduli=p_moduli,
        )

    return (
        auxiliary((source_q,), low_p_moduli),
        auxiliary(((return_prime,), config.q_moduli), return_p_moduli),
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

    low_config, return_config = encapsulation_configs(
        rns_context.config,
        source_depth,
        low_p_moduli,
        return_prime,
        return_p_moduli,
    )
    config = rns_context.config
    policy = ntt_context.ntt_backend_name
    target_ids = rns_context.rns_layout.prime_ids(target_depth)
    names, values = ckks_ops.encapsulation_table_set(
        rns_context.rns_parameter_tensor,
        list(config.moduli),
        config.logN,
        policy,
        config.scales[0],
        list(config.active_q_moduli(source_depth)),
        list(low_p_moduli),
        return_prime,
        list(return_p_moduli),
        list(config.q_moduli),
        list(target_ids),
        rounding,
    )
    tensors = shared_parameter_views(
        rns_context, dict(zip(names, values, strict=True))
    )
    tensors["return_parameters"] = tensors["return_forward_0"]
    low_facts = key_decomposition_facts(
        low_config,
        0,
        tuple(range(low_config.total_num_primes)),
        tuple(
            spec.prime_ids
            for spec in RnsLayout.from_config(low_config).digit_specs(0)
        ),
    )
    low_facts.update(
        moddown_rounding=rounding,
        basis_conversion="exact",
        galois_generator=low_config.galois_generator,
        ntt_backend=policy,
    )
    facts = {
        "low_" + name: value
        for name, value in key_switch_runtime_attributes(low_facts).items()
    }
    q_count = config.num_q_primes
    facts.update(
        ntt_backend=policy,
        q_count=q_count,
        target_row_start=target_ids[0],
        target_row_count=len(target_ids),
        return_drop_count=len(return_p_moduli) + 1,
        moddown_rounding=rounding,
    )
    return tensors, facts


@dataclass(frozen=True)
class KeySwitchTables:
    """Own the tables used by hybrid CKKS key switching."""

    rns_context: RnsContext
    ntt_context: NttContext | None
    galois_generator: int = 3
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
    def device(self) -> torch.device:
        return self.rns_context.device

    def _native_tables(
        self,
        q_ids: tuple[int, ...],
        p_ids: tuple[int, ...],
        digits: tuple[tuple[int, int, int], ...],
        *,
        basis_conversion: str,
        moddown_rounding: str,
        rescale: bool = False,
        drop_ids: tuple[int, ...] = (),
        target_ids: tuple[int, ...] = (),
        rescale_rounding: str = "nearest",
    ) -> dict[str, torch.Tensor]:
        rns = self.rns_context
        names, values = ckks_ops.key_switch_table_set(
            rns.rns_parameter_tensor,
            list(rns.config.moduli),
            rns.config.logN,
            list(q_ids),
            list(p_ids),
            [value for digit in digits for value in digit],
            basis_conversion,
            moddown_rounding,
            rescale,
            list(drop_ids),
            list(target_ids),
            rescale_rounding,
        )
        return shared_parameter_views(
            rns, dict(zip(names, values, strict=True))
        )

    def moddown_tensor_operands(
        self,
        q_ids: tuple[int, ...],
        p_ids: tuple[int, ...],
        *,
        rounding: str = "nearest",
        depth: int | None = None,
        basis_conversion: str = "exact",
    ) -> dict[str, torch.Tensor]:
        """Prepare quotient data for the represented ordered Q and P rows.

        ``basis_conversion="normalized_crt"`` adds the ``moddown_crt_`` centered
        quotient tables, which replace the exact remainder of the P rows.
        """
        normalized = basis_conversion == "normalized_crt"
        token = ("moddown", q_ids, p_ids, rounding, normalized)
        if token in self._tensor_views:
            return self._tensor_views[token][0]
        tensors = self._native_tables(
            q_ids,
            p_ids,
            (),
            basis_conversion=basis_conversion,
            moddown_rounding=rounding,
        )
        self._tensor_views[token] = (tensors, {})
        return tensors

    def tensor_operands(
        self,
        depth: int,
        *,
        moddown_rounding: str = "nearest",
        basis_conversion: str = "exact",
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
        if basis_conversion not in {"exact", "crt", "normalized_crt"}:
            raise ValueError(
                "basis_conversion must be exact, crt, or normalized_crt"
            )
        cache_key = (
            depth,
            moddown_rounding,
            basis_conversion,
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
        digits = cast(tuple[tuple[int, int, int], ...], facts["digits"])
        drop_ids: tuple[int, ...] = ()
        target_ids: tuple[int, ...] = ()
        if rescale:
            from fhelium.backend.rns._preparation import joint_rescale_facts

            joint = joint_rescale_facts(rns.config, depth, q_ids)
            drop_ids = cast(tuple[int, ...], joint["joint_drop_prime_ids"])
            target_ids = cast(tuple[int, ...], joint["joint_target_prime_ids"])
            facts.update(
                joint,
                rescale_rounding=rescale_rounding,
                output_domain=output_domain,
            )
        tensors = self._native_tables(
            q_ids,
            p_ids,
            digits,
            basis_conversion=basis_conversion,
            moddown_rounding=moddown_rounding,
            rescale=rescale,
            drop_ids=drop_ids,
            target_ids=target_ids,
            rescale_rounding=rescale_rounding,
        )
        if self.ntt_context is not None:
            transforms = [
                (basis, ids, inverse)
                for basis, ids in (("q", q_ids), ("qp", qp_ids), ("p", p_ids))
                for inverse in (False, True)
            ]
            if rescale:
                transforms = [
                    ("q", q_ids, True),
                    ("qp", qp_ids, False),
                    ("joint_dropped", (*drop_ids, *p_ids), True),
                    (
                        "joint_remaining",
                        target_ids,
                        output_domain == "coefficient",
                    ),
                ]
            for basis, ids, inverse in transforms:
                direction = "inverse" if inverse else "forward"
                for index, tensor in enumerate(
                    self.ntt_context.tensor_operands(ids, inverse=inverse)
                ):
                    tensors[f"{basis}_{direction}_{index}"] = tensor
        facts.update(
            basis_conversion=basis_conversion,
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
    basis_conversion: str = "exact",
    key_prime_ids: tuple[int, ...] | None = None,
    digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
) -> RotationSumTables:
    """Prepare ordinary numerical data; key material remains caller supplied."""
    rns = tables.rns_context
    config = rns.config
    tensors, facts = tables.tensor_operands(
        depth,
        moddown_rounding=rounding,
        basis_conversion=basis_conversion,
        key_prime_ids=key_prime_ids,
        digit_prime_ids=digit_prime_ids,
    )
    q_ids = cast(tuple[int, ...], facts["q_prime_ids"])
    p_ids = cast(tuple[int, ...], facts["p_prime_ids"])
    next_ids = tuple(rns.rns_layout.prime_ids(depth + 1)) if rescale else q_ids
    dropped = tuple(i for i in q_ids if i not in next_ids)
    drop = len(dropped)
    if rescale and (
        not drop or not next_ids or not set(next_ids) <= set(q_ids)
    ):
        raise ValueError(
            "Rotation-sum rescale requires a transition that only drops Q rows"
        )
    if rounding not in {"floor", "nearest"}:
        raise ValueError(
            "Rotation-sum quotient rounding must be floor or nearest"
        )
    p_moduli = [config.moduli[i] for i in p_ids]
    m_product = prod(config.moduli[i] for i in dropped)
    names, values = ckks_ops.rotation_sum_table_set(
        rns.rns_parameter_tensor,
        list(config.moduli),
        config.logN,
        list(q_ids),
        list(p_ids),
        list(next_ids) if rescale else [],
        rounding,
    )
    sums = dict(zip(names, values, strict=True))
    return RotationSumTables(
        tensors=dict(tensors),
        attributes={
            **key_switch_runtime_attributes(facts),
            "moddown_rounding": rounding,
        },
        p_montgomery=sums["sum_p_montgomery"],
        quotient_row_order=sums.get("sum_quotient_order"),
        quotient_parameters=(
            tensors["qp_parameters"]
            if "sum_quotient_order" not in sums
            else sums["sum_quotient_parameters"]
        ),
        quotient_inverses=sums["sum_quotient_inverses"],
        quotient_half=sums.get("sum_quotient_half"),
        remaining_q_count=len(next_ids),
        dropped_q_count=drop,
        divisor=prod(p_moduli) * m_product,
        rescale_divisor=m_product,
    )
