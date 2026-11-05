"""Adapt streaming rotation kernels to CKKS values and Engine-owned keys."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, cast

from fhelium.backend.ckks.rotation._aggregate import RotationSumAccumulator
from fhelium.backend.ckks.rotation._hoisted import iter_rotations
from fhelium.ir.ckks_state import (
    depth_prime_ids,
    output_residues,
    quotient_scale,
)
from fhelium.values import Ciphertext, RotationKey
from fhelium.values.state import PolynomialDomain

if TYPE_CHECKING:
    from ._engine import Engine


def iter_rotations_with_keys(
    engine: Engine,
    value: Ciphertext,
    keys: Sequence[RotationKey | None],
    *,
    output_domain: PolynomialDomain,
    moddown_rounding: Literal["nearest", "floor"],
    modup_method: Literal["exact", "crt", "normalized_crt"],
) -> Iterator[Ciphertext]:
    from ._engine import _ciphertext_result

    residues = output_residues(output_domain)
    engine.validate_ciphertext(value)
    value.assert_state(
        components=2,
        modulus_basis="Q",
        polynomial_domain="coefficient",
        residue_representation="standard",
    )
    placed = []
    for key in keys:
        if key is not None:
            key = cast(
                RotationKey,
                engine._key_on_device(
                    key, value.device, operation_name="iter_rotations_with_keys"
                ),
            )
        placed.append(key)
    if not placed:
        return
    tensors, attributes = engine._dispatcher_for(
        value.device
    ).device_resources.key_switch_operands(
        value.depth,
        moddown_rounding=moddown_rounding,
        modup_method=modup_method,
        **engine._key_switch_attributes(
            value, [key for key in placed if key is not None]
        ),
    )
    for data in iter_rotations(
        value.data,
        [None if key is None else key.data for key in placed],
        [0 if key is None else key.rotation_step for key in placed],
        tensors,
        attributes,
        output_ntt=output_domain == "ntt",
    ):
        yield _ciphertext_result(
            value,
            data,
            polynomial_domain=output_domain,
            residue_representation=residues,
        )
        del data


def sum_rotations(
    engine: Engine,
    terms: Iterable[tuple[Ciphertext, RotationKey | None]],
    *,
    rescale: bool,
    moddown_rounding: Literal["nearest", "floor"],
    modup_method: Literal["exact", "crt", "normalized_crt"],
) -> Ciphertext:
    iterator = iter(terms)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Rotation sum requires at least one term") from None
    source = first[0]
    engine.validate_ciphertext(source)
    source.assert_state(components=2, modulus_basis="Q")
    if source.residue_representation != output_residues(
        source.polynomial_domain
    ):
        raise ValueError(
            "Rotation-sum inputs require coefficient/standard or NTT/Montgomery residues"
        )
    tables = engine._dispatcher_for(
        source.device
    ).device_resources.rotation_sum_tables(
        source.depth,
        rescale=rescale,
        rounding=moddown_rounding,
        modup_method=modup_method,
    )
    accumulator = RotationSumAccumulator(
        tables, input_domain=source.polynomial_domain
    )
    state = (
        source.depth,
        source.scale,
        source.prime_ids,
        source.modulus_basis,
        source.polynomial_domain,
        source.residue_representation,
        source.data.shape,
        source.data.dtype,
        source.device,
    )
    depth = source.depth + int(rescale)
    scale = quotient_scale(source.scale, tables.rescale_divisor)
    device = source.device
    value, key = first
    key_layout = None
    del first, source
    while True:
        actual = (
            value.depth,
            value.scale,
            value.prime_ids,
            value.modulus_basis,
            value.polynomial_domain,
            value.residue_representation,
            value.data.shape,
            value.data.dtype,
            value.device,
        )
        if actual != state:
            raise ValueError(
                "Rotation-sum terms must share state, shape, dtype and device"
            )
        if key is not None:
            key = cast(
                RotationKey,
                engine._key_on_device(
                    key, device, operation_name="sum_rotations"
                ),
            )
            layout = engine._key_switch_attributes(value, [key])
            if key_layout is None:
                key_layout = layout
                accumulator.tables = engine._dispatcher_for(
                    device
                ).device_resources.rotation_sum_tables(
                    value.depth,
                    rescale=rescale,
                    rounding=moddown_rounding,
                    modup_method=modup_method,
                    **layout,
                )
            elif layout != key_layout:
                raise ValueError(
                    "A rotation-sum execution group requires matching key rows and digit supports"
                )
        accumulator.add(
            value.data,
            None if key is None else key.data,
            0 if key is None else key.rotation_step,
        )
        del value, key
        try:
            value, key = next(iterator)
        except StopIteration:
            break
    return Ciphertext._from_fields(
        data=accumulator.finish(),
        depth=depth,
        scale=scale,
        prime_ids=depth_prime_ids(engine.config, depth, "Q"),
        modulus_basis="Q",
        polynomial_domain="coefficient",
        residue_representation="standard",
    )
