r"""Share coefficient-digit preparation across direct rotation key products."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import torch

from fhelium.backend.rns.moddown import moddown_ntt_qp_to_q, moddown_qp_to_q
from fhelium.backend.ckks.key_switch import lift_key_switch_digit
from fhelium.backend.ntt.operations import execute_named_transform
from fhelium.backend.ntt.automorphism import ntt_galois_indices
from ._galois import rotation_galois_element
from fhelium.backend.rns.automorphism import (
    coefficient_galois_gather_indices,
)
from fhelium.native.wrapper import ckks_ops, rns_ops


def rotate_component(
    source: torch.Tensor,
    step: int,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
) -> torch.Tensor:
    generator = cast(int, attributes["galois_generator"])
    n = source.size(-1)
    indices, signs = coefficient_galois_gather_indices(
        n, rotation_galois_element(n, step, generator), source.device
    )
    return ckks_ops.apply_coefficient_galois_automorphism(
        source, indices, signs, tensors["q_parameters"][0]
    )


def prepare_digits(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
) -> torch.Tensor:
    """Materialize each active hybrid digit once in QP NTT/Montgomery form."""
    spans = cast(tuple[tuple[int, int, int], ...], attributes["digits"])
    if not spans:
        raise ValueError("Rotation hoisting requires an active RNS digit")
    digits = source.new_empty(
        (
            len(spans),
            *source.shape[:-2],
            tensors["qp_parameters"].size(1),
            source.size(-1),
        )
    )
    width = max(stop - start for start, stop, _ in spans)
    scratch = (
        source.new_empty((*source.shape[:-2], width, source.size(-1)))
        if width > 1 and attributes.get("modup_method", "exact") == "exact"
        else None
    )
    for index, span in enumerate(spans):
        lifted = lift_key_switch_digit(
            source,
            tensors,
            attributes,
            index,
            span,
            out=digits[index],
            scratch=scratch,
        )
        execute_named_transform(
            lifted, tensors, attributes, "qp", "forward_montgomery_"
        )
    return digits


def accumulate(
    prepared: torch.Tensor,
    key: torch.Tensor,
    step: int,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
) -> torch.Tensor:
    """Gather each digit's NTT permutation in the key-product load."""
    prototype = prepared[0]
    accumulator = torch.zeros(
        (2, *prototype.shape), dtype=prototype.dtype, device=prototype.device
    )
    spans = cast(tuple[tuple[int, int, int], ...], attributes["digits"])
    start = cast(int, attributes["key_row_start"])
    indices = ntt_galois_indices(
        prototype.size(-1),
        rotation_galois_element(
            prototype.size(-1), step, cast(int, attributes["galois_generator"])
        ),
        prototype.device,
    )
    if (
        prototype.is_cuda
        and len(prepared) <= 5
        and tuple(span[2] for span in spans) == tuple(range(len(spans)))
    ):
        ckks_ops.keyswitch_accumulate_products_(
            accumulator[0],
            accumulator[1],
            list(prepared),
            key,
            tensors["qp_parameters"],
            start,
            indices,
        )
    else:
        for span, digit in zip(spans, prepared, strict=True):
            ckks_ops.keyswitch_accumulate_digit_products_(
                accumulator[0],
                accumulator[1],
                digit,
                key[span[2]],
                tensors["qp_parameters"],
                start,
                indices,
            )
    return accumulator


def apply_rotation(
    c0: torch.Tensor,
    prepared: torch.Tensor,
    key: torch.Tensor,
    step: int,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    *,
    output_ntt: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulator = accumulate(prepared, key, step, tensors, attributes)
    if output_ntt:
        corrections = moddown_ntt_qp_to_q(accumulator, tensors, attributes)
        indices = ntt_galois_indices(
            c0.size(-1),
            rotation_galois_element(
                c0.size(-1), step, cast(int, attributes["galois_generator"])
            ),
            c0.device,
        )
        rotated = ckks_ops.apply_ntt_galois_automorphism(c0, indices)
        return rns_ops.add_lazy(
            rotated, corrections[0], tensors["q_parameters"]
        ), corrections[1]
    corrections = []
    for component in accumulator:
        execute_named_transform(
            component, tensors, attributes, "qp", "inverse_to_standard_"
        )
        corrections.append(moddown_qp_to_q(component, tensors, attributes))
    rotated = rotate_component(c0, step, tensors, attributes)
    return rns_ops.add_standard(
        rotated, corrections[0], tensors["q_parameters"]
    ), corrections[1]


def iter_rotations(
    source: torch.Tensor,
    keys: Sequence[torch.Tensor | None],
    steps: Sequence[int],
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    *,
    output_ntt: bool = False,
) -> Iterator[torch.Tensor]:
    """Yield direct rotations from one shared decomposition of component one.

    A None key denotes the identity. Outputs own their payloads; consuming
    one does not retain previous outputs. The prepared digits live only until
    the iterator completes or is closed.
    """
    prepared = (
        prepare_digits(source[1], tensors, attributes)
        if any(k is not None for k in keys)
        else None
    )
    c0 = (
        execute_named_transform(
            source[0],
            tensors,
            attributes,
            "q",
            "forward_to_montgomery_",
            in_place=False,
        )
        if output_ntt
        else source[0]
    )
    for key, step in zip(keys, steps, strict=True):
        if key is None:
            yield (
                execute_named_transform(
                    source,
                    tensors,
                    attributes,
                    "q",
                    "forward_to_montgomery_",
                    in_place=False,
                )
                if output_ntt
                else source.clone()
            )
        else:
            assert prepared is not None
            yield torch.stack(
                apply_rotation(
                    c0,
                    prepared,
                    key,
                    step,
                    tensors,
                    attributes,
                    output_ntt=output_ntt,
                )
            )
