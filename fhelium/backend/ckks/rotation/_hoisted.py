r"""Direct rotations that share one decomposition of component one."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch

from fhelium.backend._native_operands import key_switch_options, mapping_tables
from fhelium.backend.implementation import InvocationAttributes
from fhelium.native.wrapper import ckks_ops, ntt_ops


def iter_rotations(
    source: torch.Tensor,
    keys: Sequence[torch.Tensor | None],
    steps: Sequence[int],
    tensors: Mapping[str, torch.Tensor],
    attributes: InvocationAttributes,
    *,
    seeds: Sequence[torch.Tensor | None] | None = None,
    output_ntt: bool = False,
) -> Iterator[torch.Tensor]:
    """Yield direct rotations from one shared decomposition of component one.

    ``source`` is a coefficient standard CT2. A None key denotes the identity.
    Outputs own their payloads; consuming one does not retain previous
    outputs. The hoisted digits (``rotation_digits``) live only until the
    iterator completes or is closed. Seeded keys have entries in ``seeds``
    and the round count ``attributes["key_rounds"]``.
    """
    names, tables = mapping_tables(tensors)
    options = key_switch_options(attributes)
    options["output_domain"] = "ntt" if output_ntt else "coefficient"
    backend = options["ntt_backend"]
    digits = (
        ckks_ops.rotation_digits(source[1], names, tables, **options)
        if any(key is not None for key in keys)
        else None
    )
    c0 = (
        ntt_ops.transform(
            source[0], names, tables, backend, "q", "forward_to_montgomery"
        )
        if output_ntt
        else source[0]
    )
    seeds = [None] * len(keys) if seeds is None else seeds
    for key, step, seed in zip(keys, steps, seeds, strict=True):
        if key is None:
            yield (
                ntt_ops.transform(
                    source, names, tables, backend, "q", "forward_to_montgomery"
                )
                if output_ntt
                else source.clone()
            )
        else:
            assert digits is not None
            yield ckks_ops.rotate_hoisted(
                c0,
                digits,
                key,
                int(step),
                names,
                tables,
                **{**options, "key_seeds": [] if seed is None else [seed]},
            )
