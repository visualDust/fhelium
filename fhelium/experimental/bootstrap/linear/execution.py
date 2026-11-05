r"""Apply compiled diagonal transforms with caller-owned keys and caches."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Literal

import numpy as np
import torch

from fhelium.eager import Engine
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    Plaintext,
    RotationKey,
    EvaluationKeySet,
)
from fhelium.experimental.bootstrap.linear.transform import (
    DiagonalLinearTransform,
)


def _rotate_bound_path(
    engine: Engine,
    ciphertext: Ciphertext,
    step: int,
    *,
    rotation_paths: Mapping[int, tuple[RotationKey, ...]],
    modup_method: Literal["exact", "crt", "normalized_crt"] = "exact",
) -> Ciphertext:
    r"""Apply the selected primitive sequence for $\operatorname{Rot}_r(c)$.

    Every switch preserves the Q basis, actual scale and depth. The prepared
    path can use different stored key layouts for its independent switches.
    A zero rotation returns an owning clone.
    """
    normalized = RotationKey.normalize_step(
        step, ring_dimension=engine.config.N
    )
    if normalized == 0:
        return ciphertext.clone()
    result = ciphertext
    for key in rotation_paths[normalized]:
        result = engine.rotate_with_key(result, key, modup_method=modup_method)
    return result


def _encode_diagonal(
    engine: Engine,
    cache: MutableMapping[
        tuple[torch.device, int, int, int, int, float],
        Plaintext | CompressedPlaintext,
    ],
    *,
    device: torch.device,
    retain: bool,
    compress: bool = True,
    prepare_materials: bool = True,
    transform: DiagonalLinearTransform,
    offset: int,
    giant: int,
    depth: int,
    scale: float,
    diagonal: np.ndarray,
) -> Plaintext | CompressedPlaintext:
    r"""Encode $\operatorname{Rot}_{-g}(d_k)$ for plaintext multiplication.

    `diagonal` has CPU axes `[slot]`. After rolling by `-giant`, encoding and
    preparation produce an unbatched tensor with axes `[limb, ntt_index]`,
    engine integral dtype/device, NTT domain, Montgomery residues, Q basis,
    `prime_ids` for `depth`, and the supplied actual ``scale``. The cached plaintext
    may be returned by identity; callers must treat it as immutable. The input
    NumPy array is not mutated. Automatic compression preserves the first
    encoded sample and chooses the minimum contiguous repeated-block extent.
    Prepared execution reads the existing cache without encoding or probing.
    """

    cache_key = (device, id(transform), offset, giant, depth, scale)
    plaintext = cache.get(cache_key) if prepare_materials else cache[cache_key]
    if plaintext is not None:
        return plaintext
    if compress and device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Prepare Bootstrap core materials before CUDA Graph capture"
                )
    plaintext = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.as_tensor(
                np.roll(diagonal, -giant),
                dtype=torch.complex128,
                device=device,
            ),
            depth=depth,
            scale=scale,
            device=device,
        ),
        modulus_basis='Q',
    )
    if compress:
        compact = CompressedPlaintext._from_plaintext_if_representable(
            plaintext, unique_count=None, compression_layout="contiguous"
        )
        if compact is not None:
            plaintext = compact
    if retain:
        cache[cache_key] = plaintext
    return plaintext


def _apply_linear_transform(
    arithmetic: BootstrapArithmetic,
    stages: tuple[Any, ...],
    evaluator,
    ciphertext: Ciphertext,
    *,
    evaluation_keys: EvaluationKeySet | None,
    key_bindings: MutableMapping[
        tuple[int, int, int], dict[int, tuple[RotationKey, ...]]
    ],
    diagonal_cache: MutableMapping[
        tuple[torch.device, int, int, int, int, float],
        Plaintext | CompressedPlaintext,
    ],
    retain_diagonals: bool,
    compress_diagonals: bool = True,
    prepare_materials: bool = True,
) -> Ciphertext:
    r"""Evaluate ordered maps $L_{m-1}\circ\cdots\circ L_0$.

    Each stage removes its Q group with product $M_d$. The supplied
    arithmetic owner chooses plaintext scale $M_d s_{d+1}/\Delta_{in}$,
    so the resulting actual scale is $s_{d+1}$.
    Coefficient-domain output keeps two components and the input batch axes;
    its prime rows are the suffix after the dropped group.
    """

    engine = arithmetic.engine
    if not stages or any(stage.slots != engine.num_slots for stage in stages):
        raise ValueError('linear transform has the wrong slot count')

    def rotate_with_keys(value: Ciphertext, step: int) -> Ciphertext:
        """Execute the primitive references selected for this stage."""
        return _rotate_bound_path(
            engine,
            value,
            step,
            rotation_paths=stage_paths,
            modup_method=getattr(evaluator, "modup_method", None)
            or arithmetic.modup_method,
        )

    def encode_cached_diagonal(**kwargs) -> Plaintext | CompressedPlaintext:
        """Encode through this invocation's optional diagonal cache."""

        return _encode_diagonal(
            engine,
            diagonal_cache,
            device=device,
            retain=retain_diagonals,
            compress=compress_diagonals,
            prepare_materials=prepare_materials,
            scale=arithmetic.plaintext_scale(result.scale, result.depth),
            **kwargs,
        )

    device = ciphertext.device
    result = ciphertext
    del ciphertext
    from .._keys import _select_rotation_paths, _stage_rotation_uses

    for transform in stages:
        binding = (id(evaluator), id(transform), result.depth)
        if prepare_materials:
            assert evaluation_keys is not None
            stage_paths: dict[int, tuple[RotationKey, ...]] = {}
            for depth, offsets, kind in _stage_rotation_uses(
                evaluator, transform, result.depth
            ):
                selected = _select_rotation_paths(
                    engine, evaluation_keys, depth, offsets, kind, result.device
                )
                if selected is None:
                    raise ValueError(
                        f"Missing compatible rotation paths for stage {transform.name!r} at depth {depth}: {offsets}"
                    )
                stage_paths.update(selected)
            key_bindings[binding] = stage_paths
        else:
            stage_paths = key_bindings[binding]
        result = evaluator.evaluate(
            arithmetic,
            result,
            transform,
            rotation_paths=stage_paths,
            rotate=rotate_with_keys,
            encode_diagonal=encode_cached_diagonal,
        )
    return result
