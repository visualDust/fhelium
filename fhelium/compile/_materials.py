"""Describe numerical materials and optionally prepare missing Tensor bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

import torch
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Operation, SSAValue

from fhelium.config import (
    DEFAULT_CPU_NTT_BACKEND,
    CkksConfig,
    compatible_ntt_backends,
)
from fhelium.ir.dialects import core
from fhelium.values import (
    EvaluationKeySet,
    KeySwitchKey,
    PublicKey,
    SecretKey,
    TensorResident,
)

if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext

    from ._compilation import Compilation


def material_sources(resources):
    """Retain supplied data providers as a reusable tuple."""
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext

    return (
        (resources,)
        if isinstance(resources, (CkksDeviceResources, RnsContext, NttContext))
        else tuple(resources)
    )


def material_symbol(operation, prefix: str) -> str:
    """Allocate a symbol in the containing Program without renumbering old ones."""
    from xdsl.dialects.builtin import DictionaryAttr

    root = operation
    while root.parent_op() is not None:
        root = root.parent_op()
    used = {
        op.symbol.data
        for op in root.walk()
        if isinstance(op, core.MaterialRefOp) and op.symbol is not None
    }
    descriptions = root.attributes.get("fhelium.material_descriptions")
    if isinstance(descriptions, DictionaryAttr):
        used.update(descriptions.data)
    result = prefix
    index = 2
    while result in used:
        result = f"{prefix}_{index}"
        index += 1
    return result


def rns_parameter_identity(
    config: CkksConfig | None, prime_ids, physical: Mapping[str, Attribute]
) -> tuple[object, ...] | None:
    """Identify a compiler-requested default RNS table, not a caller binding."""
    if (
        config is None
        or not prime_ids
        or any(
            physical.get(name) in (None, StringAttr("unknown"))
            for name in ("dtype", "device")
        )
    ):
        return None
    return (
        "default-rns-parameters",
        config.N,
        config.q_moduli,
        config.p_moduli,
        config.galois_generator,
        tuple(prime_ids),
        physical.get("dtype"),
        physical.get("device"),
    )


def declare_material(
    compilation: Compilation,
    operation: Operation,
    prefix: str,
    value_type: Attribute,
    description: dict[str, object],
    *,
    identity: tuple[object, ...] | None = None,
) -> tuple[tuple[core.MaterialRefOp, ...], SSAValue]:
    """Declare one material input, sharing only an explicit default request.

    The request-local construction cache never scans caller descriptions or
    compares Tensor contents. Existing caller symbols stay independent. Shared
    declarations reuse a dominating reference, or the same symbol in another
    block. The resulting Program records the sharing through ordinary SSA and
    material symbols; no execution-time request registry is needed.
    """
    entries = cast(
        tuple[
            Operation,
            dict[
                tuple[object, ...], list[tuple[core.MaterialRefOp, Operation]]
            ],
        ]
        | None,
        compilation.workspace.get(declare_material),
    )
    if entries is None or entries[0] is not compilation.program.module:
        entries = (compilation.program.module, {})
        compilation.workspace[declare_material] = entries
    requests = entries[1]
    references = requests.get(identity, ()) if identity is not None else ()
    block = operation.parent_block()
    for reference, anchor in references:
        if reference.parent_block() is None and anchor is operation:
            return (), reference.value
        if reference.parent_block() is block:
            if not reference.is_before_in_block(operation):
                assert block is not None
                block.detach_op(reference)
                block.insert_op_before(reference, operation)
            return (), reference.value
    symbol = (
        cast(StringAttr, references[0][0].symbol).data
        if references
        else material_symbol(operation, prefix)
    )
    reference = core.MaterialRefOp(value_type, symbol=symbol)
    if identity is not None:
        requests.setdefault(identity, []).append((reference, operation))
    if symbol not in compilation.program.material_descriptions:
        compilation.program.set_material_description(symbol, description)
    return (reference,), reference.value


def parameter_description(
    config: CkksConfig, kind: str, **fields: object
) -> dict[str, object]:
    """Describe the mathematical source of a numerical table, not its owner."""
    return {
        "kind": kind,
        "ring_dimension": config.N,
        "q_depth_groups": [list(group) for group in config.q_depth_groups],
        "p_moduli": list(config.p_moduli),
        "galois_generator": config.galois_generator,
        **fields,
    }


def key_description(
    key: SecretKey | PublicKey | KeySwitchKey,
) -> dict[str, object]:
    """Describe known key metadata without identifying its secret or participant."""
    result: dict[str, object] = {
        "kind": type(key).__name__,
        "prime_ids": list(key.prime_ids),
        "polynomial_domain": key.polynomial_domain,
        "modulus_basis": key.modulus_basis,
        "residue_representation": key.residue_representation,
    }
    step = getattr(key, "rotation_step", None)
    if step is not None:
        result["rotation_step"] = step
    return result


def named_material_tensors(
    names: Mapping[str, object] | None,
) -> dict[int, str]:
    """Associate caller names with supplied Tensor leaves for capture."""
    result: dict[int, str] = {}
    for name, value in (names or {}).items():
        if isinstance(value, torch.Tensor):
            result[id(value)] = name
        elif isinstance(value, TensorResident):
            tensors = value._resident_tensors
            for index, tensor in enumerate(tensors):
                result[id(tensor)] = (
                    name if len(tensors) == 1 else f"{name}.{index}"
                )
        else:
            raise TypeError(
                "Material names refer to Tensors or TensorResident values"
            )
    return result


def prepare_material_bindings(
    compilation: Compilation,
    *,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
) -> tuple[str, ...]:
    """Fill only missing bindings from supplied data providers and keys.

    Descriptions guide this optional selection step. Named key entries matching
    a symbol or its label are direct assignments, without metadata checks.
    Semantic selection accepts one distinct candidate and leaves absent or
    ambiguous candidates unresolved. Existing bindings are never inspected or
    overwritten. The returned symbols remain unbound. No keys are generated.
    """
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext

    providers = (
        (resources,)
        if isinstance(resources, (CkksDeviceResources, RnsContext, NttContext))
        else tuple(cast(Iterable[object], resources))
    )
    if isinstance(keys, EvaluationKeySet):
        keys = keys._keys()
    named_keys = dict(keys) if isinstance(keys, Mapping) else {}
    key_values = (
        tuple(named_keys.values()) if isinstance(keys, Mapping) else tuple(keys)
    )
    descriptions = compilation.program.material_descriptions
    cache: dict[tuple[object, ...], object] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for operation in compilation.program.walk():
        if (
            not isinstance(operation, core.MaterialRefOp)
            or operation.symbol is None
        ):
            continue
        symbol = operation.symbol.data
        if symbol in seen or symbol in compilation.material_bindings:
            continue
        seen.add(symbol)
        description = descriptions.get(symbol, {})
        selected = named_keys.get(
            symbol, named_keys.get(str(description.get("label", "")))
        )
        if selected is not None:
            compilation.material_bindings[symbol] = _key_tensor(selected)
            continue
        kind = description.get("kind")
        if not isinstance(kind, str):
            missing.append(symbol)
            continue
        state = getattr(
            getattr(operation.value.type, "state", None), "data", {}
        )
        device_hint = state.get("device")
        selection = dict(description)
        if (
            isinstance(device_hint, StringAttr)
            and device_hint.data != "unknown"
        ):
            selection.setdefault("device", device_hint.data)
        candidates: dict[int, torch.Tensor] = {}
        if kind in {
            "SecretKey",
            "PublicKey",
            "KeySwitchKey",
            "RotationKey",
            "RelinearizationKey",
            "ConjugationKey",
        }:
            for key in key_values:
                if not isinstance(key, (SecretKey, PublicKey, KeySwitchKey)):
                    continue
                if selection.get("device") not in (None, str(key.device)):
                    continue
                known = key_description(key)
                if all(
                    description[name] == value
                    for name, value in known.items()
                    if name in description
                ):
                    candidates[id(key.data)] = key.data
        else:
            for provider in providers:
                tensor = _provider_tensor(provider, selection, cache)
                if tensor is not None:
                    candidates[id(tensor)] = tensor
        if len(candidates) == 1:
            compilation.material_bindings[symbol] = next(
                iter(candidates.values())
            )
        else:
            missing.append(symbol)
    return tuple(missing)


def _key_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (SecretKey, PublicKey, KeySwitchKey)):
        return value.data
    raise TypeError("A named key binding supplies a Tensor or key value")


def _provider_tensor(
    provider: object,
    description: Mapping[str, object],
    cache: dict[tuple[object, ...], object],
) -> torch.Tensor | None:
    from fhelium.backend.ckks.materialization import (
        CkksDeviceResources,
    )
    from fhelium.backend.ckks.tables import KeySwitchTables
    from fhelium.backend.rns.tables import (
        RescaleTables,
        moddown_inverse_tables,
        rescale_inverse_matrix,
    )
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext

    if not isinstance(provider, (CkksDeviceResources, RnsContext, NttContext)):
        return None
    kind = description.get("kind")
    if not isinstance(kind, str) or kind not in {
        "rns_parameters",
        "twice_modulus",
        "ntt_table",
        "key_switch_table",
        "rescale_table",
        "encode_table",
        "periodic_plaintext_table",
        "rounding_state",
        "integer_scalar",
    }:
        return None
    config = provider.config
    expected = parameter_description(config, cast(str, kind))
    if any(
        description[name] != value
        for name, value in expected.items()
        if name in description
    ):
        return None
    if description.get("device") not in (None, str(provider.device)):
        return None
    context = (
        provider if isinstance(provider, RnsContext) else provider.rns_context
    )
    described_ids = description.get("prime_ids")
    ids = (
        tuple(described_ids)
        if isinstance(described_ids, (list, tuple))
        and all(isinstance(value, int) for value in described_ids)
        else ()
    )
    if kind in {"rns_parameters", "twice_modulus", "integer_scalar"}:
        if not ids:
            return None
        parameters = context.rns_parameters_for_prime_ids(ids)
        if kind == "integer_scalar":
            from fhelium.backend.ckks.scalar import scalar_operands

            if "scalar" not in description:
                return None
            tensors, _ = scalar_operands(
                context, ids, cast(int, description["scalar"]), None, None
            )
            return tensors[1]
        return parameters[0] if kind == "twice_modulus" else parameters
    if kind == "rounding_state":
        return (
            provider.rng.rounding_state
            if isinstance(provider, CkksDeviceResources)
            else None
        )
    if kind == "periodic_plaintext_table":
        if not isinstance(provider, CkksDeviceResources):
            return None
        index = description.get("index")
        period = description.get("period")
        depth = description.get("depth")
        basis = description.get("basis")
        if (
            not isinstance(index, int)
            or not 0 <= index < 8
            or not isinstance(period, int)
            or not isinstance(depth, int)
            or basis not in ("Q", "QP")
        ):
            return None
        return provider.periodic_encode_operands(
            period, depth, cast(str, basis)
        )[index]
    if kind == "encode_table":
        index = description.get("index")
        if not isinstance(index, int) or index not in (0, 1, 2):
            return None
        if isinstance(provider, CkksDeviceResources):
            return provider.encode_operands()[index]
        if index == 2:
            return None
        from fhelium.backend.ckks.codec import _embedding

        pre, _ = _embedding._permutations(
            config.N, str(context.device), config.galois_generator
        )
        return (
            pre
            if index == 0
            else _embedding._twister(config.N, str(context.device))
        )
    policy = description.get("ntt_backend")
    ntt: NttContext | None = (
        provider if isinstance(provider, NttContext) else None
    )
    needs_ntt = kind == "ntt_table" or description.get("input_domain") == "ntt"
    if needs_ntt:
        if not isinstance(policy, str) or policy not in compatible_ntt_backends(
            config.logN
        ):
            return None
        if context.device.type == "cpu" and policy != DEFAULT_CPU_NTT_BACKEND:
            return None
        if isinstance(provider, CkksDeviceResources):
            ntt = provider.ntt_context
        elif ntt is None:
            token = (id(context), "ntt", policy)
            if token not in cache:
                cache[token] = NttContext(
                    context, ntt_backend=cast(str, policy)
                )
            ntt = cast(NttContext, cache[token])
        if ntt.ntt_backend_name != policy:
            from fhelium.backend.ntt._selection import table_layout
            from fhelium.config.ntt import resolve_ntt_backend_policy

            if kind != "ntt_table" or table_layout(
                ntt.ntt_policy
            ) != table_layout(resolve_ntt_backend_policy(policy)):
                return None
    if kind == "ntt_table":
        if (
            not ids
            or "index" not in description
            or description.get("direction") not in ("forward", "inverse")
        ):
            return None
        assert ntt is not None
        return ntt.tensor_operands(
            ids, inverse=description["direction"] == "inverse"
        )[int(cast(int, description["index"]))]
    if "depth" not in description or "name" not in description:
        return None
    depth = int(cast(int, description["depth"]))
    name = cast(str, description["name"])
    if kind == "key_switch_table":
        token = (id(context), "key_switch_rns")
        if token not in cache:
            cache[token] = KeySwitchTables(
                context,
                None,
                moddown_inverse_tables(context),
                config.galois_generator,
            )
        tensors, facts = cast(KeySwitchTables, cache[token]).tensor_operands(
            depth
        )
        if "digits" in description and description["digits"] != [
            list(span)
            for span in cast(Iterable[Iterable[int]], facts["digits"])
        ]:
            return None
        return tensors.get(name)
    include_p = description.get("basis", "Q") == "QP"
    rows = context.rns_layout.row_count(depth, include_p=include_p)
    count = len(config.q_depth_groups[depth])
    if isinstance(provider, CkksDeviceResources):
        tensors, _ = provider.rescale_operands(
            rows,
            count,
            include_p=include_p,
            input_domain=cast(
                str, description.get("input_domain", "coefficient")
            ),
        )
    else:
        token = (id(context), "rescale")
        if token not in cache:
            cache[token] = RescaleTables(
                context, rescale_inverse_matrix(context)
            )
        tensors, _ = cast(RescaleTables, cache[token]).tensor_operands(
            rows,
            count,
            include_p=include_p,
            ntt_context=ntt if needs_ntt else None,
        )
    return tensors.get(name)


__all__ = ["prepare_material_bindings"]
