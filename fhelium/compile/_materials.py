"""Describe numerical materials and optionally prepare missing Tensor bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
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
    """Identify a default RNS-table request for declaration reuse."""
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


def parameter_request_identity(
    description: Mapping[str, object], physical: Mapping[str, Attribute]
) -> tuple[object, ...] | None:
    """Share newly requested deterministic key-switch tables with known placement."""
    import json

    if description.get("kind") not in {
        "ntt_table",
        "key_switch_table",
        "rotation_sum_table",
        "joint_quotient_table",
    } or any(
        physical.get(name) in (None, StringAttr("unknown"))
        for name in ("dtype", "device")
    ):
        return None
    return (
        "default-numerical-table",
        json.dumps(description, sort_keys=True),
        physical["dtype"],
        physical["device"],
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
    """Declare a Tensor material input and reuse matching default requests.

    The request-local cache shares declarations with the same default-request identity. Caller symbols remain independent. A shared declaration reuses a dominating reference, or the same symbol in another block; the Program records this sharing through SSA operands and material symbols.
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
    """Describe the parameters from which a numerical table is constructed."""
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
    if isinstance(key, KeySwitchKey):
        result["digit_prime_ids"] = [list(rows) for rows in key.digit_prime_ids]
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
    """Fill missing Tensor bindings from supplied data providers and keys.

    Key entries named by a symbol or its label are direct assignments and bypass metadata selection. Otherwise, descriptions select a single candidate using CKKS operand rows, placement, and compatible execution groups. Absent, ambiguous, or conflicting matches remain unresolved. Existing bindings are preserved. The returned symbols identify entries still requiring data; key generation is a separate provisioning operation.
    """
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext

    providers = (
        (resources,)
        if isinstance(resources, (CkksDeviceResources, RnsContext, NttContext))
        else tuple(cast(Iterable[object], resources))
    )
    inventory = keys if isinstance(keys, EvaluationKeySet) else None
    named_keys = dict(keys) if isinstance(keys, Mapping) else {}
    key_values = (
        ()
        if inventory is not None
        else tuple(named_keys.values())
        if isinstance(keys, Mapping)
        else tuple(cast(Iterable[object], keys))
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
            _represent_key_layout(compilation, symbol, selected)
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
        candidate_keys: dict[int, KeySwitchKey] = {}
        if kind in {
            "SecretKey",
            "PublicKey",
            "KeySwitchKey",
            "RotationKey",
            "RelinearizationKey",
            "ConjugationKey",
        }:
            selected_values = key_values
            if inventory is not None:
                selected_key = _inventory_key_for_symbol(
                    compilation, symbol, description, inventory
                )
                selected_values = (
                    () if selected_key is None else (selected_key,)
                )
            for key in selected_values:
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
                    if isinstance(key, KeySwitchKey):
                        candidate_keys[id(key.data)] = key
        else:
            for provider in providers:
                tensor = _provider_tensor(provider, selection, cache)
                if tensor is not None:
                    candidates[id(tensor)] = tensor
        if len(candidates) == 1:
            tensor = next(iter(candidates.values()))
            _represent_key_layout(
                compilation, symbol, candidate_keys.get(id(tensor))
            )
            compilation.material_bindings[symbol] = tensor
        else:
            missing.append(symbol)
    return tuple(missing)


def _inventory_key_for_symbol(
    compilation: Compilation,
    symbol: str,
    description: Mapping[str, object],
    inventory: EvaluationKeySet,
) -> KeySwitchKey | None:
    """Select references covering the actual rows of all CKKS key consumers."""
    import json
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr
    from fhelium.ir.dialects import ckks, rns

    kind = description.get("kind")
    if kind not in {"RotationKey", "RelinearizationKey", "ConjugationKey"}:
        return None
    requested_step = description.get("rotation_step")
    if kind == "RotationKey" and type(requested_step) is not int:
        return None

    def integer_tuple(attribute):
        return (
            tuple(int(item.value.data) for item in attribute)
            if isinstance(attribute, ArrayAttr)
            else None
        )

    def rotation_step(key):
        step = getattr(getattr(key.type, "state", None), "data", {}).get(
            "rotation_step"
        )
        if isinstance(step, IntegerAttr):
            return int(step.value.data)
        owner = key.owner
        if isinstance(owner, core.MaterialRefOp) and owner.symbol is not None:
            step = compilation.program.material_descriptions.get(
                owner.symbol.data, {}
            ).get("rotation_step")
            return step if type(step) is int else None
        return None

    source_facts = []
    group_keys = []
    required_steps = []
    for reference in compilation.program.walk():
        if (
            not isinstance(reference, core.MaterialRefOp)
            or reference.symbol is None
            or reference.symbol.data != symbol
        ):
            continue
        for use in reference.value.uses:
            consumer = use.operation
            if (
                isinstance(
                    consumer,
                    (
                        ckks.RotateOp,
                        ckks.RelinearizeOp,
                        ckks.ConjugateOp,
                        ckks.SwitchKeyOp,
                    ),
                )
                and use.index == 1
            ):
                sources, operands = (consumer.value,), (consumer.key,)
                steps = (
                    (rotation_step(consumer.key),)
                    if kind == "RotationKey"
                    else ()
                )
            elif (
                kind == "RotationKey"
                and isinstance(
                    consumer,
                    (ckks.RotateManyOp, ckks.GroupedRotationWeightedSumOp),
                )
                and 1 <= use.index < 1 + len(consumer.keys)
            ):
                operands = tuple(consumer.keys)
                sources = (consumer.value,) * len(operands)
                steps = tuple(rotation_step(key) for key in operands)
            elif (
                kind == "RotationKey"
                and isinstance(consumer, ckks.RotationSumOp)
                and len(consumer.values)
                <= use.index
                < len(consumer.values) + len(consumer.keys)
            ):
                pairs = tuple(
                    (value, int(step.value.data))
                    for value, step in zip(
                        consumer.values, consumer.rotation_steps, strict=True
                    )
                    if step.value.data
                )
                sources = tuple(value for value, _ in pairs)
                steps = tuple(step for _, step in pairs)
                operands = tuple(consumer.keys)
            else:
                # A lowered consumer uses the already declared stored layout.
                state = getattr(
                    getattr(reference.value.type, "state", None), "data", {}
                )
                rows = integer_tuple(state.get("prime_ids"))
                supports = state.get("digit_prime_ids")
                if rows is None or not isinstance(supports, ArrayAttr):
                    return None
                q_ids = tuple(
                    i
                    for digit in supports
                    for i in cast(tuple[int, ...], integer_tuple(digit))
                )
                physical = dict(state)
                if (
                    isinstance(consumer, rns.KeySwitchDigitProductOp)
                    and use.index == 1
                ):
                    source_state = getattr(
                        getattr(consumer.operands[0].type, "state", None),
                        "data",
                        {},
                    )
                    for name in ("ring_dimension", "dtype", "device"):
                        if (
                            physical.get(name) in (None, StringAttr("unknown"))
                            and name in source_state
                        ):
                            physical[name] = source_state[name]
                ring = physical.get("ring_dimension")
                dtype = physical.get("dtype")
                device = physical.get("device")
                source_facts.append(
                    {
                        "q_prime_ids": q_ids,
                        "p_prime_ids": rows[len(q_ids) :],
                        "ring_dimension": int(ring.value.data)
                        if isinstance(ring, IntegerAttr)
                        else None,
                        "dtype": getattr(
                            torch, dtype.data.removeprefix("torch.")
                        )
                        if isinstance(dtype, StringAttr)
                        and dtype.data != "unknown"
                        else None,
                        "device": device.data
                        if isinstance(device, StringAttr)
                        and device.data != "unknown"
                        else None,
                    }
                )
                group_keys.append(reference.value)
                if kind == "RotationKey":
                    required_steps.append(cast(int, requested_step))
                continue
            if any(step is None for step in steps):
                return None
            group_keys.extend(operands)
            for operand, step in zip(operands, steps):
                owner = operand.owner
                fixed = not isinstance(owner, core.MaterialRefOp) or (
                    owner.symbol is not None
                    and owner.symbol.data in compilation.material_bindings
                )
                if fixed:
                    state = getattr(
                        getattr(operand.type, "state", None), "data", {}
                    )
                    if not isinstance(
                        state.get("prime_ids"), ArrayAttr
                    ) or not isinstance(
                        state.get("digit_prime_ids"), ArrayAttr
                    ):
                        return None
                else:
                    required_steps.append(step)
            config_attr = consumer.attributes.get("ckks_config")
            config = (
                CkksConfig.parse(json.loads(config_attr.data))
                if isinstance(config_attr, StringAttr)
                and config_attr.data != "unknown"
                else compilation.workspace.get(CkksConfig)
            )
            config = config if isinstance(config, CkksConfig) else None
            for source in sources:
                if not isinstance(source.type, ckks.CiphertextType):
                    return None
                state = source.type.state.data
                q_ids = integer_tuple(state.get("prime_ids"))
                depth = state.get("depth")
                if (
                    q_ids is None
                    and config is not None
                    and isinstance(depth, IntegerAttr)
                ):
                    q_ids = tuple(
                        range(
                            config.q_row_start(int(depth.value.data)),
                            config.num_q_primes,
                        )
                    )
                ring = state.get("ring_dimension")
                dtype = state.get("dtype")
                device = state.get("device")
                source_facts.append(
                    {
                        "q_prime_ids": q_ids,
                        "p_prime_ids": None,
                        "ring_dimension": int(ring.value.data)
                        if isinstance(ring, IntegerAttr)
                        else config.N
                        if config is not None
                        else None,
                        "dtype": getattr(
                            torch, dtype.data.removeprefix("torch.")
                        )
                        if isinstance(dtype, StringAttr)
                        and dtype.data != "unknown"
                        else None,
                        "device": device.data
                        if isinstance(device, StringAttr)
                        and device.data != "unknown"
                        else None,
                    }
                )
    if not source_facts:
        return None
    facts = {}
    known_facts = {}
    for name in source_facts[0]:
        values = [source[name] for source in source_facts]
        known = [value for value in values if value is not None]
        if name == "q_prime_ids":
            known_facts[name] = (
                tuple(sorted({i for rows in known for i in rows}))
                if known
                else None
            )
            facts[name] = (
                known_facts[name] if len(known) == len(values) else None
            )
        else:
            if known and any(value != known[0] for value in known[1:]):
                return None
            known_facts[name] = known[0] if known else None
            facts[name] = known[0] if len(known) == len(values) else None
    for name, field in (
        ("key_prime_ids", "prime_ids"),
        ("digit_prime_ids", "digit_prime_ids"),
    ):
        constraints = []
        for key in group_keys:
            value = getattr(getattr(key.type, "state", None), "data", {}).get(
                field
            )
            if isinstance(value, ArrayAttr):
                constraints.append(
                    integer_tuple(value)
                    if field == "prime_ids"
                    else tuple(integer_tuple(rows) for rows in value)
                )
        if constraints:
            if any(value != constraints[0] for value in constraints[1:]):
                return None
            facts[name] = constraints[0]
    if any(
        facts[name] is None for name in source_facts[0] if name != "p_prime_ids"
    ):
        inventory = EvaluationKeySet(
            inventory._matching(
                KeySwitchKey,
                **known_facts,
                key_prime_ids=facts.get("key_prime_ids"),
                digit_prime_ids=facts.get("digit_prime_ids"),
            )
        )
    if kind == "RotationKey":
        selected = inventory.select_rotations(required_steps, **facts)
        return (
            None
            if selected is None
            else selected.get(cast(int, requested_step))
        )
    if kind == "RelinearizationKey":
        return inventory.select_relinearization(**facts)
    return inventory.select_conjugation(**facts)


def _key_state_attributes(fields: Mapping[str, object]) -> dict[str, Attribute]:
    """Convert ordinary key-layout fields to the existing open type state."""
    from .frontend._eager_values import state_attribute

    facts = {}
    for name, value in fields.items():
        if name in {"prime_ids", "shape", "strides"}:
            value = tuple(cast(Iterable[int], value))
        elif name == "digit_prime_ids":
            value = tuple(
                tuple(rows) for rows in cast(Iterable[Iterable[int]], value)
            )
        attribute = state_attribute(value)
        if attribute is None:
            raise TypeError(f"Unsupported key layout field {name!r}: {value!r}")
        facts[name] = attribute
    return facts


def _key_layout_description(
    facts: Mapping[str, Attribute],
) -> dict[str, object]:
    """Describe the mathematical layout carried by a key operand."""
    result: dict[str, object] = {}
    for name in (
        "prime_ids",
        "digit_prime_ids",
        "polynomial_domain",
        "residue_representation",
        "rotation_step",
        "basis",
    ):
        value = facts.get(name)
        if isinstance(value, ArrayAttr):
            result[name] = (
                [
                    [
                        int(cast(IntegerAttr, i).value.data)
                        for i in cast(ArrayAttr, rows)
                    ]
                    for rows in value
                ]
                if name == "digit_prime_ids"
                else [int(cast(IntegerAttr, i).value.data) for i in value]
            )
        elif isinstance(value, StringAttr) and value.data != "unknown":
            result["modulus_basis" if name == "basis" else name] = value.data
        elif isinstance(value, IntegerAttr):
            result[name] = int(value.value.data)
    return result


def _assign_key_facts(
    compilation: Compilation,
    symbol: str,
    facts: Mapping[str, Attribute],
    description: Mapping[str, object],
) -> bool:
    """Fill key facts without changing a represented execution contract."""
    from xdsl.rewriter import Rewriter
    from fhelium.ir.dialects import ckks

    previous = compilation.program.material_descriptions.get(symbol, {})
    for name in (
        "prime_ids",
        "digit_prime_ids",
        "polynomial_domain",
        "residue_representation",
        "modulus_basis",
        "rotation_step",
    ):
        known = previous.get(name)
        value = description.get(name)
        if (
            known is not None
            and known != "unknown"
            and value is not None
            and known != value
        ):
            raise ValueError(
                f"Key material {symbol!r} conflicts with declared {name}"
            )
    updates = []
    for operation in compilation.program.walk():
        if not (
            isinstance(operation, core.MaterialRefOp)
            and operation.symbol is not None
            and operation.symbol.data == symbol
            and isinstance(operation.value.type, ckks.EvaluationKeyType)
        ):
            continue
        typ = operation.value.type
        additions = {}
        for name, value in facts.items():
            represented = typ.state.data.get(name)
            if represented is None or represented == StringAttr("unknown"):
                additions[name] = value
            elif represented != value:
                if (
                    name == "dtype"
                    and isinstance(represented, StringAttr)
                    and isinstance(value, StringAttr)
                    and represented.data.removeprefix("torch.")
                    == value.data.removeprefix("torch.")
                ):
                    continue
                # Physical extents may still contain unknown dimensions.
                if (
                    name in {"shape", "strides"}
                    and isinstance(represented, ArrayAttr)
                    and isinstance(value, ArrayAttr)
                    and len(represented) == len(value)
                    and all(
                        known == actual
                        or isinstance(known, IntegerAttr)
                        and known.value.data < 0
                        for known, actual in zip(
                            represented, value, strict=True
                        )
                    )
                ):
                    additions[name] = value
                else:
                    raise ValueError(
                        f"Key material {symbol!r} conflicts with declared {name}"
                    )
        if additions:
            updates.append((operation.value, typ.with_state(additions)))
    for value, typ in updates:
        Rewriter.replace_value_with_new_type(value, typ)
    merged = {**previous, **description}
    # A named binding can supply a generic key object for a declared role.
    if "kind" in previous:
        merged["kind"] = previous["kind"]
    if merged != previous:
        compilation.program.set_material_description(symbol, merged)
    return bool(updates) or merged != previous


def _prepare_key_layouts(
    compilation: Compilation,
    layouts: Mapping[str, Mapping[str, object]],
    *,
    defaults: bool,
) -> int:
    """Declare layouts without numerical data, sharing facts across aliases.

    Explicit entries use material symbols or labels and can retain partial
    state. Defaults select the configured depth-zero layout only when no
    layout has been declared. Bound-key facts are supplied between these steps.
    """
    import json
    from fhelium.backend.rns.layout import RnsLayout
    from fhelium.ir.dialects import ckks

    references: dict[str, list[core.MaterialRefOp]] = {}
    for operation in compilation.program.walk():
        if (
            isinstance(operation, core.MaterialRefOp)
            and operation.symbol is not None
            and isinstance(operation.value.type, ckks.EvaluationKeyType)
        ):
            references.setdefault(operation.symbol.data, []).append(operation)
    changed = 0
    for symbol, aliases in references.items():
        description = compilation.program.material_descriptions.get(symbol, {})
        supplied = layouts.get(
            symbol, layouts.get(str(description.get("label", "")))
        )
        if supplied is None and not defaults:
            continue
        facts = _key_state_attributes(supplied) if supplied is not None else {}
        for alias in aliases:
            for name, value in cast(
                ckks.EvaluationKeyType, alias.value.type
            ).state.data.items():
                if name not in {"shape", "strides"} and value != StringAttr(
                    "unknown"
                ):
                    facts.setdefault(name, value)
        for name in (
            "prime_ids",
            "digit_prime_ids",
            "polynomial_domain",
            "residue_representation",
            "rotation_step",
            "modulus_basis",
        ):
            value = description.get(name)
            if value is not None and value != "unknown":
                state_name = "basis" if name == "modulus_basis" else name
                facts.setdefault(
                    state_name,
                    _key_state_attributes({state_name: value})[state_name],
                )
        if supplied is None and not any(
            isinstance(facts.get(name), ArrayAttr)
            for name in ("prime_ids", "digit_prime_ids")
        ):
            if description.get("kind") not in {
                "KeySwitchKey",
                "RotationKey",
                "RelinearizationKey",
                "ConjugationKey",
            }:
                continue
            config = compilation.workspace.get(CkksConfig)
            for alias in aliases:
                for use in alias.value.uses:
                    represented = use.operation.attributes.get("ckks_config")
                    if (
                        isinstance(represented, StringAttr)
                        and represented.data != "unknown"
                    ):
                        config = CkksConfig.parse(json.loads(represented.data))
                        break
            if not isinstance(config, CkksConfig):
                continue
            layout = RnsLayout.from_config(config)
            for name, value in _key_state_attributes(
                {
                    "prime_ids": layout.prime_ids(0, include_p=True),
                    "digit_prime_ids": layout.hybrid_decomposition.depth_zero_digits,
                    "polynomial_domain": "ntt",
                    "residue_representation": "montgomery",
                    "basis": "QP",
                    "ring_dimension": config.N,
                }
            ).items():
                facts.setdefault(name, value)
        changed += _assign_key_facts(
            compilation, symbol, facts, _key_layout_description(facts)
        )
    return changed


def _represent_key_layout(
    compilation: Compilation, symbol: str, key: object
) -> None:
    """Assign missing key facts and check the already declared layout."""
    if not isinstance(key, KeySwitchKey):
        return
    from .frontend._eager_values import state_attributes

    facts = state_attributes(key)
    facts["ring_dimension"] = IntegerAttr(key.data.size(-1), 64)
    _assign_key_facts(compilation, symbol, facts, key_description(key))


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
        "rotation_sum_table",
        "centered_basis_extension_table",
        "encapsulated_modulus_raise_table",
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
    if kind == "joint_quotient_table":
        from fhelium.backend.rns.tables import moddown_rescale_tables

        q_ids = tuple(cast(Iterable[int], description["q_prime_ids"]))
        p_ids = tuple(cast(Iterable[int], description["p_prime_ids"]))
        drop_ids = tuple(cast(Iterable[int], description["drop_prime_ids"]))
        rp, rd = (
            str(description.get("moddown_rounding", "nearest")),
            str(description.get("rescale_rounding", "nearest")),
        )
        token = (id(context), "joint_quotient", q_ids, p_ids, drop_ids, rp, rd)
        if token not in cache:
            cache[token] = moddown_rescale_tables(
                context,
                q_ids,
                p_ids,
                drop_ids,
                moddown_rounding=rp,
                rescale_rounding=rd,
            )
        return cast(dict[str, torch.Tensor], cache[token]).get(
            str(description["name"])
        )
    if "depth" not in description or "name" not in description:
        return None
    depth = int(cast(int, description["depth"]))
    name = cast(str, description["name"])
    if kind == "centered_basis_extension_table":
        from fhelium.backend.rns.tables import centered_basis_extension_tables

        target_depth = description.get("target_depth")
        if not isinstance(target_depth, int):
            return None
        token = (id(context), "centered_basis_extension", depth, target_depth)
        if token not in cache:
            cache[token] = centered_basis_extension_tables(
                context,
                context,
                context.rns_layout.prime_ids(depth),
                context.rns_layout.prime_ids(target_depth),
            )
        return cast(dict[str, torch.Tensor], cache[token]).get(name)
    if kind == "encapsulated_modulus_raise_table":
        from fhelium.backend.ckks.tables import (
            prepare_encapsulated_raise_operands,
        )

        target_depth, guard = (
            description.get("target_depth"),
            description.get("return_prime"),
        )
        low_p, return_p = (
            description.get("low_p_moduli"),
            description.get("return_p_moduli"),
        )
        policy = description.get("ntt_backend")
        if (
            not isinstance(target_depth, int)
            or not isinstance(guard, int)
            or not isinstance(low_p, (list, tuple))
            or not isinstance(return_p, (list, tuple))
            or not isinstance(policy, str)
        ):
            return None
        low_p, return_p = tuple(low_p), tuple(return_p)
        rounding = str(description.get("rounding", "nearest"))
        token = (
            id(context),
            "encapsulated_modulus_raise",
            depth,
            target_depth,
            low_p,
            guard,
            return_p,
            policy,
            rounding,
        )
        if token not in cache:
            if isinstance(provider, CkksDeviceResources):
                selected_ntt = provider.ntt_context
            elif isinstance(provider, NttContext):
                selected_ntt = provider
            else:
                ntt_token = (id(context), "ntt", policy)
                if ntt_token not in cache:
                    cache[ntt_token] = NttContext(context, ntt_backend=policy)
                selected_ntt = cast(NttContext, cache[ntt_token])
            if selected_ntt.ntt_backend_name != policy:
                return None
            cache[token] = prepare_encapsulated_raise_operands(
                context,
                selected_ntt,
                source_depth=depth,
                target_depth=target_depth,
                low_p_moduli=low_p,
                return_prime=guard,
                return_p_moduli=return_p,
                rounding=rounding,
            )[0]
        return cast(dict[str, torch.Tensor], cache[token]).get(name)
    if kind == "rotation_sum_table":
        if name.startswith("sum_galois_"):
            from fhelium.backend.ntt.automorphism import ntt_galois_indices
            from fhelium.backend.ckks.rotation._galois import (
                rotation_galois_element,
            )

            step = int(name.removeprefix("sum_galois_"))
            return ntt_galois_indices(
                config.N,
                rotation_galois_element(
                    config.N, step, config.galois_generator
                ),
                context.device,
            )
        from fhelium.backend.ckks.tables import (
            prepare_rotation_sum_tables,
        )

        key_rows = (
            tuple(cast(Iterable[int], description["key_prime_ids"]))
            if "key_prime_ids" in description
            else None
        )
        key_digits = (
            tuple(
                tuple(rows)
                for rows in cast(
                    Iterable[Iterable[int]], description["digit_prime_ids"]
                )
            )
            if "digit_prime_ids" in description
            else None
        )
        token = (
            id(context),
            "rotation_sum",
            depth,
            key_rows,
            key_digits,
            bool(description.get("rescale", False)),
            description.get("rounding", "nearest"),
            description.get("modup_method", "exact"),
        )
        if token not in cache:
            base_token = (id(context), "key_switch_rns")
            if base_token not in cache:
                cache[base_token] = KeySwitchTables(
                    context,
                    None,
                    moddown_inverse_tables(context),
                    config.galois_generator,
                )
            base = cast(KeySwitchTables, cache[base_token])
            cache[token] = prepare_rotation_sum_tables(
                base,
                depth,
                rescale=bool(description.get("rescale", False)),
                rounding=str(description.get("rounding", "nearest")),
                modup_method=str(description.get("modup_method", "exact")),
                key_prime_ids=key_rows,
                digit_prime_ids=key_digits,
            ).tensor_operands()
        return cast(dict[str, torch.Tensor], cache[token]).get(name)
    if kind == "key_switch_table":
        token = (id(context), "key_switch_rns")
        if token not in cache:
            cache[token] = KeySwitchTables(
                context,
                None,
                (),
                config.galois_generator,
            )
        if (
            name in {"moddown_inverses", "p_inverse", "moddown_half"}
            and "q_prime_ids" in description
            and "p_prime_ids" in description
        ):
            return (
                cast(KeySwitchTables, cache[token])
                .moddown_tensor_operands(
                    tuple(cast(Iterable[int], description["q_prime_ids"])),
                    tuple(cast(Iterable[int], description["p_prime_ids"])),
                    rounding="nearest" if name == "moddown_half" else "floor",
                    depth=depth,
                )
                .get(name)
            )
        tensors, facts = cast(KeySwitchTables, cache[token]).tensor_operands(
            depth,
            moddown_rounding=str(
                description.get(
                    "moddown_rounding",
                    "nearest" if name == "moddown_half" else "floor",
                )
            ),
            modup_method=str(description.get("modup_method", "exact")),
            rescale=bool(description.get("rescale", False)),
            rescale_rounding=str(
                description.get("rescale_rounding", "nearest")
            ),
            output_domain=str(description.get("output_domain", "coefficient")),
            key_prime_ids=tuple(
                cast(Iterable[int], description["key_prime_ids"])
            )
            if "key_prime_ids" in description
            else None,
            digit_prime_ids=tuple(
                tuple(rows)
                for rows in cast(
                    Iterable[Iterable[int]], description["digit_prime_ids"]
                )
            )
            if "digit_prime_ids" in description
            else None,
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
