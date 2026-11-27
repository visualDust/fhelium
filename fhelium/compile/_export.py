"""Export compiled callables as artifacts and load them for execution."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from ._binding_time import PREPARATION
from ._materials import (
    _provider_tensor,
    provided_material_selections,
)
from ._preparation import Specialization, _CompiledVariant
from ._specialization import (
    ArgumentSignature,
    CallSignature,
    SpecializationMiss,
)

if TYPE_CHECKING:
    from fhelium.backend import OperationBackend
    from fhelium.eager import Engine

    from ._callable import CompiledCallable

_KEY_KINDS = frozenset(
    {
        "SecretKey",
        "PublicKey",
        "KeySwitchKey",
        "RotationKey",
        "RelinearizationKey",
        "ConjugationKey",
    }
)
_SCALARS: dict[str, type] = {
    "NoneType": type(None),
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
}


class ExportError(ValueError):
    """The compiled callable cannot be stored as an artifact."""


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _frozen(value: object) -> object:
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _signature_record(signature: CallSignature) -> list[dict[str, object]]:
    return [
        {
            "name": argument.name,
            "kind": argument.kind,
            "properties": _plain(argument.properties),
        }
        for argument in signature.arguments
    ]


def _signature(
    arguments: Sequence[Mapping[str, Any]], device: torch.device | None
) -> CallSignature:
    def property_value(name: str, value: object) -> object:
        if (
            device is not None
            and name.endswith("device")
            and isinstance(value, str)
            and value.startswith("cuda")
        ):
            return str(device)
        return _frozen(value)

    return CallSignature(
        tuple(
            ArgumentSignature(
                item["name"],
                item["kind"],
                tuple(
                    (name, property_value(name, value))
                    for name, value in item["properties"]
                ),
            )
            for item in arguments
        )
    )


def _results(output: object) -> list[dict[str, object]]:
    """Value-file type and metadata of each output leaf, in output order."""
    from torch.utils._pytree import tree_flatten

    from fhelium.serialization.value import ValueEnvelope

    results: list[dict[str, object]] = []
    for leaf in tree_flatten(output)[0]:
        if isinstance(leaf, torch.Tensor):
            results.append({"value_type": "Tensor", "value_metadata": {}})
            continue
        if leaf is None:
            results.append({"value_type": None, "value_metadata": {}})
            continue
        envelope = ValueEnvelope.from_value(cast(Any, leaf))
        results.append(
            {
                "value_type": envelope.value_type,
                "value_metadata": envelope.metadata,
            }
        )
    return results


def _argument_type(kind: str) -> type:
    import fhelium.values as values

    if kind == "Tensor":
        return torch.Tensor
    return _SCALARS.get(kind) or cast(type, getattr(values, kind))


def export(
    function: CompiledCallable,
    path: str | os.PathLike[str],
    *,
    signatures: Iterable[Sequence[object] | Mapping[str, object]],
    weights: str = "raw",
    targets: Sequence[str] = (),
    overwrite: bool = False,
) -> None:
    """Write the specializations of ``function`` for the example calls ``signatures`` to the artifact ``path``.

    Each example is a sequence of positional arguments or a mapping of keyword arguments; ``function`` is prepared for it and its linked executable is stored as entry ``call<i>``, with a second entry ``call<i>.graphs`` planned with CUDA Graph segments. The artifact records materials by binding:

    - keys are ``caller`` materials with their key description; the loader binds keys that match it;
    - NTT, RNS and key-switching tables are ``built`` outputs of the ``tables`` entry, the registered builder operator calls that produce them, which the loader runs once on its device; a table produced by Python code is ``embedded``;
    - the rounding seed of plaintext encoding is a ``caller`` material bound from the loading Engine;
    - plaintexts prepared inside ``function`` (binding-time preparation) are, with ``weights="raw"``, ``prepare`` outputs of entry ``call<i>.prepare``, which encodes their stored raw inputs at load; ``weights="embedded"`` stores the encoded plaintexts. When preparation reads call arguments, the prepare entry is marked ``per_call`` and the loaded callable runs it on calls whose argument Tensors changed;
    - other material Tensors, such as plaintexts encoded before capture, are ``embedded``.

    ``targets`` adds kernel images compiled for further CUDA architectures, for example ``("sm_90",)``. Raises :class:`ExportError` when a schedule calls Python or when an executable is lowered per runtime input layout.
    """
    from fhelium.backend._artifact import (
        ArtifactIneligible,
        embedded_record,
        export_executable,
        export_tables,
        material_record,
    )
    from fhelium.backend.rns.context import RnsContext
    from fhelium.backend.runner import State
    from fhelium.config import CkksConfig
    from fhelium.native.artifact import (
        ArtifactWriter,
        ExportError as NativeExportError,
        save_artifact,
    )

    if weights not in ("raw", "embedded"):
        raise ValueError("weights must be 'raw' or 'embedded'")
    writer = ArtifactWriter(targets=tuple(targets))
    names: dict[int, tuple[str, torch.Tensor]] = {}

    def name(symbol: str, tensor: torch.Tensor) -> str:
        if id(tensor) not in names:
            names[id(tensor)] = (f"m{len(names)}", tensor)
        return names[id(tensor)][0]

    records: dict[str, dict[str, dict[str, object]]] = {
        "keys": {},
        "tables": {},
        "weights": {},
        "rounding": {},
    }
    tensors: dict[str, torch.Tensor] = {}
    tables: dict[str, tuple[Mapping[str, object], torch.Tensor]] = {}
    prepare: dict[str, dict[str, object]] = {}
    signature_records = []
    config: CkksConfig | None = None
    fixed_rank = False
    for index, example in enumerate(signatures):
        args, kwargs = (
            ((), dict(example))
            if isinstance(example, Mapping)
            else (tuple(example), {})
        )
        arguments = function._arguments(args, kwargs)
        prepared = function.prepare(*args, **kwargs)
        compilation = prepared.compilation
        from .frontend._eager_capture import (
            CAPTURED_CONFIGS,
            CAPTURED_RESOURCES,
        )

        plan = compilation.workspace.get(PREPARATION)
        for resources in (
            *function._resources,
            *compilation.workspace.get(CAPTURED_RESOURCES, ()),
            *(plan.engines if plan is not None else ()),
        ):
            config = cast(CkksConfig, getattr(resources, "config", config))
        for captured in compilation.workspace.get(CAPTURED_CONFIGS, ()):
            if config is not None and captured != config:
                raise ExportError(
                    "the callable uses Engines of different CKKS configurations"
                )
            config = captured
        if config is None:
            config = compilation.workspace.get(CkksConfig)
        entry = f"call{index}"
        try:
            record = export_executable(
                writer, prepared.executable, entry, symbols=name
            )
            fixed_rank = fixed_rank or any(
                lowered.schedule.fixed_rank
                for lowered in prepared.executable.lowered
            )
        except ArtifactIneligible as error:
            raise ExportError(str(error)) from None
        bound = record.pop("materials")
        selections = provided_material_selections(compilation)
        descriptions = compilation.program.material_descriptions
        prepared_symbols = set(plan.symbols) if plan is not None else set()
        per_call = plan is not None and any(
            source["kind"] == "argument" for source in plan.inputs
        )
        if plan is not None and (weights == "raw" or per_call):

            def state(bound_value: object) -> str | None:
                if (
                    isinstance(bound_value, State)
                    and bound_value.field == "seed"
                ):
                    records["rounding"]["rounding.seed"] = {
                        **material_record(
                            "rounding.seed",
                            getattr(bound_value.resource, "seed"),
                        ),
                        "kind": "rounding_state",
                    }
                    return "rounding.seed"
                if isinstance(bound_value, State):
                    raise ExportError(
                        f"plaintext preparation reads {bound_value.field} of a live resource"
                    )
                return None

            try:
                writer.entry(
                    f"{entry}.prepare", plan.schedule(arguments), binding=state
                )
            except NativeExportError as error:
                raise ExportError(str(error)) from None
            inputs: list[object] = []
            for source, value in zip(
                plan.inputs, plan.values(arguments), strict=True
            ):
                if source["kind"] == "argument":
                    inputs.append(
                        {
                            "argument": source["name"],
                            "field": source.get("field"),
                        }
                    )
                    continue
                raw = record["symbols"].get(source["symbol"]) or name(
                    source["symbol"], value
                )
                if raw not in records["weights"]:
                    records["weights"][raw] = embedded_record(
                        raw, value, tensors
                    )
                inputs.append(raw)
            outputs = []
            for symbol in (*plan.exports, *plan.views):
                output = record["symbols"].get(symbol) or name(
                    symbol, plan.bindings[symbol]
                )
                outputs.append(output)
                if output in inputs:
                    # The recipe returns this raw input unchanged.
                    continue
                records["weights"][output] = {
                    **material_record(output, plan.bindings[symbol], "prepare"),
                    "prepare_entry": f"{entry}.prepare",
                }
            prepare[f"{entry}.prepare"] = {
                "entry": f"{entry}.prepare",
                "inputs": inputs,
                "outputs": outputs,
                "per_call": per_call,
            }
        for symbol, artifact_symbol in record["symbols"].items():
            if any(artifact_symbol in group for group in records.values()):
                continue
            tensor = bound[artifact_symbol]
            description = descriptions.get(symbol, {})
            kind = description.get("kind")
            selection = selections.get(symbol)
            if kind in _KEY_KINDS:
                records["keys"][artifact_symbol] = {
                    **material_record(artifact_symbol, tensor),
                    "role": kind,
                    "layout": _plain(description),
                }
            elif kind == "rounding_state":
                records["rounding"][artifact_symbol] = {
                    **material_record(artifact_symbol, tensor),
                    "kind": "rounding_state",
                }
            elif selection is not None and str(kind)[:1].islower():
                tables[artifact_symbol] = (selection, tensor)
            elif symbol not in prepared_symbols or weights == "embedded":
                records["weights"][artifact_symbol] = embedded_record(
                    artifact_symbol, tensor, tensors
                )
        record["states"] = {
            symbol: list(item) for symbol, item in record["states"].items()
        }
        variant = prepared._variant
        signature_records.append(
            {
                "name": entry,
                "entry": entry,
                "arguments": _signature_record(prepared.signature),
                "input_names": list(variant.input_names),
                "input_fields": {
                    key: list(item)
                    for key, item in variant.input_fields.items()
                },
                "output_structure": variant.output_structure,
                "executable": record,
                # Materials the entry reads; prefetch planning uses them later.
                "working_set": sorted(set(record["symbols"].values())),
                "prepare": f"{entry}.prepare"
                if f"{entry}.prepare" in prepare
                else None,
                "results": _results(function(*args, **kwargs)),
            }
        )
    if not signature_records:
        raise ValueError("export requires at least one example call")
    if tables and config is None:
        raise ExportError(
            "tables are read without a CKKS configuration to rebuild them"
        )

    def builders():
        contexts: dict[torch.device, tuple[RnsContext, dict]] = {}

        def build(symbol: str, selection: Mapping[str, object]):
            device = tables[symbol][1].device
            if device not in contexts:
                contexts[device] = (
                    RnsContext(cast(CkksConfig, config), device=device),
                    {},
                )
            provider, cache = contexts[device]
            return _provider_tensor(provider, selection, cache)

        return build

    built, embedded = export_tables(writer, tables, builders)
    records["tables"].update(built)
    tensors.update(embedded)
    manifest, stored = writer.build()
    stored.update(tensors)
    manifest["materials"] = {
        kind: list(group.values()) for kind, group in records.items()
    }
    manifest["prepare"] = prepare
    manifest["signatures"] = signature_records
    manifest["config"] = config.dumps() if config is not None else None
    manifest["rank"] = _schedule_rank(function, fixed_rank)
    save_artifact(path, manifest, stored, overwrite=overwrite)


def _schedule_rank(function: CompiledCallable, fixed_rank: bool) -> int | None:
    """Return the process rank the schedules were captured or lowered for when they depend on it, else None.

    Collectives read the group rank and size when they run. A schedule whose structured control flow was expanded for the rank (``fixed_rank``), a source that reads ``torch.distributed.get_rank`` while it is captured, or a source whose reads cannot be identified records the rank of the exporting process.
    """
    import torch.distributed as torch_distributed

    from fhelium.compile._cache import CacheIneligible, source_identity
    from fhelium.compile._compilation import Compilation
    from fhelium.ir import Program

    if not torch_distributed.is_initialized():
        return None
    if fixed_rank:
        return torch_distributed.get_rank()
    if isinstance(function._source, (Program, Compilation)):
        return None
    try:
        identity = source_identity(
            function._source, material_names=function._material_names
        )
    except CacheIneligible:
        return torch_distributed.get_rank()
    return torch_distributed.get_rank() if identity.reads_rank else None


def _bind_groups(
    loaded: Any, rank: int | None, groups: Mapping[str, object] | None
) -> None:
    import torch.distributed as torch_distributed

    symbols = {
        group["symbol"]
        for entry in loaded.manifest["entries"].values()
        for group in entry.get("groups", ())
    }
    if rank is not None or symbols:
        if not torch_distributed.is_initialized():
            raise RuntimeError(
                "the artifact runs collectives or was captured for one process rank; initialize torch.distributed before loading it"
            )
    if rank is not None and torch_distributed.get_rank() != rank:
        raise RuntimeError(
            f"the artifact was captured for process rank {rank}; this process has rank {torch_distributed.get_rank()}"
        )
    for symbol in sorted(symbols):
        group = (groups or {}).get(symbol)
        if group is None and symbol == "world":
            group = torch_distributed.group.WORLD
        if group is None:
            raise ValueError(
                f"supply the process group for artifact group {symbol!r}"
            )
        loaded.bind_group(symbol, cast(Any, group).group_name)


class _PerCallPreparation:
    """Run a ``per_call`` prepare entry when the argument Tensors it reads change."""

    def __init__(
        self, loaded: Any, name: str, inputs: list, names: tuple[str, ...]
    ) -> None:
        self.loaded = loaded
        self.name = name
        self.sources = tuple(
            (names.index(item["argument"]), item["field"])
            if isinstance(item, Mapping)
            else item
            for item in inputs
        )
        self.versions: tuple | None = None

    def __call__(self, values: tuple[object, ...]) -> None:
        tensors = []
        for source in self.sources:
            if isinstance(source, str):
                tensors.append(self.loaded.artifact.material(source))
                continue
            position, field = source
            value = values[position]
            tensors.append(getattr(value, field) if field else value)
        versions = tuple((id(t), t._version) for t in tensors)
        if versions != self.versions:
            self.loaded.artifact.refresh(self.name, tensors)
            self.versions = versions


class LoadedFunction:
    """Call the specializations stored in an artifact.

    A call selects the specialization whose recorded argument conditions match the arguments and runs its entry through the artifact's Runners. A specialization whose plaintext preparation reads call arguments first runs its prepare entry when those argument Tensors differ from the previous call's, by identity or version counter.
    """

    def __init__(
        self,
        loaded: Any,
        specializations: tuple[Specialization, ...],
        names: tuple[str, ...],
        preparations: tuple[_PerCallPreparation | None, ...] = (),
    ) -> None:
        self.artifact = loaded
        self.specializations = specializations
        self._names = names
        self._preparations = preparations or (None,) * len(specializations)

    def __call__(self, *args: object, **kwargs: object) -> object:
        values = list(args)
        for name in self._names[len(args) :]:
            if name not in kwargs:
                raise TypeError(f"missing argument {name!r}")
            values.append(kwargs[name])
        call = tuple(values)
        for specialization, preparation in zip(
            self.specializations, self._preparations, strict=True
        ):
            if specialization._matches(call):
                if preparation is not None:
                    preparation(call)
                return specialization._execute(call)
        raise SpecializationMiss(
            "No specialization stored in the artifact matches these arguments"
        )


def load(
    path: str | os.PathLike[str],
    *,
    keys: Mapping[str, object] | Iterable[object] = (),
    engine: Engine | None = None,
    backend: OperationBackend | None = None,
    device: torch.device | str | None = None,
    graph: bool | str = "auto",
    groups: Mapping[str, object] | None = None,
) -> LoadedFunction:
    """Load an artifact written by :func:`export` and return its callable.

    ``keys`` supplies the keys the artifact declares, as a mapping from material symbol or as key objects matched to the recorded key descriptions. ``engine`` provides the rounding stream of plaintext encoding; by default an Engine of the recorded CKKS configuration is created; an artifact that reads no tables records no configuration and requires ``engine``. ``device`` places CUDA materials and execution on another device. ``graph`` selects Graph execution as in :func:`fhelium.compile.compile`. ``groups`` maps the process-group symbols of the artifact's collectives to ``torch.distributed`` process groups; the symbol ``world`` defaults to the default group. An artifact captured from a source that reads the process rank runs only on the rank that exported it. The artifact's ``tables`` entry and its prepare entries run once while loading.
    """
    from fhelium.backend._artifact import load_device, restore_executable
    from fhelium.backend.execution import OperationBackend
    from fhelium.backend.runner import LoadedArtifact
    from fhelium.config import CkksConfig
    from fhelium.eager import Engine
    from fhelium.serialization.value import ValueEnvelope
    from fhelium.values import KeySwitchKey

    import json

    from fhelium.native.artifact import read_container_header

    metadata, _ = read_container_header(path)
    manifest = json.loads(metadata["fhelium.manifest"])
    signatures = manifest["signatures"]
    placed = (
        torch.device(device)
        if device is not None
        else load_device(signatures[0]["executable"])
    )
    loaded = LoadedArtifact(os.fspath(path), placed)
    _bind_groups(loaded, manifest.get("rank"), groups)
    if engine is None:
        if manifest["config"] is None:
            raise ValueError(
                "the artifact records no CKKS configuration; supply engine"
            )
        engine = Engine(CkksConfig.parse(manifest["config"]))
    materials = loaded.manifest["materials"]
    if isinstance(keys, Mapping):
        objects = []
        for symbol, value in keys.items():
            if symbol in loaded.materials:
                loaded.artifact.bind(
                    symbol, cast(torch.Tensor, getattr(value, "data", value))
                )
            else:
                objects.append(value)
    else:
        objects = list(keys)
    shapes = {tuple(record["shape"]) for record in materials.get("keys", ())}
    for key in objects:
        if (
            isinstance(key, KeySwitchKey)
            and tuple(key.data.shape) not in shapes
        ):
            # A compact key, such as one returned by load_value, is expanded
            # on its device unless the artifact was captured with compact keys.
            key = key.expand()
        envelope = ValueEnvelope.from_value(cast(Any, key))
        # A live key holds its expanded rows; binding them shares the key's storage.
        loaded.artifact.bind_key(
            envelope.value_type,
            json.dumps(envelope.metadata),
            {"data": cast(Any, key).data},
        )
    unbound = set(loaded.artifact.unbound())
    missing = [
        f"{record['symbol']} ({record['role']})"
        for record in materials.get("keys", ())
        if record["symbol"] in unbound
    ]
    if missing:
        raise ValueError(
            f"No supplied key matches artifact keys {', '.join(missing)}"
        )
    for record in materials.get("rounding", ()):
        rounding_device = (
            torch.device("cpu") if record["device"] == "cpu" else placed
        )
        loaded.artifact.bind(
            record["symbol"],
            engine._dispatcher_for(rounding_device).device_resources.rng.seed,
        )
    loaded.artifact.prepare()
    backend = backend or OperationBackend()
    if torch.distributed.is_initialized():
        from fhelium.backend.distributed import ProcessGroupExecutionResource
        from fhelium.backend.resources import BoundResource, ResourceBindings

        # Collectives of the default process group name it ``world``.
        backend = backend.with_named_resources(
            ResourceBindings(
                (
                    BoundResource(
                        "world",
                        "process-group",
                        ProcessGroupExecutionResource(),
                    ),
                )
            ),
            override=False,
        )
    specializations = []
    for item in signatures:
        signature = _signature(
            item["arguments"], placed if device is not None else None
        )
        executable = restore_executable(
            loaded, item["executable"], backend, {}, graphs=graph
        )
        variant = _CompiledVariant(
            signature,
            cast(Any, None),
            cast(Any, None),
            tuple(item["input_names"]),
            item["output_structure"],
            tuple(_argument_type(a.kind) for a in signature.arguments),
            {
                key: cast(tuple[str, str], tuple(v))
                for key, v in item["input_fields"].items()
            },
        )
        specializations.append(Specialization(variant, backend, executable))
    names = tuple(a["name"] for a in signatures[0]["arguments"])
    prepare = loaded.manifest.get("prepare", {})
    preparations = tuple(
        _PerCallPreparation(
            loaded, item["prepare"], prepare[item["prepare"]]["inputs"], names
        )
        if item.get("prepare") and prepare[item["prepare"]].get("per_call")
        else None
        for item in signatures
    )
    return LoadedFunction(loaded, tuple(specializations), names, preparations)


__all__ = ["ExportError", "LoadedFunction", "export", "load"]
