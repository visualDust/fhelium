"""Represent and execute plaintext preparation independently of ciphertext work."""

from __future__ import annotations

import dataclasses
import json
import operator
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from xdsl.dialects.builtin import StringAttr

from fhelium.backend.execution import OperationBackend
from fhelium.backend.implementation import OperationImplementationRegistry
from fhelium.backend.torch import TorchCallImplementation, prepare_tensor_call
from fhelium.ir import Program
from fhelium.ir.dialects.torch import CallOp
from fhelium.values import CompressedPlaintext, Plaintext

from ._compilation import Compilation

PREPARATION = "plaintext_preparation"


def pack_value(value):
    """Describe a plaintext envelope using its state and ordinary Tensor fields."""
    if isinstance(value, (Plaintext, CompressedPlaintext)):
        return {
            "__plaintext__": type(value).__name__,
            "fields": {
                item.name: pack_value(getattr(value, item.name))
                for item in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return tuple(pack_value(item) for item in value)
    if isinstance(value, list):
        return [pack_value(item) for item in value]
    if isinstance(value, dict):
        return {key: pack_value(item) for key, item in value.items()}
    return value


def unpack_value(value):
    if isinstance(value, dict):
        if "__plaintext__" in value:
            cls = {
                "Plaintext": Plaintext,
                "CompressedPlaintext": CompressedPlaintext,
            }[value["__plaintext__"]]
            return cls(**unpack_value(value["fields"]))
        return {key: unpack_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(unpack_value(item) for item in value)
    return value


def tensors(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (Plaintext, CompressedPlaintext)):
        return tensors(pack_value(value))
    if isinstance(value, Mapping):
        return tensors(tuple(value.values()))
    if isinstance(value, (tuple, list)):
        return tuple(t for item in value for t in tensors(item))
    return ()


def payload_bytes(values):
    storages = {
        value.untyped_storage()._cdata: value.untyped_storage().nbytes()
        for value in values
    }
    return sum(storages.values())


def tensor_layout(value):
    return {
        "shape": list(value.shape),
        "strides": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def tensor_version(value):
    try:
        version = value._version
    except RuntimeError:
        # Inference tensors have no version counter. Their preparation is refreshed.
        version = None
    return id(value), version


def engine_call(index, method, parameters):
    """Name a high-level preparation node in a make_fx trace."""
    raise RuntimeError("An engine preparation node requires its bound Engine")


def tensor_factory(name, arguments, keywords):
    return getattr(torch, name)(*arguments, **keywords)


@dataclass(frozen=True)
class _EngineMethod:
    """Call an Engine encoding method recorded by preparation capture."""

    engines: tuple

    def __call__(self, index, method, parameters):
        return tensors(
            getattr(self.engines[index], method)(**unpack_value(parameters))
        )


@dataclass(frozen=True)
class PreparationCallImplementation(TorchCallImplementation):
    """Execute ordered ATen calls and high-level plaintext codec nodes."""

    name: str = "plaintext-preparation"
    operation_types: tuple = (CallOp,)
    engines: tuple = ()

    def supports_operation(self, operation):
        target = operation.attributes.get("fhelium.call.target")
        return isinstance(target, StringAttr) and target.data.startswith(
            ("aten.", "prepare.")
        )

    def prepare_operation(self, operation):
        target = operation.attributes["fhelium.call.target"].data
        if target.startswith("aten."):
            _, name, overload = target.split(".", 2)
            function = getattr(getattr(torch.ops.aten, name), overload)
        elif target == "prepare.engine":
            function = _EngineMethod(self.engines)
        else:
            function = {
                "prepare.getitem": operator.getitem,
                "prepare.factory": tensor_factory,
            }[target]
        descriptor = json.loads(
            operation.attributes["fhelium.call.arguments"].data
        )
        return dataclasses.replace(
            self, _call=prepare_tensor_call(function, descriptor)
        )


@dataclass(frozen=True)
class PreparationRecipe:
    """Recorded preparation calls from which the preparation Program is built on request.

    Values are numbered with the ``inputs`` Program arguments first and then one result per call. Each call holds its target, its JSON argument descriptor and the value numbers of its operands.
    """

    inputs: int
    calls: tuple[tuple[str, str, tuple[int, ...]], ...]
    outputs: tuple[int, ...]

    def digest(self) -> str:
        import hashlib

        return hashlib.blake2b(
            json.dumps([self.inputs, self.calls, self.outputs]).encode(),
            digest_size=16,
        ).hexdigest()

    def program(self) -> Program:
        from xdsl.dialects.func import ReturnOp
        from xdsl.ir import Block

        from fhelium.ir.dialects import core

        block = Block(
            arg_types=[core.MessageType() for _ in range(self.inputs)]
        )
        values = list(block.args)
        for target, descriptor, operands in self.calls:
            operation = CallOp(
                [values[index] for index in operands],
                core.MessageType(),
                kind="function",
                target=target,
                argument_descriptor=descriptor,
                role="message",
            )
            block.add_op(operation)
            values.append(operation.result)
        outputs = [values[index] for index in self.outputs]
        block.add_op(ReturnOp(*outputs))
        program = Program.from_function(
            block, [value.type for value in outputs]
        )
        program.verify_structure()
        return program


@dataclass
class PreparationPlan:
    """Own a preparation Program, its leaf bindings, and its current generation.

    Inputs identify callable arguments or external material leaves. Outputs bind ordinary MaterialRef symbols in the ciphertext Program. Tensor identity and version counters govern resident reuse; inference tensors are refreshed on each invocation.
    """

    source: Program | PreparationRecipe
    inputs: tuple[dict, ...]
    exports: tuple[str, ...]
    engines: tuple
    leaves: dict[str, torch.Tensor] = field(default_factory=dict)
    getters: dict[str, Any] = field(default_factory=dict)
    mode: str = "resident"
    merged: bool = True
    memo_writes: tuple = ()
    views: dict = field(default_factory=dict)
    generation: int = 0
    versions: tuple = ()
    bindings: dict[str, torch.Tensor] = field(default_factory=dict)
    before_merge_bytes: int = 0
    prepare_seconds: float = 0.0
    rebind_seconds: float = 0.0
    _executable: Any = None
    _initial: bool = True
    layouts: dict = field(default_factory=dict)
    last_material_bytes: int = 0
    resident_material_bytes: int = 0
    link_seconds: float = 0.0
    link_source: str | None = None

    @cached_property
    def program(self) -> Program:
        """The preparation Program, built from a recorded recipe on first access."""
        source = self.source
        return source if isinstance(source, Program) else source.program()

    def values(self, arguments):
        values = []
        for source in self.inputs:
            if source["kind"] == "argument":
                value = arguments[source["name"]]
                if source.get("field"):
                    value = getattr(value, source["field"])
            else:
                symbol = source["symbol"]
                getter = self.getters.get(symbol)
                value = getter() if getter is not None else self.leaves[symbol]
            if tensor_layout(value) != source["layout"]:
                raise ValueError(
                    "Preparation input layout changed; clear or prepare a matching specialization"
                )
            values.append(value)
        return tuple(values)

    @cached_property
    def symbols(self):
        return (*self.exports, *self.views)

    def materialize_views(self, bindings):
        for symbol, view in self.views.items():
            owner = bindings[view["owner"]]
            bindings[symbol] = owner.as_strided(
                view["shape"],
                view["strides"],
                owner.storage_offset() + view["offset"],
            )
        return bindings

    def initialize(self, arguments, bindings):
        bindings = self.materialize_views(dict(bindings))
        self.bindings = bindings
        self.versions = tuple(tensor_version(t) for t in self.values(arguments))
        self.generation = 1
        self.layouts = {
            symbol: (
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
            )
            for symbol, value in bindings.items()
        }
        self.last_material_bytes = payload_bytes(bindings.values())
        self.resident_material_bytes = self.last_material_bytes

    def link(self, cache=None):
        """Link the preparation Program, restoring its executable from ``cache`` when recorded."""
        if self._executable is not None:
            return
        from fhelium.backend._artifact import (
            ArtifactIneligible,
            export_executable,
            restore_executable,
        )

        started = time.perf_counter()
        backend = OperationBackend(
            OperationImplementationRegistry(
                (PreparationCallImplementation(engines=self.engines),)
            )
        )
        key = record = None
        if cache is not None:
            from ._cache import preparation_artifact_key

            key = preparation_artifact_key(self.source)
            record = cache.load_artifact(key)
        if record is not None:
            self._executable = restore_executable(
                record["payload"], backend, {}, externals=self.engines
            )
            self.link_source = "artifact"
        else:
            self._executable = backend.link(Compilation(self.program))
            self.link_source = "linked"
            if key is not None:
                try:
                    cache.store_artifact(
                        key,
                        {
                            "payload": export_executable(
                                self._executable,
                                backend.registry,
                                externals=self.engines,
                            )
                        },
                    )
                except (ArtifactIneligible, OSError):
                    self.link_source = "linked-unrecorded"
        self.link_seconds += time.perf_counter() - started

    def refresh(self, arguments, *, force=False):
        values = self.values(arguments)
        versions = tuple(tensor_version(value) for value in values)
        unchanged = versions == self.versions and all(
            item[1] is not None for item in versions
        )
        if (
            not force
            and unchanged
            and (self.mode == "resident" or self._initial)
        ):
            self._initial = False
            return None
        started = time.perf_counter()
        self.link()
        outputs = self._executable.run(*values)
        if len(self.exports) == 1:
            outputs = (outputs,)
        bindings = self.materialize_views(
            dict(zip(self.exports, outputs, strict=True))
        )
        # A recipe may expose different views of one allocation. Layout is checked
        # against the linked material interface before publishing the generation.
        for symbol, value in bindings.items():
            layout = (
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
            )
            prior = self.layouts.setdefault(symbol, layout)
            if layout != prior:
                raise ValueError(
                    f"Preparation changed the layout of material {symbol!r}"
                )
        for mapping, key, descriptor in self.memo_writes:
            mapping[key] = unpack_value(_restore_export(descriptor, bindings))
        self.prepare_seconds += time.perf_counter() - started
        for source, value in zip(self.inputs, values, strict=True):
            if source["kind"] == "material":
                self.leaves[source["symbol"]] = value
        self.versions = versions
        self.generation += 1
        self._initial = False
        self.bindings = bindings
        self.last_material_bytes = payload_bytes(bindings.values())
        self.resident_material_bytes = self.last_material_bytes
        return bindings

    @cached_property
    def operation_count(self):
        source = self.source
        if isinstance(source, PreparationRecipe):
            return len(source.calls)
        return sum(1 for op in source.walk() if isinstance(op, CallOp))

    def release(self):
        self.bindings.clear()
        self.resident_material_bytes = 0

    def fork(self):
        return dataclasses.replace(
            self,
            leaves=dict(self.leaves),
            getters=dict(self.getters),
            bindings=dict(self.bindings),
            layouts=dict(self.layouts),
        )

    def stats(self):
        return {
            "mode": self.mode,
            "merge_encodings": self.merged,
            "operations": self.operation_count,
            "material_owners": len(self.exports),
            "material_views": len(self.views),
            "prepare_execute_seconds": self.prepare_seconds,
            "material_rebind_seconds": self.rebind_seconds,
            "derived_material_bytes": self.last_material_bytes,
            "resident_material_bytes": self.resident_material_bytes,
            "preparation_link_seconds": self.link_seconds,
            "preparation_link": self.link_source,
            "derived_material_bytes_before_merge": self.before_merge_bytes,
            "binding_generation": self.generation,
        }

    def descriptor(self, identity):
        from ._cache import build_manifest

        leaf_manifest = build_manifest(identity, self.leaves, {})
        if leaf_manifest is None:
            raise ValueError(
                "A preparation source leaf has no persistent locator"
            )
        engine_paths = []
        for engine in self.engines:
            if id(engine) not in identity.engines:
                raise ValueError(
                    "Preparation Engine has no persistent source locator"
                )
            engine_paths.append(identity.engines[id(engine)][0])
        if self.memo_writes:
            raise ValueError(
                "Preparation memoization writes retain live object bindings"
            )
        return {
            "program": self.program.to_text(),
            "inputs": self.inputs,
            "exports": self.exports,
            "views": self.views,
            "leaves": leaf_manifest,
            "engines": engine_paths,
            "mode": self.mode,
            "merge_encodings": self.merged,
            "before_merge_bytes": self.before_merge_bytes,
        }

    @classmethod
    def restore(
        cls, descriptor, *, source, arguments, bindings, material_names=None
    ):
        from ._cache import _resolve_path

        roots = {"fn": source, "names": material_names or {}}

        def getter(origins):
            def read():
                values = []
                for origin in origins:
                    owner = _resolve_path(roots, origin["path"])
                    values.append(
                        getattr(owner, origin["field"])
                        if origin["field"]
                        else owner
                    )
                if any(value is not values[0] for value in values[1:]):
                    raise ValueError(
                        "Preparation leaf aliases changed; clear the callable before recapture"
                    )
                return values[0]

            return read

        plan = cls(
            Program.parse(descriptor["program"]),
            tuple(descriptor["inputs"]),
            tuple(descriptor["exports"]),
            tuple(_resolve_path(roots, path) for path in descriptor["engines"]),
            leaves={
                symbol: bindings[symbol] for symbol in descriptor["leaves"]
            },
            getters={
                symbol: getter(entry["origins"])
                for symbol, entry in descriptor["leaves"].items()
            },
            views=descriptor.get("views", {}),
            mode=descriptor["mode"],
            merged=descriptor["merge_encodings"],
            before_merge_bytes=descriptor["before_merge_bytes"],
        )
        plan.refresh(arguments, force=True)
        plan._initial = True
        return plan


def _restore_export(value, bindings):
    if isinstance(value, dict):
        if "__prepared_symbol__" in value:
            return bindings[value["__prepared_symbol__"]]
        return {
            key: _restore_export(item, bindings) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(_restore_export(item, bindings) for item in value)
    return value


def preparation_guards(arguments):
    """Keep metadata-dependent recipes within the input state that was traced."""
    from ._specialization import describe_call
    from .frontend._template_provenance import json_value

    return json_value(
        tuple(
            (item.name, item.kind, item.properties)
            for item in describe_call(arguments).arguments
        )
    )
