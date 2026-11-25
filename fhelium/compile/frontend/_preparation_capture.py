"""Record real plaintext evaluation as a PyTorch FX graph of ATen calls."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import operator
import time
import weakref
from contextlib import contextmanager

import torch
from torch.fx import Graph, Node
from torch.utils._python_dispatch import TorchDispatchMode

from fhelium.values import Ciphertext, CompressedPlaintext, Plaintext

from .._binding_time import (
    PreparationRecipe,
    PREPARATION,
    PreparationPlan,
    engine_call,
    pack_value,
    tensor_factory,
    tensor_layout,
    tensors,
)
from .._errors import CaptureError
from ._pytorch_encoding import encode_literal


PREPARATION_METHODS = frozenset(
    {
        "encode",
        "prepare_compressed_plaintext",
        "prepare_plaintext_for_addition",
        "prepare_plaintext_for_multiplication",
        "integer_coefficients_to_rns",
        "coefficient_domain_to_ntt_domain",
        "ntt_domain_to_coefficient_domain",
        "standard_residues_to_montgomery_residues",
        "montgomery_residues_to_standard_residues",
        "zero_plaintext_like",
        "mod_switch_to_depth",
    }
)


def contains_proxy(value):
    from ._eager_capture import _Value

    if isinstance(value, (_Value, Ciphertext)):
        return True
    if isinstance(value, dict):
        return any(contains_proxy(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_proxy(item) for item in value)
    return False


def _digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


class _Tracer:
    """Map live Tensors to the FX nodes that compute them.

    Each ATen call on the recorded path becomes a ``call_function`` node whose Tensor arguments are the nodes of their current producers. Multi-Tensor results are read through ``operator.getitem`` nodes, and an in-place result rebinds its Tensor to the writing node. A Tensor without a producer is an unlocated constant, which the preparation Program cannot persist.
    """

    def __init__(self):
        self.graph = Graph()
        self.nodes = {}

    def node(self, tensor):
        entry = self.nodes.get(id(tensor))
        return entry[1] if entry is not None and entry[0]() is tensor else None

    def track(self, value, node):
        if isinstance(value, torch.Tensor):
            self.nodes[id(value)] = (weakref.ref(value), node)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                if isinstance(item, (torch.Tensor, tuple, list)):
                    self.track(
                        item,
                        self.graph.call_function(
                            operator.getitem, (node, index)
                        ),
                    )

    def argument(self, value):
        if isinstance(value, torch.Tensor):
            node = self.node(value)
            if node is None:
                # The preparation Program rejects this node if it is retained.
                return self.graph.create_node("get_attr", "_tensor_constant")
            return node
        if isinstance(value, (tuple, list)):
            return type(value)(self.argument(item) for item in value)
        if isinstance(value, dict):
            return {key: self.argument(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self.argument(value.start),
                self.argument(value.stop),
                self.argument(value.step),
            )
        return value

    def clear(self):
        self.nodes.clear()


class _Effects(TorchDispatchMode):
    """Track storage effects for merge proofs and reject writes to source leaves.

    The recorder's tracer receives the ATen calls. This observer also maintains storage ownership and expression identities, including ordered writes through local views.
    """

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.values = {}
        self.storage = {}
        self.owners = {}
        self.external = set()
        self.serial = 0

    def information(self, tensor):
        entry = self.values.get(id(tensor))
        return entry[1:] if entry is not None and entry[0]() is tensor else None

    def record(self, tensor, digest, virtual, *, external=False, owner=None):
        storage = tensor.untyped_storage()._cdata
        self.storage[storage] = digest
        self.owners[id(tensor)] = tensor if owner is None else owner
        self.values[id(tensor)] = (weakref.ref(tensor), storage, virtual)
        if external:
            self.external.add(storage)

    def signature(self, value):
        if isinstance(value, torch.Tensor):
            info = self.information(value)
            if info is None:
                # An unknown leaf cannot prove equal encodings.
                return ("unknown", id(value))
            storage, _ = info
            return (
                self.storage[storage],
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                str(value.dtype),
                str(value.device),
            )
        if isinstance(value, (Plaintext, CompressedPlaintext)):
            return self.signature(pack_value(value))
        if isinstance(value, dict):
            return tuple(
                (key, self.signature(item)) for key, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            return (
                type(value).__name__,
                tuple(self.signature(item) for item in value),
            )
        return value

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        recorder = self.recorder
        if recorder.internal:
            return function(*args, **kwargs)
        self.serial += 1
        input_tensors = tensors((args, kwargs))
        input_storage = {
            t.untyped_storage()._cdata: (
                self.information(t),
                self.owners.get(id(t), t),
            )
            for t in input_tensors
        }
        writes = []
        for index, argument in enumerate(function._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                value = (
                    args[index]
                    if index < len(args)
                    else kwargs.get(argument.name)
                )
                for tensor in tensors(value):
                    storage = tensor.untyped_storage()._cdata
                    if (
                        storage in self.external
                        or self.information(tensor) is None
                    ):
                        raise CaptureError(
                            "Plaintext preparation cannot mutate an external Tensor"
                        )
                    writes.append(storage)
        signature = (str(function), self.signature((args, kwargs)))
        if torch.Tag.nondeterministic_seeded in function.tags:
            signature += ("random-event", self.serial)
        digest = _digest(signature)
        view = recorder.traced_view(function, args, kwargs)
        if view is None and recorder.pending:
            for tensor in input_tensors:
                recorder.materialize(tensor)
            if writes:
                recorder.materialize_views(writes)
        started = time.perf_counter()
        if view is None:
            if torch.Tag.data_dependent_output in function.tags and any(
                recorder.tracer.node(tensor) is not None
                for tensor in input_tensors
            ):
                raise CaptureError(
                    "Plaintext preparation cannot read traced Tensor values into Python"
                )
            # A fresh Tensor literal is copied so that later writes cannot alias it.
            if function is torch.ops.aten.lift_fresh.default:
                function = torch.ops.aten.lift_fresh_copy.default
            tracer = recorder.tracer
            node = tracer.graph.call_function(
                function, tracer.argument(args), tracer.argument(kwargs)
            )
            result = function(*args, **kwargs)
            tracer.track(result, node)
        else:
            # Most views of traced Tensors are consumed only through owner/offset
            # descriptors; their graph node is created on first traced use.
            result = function(*args, **kwargs)
            reference = weakref.ref(result)
            recorder.pending[id(result)] = (reference, function, view)
            recorder.pending_storage.setdefault(
                result.untyped_storage()._cdata, []
            ).append(reference)
        recorder.execute_seconds += time.perf_counter() - started
        for storage in writes:
            self.storage[storage] = _digest(
                ("write", self.storage[storage], digest)
            )
        for index, tensor in enumerate(tensors(result)):
            storage = tensor.untyped_storage()._cdata
            source = input_storage.get(storage)
            if (
                source is not None
                and source[0] is not None
                and any(
                    result.alias_info is not None
                    for result in function._schema.returns
                )
            ):
                virtual = source[0][1]
                owner = source[1]
                current = self.storage[storage]
            else:
                current = _digest((digest, index))
                virtual = ("aten", self.serial, index)
                owner = tensor
            self.record(tensor, current, virtual, owner=owner)
        return result


class _TorchModule:
    def __init__(self, recorder):
        self.recorder = recorder

    def __getattr__(self, name):
        if name in {"tensor", "as_tensor"}:
            return lambda *args, **kwargs: self.recorder.factory(
                name, args, kwargs
            )
        return getattr(torch, name)


class PreparationRecorder:
    def __init__(self, capture, arguments):
        self.capture = capture
        self.arguments = arguments
        self.inputs = []
        self.input_values = []
        self.placeholders = {}
        self.leaves = {}
        self.getters = {}
        self.outputs = []
        self.exports = []
        self.exported = {}
        self.symbols = set()
        self.views = {}
        self.before_storages = {}
        self.engines = []
        self.codecs = {}
        self.memo_writes = []
        self.internal = 0
        self.execute_seconds = 0.0
        self.tracer = None
        self.pending = {}
        self.pending_storage = {}
        self.event = 0
        self.effects = _Effects(self)
        self.torch = _TorchModule(self)
        self.merged = capture.workspace.get("merge_plaintext_encodings", True)
        self.residency = capture.workspace.get(
            "plaintext_preparation_mode", "resident"
        )
        if self.residency not in {"resident", "per_call"}:
            raise ValueError(
                "plaintext_preparation_mode must be resident or per_call"
            )
        for name, value in arguments.items():
            if isinstance(value, torch.Tensor):
                self._argument(name, None, value)
            elif isinstance(value, (Plaintext, CompressedPlaintext)):
                for item in dataclasses.fields(value):
                    tensor = getattr(value, item.name)
                    if isinstance(tensor, torch.Tensor):
                        self._argument(name, item.name, tensor)

    def _argument(self, name, field, tensor):
        if any(tensor is prior for prior in self.input_values):
            raise CaptureError(
                "Aliased preparation arguments require full callable capture"
            )
        self.inputs.append(
            {
                "kind": "argument",
                "name": name,
                "field": field,
                "layout": tensor_layout(tensor),
            }
        )
        self.input_values.append(tensor)
        self.placeholders[id(tensor)] = tensor
        self.effects.record(
            tensor,
            _digest(("argument", name, field)),
            ("input", name, field),
            external=True,
        )

    @contextmanager
    def pause(self):
        self.internal += 1
        try:
            yield
        finally:
            self.internal -= 1

    def proxy(self, tensor):
        if self.pending:
            self.materialize(tensor)
        return self.tracer.node(tensor)

    def traced(self, tensor):
        """Return whether a Tensor has a graph value, including a deferred view."""
        entry = self.pending.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return True
        return self.tracer.node(tensor) is not None

    def placeholder(self, tensor):
        """Return whether a Tensor is a graph input of the preparation trace."""
        return self.placeholders.get(id(tensor)) is tensor

    def traced_view(self, function, args, kwargs):
        """Return graph operands of a single-Tensor view of traced Tensors, else None."""
        schema = function._schema
        if (
            self.tracer is None
            or schema.is_mutable
            or len(schema.returns) != 1
            or schema.returns[0].alias_info is None
            or not isinstance(schema.returns[0].type, torch.TensorType)
        ):
            return None
        try:
            return self.mapped(args), self.mapped(kwargs)
        except CaptureError:
            return None

    def materialize(self, tensor):
        """Add the graph node of a deferred view before its first traced use."""
        entry = self.pending.pop(id(tensor), None)
        if entry is None or entry[0]() is not tensor:
            return
        _, function, (args, kwargs) = entry
        self.tracer.track(
            tensor, self.tracer.graph.call_function(function, args, kwargs)
        )

    def materialize_views(self, storages):
        """Place deferred views of written storage before the write."""
        for storage in storages:
            for reference in self.pending_storage.pop(storage, ()):
                tensor = reference()
                if tensor is not None:
                    self.materialize(tensor)

    def leaf(self, value, getter=None):
        for field, tensor in self._fields(value):
            if self.traced(tensor):
                if getter is not None and any(
                    tensor is item
                    for item in self.input_values[
                        : sum(
                            source["kind"] == "argument"
                            for source in self.inputs
                        )
                    ]
                ):
                    raise CaptureError(
                        "A preparation leaf aliases a runtime argument; use full callable capture"
                    )
                continue
            symbol = self.capture.tensor_symbols.get(id(tensor))
            if symbol is None:
                symbol = self.capture.names.get(
                    id(tensor), f"tensor{len(self.capture.materials)}"
                )
                reserved = (
                    set(self.capture.names.values())
                    if id(tensor) not in self.capture.names
                    else set()
                )
                while symbol in self.capture.materials or symbol in reserved:
                    symbol += "_"
                self.capture.tensor_symbols[id(tensor)] = symbol
                self.capture.materials[symbol] = tensor
                self.capture.descriptions[symbol] = {"kind": "Tensor"}
            index = len(self.inputs)
            self.tracer.track(
                tensor,
                self.tracer.graph.placeholder(f"preparation_leaf_{index}"),
            )
            self.inputs.append(
                {
                    "kind": "material",
                    "symbol": symbol,
                    "layout": tensor_layout(tensor),
                }
            )
            self.input_values.append(tensor)
            self.placeholders[id(tensor)] = tensor
            self.leaves[symbol] = tensor
            if getter is not None:
                self.getters[symbol] = (
                    getter
                    if field is None
                    else lambda get=getter, name=field: getattr(get(), name)
                )
            self.effects.record(
                tensor,
                _digest(("material", symbol)),
                ("leaf", symbol),
                external=True,
            )
        return value

    @staticmethod
    def _fields(value):
        if isinstance(value, torch.Tensor):
            return ((None, value),)
        if isinstance(value, (Plaintext, CompressedPlaintext)):
            return tuple(
                (item.name, getattr(value, item.name))
                for item in dataclasses.fields(value)
                if isinstance(getattr(value, item.name), torch.Tensor)
            )
        return ()

    def mapped(self, value):
        if isinstance(value, torch.Tensor):
            proxy = self.proxy(value)
            if proxy is None:
                raise CaptureError(
                    "Plaintext preparation encountered a Tensor without a source recipe"
                )
            return proxy
        if isinstance(value, (tuple, list)):
            return type(value)(self.mapped(item) for item in value)
        if isinstance(value, dict):
            return {key: self.mapped(item) for key, item in value.items()}
        return value

    def factory(self, name, args, kwargs):
        # Literal factories retain the literal construction, not a captured payload.
        if contains_proxy((args, kwargs)):
            return getattr(torch, name)(*args, **kwargs)
        self.event += 1
        started = time.perf_counter()
        with self.pause():
            result = tensor_factory(name, args, kwargs)
        self.execute_seconds += time.perf_counter() - started
        self.tracer.track(
            result,
            self.tracer.graph.call_function(
                tensor_factory, (name, self.mapped(args), self.mapped(kwargs))
            ),
        )
        self.effects.record(
            result,
            _digest(("factory", name, self.effects.signature((args, kwargs)))),
            ("factory", self.event),
        )
        return result

    def engine(self, engine, method, parameters):
        if engine not in self.engines:
            self.engines.append(engine)
        index = self.engines.index(engine)
        packed = pack_value(parameters)
        signature = _digest((index, method, self.effects.signature(packed)))
        prior = self.codecs.get(signature) if self.merged else None
        self.event += 1
        event = self.event
        if prior is None:
            started = time.perf_counter()
            with self.pause():
                result = getattr(engine, method)(**parameters)
            self.execute_seconds += time.perf_counter() - started
            proxy = self.tracer.graph.call_function(
                engine_call, (index, method, self.mapped(packed))
            )
            self.codecs[signature] = (result, proxy)
        else:
            result, proxy = prior
            # New envelope/view identities preserve the pre-merge storage accounting.
            with self.pause():
                if isinstance(result, torch.Tensor):
                    result = result.view_as(result)
                else:
                    result = dataclasses.replace(
                        result,
                        **{
                            field: tensor.view_as(tensor)
                            for field, tensor in self._fields(result)
                        },
                    )
        for output_index, (field, tensor) in enumerate(self._fields(result)):
            self.tracer.track(
                tensor,
                self.tracer.graph.call_function(
                    operator.getitem, (proxy, output_index)
                ),
            )
            self.effects.record(
                tensor,
                _digest(("codec", signature, field)),
                ("codec", event, field),
                owner=(
                    self.effects.owners.get(
                        id(dict(self._fields(prior[0]))[field])
                    )
                    if prior is not None
                    else None
                ),
            )
        return result

    def export(self, tensor):
        self.leaf(tensor)
        prior = self.exported.get(id(tensor))
        if prior is not None:
            return prior
        info = self.effects.information(tensor)
        owner = (
            self.effects.owners.get(id(tensor)) if info is not None else None
        )
        if (
            owner is not None
            and owner is not tensor
            and self.proxy(owner) is not None
        ):
            owner_symbol = self.export(owner)
        else:
            owner_symbol = None
        symbol = f"prepared{len(self.symbols)}"
        while symbol in self.capture.materials:
            symbol += "_"
        self.symbols.add(symbol)
        if owner_symbol is None:
            self.materialize(tensor)
            self.exports.append(symbol)
            self.outputs.append(tensor)
        else:
            self.views[symbol] = {
                "owner": owner_symbol,
                "offset": tensor.storage_offset() - owner.storage_offset(),
                "shape": list(tensor.shape),
                "strides": list(tensor.stride()),
            }
        self.exported[id(tensor)] = symbol
        self.capture.tensor_symbols[id(tensor)] = symbol
        self.capture.materials[symbol] = tensor
        self.capture.descriptions[symbol] = {"kind": "PreparedTensor"}
        info = self.effects.information(tensor)
        virtual = info[1] if info is not None else ("unknown", id(tensor))
        self.before_storages[virtual] = tensor.untyped_storage().nbytes()
        return symbol

    def memoize(self, mapping, key, value):
        if contains_proxy(value) or not tensors(value):
            raise CaptureError(
                "Captured computation cannot store proxy values or arbitrary objects in bound mappings"
            )

        def descriptor(item):
            if isinstance(item, torch.Tensor):
                return {"__prepared_symbol__": self.export(item)}
            if isinstance(item, dict):
                return {name: descriptor(child) for name, child in item.items()}
            if isinstance(item, (tuple, list)):
                return type(item)(descriptor(child) for child in item)
            return item

        from ._eager_capture import _EngineCalls
        from ._objects import unwrap

        def real_key(item):
            if isinstance(item, _EngineCalls):
                return item.engine
            if isinstance(item, tuple):
                return tuple(real_key(child) for child in item)
            return unwrap(item)

        key = real_key(key)
        encoded = descriptor(pack_value(value))
        mapping[key] = value
        self.memo_writes.append((mapping, key, encoded))

    def run(self, function):
        try:
            return self._run(function)
        finally:
            self.close()

    def close(self):
        """Release tracing payloads after transferring the executable preparation recipe."""
        if self.tracer is not None:
            self.tracer.clear()
        self.tracer = None
        self.pending.clear()
        self.pending_storage.clear()
        self.effects.owners.clear()
        self.effects.values.clear()
        self.effects.storage.clear()
        self.effects.external.clear()
        self.effects.recorder = None
        self.torch.recorder = None
        self.arguments = {}
        self.input_values.clear()
        self.placeholders.clear()
        self.outputs.clear()
        self.codecs.clear()
        self.leaves = {}
        self.getters = {}
        self.engines = []
        self.memo_writes = []
        self.capture = None

    def _run(self, function):
        tracer = self.tracer = _Tracer()
        for index, tensor in enumerate(self.input_values):
            tracer.track(tensor, tracer.graph.placeholder(f"input_{index}"))
        with self.effects:
            function()
        tracer.graph.output(tuple(tracer.argument(tuple(self.outputs))))
        graph = tracer.graph
        if not self.exports:
            return None
        # View recipes are serialized as owner/offset descriptors. Keep codec and
        # ordered write events even when their individual results are unused.
        for node in reversed(tuple(graph.nodes)):
            if node.op != "call_function" or node.users:
                continue
            target = node.target
            if target is operator.getitem or (
                isinstance(target, torch._ops.OpOverload)
                and not target._schema.is_mutable
                and any(
                    result.alias_info is not None
                    for result in target._schema.returns
                )
            ):
                graph.erase_node(node)
        plan = PreparationPlan(
            self._recipe(graph),
            tuple(self.inputs),
            tuple(self.exports),
            tuple(self.engines),
            leaves=self.leaves,
            getters=self.getters,
            mode=self.residency,
            merged=self.merged,
            memo_writes=tuple(self.memo_writes),
            views=self.views,
            before_merge_bytes=sum(self.before_storages.values()),
            prepare_seconds=self.execute_seconds,
        )
        plan.initialize(
            self.arguments,
            {symbol: self.capture.materials[symbol] for symbol in self.exports},
        )
        self.capture.workspace[PREPARATION] = plan
        return plan

    def _recipe(self, graph):
        values = {}
        calls = []
        targets = {
            engine_call: "prepare.engine",
            operator.getitem: "prepare.getitem",
            tensor_factory: "prepare.factory",
        }
        placeholders = 0
        for node in graph.nodes:
            if node.op == "placeholder":
                # Placeholders can follow calls; Program arguments precede call results.
                values[node] = placeholders
                placeholders += 1
                continue
            if node.op == "get_attr":
                raise CaptureError(
                    "Plaintext preparation cannot persist an unlocated Tensor constant"
                )
            if node.op == "output":
                outputs = tuple(values[item] for item in node.args[0])
                break
            if node.op != "call_function":
                raise CaptureError(f"Unsupported preparation FX node {node.op}")
            target = (
                "ops." + str(node.target)
                if isinstance(node.target, torch._ops.OpOverload)
                else targets.get(node.target)
            )
            if target is None:
                raise CaptureError(
                    f"Unsupported preparation target {node.target}"
                )
            operands = []

            def encode(item):
                if isinstance(item, Node):
                    operands.append(values[item])
                    return {"kind": "ssa", "operand": len(operands) - 1}
                if isinstance(item, (tuple, list)):
                    return {
                        "kind": "tuple" if isinstance(item, tuple) else "list",
                        "items": [encode(child) for child in item],
                    }
                if isinstance(item, dict):
                    return {
                        "kind": "mapping",
                        "entries": [
                            [encode_literal(key), encode(child)]
                            for key, child in item.items()
                        ],
                    }
                if isinstance(item, slice):
                    return {
                        "kind": "slice",
                        "start": encode(item.start),
                        "stop": encode(item.stop),
                        "step": encode(item.step),
                    }
                return {"kind": "literal", "value": encode_literal(item)}

            descriptor = json.dumps(
                {"args": encode(node.args), "kwargs": encode(node.kwargs)}
            )
            values[node] = len(self.inputs) + len(calls)
            calls.append((target, descriptor, tuple(operands)))
        return PreparationRecipe(len(self.inputs), tuple(calls), outputs)
