"""Persist prepared Programs and relocate current numerical materials.

Entries are artifact files that contain encoded Programs, JSON metadata and, once linked, the executable's entries and kernels; material Tensors remain in the calling process. Source entries are keyed before capture and require pure Python expressions, static control flow, and supported Tensor and Engine operations. Trace entries are keyed by the captured Program and accept any captured source.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import hashlib
import inspect
import json
import marshal
import math
import os
import platform
import struct
import sys
import sysconfig
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from fhelium._version import __version__
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    ConjugationKey,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)

from ._specialization import CallSignature, _scalar_key

if TYPE_CHECKING:
    from fhelium.eager import Engine
    from fhelium.ir import Program

_SCALARS = (type(None), bool, int, float, complex, str, bytes)
_LEAVES = (
    torch.Tensor,
    Ciphertext,
    Plaintext,
    CompressedPlaintext,
    SecretKey,
    PublicKey,
    KeySwitchKey,
    RotationKey,
    RelinearizationKey,
    ConjugationKey,
)
_SCHEMA = 10
# Paths are JSON arrays of [operation, argument] steps, never Python expressions.
_Path = list[list[object]]
_PURE_BUILTINS = {
    value: name
    for name in (
        "range",
        "len",
        "enumerate",
        "zip",
        "abs",
        "min",
        "max",
        "sum",
        "tuple",
        "list",
        "reversed",
    )
    if (value := vars(builtins)[name]) is not None
}
_MATH_FUNCTIONS = {
    value: name
    for name, value in vars(math).items()
    if isinstance(value, types.BuiltinFunctionType)
}
_METADATA = {
    "shape",
    "ndim",
    "dtype",
    "device",
    "requires_grad",
    "depth",
    "scale",
    "prime_ids",
    "digit_prime_ids",
    "polynomial_domain",
    "modulus_basis",
    "residue_representation",
    "ring_dimension",
    "compression_layout",
    "rotation_step",
    "batch_shape",
    "component_count",
    "unique_count",
    "config",
    "num_slots",
    "max_depth",
    "scales",
    "N",
    "logN",
    "q_bases",
    "q_moduli",
    "p_moduli",
    "galois_generator",
}
_TENSOR_METHODS = {
    "size",
    "view",
    "reshape",
    "contiguous",
    "clone",
    "to",
    "select",
    "narrow",
    "neg",
    "remainder",
    "select_batch",
    "slice_batch",
    "slice_limbs",
}


# Class methods of value types called on the class, such as ``Ciphertext.stack_batch``.
_VALUE_CONSTRUCTORS = {"stack_batch"}


def _value_classes() -> tuple[type, ...]:
    from fhelium.values import CompressedPlaintext, Ciphertext, Plaintext

    return (Ciphertext, Plaintext, CompressedPlaintext)


class CacheIneligible(Exception):
    """Report a source whose reuse conditions are outside the supported subset."""


def cache_directory() -> Path | None:
    configured = os.environ.get("FHELIUM_CACHE_DIR")
    if configured is not None:
        return None if configured in ("", "0", "off") else Path(configured)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(base) / "fhelium" / "compile"


def _digest(payload: object) -> str:
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode(),
        digest_size=16,
    ).hexdigest()


@lru_cache(maxsize=4096)
def _file_digest(path: str, size: int, mtime: int, ctime: int) -> str:
    return hashlib.blake2b(Path(path).read_bytes(), digest_size=16).hexdigest()


def _source_file(path: str) -> str:
    try:
        source = Path(path)
        stat = source.stat()
        return _file_digest(
            str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        )
    except OSError as error:
        raise CacheIneligible(f"Source file unavailable: {path}") from error


def environment_identity() -> dict[str, object]:
    """Fingerprint current Python sources, reusing unchanged file digests."""
    import xdsl

    package = Path(__file__).resolve().parents[1]
    native_manifest = (
        package
        / "native"
        / "torchops"
        / (f"_build_manifest.{sysconfig.get_config_var('SOABI')}.json")
    )
    source_files = [
        [str(path.relative_to(package)), _source_file(str(path))]
        for path in sorted(package.rglob("*.py"))
    ]
    return {
        "fhelium": str(__version__),
        "sources": _digest(source_files),
        "python": platform.python_version(),
        "python_tag": sys.implementation.cache_tag,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "native_manifest": (
            _source_file(str(native_manifest))
            if native_manifest.is_file()
            else None
        ),
        "xdsl": str(getattr(xdsl, "__version__", "unknown")),
        "platform": sys.platform,
        "schema": _SCHEMA,
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "grad": torch.is_grad_enabled(),
        "inference": torch.is_inference_mode_enabled(),
        "autocast_cpu": torch.is_autocast_enabled("cpu"),
        "autocast_cuda": torch.is_autocast_enabled("cuda"),
    }


def _step(path: _Path, operation: str, argument: object) -> _Path:
    return [*path, [operation, argument]]


def _primitive(value: object) -> str | None:
    from fhelium.backend.torch import TORCH_FUNCTIONS

    # Identity comparisons avoid invoking arbitrary user __hash__/__eq__ methods.
    for function, name in _PURE_BUILTINS.items():
        if value is function:
            return "builtins." + name
    for function, name in _MATH_FUNCTIONS.items():
        if value is function:
            return "math." + name
    for name, function in TORCH_FUNCTIONS.items():
        if value is function:
            return name
    return None


@lru_cache(maxsize=256)
def _python_source(path: str, size: int, mtime: int, ctime: int, flags: int):
    text = Path(path).read_bytes()
    tree = ast.parse(text, filename=path)
    code = compile(text, path, "exec", flags=flags, dont_inherit=True)
    codes = []

    def collect(current):
        codes.append(current)
        for constant in current.co_consts:
            if isinstance(constant, types.CodeType):
                collect(constant)

    collect(code)
    return tree, codes


def _function_definition(function):
    code = function.__code__
    flags = 0
    for name in __future__.all_feature_names:
        flags |= getattr(__future__, name).compiler_flag
    try:
        stat = Path(code.co_filename).stat()
        tree, candidates = _python_source(
            code.co_filename,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            code.co_flags & flags,
        )
    except (OSError, SyntaxError, ValueError) as error:
        raise CacheIneligible(
            "Python function source is unavailable"
        ) from error
    if not any(candidate == code for candidate in candidates):
        raise CacheIneligible(
            "Current source does not describe the live Python code"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == code.co_name:
            first = min(
                [node.lineno, *(item.lineno for item in node.decorator_list)]
            )
            if first == code.co_firstlineno:
                return node
    raise CacheIneligible(
        "Only ordinary Python function definitions are cached"
    )


# Marks a function scanned without a bound instance.
_NO_RECEIVER = object()


class _Eligibility(ast.NodeVisitor):
    """Check a small expression/loop subset without evaluating user code."""

    _nodes = (
        ast.Return,
        ast.Assign,
        ast.Expr,
        ast.If,
        ast.For,
        ast.Pass,
        ast.Name,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.List,
        ast.Tuple,
        ast.Starred,
        ast.Dict,
        ast.Subscript,
        ast.Slice,
        ast.Call,
        ast.Attribute,
        ast.keyword,
        ast.operator,
        ast.unaryop,
        ast.boolop,
        ast.cmpop,
        ast.expr_context,
    )

    def __init__(self, function, path, scan, receiver=_NO_RECEIVER):
        self.function, self.path, self.scan = function, path, scan
        self.receiver = None
        self.environment = {
            name: (value, _step(path, "global", name))
            for name, value in function.__globals__.items()
            if name not in function.__code__.co_varnames
        }
        self.environment.update(
            {
                name: (cell.cell_contents, _step(path, "closure", name))
                for name, cell in zip(
                    function.__code__.co_freevars, function.__closure__ or ()
                )
            }
        )
        if receiver is not _NO_RECEIVER:
            # The bound instance is read only through its attributes.
            self.receiver = function.__code__.co_varnames[0]
            self.environment[self.receiver] = (
                receiver,
                _step(path, "self", None),
            )
        self.locals = set(function.__code__.co_varnames)
        self.module_members: list[object] = []
        self.local_lists: set[str] = set()

    def check(self):
        definition = _function_definition(self.function)
        # Only lists constructed locally, never bound/aliased lists, may be appended to.
        writes: dict[str, list[ast.AST]] = {}
        for node in ast.walk(definition):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                writes.setdefault(node.id, []).append(node)
        if self.receiver in writes:
            raise CacheIneligible(
                "Assignments to the bound instance are not cached"
            )
        for node in ast.walk(definition):
            if isinstance(node, ast.Assign) and isinstance(
                node.value, ast.List
            ):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and len(writes[target.id]) == 1
                    ):
                        self.local_lists.add(target.id)
        for statement in definition.body:
            self.visit(statement)
        return self.module_members

    def generic_visit(self, node):
        if not isinstance(node, self._nodes):
            raise CacheIneligible(
                f"Unsupported Python cache syntax: {type(node).__name__}"
            )
        super().generic_visit(node)

    def _external(self, node):
        if isinstance(node, ast.Name):
            found = self.environment.get(node.id)
            if found is not None:
                return found
            if (
                node.id not in self.locals
                and node.id in self.function.__builtins__
            ):
                return self.function.__builtins__[node.id], _step(
                    self.path, "builtin", node.id
                )
        elif isinstance(node, ast.Attribute):
            parent = self._external(node.value)
            if parent is not None and type(parent[0]) is types.ModuleType:
                module, path = parent
                if module.__name__.split(".")[0] in {
                    "os",
                    "sys",
                    "time",
                    "random",
                    "numpy",
                    "secrets",
                }:
                    raise CacheIneligible(
                        f"External-state module: {module.__name__}"
                    )
                if node.attr.startswith("_") or node.attr not in vars(module):
                    raise CacheIneligible(
                        "Dynamic module attributes are not cached"
                    )
                return vars(module)[node.attr], _step(
                    path, "module_attr", node.attr
                )
            if parent is not None and _instance(parent[0]):
                owner, path = parent
                member, step = _instance_attribute(owner, node.attr)
                return member, _step(path, step, node.attr)
        return None

    def visit_Name(self, node):
        found = self._external(node)
        if isinstance(node.ctx, ast.Load) and found is not None:
            if isinstance(found[0], types.ModuleType):
                raise CacheIneligible(
                    "Modules must be used through static supported attributes"
                )
            if _instance(found[0]):
                raise CacheIneligible(
                    "Objects are read only through their attributes"
                )

    def visit_Compare(self, node):
        operands = (node.left, *node.comparators)
        for left, operation, right in zip(operands, node.ops, operands[1:]):
            if isinstance(operation, (ast.Is, ast.IsNot)) and not any(
                isinstance(value, ast.Constant)
                and (value.value is None or type(value.value) is bool)
                for value in (left, right)
            ):
                raise CacheIneligible(
                    "Identity comparisons require a None or boolean singleton"
                )
        self.generic_visit(node)

    def visit_Assign(self, node):
        def target(value):
            if isinstance(value, ast.Name):
                return
            if isinstance(value, (ast.Tuple, ast.List)):
                for item in value.elts:
                    target(item)
                return
            raise CacheIneligible("Assignments to bound state are not cached")

        for value in node.targets:
            target(value)
        self.visit(node.value)

    def visit_For(self, node):
        if any(
            not isinstance(part, (ast.Name, ast.Tuple, ast.Store))
            for part in ast.walk(node.target)
        ):
            raise CacheIneligible("Unsupported loop target")
        self.visit(node.target)
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_Attribute(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise CacheIneligible("Assignments to bound state are not cached")
        external = self._external(node)
        if external is not None:
            value, path = external
            if type(value) is types.ModuleType:
                raise CacheIneligible("Bare module attributes are not cached")
            self.module_members.append(self.scan.value(value, path))
            return
        if node.attr not in _METADATA:
            raise CacheIneligible(
                f"Unsupported captured attribute: {node.attr}"
            )
        self.visit(node.value)

    def visit_Call(self, node):
        external = self._external(node.func)
        if external is not None:
            value, path = external
            if (
                _primitive(value) is None
                and not _python_function(value)
                and _compiled_source(value) is None
            ):
                raise CacheIneligible("Unsupported external callable")
            self.module_members.append(self.scan.value(value, path))
        elif isinstance(node.func, ast.Attribute):
            from fhelium.eager import Engine
            from .frontend._eager_capture import _COMPOSITES, _METHODS

            receiver = self._external(node.func.value)
            if receiver is not None and type(receiver[0]) is Engine:
                if node.func.attr not in _METHODS | _COMPOSITES:
                    raise CacheIneligible(
                        f"Unsupported Engine operation: {node.func.attr}"
                    )
                self.visit(node.func.value)
            elif (
                node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.local_lists
            ):
                pass
            elif node.func.attr in _TENSOR_METHODS:
                self.visit(node.func.value)
            elif (
                receiver is not None
                and node.func.attr in _VALUE_CONSTRUCTORS
                and receiver[0] in _value_classes()
            ):
                # The class belongs to fhelium, whose sources the environment
                # identity covers; its name identifies the call.
                self.module_members.append(
                    ["class", cast(type, receiver[0]).__qualname__]
                )
            else:
                raise CacheIneligible(
                    f"Unsupported method call: {node.func.attr}"
                )
        else:
            raise CacheIneligible("Dynamic call targets are not cached")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise CacheIneligible(
                    "Dynamic keyword arguments are not cached"
                )
            self.visit(keyword.value)


@dataclass
class _Scan:
    materials: dict[int, list[tuple[_Path, object]]] = field(
        default_factory=dict
    )
    engines: dict[int, tuple[_Path, object]] = field(default_factory=dict)
    seen: dict[object, int] = field(default_factory=dict)
    active: set[object] = field(default_factory=set)
    storages: dict[int, int] = field(default_factory=dict)
    reads_rank: bool = False

    def value(self, value: object, path: _Path) -> object:
        if type(value) in _SCALARS:
            scalar = value.hex() if type(value) is bytes else _scalar_key(value)
            return [type(value).__name__, scalar]
        primitive = _primitive(value)
        if primitive is not None:
            return ["primitive", primitive]
        if type(value) in (torch.dtype, torch.device):
            return [type(value).__name__, str(value)]
        if type(value) is types.ModuleType:
            # Every module use is checked separately by the syntax visitor.
            return ["module", value.__name__]
        # Bound methods are created on each attribute read; they are
        # identified by their function and instance.
        identity = (
            (id(value.__func__), id(value.__self__))
            if type(value) is types.MethodType
            else id(value)
        )
        if identity in self.active:
            raise CacheIneligible("Reference cycles are not cached")
        if type(value) in _LEAVES:
            self.materials.setdefault(id(value), []).append((path, value))
        if identity in self.seen:
            return ["alias", self.seen[identity]]
        reference = self.seen.setdefault(identity, len(self.seen))
        self.active.add(identity)
        try:
            return ["object", reference, self._value(value, path)]
        finally:
            self.active.remove(identity)

    def _value(self, value, path):
        if type(value) in _LEAVES:
            layout = _leaf_layout(value)
            if type(value) is torch.Tensor:
                storage = value.untyped_storage()._cdata
                return [
                    "Tensor",
                    layout,
                    self.storages.setdefault(storage, len(self.storages)),
                ]
            return [
                type(value).__name__,
                layout,
                [
                    [
                        name,
                        self.value(tensor, _step(path, "field", name)),
                    ]
                    for name, tensor in _leaf_tensors(value).items()
                ],
            ]
        if type(value) in (tuple, list):
            return [
                type(value).__name__,
                [
                    self.value(v, _step(path, "index", i))
                    for i, v in enumerate(value)
                ],
            ]
        if type(value) is dict:
            if any(type(key) not in _SCALARS for key in value):
                raise CacheIneligible("Only scalar dictionary keys are cached")
            return [
                "dict",
                [
                    [
                        self.value(k, _step(path, "dict_key", i)),
                        self.value(v, _step(path, "dict_value", i)),
                    ]
                    for i, (k, v) in enumerate(value.items())
                ],
            ]
        from fhelium.eager import Engine
        from fhelium.config import CkksConfig

        if type(value) is Engine:
            self.engines[id(value)] = (path, value)
            keys = value._keys
            return [
                "engine",
                engine_identity(value),
                [
                    [
                        step,
                        self.value(
                            key, _step(path, "engine_key", ["rotation", step])
                        ),
                    ]
                    for step, key in keys.rotation_keys.items()
                ],
                [
                    [
                        role,
                        self.value(
                            getattr(keys, role),
                            _step(path, "engine_key", [role]),
                        ),
                    ]
                    for role in ("relinearization_key", "conjugation_key")
                ],
            ]
        if type(value) is CkksConfig:
            return ["config", value.dumps()]
        distributed = _distributed_function(value)
        if distributed is not None:
            return distributed(self)
        if _python_function(value):
            return self.function(value, path)
        source = _compiled_source(value)
        if source is not None:
            # Capture inlines the source function of a nested compiled
            # function under the outer options, so its identity is that of
            # the source function.
            return self.function(source, _step(path, "compiled_source", None))
        raise CacheIneligible(
            f"{type(value).__name__} has no supported cache identity"
        )

    def function(
        self, function: types.FunctionType | types.MethodType, path: _Path
    ) -> object:
        # A bound method is its function with the instance attributes it
        # reads; the path steps below read through the method to its function.
        receiver = _NO_RECEIVER
        if type(function) is types.MethodType:
            receiver, function = function.__self__, function.__func__
        code = function.__code__
        members = _Eligibility(function, path, self, receiver).check()
        globals_ = function.__globals__
        entry = {
            "code": hashlib.blake2b(
                marshal.dumps(code), digest_size=16
            ).hexdigest(),
            "source": _source_file(code.co_filename),
            "defaults": [
                self.value(v, _step(path, "default", i))
                for i, v in enumerate(function.__defaults__ or ())
            ],
            "kwdefaults": [
                [k, self.value(v, _step(path, "kwdefault", k))]
                for k, v in (function.__kwdefaults__ or {}).items()
            ],
            "closure": [
                [
                    name,
                    self.value(
                        cell.cell_contents, _step(path, "closure", name)
                    ),
                ]
                for name, cell in zip(
                    code.co_freevars, function.__closure__ or ()
                )
            ],
            "globals": [
                [
                    name,
                    self.value(
                        globals_.get(name, function.__builtins__.get(name)),
                        _step(
                            path,
                            "global" if name in globals_ else "builtin",
                            name,
                        ),
                    ),
                ]
                for name in sorted(_global_reads(code))
            ],
            "members": members,
            "bound": receiver is not _NO_RECEIVER,
        }
        return ["function", entry]


def _distributed_function(value: object) -> Any:
    """Return the identity of a process-group query or captured collective function.

    Rank and group size enter the identity by their current values; the collectives that capture records are identified by name, and their sources are part of the environment identity. A process group is identified by its c10d registry name and global ranks, and a stored specialization binds the live group registered under that name.
    """
    import torch.distributed as torch_distributed

    from fhelium.distributed import all_reduce_ciphertext, reduce_ciphertext

    if (
        value is torch_distributed.get_rank
        or value is torch_distributed.get_world_size
    ):
        name = cast(Any, value).__name__

        def observed(scan: _Scan) -> object:
            if name == "get_rank":
                scan.reads_rank = True
            current = (
                cast(Any, value)()
                if torch_distributed.is_initialized()
                else None
            )
            return ["distributed", name, current]

        return observed
    if value is all_reduce_ciphertext or value is reduce_ciphertext:
        return lambda scan: ["collective", cast(Any, value).__name__]
    if isinstance(value, torch_distributed.ProcessGroup):
        return lambda scan: [
            "process-group",
            value.group_name,
            torch_distributed.get_process_group_ranks(value),
        ]
    return None


def _python_function(value: object) -> bool:
    return type(value) is types.FunctionType or (
        type(value) is types.MethodType
        and type(value.__func__) is types.FunctionType
    )


def _compiled_source(
    value: object,
) -> types.FunctionType | types.MethodType | None:
    """Return the source function or bound method of a compiled callable."""
    from ._callable import CompiledCallable

    if type(value) is CompiledCallable and _python_function(value._source):
        return cast(Any, value._source)
    return None


def _callable_root(function: Any) -> Any:
    """Identify a callable object as its class ``__call__`` bound to the object."""
    if _python_function(function):
        return function
    method = inspect.getattr_static(type(function), "__call__", None)
    if type(method) is types.FunctionType and _instance(function):
        return types.MethodType(method, function)
    return function


def _instance(value: object) -> bool:
    """Whether ``value`` is a user object whose attributes are read statically."""
    from fhelium.config import CkksConfig
    from fhelium.eager import Engine

    from ._callable import CompiledCallable

    kind = type(value)
    return (
        kind.__module__ != "builtins"
        and kind not in _LEAVES
        and kind not in (Engine, CkksConfig, CompiledCallable)
        and not isinstance(value, (types.ModuleType, torch.Tensor))
        and not _python_function(value)
        and isinstance(getattr(value, "__dict__", None), dict)
        and kind.__getattribute__ is object.__getattribute__
        and inspect.getattr_static(kind, "__getattr__", None) is None
    )


def _instance_attribute(owner: object, name: str) -> tuple[object, str]:
    """Read ``owner.name`` from the instance dictionary or bind a class function."""
    from ._callable import CompiledCallable

    member = inspect.getattr_static(type(owner), name, _NO_RECEIVER)
    if type(member) is CompiledCallable:
        # A compiled method is inlined as its source method.
        member = _compiled_source(member)
    elif hasattr(type(member), "__set__"):
        raise CacheIneligible(f"Data descriptors are not cached: {name}")
    elif name in vars(owner):
        return vars(owner)[name], "attr"
    if type(member) is types.FunctionType:
        return types.MethodType(member, owner), "method"
    raise CacheIneligible(f"Unsupported instance attribute: {name}")


def _global_reads(code: types.CodeType) -> set[str]:
    import dis

    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
    }
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _global_reads(constant)
    return names


def _tensor_layout(tensor: torch.Tensor) -> list[object]:
    if (
        type(tensor) is not torch.Tensor
        or tensor.layout != torch.strided
        or tensor.is_quantized
    ):
        raise CacheIneligible("Only ordinary strided Tensors are cached")
    return [
        list(tensor.shape),
        list(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        tensor.requires_grad,
        tensor.storage_offset(),
        tensor.is_conj(),
        tensor.is_neg(),
    ]


def _leaf_layout(value: object) -> object:
    if type(value) not in _LEAVES:
        raise CacheIneligible("Custom material classes are not cached")
    if type(value) is torch.Tensor:
        return _tensor_layout(value)
    if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
        from .frontend._eager_values import value_state

        layout = dict(value_state(value))
        layout["tensors"] = {
            name: _tensor_layout(tensor)
            for name, tensor in _leaf_tensors(value).items()
        }
    else:
        from ._materials import key_description

        assert isinstance(value, (SecretKey, PublicKey, KeySwitchKey))
        layout = key_description(value)
        layout["data"] = _tensor_layout(value.data)
    # Normalize tuples identically before and after JSON persistence.
    return json.loads(json.dumps(layout))


def engine_identity(engine: Any) -> dict[str, object]:
    from .frontend._eager_capture import _COMPOSITES, _METHODS

    if any(name in vars(engine) for name in _METHODS | _COMPOSITES):
        raise CacheIneligible("Engine operation overrides are not cached")
    return {
        "config": engine.config.dumps(),
        "dtype": str(engine.dtype),
        "ntt_backend": engine._ntt_backend,
        "automatic_keys": engine.allow_automatic_key_generation,
        "automatic_replication": engine.allow_automatic_key_replication,
    }


def _signature_payload(signature: CallSignature) -> list[object]:
    return [
        list(signature.context),
        [
            [argument.name, argument.kind, list(argument.properties)]
            for argument in signature.arguments
        ],
    ]


@dataclass(frozen=True)
class SourceIdentity:
    """Source identity and current objects from which materials can be relocated."""

    key: str
    materials: dict[int, list[tuple[_Path, object]]]
    engines: dict[int, tuple[_Path, object]]
    reads_rank: bool = False
    """The source reads the process rank, so its Program may differ between ranks."""


def source_identity(
    function: Any,
    *,
    material_names: Mapping[str, object] | None,
) -> SourceIdentity:
    function = _callable_root(function)
    if not _python_function(function):
        raise CacheIneligible(
            "Only ordinary Python functions, bound methods and callable objects are cached"
        )
    if material_names is not None and type(material_names) is not dict:
        raise CacheIneligible("Material names require an ordinary dictionary")
    scan = _Scan()
    root = scan.value(function, [["root", "fn"]])
    names = scan.value(material_names or {}, [["root", "names"]])
    return SourceIdentity(
        _digest([environment_identity(), root, names]),
        scan.materials,
        scan.engines,
        scan.reads_rank,
    )


def runtime_cacheable(arguments: Mapping[str, object]) -> bool:
    """Accept flat built-in scalars and ordinary Tensor/value arguments."""
    for value in arguments.values():
        if type(value) in _SCALARS:
            continue
        if type(value) not in _LEAVES:
            return False
        try:
            _leaf_layout(value)
        except CacheIneligible:
            return False
    return True


def supplied_keys(keys: object) -> tuple[Any, ...]:
    """The key objects of a ``keys`` argument: a mapping's values, a key set's keys or a sequence."""
    return tuple(
        keys.values() if isinstance(keys, Mapping) else cast(Any, keys)
    )


def key_inventory(keys: object) -> list[str]:
    """Layouts of the supplied keys; compilation selects keys by these layouts."""
    from ._materials import key_description

    return sorted(
        json.dumps(key_description(key), sort_keys=True)
        for key in supplied_keys(keys)
    )


def entry_key(
    identity: SourceIdentity, signature: CallSignature, keys: object = ()
) -> str:
    return _digest(
        [identity.key, _signature_payload(signature), key_inventory(keys)]
    )


def trace_entry_key(
    program: Program,
    bindings: Mapping[str, torch.Tensor],
    signature: CallSignature,
    keys: object = (),
) -> str:
    """Key a prepared Program by the captured Program it was compiled from.

    The captured Program records every Tensor operation, Engine operation, material reference and type that the Python source produced for this call. Together with the layouts of the bound materials, the call signature and the layouts of the keys supplied to the callable it determines the pipeline result, independently of how the Python source computed it.
    """
    from fhelium.ir._structure import program_digest

    return _digest(
        [
            "trace",
            environment_identity(),
            _signature_payload(signature),
            program_digest(program.module),
            sorted(
                [symbol, _tensor_layout(tensor)]
                for symbol, tensor in bindings.items()
            ),
            key_inventory(keys),
        ]
    )


def artifact_key(registry) -> str | None:
    """Key the linked executables of one implementation registry.

    Returns None when the registry contains implementations defined outside FHElium.
    """
    from fhelium.backend._artifact import registry_identity

    identity = registry_identity(registry)
    return None if identity is None else _digest(["linked", identity])


def _leaf_tensors(leaf: object) -> dict[str, torch.Tensor]:
    if type(leaf) not in _LEAVES:
        raise CacheIneligible("Unsupported material class")
    if type(leaf) is torch.Tensor:
        return {"": leaf}
    if type(leaf) is Plaintext:
        return {
            name: tensor
            for name in ("data", "message")
            if (tensor := getattr(leaf, name)) is not None
        }
    if type(leaf) is CompressedPlaintext:
        return {
            name: tensor
            for name in ("data", "implicit_data")
            if (tensor := getattr(leaf, name)) is not None
        }
    return {
        name: tensor
        for name in ("data", "seed")
        if (tensor := getattr(leaf, name, None)) is not None
    }


def _resolve_path(roots: Mapping[str, object], path: _Path) -> object:
    # Each step reads an attribute, item or field of the object reached so far.
    value: Any = roots
    argument: Any
    for operation, argument in path:
        if operation == "root":
            value = roots[argument]
        elif operation == "global":
            value = value.__globals__[argument]
        elif operation == "builtin":
            value = value.__builtins__[argument]
        elif operation == "closure":
            value = dict(
                zip(value.__code__.co_freevars, value.__closure__ or ())
            )[argument].cell_contents
        elif operation == "default":
            value = value.__defaults__[argument]
        elif operation == "kwdefault":
            value = value.__kwdefaults__[argument]
        elif operation == "index" and type(value) in (tuple, list):
            value = value[argument]
        elif operation in {"dict_key", "dict_value"} and type(value) is dict:
            value = list(
                value.keys() if operation == "dict_key" else value.values()
            )[argument]
        elif operation == "module_attr" and type(value) is types.ModuleType:
            value = vars(value)[argument]
        elif operation == "field":
            value = _leaf_tensors(value)[argument]
        elif operation == "self":
            value = value.__self__
        elif operation == "attr":
            value = vars(value)[argument]
        elif operation == "method":
            value = _instance_attribute(value, argument)[0]
        elif operation == "compiled_source":
            value = _compiled_source(value)
            if value is None:
                raise CacheIneligible("Compiled function root changed")
        elif operation == "engine_key":
            from fhelium.eager import Engine

            if type(value) is not Engine:
                raise CacheIneligible("Engine material root changed")
            value = (
                value._keys.rotation_keys[argument[1]]
                if argument[0] == "rotation"
                else getattr(value._keys, argument[0])
            )
        else:
            raise CacheIneligible("Unsupported material path step")
    return value


_RESOURCE_KINDS = {
    "rns_parameters",
    "twice_modulus",
    "ntt_table",
    "key_switch_table",
    "rotation_sum_table",
    "galois_table",
    "centered_basis_extension_table",
    "encapsulated_modulus_raise_table",
    "rescale_table",
    "encode_table",
    "periodic_plaintext_table",
    "rounding_state",
    "integer_scalar",
}


def build_manifest(
    identity: SourceIdentity,
    bindings: Mapping[str, torch.Tensor],
    descriptions: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    """Locate bound materials by their paths from the source function."""
    located: dict[int, list[dict[str, object]]] = {}
    for occurrences in identity.materials.values():
        for path, leaf in occurrences:
            for name, tensor in _leaf_tensors(leaf).items():
                origin = {
                    "path": path,
                    "field": name,
                    "owner_layout": _leaf_layout(leaf),
                }
                located.setdefault(id(tensor), []).append(origin)
    materials: dict[str, object] = {}
    aliases: dict[int, int] = {}
    for symbol, tensor in bindings.items():
        if descriptions.get(symbol, {}).get("kind") == "PreparedTensor":
            continue
        origins = located.get(id(tensor))
        if origins is not None:
            entry = {
                "origins": origins,
                "layout": _tensor_layout(tensor),
                "alias": aliases.setdefault(id(tensor), len(aliases)),
            }
        elif (
            descriptions.get(symbol, {}).get("kind") in _RESOURCE_KINDS
            and len(identity.engines) == 1
        ):
            # Linking regenerates current table requirements. Retain only the
            # provider location, never the old numerical description/state.
            entry = {
                "engine": next(iter(identity.engines.values()))[0],
                "resource": str(tensor.device),
            }
        else:
            return None
        materials[symbol] = entry
    return materials


def relocate_materials(
    function: Any,
    manifest: Mapping[str, Any],
    *,
    material_names: Mapping[str, object] | None = None,
) -> tuple[dict[str, torch.Tensor], list[tuple[Engine, torch.device]]]:
    """Resolve current material objects and providers for an instance Program."""
    roots = {"fn": _callable_root(function), "names": material_names or {}}
    bindings: dict[str, torch.Tensor] = {}
    providers: dict[tuple[int, str], tuple[Engine, torch.device]] = {}
    aliases: dict[int, torch.Tensor] = {}
    reverse_aliases: dict[int, int] = {}
    for symbol, entry in manifest.items():
        if "resource" in entry:
            from fhelium.eager import Engine

            engine = _resolve_path(roots, entry["engine"])
            if type(engine) is not Engine:
                raise CacheIneligible("Engine material root changed")
            device = torch.device(entry["resource"])
            providers.setdefault((id(engine), str(device)), (engine, device))
            continue
        tensors = []
        for origin in entry["origins"]:
            leaf = _resolve_path(roots, origin["path"])
            if _leaf_layout(leaf) != origin["owner_layout"]:
                raise CacheIneligible(f"{symbol}: material state changed")
            tensors.append(_leaf_tensors(leaf)[origin["field"]])
        if not tensors or any(tensor is not tensors[0] for tensor in tensors):
            raise CacheIneligible(f"{symbol}: material aliases changed")
        tensor = tensors[0]
        alias = entry["alias"]
        if (
            _tensor_layout(tensor) != entry["layout"]
            or aliases.setdefault(alias, tensor) is not tensor
            or reverse_aliases.setdefault(id(tensor), alias) != alias
        ):
            raise CacheIneligible(
                f"{symbol}: material layout or aliases changed"
            )
        bindings[symbol] = tensor
    return bindings, list(providers.values())


@dataclass(frozen=True)
class CacheEntry:
    """One persisted specialization.

    ``program`` is the prepared Program, its :meth:`Program.encode` payload, or None when the file stores no Program text; linking from the Program is then unavailable. ``linked`` records the linked executable the file stores for one implementation registry: ``registry``, the provider ``selections`` of generated materials, ``keys`` (key descriptions by material symbol), ``input_fields`` and the ``executable`` record; ``path`` is added when the entry is loaded.
    """

    program: Program | dict[str, object] | None
    manifest: dict[str, object]
    output_structure: object | None
    input_names: tuple[str, ...]
    preparation: dict | None = None
    linked: dict[str, Any] | None = None
    groups: dict[str, str | None] | None = None
    """c10d registry names of the captured process groups by resource symbol; None names the default group."""


_PROGRAM = "cache.program"


class CompileCache:
    """Publish one artifact file per input specialization.

    The manifest records the specialization under ``cache``: the relocation manifest of its materials, its output structure, input names, preparation descriptor and, when one was stored, its linked executable under ``linked``, whose entries, kernels, constants and material records are the artifact's. The prepared Program is stored as the byte tensor ``cache.program``, from which a specialization is linked again when the linked executable does not apply.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / (key + ".fhelium")

    def load(self, key: str, *, decode: bool = True) -> CacheEntry | None:
        """Read one entry; ``decode=False`` leaves ``program`` as its encoded payload."""
        from fhelium.ir import Program
        from fhelium.native.artifact import (
            ARTIFACT_FORMAT,
            read_container_header,
        )

        path = self._path(key)
        try:
            metadata, header = read_container_header(path)
            if metadata.get("fhelium.format") != ARTIFACT_FORMAT:
                return None
            cache = json.loads(metadata["fhelium.manifest"])["cache"]
            materials, names = cache["materials"], cache["input_names"]
            if (
                cache["schema"] != _SCHEMA
                or cache["key"] != key
                or not isinstance(materials, dict)
                or not isinstance(names, list)
                or any(not isinstance(name, str) for name in names)
            ):
                return None
            program = None
            if _PROGRAM in header:
                start, stop = header[_PROGRAM]["data_offsets"]
                with open(path, "rb") as stream:
                    (length,) = struct.unpack("<Q", stream.read(8))
                    stream.seek(8 + length + start)
                    program = json.loads(stream.read(stop - start))
                if decode:
                    # Linking verifies the Program it prepares from this entry.
                    program = Program.decode(program, verify=False)
            linked = cache.get("linked")
            return CacheEntry(
                program,
                materials,
                cache["output_structure"],
                tuple(names),
                cache.get("preparation"),
                {**linked, "path": str(path)} if linked is not None else None,
                cache.get("groups"),
            )
        except Exception:
            # Cache files are optional local artifacts; malformed files are misses.
            return None

    def store(
        self,
        key: str,
        entry: CacheEntry,
        parts: Mapping[str, Any] | None = None,
        tensors: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Write ``entry``; ``parts`` and ``tensors`` are the artifact manifest fields and Tensors of ``entry.linked``."""
        from fhelium.native.artifact import save_artifact

        stored = dict(tensors or {})
        if entry.program is not None:
            stored[_PROGRAM] = _json_tensor(
                entry.program
                if isinstance(entry.program, dict)
                else entry.program.encode()
            )
        linked = (
            None
            if entry.linked is None
            else {k: v for k, v in entry.linked.items() if k != "path"}
        )
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_artifact(
            path,
            {
                "entries": {},
                "kernels": [],
                "constants": [],
                "materials": {},
                "signatures": [],
                "prepare": {},
                **(parts or {}),
                "cache": {
                    "schema": _SCHEMA,
                    "key": key,
                    "preparation": entry.preparation,
                    "materials": entry.manifest,
                    "output_structure": entry.output_structure,
                    "input_names": list(entry.input_names),
                    "linked": linked,
                    "groups": entry.groups,
                },
            },
            stored,
            overwrite=True,
        )


def _json_tensor(value: object) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(json.dumps(value, allow_nan=False).encode()),
        dtype=torch.uint8,
    )


__all__ = [
    "CacheEntry",
    "artifact_key",
    "CacheIneligible",
    "CompileCache",
    "SourceIdentity",
    "build_manifest",
    "cache_directory",
    "entry_key",
    "relocate_materials",
    "runtime_cacheable",
    "source_identity",
    "trace_entry_key",
]
