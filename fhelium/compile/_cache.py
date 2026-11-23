"""Persist prepared Programs and relocate current numerical materials.

Entries contain encoded Programs and JSON metadata; Tensor payloads remain in the calling process. Source entries are keyed before capture and require pure Python expressions, static control flow, and supported Tensor and Engine operations. Trace entries are keyed by the captured Program and accept any captured source.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import hashlib
import json
import marshal
import math
import os
import platform
import sys
import sysconfig
import tempfile
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
_CKKS_LEAVES = (Ciphertext, Plaintext, CompressedPlaintext)
_SCHEMA = 5
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
    "default_scale",
    "N",
    "logN",
    "q_depth_groups",
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

    def __init__(self, function, path, scan):
        self.function, self.path, self.scan = function, path, scan
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
        return None

    def visit_Name(self, node):
        found = self._external(node)
        if (
            isinstance(node.ctx, ast.Load)
            and found is not None
            and isinstance(found[0], types.ModuleType)
        ):
            raise CacheIneligible(
                "Modules must be used through static supported attributes"
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
                and type(value) is not types.FunctionType
            ):
                raise CacheIneligible("Unsupported external callable")
            self.module_members.append(self.scan.value(value, path))
        elif isinstance(node.func, ast.Attribute):
            from fhelium.eager import Engine
            from .frontend._eager_capture import _METHODS

            receiver = self._external(node.func.value)
            if receiver is not None and type(receiver[0]) is Engine:
                if node.func.attr not in _METHODS:
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
    structural: bool = False
    payload_owners: dict[int, list[tuple[_Path, object, str]]] = field(
        default_factory=dict
    )
    raw_materials: set[int] = field(default_factory=set)
    materials: dict[int, list[tuple[_Path, object]]] = field(
        default_factory=dict
    )
    engines: dict[int, tuple[_Path, object]] = field(default_factory=dict)
    seen: dict[int, int] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)
    storages: dict[int, int] = field(default_factory=dict)

    def value(self, value: object, path: _Path, *, payload=None) -> object:
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
        identity = id(value)
        if type(value) is torch.Tensor:
            if payload is None:
                self.raw_materials.add(identity)
            else:
                self.payload_owners.setdefault(identity, []).append(payload)
        if identity in self.active:
            raise CacheIneligible("Reference cycles are not cached")
        if type(value) in _LEAVES:
            self.materials.setdefault(identity, []).append((path, value))
        if identity in self.seen:
            return ["alias", self.seen[identity]]
        reference = self.seen.setdefault(identity, len(self.seen))
        self.active.add(identity)
        try:
            return ["object", reference, self._value(value, path, payload)]
        finally:
            self.active.remove(identity)

    def _value(self, value, path, payload):
        if type(value) in _LEAVES:
            layout = _leaf_layout(value, structural=self.structural)
            if self.structural and payload is not None:
                _, owner, field_name = payload
                layout = _payload_layout(owner, field_name, structural=True)
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
                        self.value(
                            tensor,
                            _step(path, "field", name),
                            payload=(path, value, name)
                            if _rns_leaf(value)
                            else None,
                        ),
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
        if type(value) is types.FunctionType:
            return self.function(value, path)
        raise CacheIneligible(
            f"{type(value).__name__} has no supported cache identity"
        )

    def function(self, function: types.FunctionType, path: _Path) -> object:
        code = function.__code__
        members = _Eligibility(function, path, self).check()
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
        }
        return ["function", entry]


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


def _rns_leaf(value: object) -> bool:
    return type(value) in _CKKS_LEAVES and (
        type(value) is not Plaintext or value.representation == "rns"
    )


def _row_layout(layout: list[object], row_axis: int) -> list[object]:
    """Normalize a standard dense row extent and its dependent strides.

    Nonstandard strides retain their full physical layout. A stride is treated as row-dependent only when the complete stride tuple matches dense contiguous storage.
    """
    shape, strides = list(layout[0]), list(layout[1])
    axis = len(shape) + row_axis
    if (
        axis < 0
        or not shape[axis]
        or any(dimension <= 0 for dimension in shape)
    ):
        return layout
    expected = []
    extent = 1
    for dimension in reversed(shape):
        expected.append(extent)
        extent *= dimension
    if strides != list(reversed(expected)):
        return layout
    rows = shape[axis]
    shape[axis] = "active-prime-rows"
    strides = [
        ["active-prime-rows", stride // rows] if index < axis else stride
        for index, stride in enumerate(strides)
    ]
    return [shape, strides, *layout[2:]]


def _payload_layout(
    owner: object, name: str, *, structural: bool
) -> list[object]:
    tensor = _leaf_tensors(owner)[name]
    layout = _tensor_layout(tensor)
    if structural and _rns_leaf(owner):
        axis = -1 if name == "implicit_data" else -2
        if tensor.ndim >= -axis and tensor.shape[axis] == len(owner.prime_ids):
            return _row_layout(layout, axis)
    return layout


def _leaf_layout(value: object, *, structural: bool = False) -> object:
    if type(value) not in _LEAVES:
        raise CacheIneligible("Custom material classes are not cached")
    if type(value) is torch.Tensor:
        return _tensor_layout(value)
    if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
        from .frontend._eager_values import value_state

        layout = dict(value_state(value))
        if structural and _rns_leaf(value):
            for name in ("depth", "scale", "prime_ids"):
                layout.pop(name)
        layout["tensors"] = {
            name: _payload_layout(value, name, structural=structural)
            for name in _leaf_tensors(value)
        }
    else:
        from ._materials import key_description

        layout = key_description(value)
        layout["data"] = _tensor_layout(value.data)
    # Normalize tuples identically before and after JSON persistence.
    return json.loads(json.dumps(layout))


def engine_identity(engine: Any) -> dict[str, object]:
    from .frontend._eager_capture import _METHODS

    if any(name in vars(engine) for name in _METHODS):
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
    payload_owners: dict[int, list[tuple[_Path, object, str]]] = field(
        default_factory=dict
    )
    raw_materials: frozenset[int] = frozenset()


def source_identity(
    function: Any,
    *,
    material_names: Mapping[str, object] | None,
    structural: bool = False,
) -> SourceIdentity:
    if type(function) is not types.FunctionType:
        raise CacheIneligible("Only ordinary Python functions are cached")
    if material_names is not None and type(material_names) is not dict:
        raise CacheIneligible("Material names require an ordinary dictionary")
    scan = _Scan(structural=structural)
    root = scan.value(function, [["root", "fn"]])
    names = scan.value(material_names or {}, [["root", "names"]])
    if structural:
        # An additional raw reference remains fixed even when the same Tensor was
        # first encountered through a CKKS owner and assigned an alias token.
        raw_layouts = [
            [
                scan.seen[identity],
                _tensor_layout(scan.materials[identity][0][1]),
            ]
            for identity in sorted(
                scan.raw_materials, key=scan.seen.__getitem__
            )
        ]
        root = ["structural", root, raw_layouts]
    return SourceIdentity(
        _digest([environment_identity(), root, names]),
        scan.materials,
        scan.engines,
        scan.payload_owners,
        frozenset(scan.raw_materials),
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


def entry_key(identity: SourceIdentity, signature: CallSignature) -> str:
    return _digest([identity.key, _signature_payload(signature)])


def trace_entry_key(
    program: Program,
    bindings: Mapping[str, torch.Tensor],
    signature: CallSignature,
) -> str:
    """Key a prepared Program by the captured Program it was compiled from.

    The captured Program records every Tensor operation, Engine operation, material reference and type that the Python source produced for this call. Together with the layouts of the bound materials and the call signature it determines the pipeline result, independently of how the Python source computed it.
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
        ]
    )


def preparation_artifact_key(source) -> str:
    """Key the linked executable of a plaintext-preparation Program or its recorded recipe."""
    from fhelium.ir import Program
    from fhelium.ir._structure import program_digest

    if isinstance(source, Program):
        content = ["program", program_digest(source.module)]
    else:
        content = ["recipe", source.digest()]
    return _digest(["preparation", environment_identity(), *content])


def artifact_key(entry: str, registry) -> str | None:
    """Key the linked executable of a cached Program for one implementation registry.

    Returns None when the registry contains implementations defined outside FHElium.
    """
    from fhelium.backend._artifact import registry_identity

    identity = registry_identity(registry)
    return None if identity is None else _digest(["linked", entry, identity])


def structural_entry_key(
    identity: SourceIdentity, signature: CallSignature
) -> str:
    """Group RNS CKKS requests while retaining fixed arguments and execution modes."""
    arguments = []
    for argument in signature.arguments:
        fields = dict(argument.properties)
        is_rns = argument.kind in {"Ciphertext", "CompressedPlaintext"} or (
            argument.kind == "Plaintext"
            and fields.get("representation") == "rns"
        )
        if is_rns:
            rows = len(fields["prime_ids"])
            for prefix, axis in (("", -2), ("implicit_", -1)):
                shape, strides = (
                    fields.get(prefix + "shape"),
                    fields.get(prefix + "strides"),
                )
                if (
                    shape is not None
                    and strides is not None
                    and len(shape) >= -axis
                    and shape[axis] == rows
                ):
                    normalized = _row_layout([shape, strides], axis)
                    fields[prefix + "shape"], fields[prefix + "strides"] = (
                        normalized
                    )
            for name in ("depth", "scale", "prime_ids"):
                fields.pop(name, None)
        arguments.append([argument.name, argument.kind, list(fields.items())])
    return _digest(
        ["structural", identity.key, list(signature.context), arguments]
    )


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
    return {"data": leaf.data}


def _resolve_path(roots: Mapping[str, object], path: _Path) -> object:
    value: object = roots
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
    *,
    structural: bool = False,
) -> dict[str, object] | None:
    """Locate bound materials and optionally retain their CKKS owner paths."""
    located: dict[int, list[dict[str, object]]] = {}
    owners: dict[int, list[tuple[object, str]]] = {}
    for occurrences in identity.materials.values():
        for path, leaf in occurrences:
            if (
                structural
                and type(leaf) is torch.Tensor
                and id(leaf) in identity.payload_owners
                and id(leaf) not in identity.raw_materials
            ):
                # The owning envelope provides both physical and mathematical
                # state; its recursive Tensor observation is not an extra input.
                continue
            dependent = structural and _rns_leaf(leaf)
            for name, tensor in _leaf_tensors(leaf).items():
                origin = {
                    "path": path,
                    "field": name,
                    "owner_layout": _leaf_layout(leaf, structural=dependent),
                }
                if dependent:
                    origin["ckks_owner"] = type(leaf).__name__
                    owners.setdefault(id(tensor), []).append((leaf, name))
                located.setdefault(id(tensor), []).append(origin)
    materials: dict[str, object] = {}
    aliases: dict[int, int] = {}
    for symbol, tensor in bindings.items():
        if descriptions.get(symbol, {}).get("kind") == "PreparedTensor":
            continue
        origins = located.get(id(tensor))
        if origins is not None:
            owner_entries = owners.get(id(tensor), ())
            if owner_entries:
                owner, field_name = owner_entries[0]
                # One material symbol must have one interpretation of CKKS state.
                if any(
                    type(other) is not type(owner)
                    or name != field_name
                    or _leaf_layout(other) != _leaf_layout(owner)
                    for other, name in owner_entries[1:]
                ):
                    return None
                if any(
                    "ckks_owner" not in origin and origin["field"] != ""
                    for origin in origins
                ):
                    return None
            dependent = (
                bool(owner_entries) and id(tensor) not in identity.raw_materials
            )
            entry = {
                "origins": origins,
                "layout": (
                    _payload_layout(owner, field_name, structural=True)
                    if dependent
                    else _tensor_layout(tensor)
                ),
                "alias": aliases.setdefault(id(tensor), len(aliases)),
            }
            if dependent:
                entry["dependent_layout"] = True
        elif (
            descriptions.get(symbol, {}).get("kind") in _RESOURCE_KINDS
            and len(identity.engines) == 1
        ):
            # The template regenerates current table requirements. Retain only
            # the provider location, never the old numerical description/state.
            entry = {
                "engine": next(iter(identity.engines.values()))[0],
                "resource": str(tensor.device),
            }
        else:
            return None
        materials[symbol] = entry
    return materials


def _relocate_materials(
    function: Any,
    manifest: Mapping[str, Any],
    material_names: Mapping[str, object] | None,
    *,
    structural: bool,
):
    roots = {"fn": function, "names": material_names or {}}
    bindings: dict[str, torch.Tensor] = {}
    providers: dict[tuple[int, str], tuple[object, torch.device]] = {}
    aliases: dict[int, torch.Tensor] = {}
    reverse_aliases: dict[int, int] = {}
    leaf_states: dict[str, object] = {}
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
        owner = None
        fixed_owner = None
        field_name = None
        for origin in entry["origins"]:
            leaf = _resolve_path(roots, origin["path"])
            dependent = structural and "ckks_owner" in origin
            if dependent:
                if (
                    not _rns_leaf(leaf)
                    or type(leaf).__name__ != origin["ckks_owner"]
                ):
                    raise CacheIneligible(
                        f"{symbol}: CKKS material role changed"
                    )
                if owner is not None and (
                    _leaf_layout(owner) != _leaf_layout(leaf)
                    or field_name != origin["field"]
                ):
                    raise CacheIneligible(
                        f"{symbol}: aliased CKKS owners disagree"
                    )
                owner, field_name = leaf, origin["field"]
            if (
                _leaf_layout(leaf, structural=dependent)
                != origin["owner_layout"]
            ):
                raise CacheIneligible(f"{symbol}: material state changed")
            if structural and isinstance(
                leaf, (SecretKey, PublicKey, KeySwitchKey)
            ):
                if fixed_owner is not None and (
                    type(fixed_owner) is not type(leaf)
                    or _leaf_layout(fixed_owner) != _leaf_layout(leaf)
                ):
                    raise CacheIneligible(
                        f"{symbol}: aliased key owners disagree"
                    )
                fixed_owner = leaf
            tensors.append(_leaf_tensors(leaf)[origin["field"]])
        if not tensors or any(tensor is not tensors[0] for tensor in tensors):
            raise CacheIneligible(f"{symbol}: material aliases changed")
        tensor = tensors[0]
        layout = _tensor_layout(tensor)
        if structural and entry.get("dependent_layout"):
            if owner is None:
                raise CacheIneligible(f"{symbol}: missing CKKS owner")
            layout = _payload_layout(owner, field_name, structural=True)
        alias = entry["alias"]
        if (
            layout != entry["layout"]
            or aliases.setdefault(alias, tensor) is not tensor
            or reverse_aliases.setdefault(id(tensor), alias) != alias
        ):
            raise CacheIneligible(
                f"{symbol}: material layout or aliases changed"
            )
        bindings[symbol] = tensor
        if owner is not None:
            leaf_states[symbol] = (
                tensor if field_name == "implicit_data" else owner
            )
        elif fixed_owner is not None:
            leaf_states[symbol] = fixed_owner
    return bindings, list(providers.values()), leaf_states


def relocate_materials(
    function: Any,
    manifest: Mapping[str, Any],
    *,
    material_names: Mapping[str, object] | None = None,
) -> tuple[dict[str, torch.Tensor], list[tuple[object, torch.device]]]:
    """Resolve current material objects and providers for an instance Program."""
    bindings, providers, _ = _relocate_materials(
        function,
        manifest,
        material_names,
        structural=False,
    )
    return bindings, providers


def relocate_structural_materials(
    function: Any,
    manifest: Mapping[str, Any],
    *,
    material_names: Mapping[str, object] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    list[tuple[object, torch.device]],
    dict[str, object],
]:
    """Resolve current bindings and states for a structural candidate.

    Main CKKS and key payloads retain their owning envelopes. An implicit compressed-plaintext payload supplies its Tensor state because its Program operand is a plain MessageType.
    """
    return _relocate_materials(
        function, manifest, material_names, structural=True
    )


@dataclass(frozen=True)
class CacheEntry:
    """One persisted specialization; ``source_encoding`` is decoded only when a caller needs the source Program.

    ``program`` is a Program or its :meth:`Program.encode` payload.
    """

    program: Program | dict[str, object]
    manifest: dict[str, object]
    output_structure: object | None
    input_names: tuple[str, ...]
    source_encoding: dict[str, object] | None = None
    preparation: dict | None = None


class CompileCache:
    """Atomically publish one JSON document per input specialization."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / (key + ".json")

    def load(self, key: str, *, decode: bool = True) -> CacheEntry | None:
        """Read one entry; ``decode=False`` leaves ``program`` as its encoded payload."""
        from fhelium.ir import Program

        try:
            payload = json.loads(self._path(key).read_text(encoding="utf-8"))
            if payload["schema"] != _SCHEMA or payload["key"] != key:
                return None
            materials, names = payload["materials"], payload["input_names"]
            if (
                not isinstance(materials, dict)
                or not isinstance(names, list)
                or any(not isinstance(name, str) for name in names)
            ):
                return None
            return CacheEntry(
                # Linking verifies the Program it prepares from this entry.
                (
                    Program.decode(payload["program"], verify=False)
                    if decode
                    else payload["program"]
                ),
                materials,
                payload["output_structure"],
                tuple(names),
                payload.get("source_program"),
                payload.get("preparation"),
            )
        except Exception:
            # Cache files are optional local artifacts; malformed JSON/IR is a miss.
            return None

    def store(self, key: str, entry: CacheEntry) -> None:
        payload = {
            "schema": _SCHEMA,
            "key": key,
            "program": (
                entry.program
                if isinstance(entry.program, dict)
                else entry.program.encode()
            ),
            "source_program": entry.source_encoding,
            "preparation": entry.preparation,
            "materials": entry.manifest,
            "output_structure": entry.output_structure,
            "input_names": list(entry.input_names),
        }
        self._publish(
            self._path(key), json.dumps(payload, allow_nan=False).encode()
        )

    def _artifact_path(self, key: str) -> Path:
        return self.directory / key[:2] / (key + ".linked")

    def load_artifact(self, key: str) -> dict[str, object] | None:
        """Return a linked-executable artifact record, or None when absent or unreadable."""
        import pickle

        try:
            record = pickle.loads(self._artifact_path(key).read_bytes())
        except Exception:
            return None
        if record.get("schema") != _SCHEMA or record.get("key") != key:
            return None
        return record

    def store_artifact(self, key: str, record: dict[str, object]) -> None:
        import pickle

        self._publish(
            self._artifact_path(key),
            pickle.dumps(
                {"schema": _SCHEMA, "key": key, **record},
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(data)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


__all__ = [
    "CacheEntry",
    "artifact_key",
    "preparation_artifact_key",
    "CacheIneligible",
    "CompileCache",
    "SourceIdentity",
    "build_manifest",
    "cache_directory",
    "entry_key",
    "relocate_materials",
    "relocate_structural_materials",
    "runtime_cacheable",
    "source_identity",
    "structural_entry_key",
    "trace_entry_key",
]
