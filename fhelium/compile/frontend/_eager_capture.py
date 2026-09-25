"""Capture mixed Tensor expressions and Engine calls into one Program.

Symbolic values preserve their numerical roles. Engine adapters expose actual
Tensor dataflow without executing numerical kernels or generating keys.
"""

from __future__ import annotations

import dis
import inspect
import json
import math
import operator
import types
from collections.abc import Callable, Mapping, Sequence
from operator import index as integer_index
from types import MappingProxyType
from typing import Any, cast

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
    f64,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.eager import Engine
from fhelium.ir import Program, value_role
from fhelium.ir.ckks_state import (
    depth_prime_ids,
    output_residues,
    product_scale,
    quotient_scale,
    reinterpreted_scale,
    transform_state,
)
from fhelium.ir.dialects import ckks, core, rns
from fhelium.ir.dialects._common import OpenStateType
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    KeySwitchKey,
    Plaintext,
    RotationKey,
)
from fhelium.values._scale import coerce_scale
from fhelium.values.state import ModulusBasis, PolynomialDomain

from .._compilation import Compilation
from .._errors import CaptureError
from .._materials import (
    key_description,
    named_material_tensors,
    parameter_description,
)
from .._workspace import CompileWorkspace
from ._captured_callable import CapturedCallable
from ._eager_values import state_attribute, state_attributes, value_type
from ._pytorch_encoding import encode_literal
from ._specs import (
    InputSpec,
    StaticValue,
    encrypted,
    message,
    plaintext,
    static,
)


class _Value:
    """Expose represented metadata while withholding a numerical Tensor payload."""

    _implicit_value: _Value | None

    def __init__(self, value: SSAValue, capture: _Capture) -> None:
        self.value = value
        self.capture = capture

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        def find(value):
            if isinstance(value, _Value):
                return value.capture
            if isinstance(value, (tuple, list)):
                return next(
                    (
                        found
                        for item in value
                        if (found := find(item)) is not None
                    ),
                    None,
                )
            if isinstance(value, dict):
                return find(tuple(value.values()))
            return None

        capture = find((args, kwargs or {}))
        if capture is None:
            return NotImplemented
        return capture.numeric_call(function, args, kwargs or {})

    def __add__(self, other):
        return self.capture.numeric_call(torch.add, (self, other), {})

    def __radd__(self, other):
        return self.capture.numeric_call(torch.add, (other, self), {})

    def __sub__(self, other):
        return self.capture.numeric_call(torch.sub, (self, other), {})

    def __rsub__(self, other):
        return self.capture.numeric_call(torch.sub, (other, self), {})

    def __mul__(self, other):
        return self.capture.numeric_call(torch.mul, (self, other), {})

    def __rmul__(self, other):
        return self.capture.numeric_call(torch.mul, (other, self), {})

    def __truediv__(self, other):
        return self.capture.numeric_call(torch.div, (self, other), {})

    def __rtruediv__(self, other):
        return self.capture.numeric_call(torch.div, (other, self), {})

    def __neg__(self):
        return self.capture.numeric_call(torch.neg, (self,), {})

    def __pow__(self, exponent):
        if value_role(self.value) == "encrypted" and exponent == 2:
            return self * self
        return self.capture.numeric_call(torch.pow, (self, exponent), {})

    def __matmul__(self, other):
        return self.capture.numeric_call(torch.matmul, (self, other), {})

    def __getitem__(self, index):
        return self.capture.numeric_call(operator.getitem, (self, index), {})

    def reshape(self, *shape):
        shape = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )
        return self.capture.numeric_call(torch.reshape, (self, shape), {})

    def contiguous(self):
        return self.capture.numeric_call(torch.Tensor.contiguous, (self,), {})

    def clone(self):
        if isinstance(
            self.value.type,
            (
                ckks.CiphertextType,
                ckks.PlaintextType,
                ckks.CompressedPlaintextType,
            ),
        ):
            return self._with_resident_tensors(
                tuple(t.clone() for t in self._resident_tensors)
            )
        return self.capture.numeric_call(torch.clone, (self,), {})

    def to(self, *args, **kwargs):
        if isinstance(
            self.value.type,
            (
                ckks.CiphertextType,
                ckks.PlaintextType,
                ckks.CompressedPlaintextType,
            ),
        ):
            return self._move_value(*args, **kwargs)
        return self.capture.numeric_call(torch.Tensor.to, (self, *args), kwargs)

    def _move_value(self, device, *, non_blocking=False, copy=False):
        target = torch.device(device)
        return self._with_resident_tensors(
            tuple(
                tensor.to(target, non_blocking=non_blocking, copy=copy)
                for tensor in self._resident_tensors
            )
        )

    def select(self, dim: int, index: int):
        return self.capture.numeric_call(torch.select, (self, dim, index), {})

    def narrow(self, dim: int, start: int, length: int):
        return self.capture.numeric_call(
            torch.narrow, (self, dim, start, length), {}
        )

    def select_batch(self, index: int, *, dim: int = 0):
        cls = (
            Ciphertext
            if isinstance(self.value.type, ckks.CiphertextType)
            else CompressedPlaintext
            if isinstance(self.value.type, ckks.CompressedPlaintextType)
            else Plaintext
        )
        return cls.select_batch(cast(Any, self), index, dim=dim)

    def slice_batch(self, start: int, stop: int, *, dim: int = 0):
        cls = (
            Ciphertext
            if isinstance(self.value.type, ckks.CiphertextType)
            else CompressedPlaintext
            if isinstance(self.value.type, ckks.CompressedPlaintextType)
            else Plaintext
        )
        return cls.slice_batch(cast(Any, self), start, stop, dim=dim)

    def slice_limbs(self, start: int, stop: int):
        if not 0 <= start < stop <= len(self.prime_ids):
            raise ValueError(
                "Limb slice must select a nonempty stored row interval"
            )
        tensors = (self.data[..., start:stop, :],)
        if self.implicit_data is not None:
            tensors += (self.implicit_data[..., start:stop],)
        result = self._with_resident_tensors(tensors)
        result.value = Rewriter.replace_value_with_new_type(
            result.value,
            cast(OpenStateType, result.value.type).with_state(
                {
                    "prime_ids": cast(
                        Attribute,
                        state_attribute(tuple(self.prime_ids[start:stop])),
                    )
                }
            ),
        )
        return result

    def _tensor_state(self):
        state = dict(cast(OpenStateType, self.value.type).state.data)
        if "shape" not in state and isinstance(
            self.value.type, (ckks.CiphertextType, ckks.PlaintextType)
        ):
            batch = self.batch_shape
            rns = (
                isinstance(self.value.type, ckks.CiphertextType)
                or self.representation == "rns"
            )
            shape = (
                *(
                    (self.component_count,)
                    if isinstance(self.value.type, ckks.CiphertextType)
                    else ()
                ),
                *batch,
                *((len(self.prime_ids),) if rns else ()),
                self.ring_dimension,
            )
            state["shape"] = cast(Attribute, state_attribute(shape))
        return state

    def _with_resident_tensors(self, tensors):
        payload = tensors[0]
        typ = cast(OpenStateType, self.value.type)
        state = dict(typ.state.data)
        physical = payload._tensor_state()
        for name in ("shape", "strides", "dtype", "device", "requires_grad"):
            state.pop(name, None)
            if name in physical:
                state[name] = physical[name]
        shape = payload.shape
        ciphertext = isinstance(typ, ckks.CiphertextType)
        rns = ciphertext or self.representation == "rns"
        batch = shape[1:-2] if ciphertext else shape[:-2] if rns else shape[:-1]
        state["batch_shape"] = cast(Attribute, state_attribute(batch))
        operation, result = UnrealizedConversionCastOp.cast_one(
            payload.value, type(typ)().with_state(state)
        )
        self.capture.block.add_op(operation)
        value = _Value(result, self.capture)
        if len(tensors) == 2:
            value._implicit_value = tensors[1]
        return value

    def __bool__(self) -> bool:
        raise CaptureError(
            "Captured computation cannot branch on Tensor or ciphertext data"
        )

    def __getattr__(self, name: str) -> Any:
        if name in {
            "is_rns",
            "is_slots",
            "is_integer_coefficients",
            "is_approximate_coefficients",
        }:
            return self.representation == name.removeprefix("is_")
        if name == "limb_count":
            return len(self.prime_ids)
        if name == "batch_size":
            return math.prod(self.batch_shape)
        if name == "ndim":
            return len(self.shape)
        if name == "is_batched":
            return bool(self.batch_shape)
        if name == "implicit_data":
            return self.__dict__.get("_implicit_value")
        if name == "_resident_tensors":
            implicit = self.__dict__.get("_implicit_value")
            return (self.data,) if implicit is None else (self.data, implicit)
        if name == "data":
            state = DictionaryAttr(self._tensor_state())
            operation, value = UnrealizedConversionCastOp.cast_one(
                self.value, core.MessageType.new((state,))
            )
            owner = self.value.owner
            block = owner if isinstance(owner, Block) else owner.parent_block()
            if block is None:
                raise CaptureError(
                    "Tensor access requires an attached captured value"
                )
            block.add_op(operation)
            return _Value(value, self.capture)
        field = {"modulus_basis": "basis", "component_count": "components"}.get(
            name, name
        )
        attribute = (
            self._tensor_state().get(field)
            if field == "shape"
            else cast(OpenStateType, self.value.type).state.data.get(field)
        )
        if isinstance(attribute, (FloatAttr, IntegerAttr)):
            return attribute.value.data
        if isinstance(attribute, StringAttr):
            return attribute.data
        if isinstance(attribute, ArrayAttr):
            return tuple(
                item.value.data
                for item in attribute
                if isinstance(item, IntegerAttr)
            )
        raise CaptureError(
            f"Captured CKKS values do not expose {name!r}; use Engine operations "
            "rather than inspecting or mutating their Tensor payloads"
        )


def _result_type(source: _Value, **updates: object) -> OpenStateType:
    typ = cast(OpenStateType, source.value.type)
    # Output allocation can change physical strides. Mathematical batch shape,
    # dtype, device, and ring metadata remain usable by later passes.
    fields = {
        name: value
        for name, value in typ.state.data.items()
        if name not in {"shape", "strides", "requires_grad"}
    }
    for name, value in updates.items():
        attribute = (
            value if isinstance(value, Attribute) else state_attribute(value)
        )
        if attribute is not None:
            fields[name] = attribute
    return type(typ)().with_state(fields)


def _symbolic(value: object) -> _Value:
    if not isinstance(value, _Value):
        raise CaptureError(
            "Pass computed ciphertext/plaintext operands as function arguments"
        )
    return value


def _float(value: object) -> float:
    if type(value) not in {int, float}:
        raise CaptureError("This operation requires a static real scalar")
    return float(cast(float, value))


_METHODS = frozenset(
    {
        "add",
        "subtract",
        "negate",
        "multiply",
        "add_plaintext",
        "multiply_plaintext",
        "prepare_compressed_plaintext",
        "sum_plaintext_products",
        "sum_plaintext_product_groups",
        "sum_rotated_plaintext_product_groups",
        "zero_plaintext_like",
        "add_scalar",
        "multiply_scalar",
        "multiply_integer_scalar",
        "coefficient_domain_to_ntt_domain",
        "ntt_domain_to_coefficient_domain",
        "standard_residues_to_montgomery_residues",
        "montgomery_residues_to_standard_residues",
        "rescale_to_next_depth",
        "mod_switch_to_depth",
        "mod_switch_to_next_depth",
        "reinterpret_at_scale",
        "relinearize",
        "conjugate",
        "switch_key",
        "rotate_by_step",
        "rotate_with_key",
        "rotate_many_by_steps",
        "sum_ciphertexts",
        "sum_ciphertext_batch",
        "rotate_many_with_keys",
        "encode",
        "integer_coefficients_to_rns",
        "prepare_plaintext_for_addition",
        "prepare_plaintext_for_multiplication",
    }
)


class _EngineCalls:
    def __init__(self, engine: Engine, capture: _Capture) -> None:
        self.engine = engine
        self.capture = capture

    def __getattr__(self, name: str) -> object:
        if name in {
            "config",
            "ring_dimension",
            "num_slots",
            "max_depth",
            "rescale_divisor",
            "rescale_output_scale",
            "depth_remaining",
        }:
            return getattr(self.engine, name)
        if name not in _METHODS:
            raise CaptureError(
                f"Engine.{name} is outside the callable frontend's operation set; "
                "keep lifecycle work outside the computation or supply a Program"
            )
        method = getattr(Engine, name)

        def emit(*args: object, **kwargs: object) -> object:
            bound = inspect.signature(method).bind(self.engine, *args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            parameters.pop("self")
            if parameters.pop("inplace", False):
                raise CaptureError(
                    "Callable capture requires out-of-place Engine operations"
                )
            before = self.capture.block.last_op
            result = self._emit(name, parameters)
            operation = (
                self.capture.block.first_op
                if before is None
                else before.next_op
            )
            configuration = StringAttr(
                json.dumps(self.engine.config.dumps(), sort_keys=True)
            )
            while operation is not None:
                if operation.name.startswith("fhelium_ckks."):
                    operation.attributes["ckks_config"] = configuration
                operation = operation.next_op
            return result

        return emit

    def _emit(self, name: str, p: dict[str, object]) -> object:
        c = self.capture
        if name == "zero_plaintext_like":
            source = cast(_Value, p["plaintext"])
            return source._with_resident_tensors(
                tuple(
                    c.numeric_call(torch.zeros_like, (t,), {})
                    for t in source._resident_tensors
                )
            )
        if name == "sum_plaintext_products":
            ciphertexts = cast(Sequence[_Value], p["ciphertexts"])
            plaintexts = cast(Sequence[_Value], p["plaintexts"])
            if not ciphertexts or len(ciphertexts) != len(plaintexts):
                raise CaptureError(
                    "Weighted sum requires matching nonempty operands"
                )
            products = [
                self._emit(
                    "multiply_plaintext", {"ciphertext": x, "plaintext": y}
                )
                for x, y in zip(ciphertexts, plaintexts, strict=True)
            ]
            result = products[0]
            for item in products[1:]:
                result = self._emit("add", {"lhs": result, "rhs": item})
            return result
        if name == "sum_plaintext_product_groups":
            groups = cast(Sequence[Sequence[_Value]], p["plaintext_groups"])
            if not groups:
                raise CaptureError("Weighted groups must be nonempty")
            group_outputs = [
                self._emit(
                    "sum_plaintext_products",
                    {"ciphertexts": p["ciphertexts"], "plaintexts": group},
                )
                for group in groups
            ]
            return Ciphertext.stack_batch(cast(Any, group_outputs))
        if name == "sum_rotated_plaintext_product_groups":
            keys = cast(Sequence[Any], p["rotation_keys"])
            if not keys or sum(key is None for key in keys) > 1:
                raise CaptureError(
                    "Rotation group requires keys and at most one unrotated entry"
                )
            source = p["ciphertext"]
            rotated = iter(
                cast(
                    Sequence[_Value],
                    self._emit(
                        "rotate_many_with_keys",
                        {
                            "value": source,
                            "keys": [key for key in keys if key is not None],
                            "output_domain": "ntt",
                            "use_hoisting": True,
                        },
                    ),
                )
            )
            terms = [
                self._emit(
                    "coefficient_domain_to_ntt_domain", {"value": source}
                )
                if key is None
                else next(rotated)
                for key in keys
            ]
            return self._emit(
                "sum_plaintext_product_groups",
                {
                    "ciphertexts": terms,
                    "plaintext_groups": p["plaintext_groups"],
                },
            )
        if name in {
            "standard_residues_to_montgomery_residues",
            "montgomery_residues_to_standard_residues",
        } and isinstance(
            cast(_Value, p.get("plaintext", p.get("value"))).value.type,
            ckks.CompressedPlaintextType,
        ):
            source = cast(_Value, p.get("plaintext", p.get("value")))
            to_montgomery = name == "standard_residues_to_montgomery_residues"
            if not to_montgomery and source.polynomial_domain != "coefficient":
                raise CaptureError(
                    "Standard compressed plaintext requires coefficient-domain data"
                )
            op_type = (
                rns.StandardToMontgomeryOp
                if to_montgomery
                else rns.MontgomeryToStandardOp
            )
            parameters = c.material(
                self.engine._dispatcher_for(
                    torch.device(source.device)
                ).rns_context.rns_parameters_for_prime_ids(source.prime_ids),
                rns.RnsParametersType(),
            )
            outputs = []
            for index, tensor in enumerate(source._resident_tensors):
                operand = (
                    tensor if index == 0 else tensor.reshape(*tensor.shape, 1)
                )
                physical = operand._tensor_state()
                state = {
                    key: physical[key]
                    for key in (
                        "shape",
                        "strides",
                        "dtype",
                        "device",
                        "requires_grad",
                    )
                    if key in physical
                }
                state.update(
                    {
                        "prime_ids": cast(
                            Attribute, state_attribute(source.prime_ids)
                        ),
                        "basis": StringAttr(source.modulus_basis),
                        "polynomial_domain": StringAttr(
                            source.polynomial_domain
                        ),
                        "residue_representation": StringAttr(
                            source.residue_representation
                        ),
                    }
                )
                bundle_type = rns.RnsBundleType().with_state(state)
                cast_op, bundle = UnrealizedConversionCastOp.cast_one(
                    operand.value, bundle_type
                )
                c.block.add_op(cast_op)
                result_type = bundle_type.with_state(
                    {
                        "residue_representation": StringAttr(
                            "montgomery" if to_montgomery else "standard"
                        )
                    }
                )
                result = c.emit(
                    op_type.create(
                        operands=[bundle, parameters],
                        result_types=[result_type],
                    )
                )
                outputs.append(
                    result if index == 0 else result.reshape(*tensor.shape)
                )
            result = source._with_resident_tensors(tuple(outputs))
            result.value = Rewriter.replace_value_with_new_type(
                result.value,
                cast(OpenStateType, result.value.type).with_state(
                    {
                        "residue_representation": StringAttr(
                            "montgomery" if to_montgomery else "standard"
                        )
                    }
                ),
            )
            return result
        if name == "prepare_compressed_plaintext":
            from fhelium.backend.ckks.codec._periodic import periodic_extent

            source = cast(_Value, c.substitute(p["message"]))
            target = torch.device(
                torch.get_default_device()
                if p["device"] is None
                else cast(torch.device | str, p["device"])
            )
            if torch.device(source.device) != target:
                source = c.numeric_call(
                    torch.Tensor.to, (source,), {"device": target}
                )
            if not source.shape:
                source = source.reshape(1)
            period = source.shape[-1]
            extent = periodic_extent(
                period,
                self.engine.config.N,
                self.engine.config.galois_generator,
            )
            depth = int(cast(int, p["depth"]))
            scale = coerce_scale(
                self.engine.config.default_scale
                if p["scale"] is None
                else p["scale"],
                value_name="encode scale",
            )
            basis = cast(ModulusBasis, p["modulus_basis"])
            domain = p["polynomial_domain"]
            if domain not in ("coefficient", "ntt"):
                raise CaptureError(
                    "polynomial_domain must be coefficient or ntt"
                )
            resources = self.engine._dispatcher_for(target).device_resources
            ids = resources.rns_context.rns_layout.prime_ids(
                depth, include_p=basis == "QP"
            )
            tables = resources.periodic_encode_operands(period, depth, basis)
            parameters = [
                c.material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config,
                        "periodic_plaintext_table",
                        period=period,
                        depth=depth,
                        basis=basis,
                        index=index,
                    ),
                )
                for index, tensor in enumerate(tables)
            ]
            state = {
                "depth": depth,
                "scale": scale,
                "prime_ids": ids,
                "polynomial_domain": domain,
                "basis": basis,
                "residue_representation": "montgomery",
                "representation": "rns",
                "compression_layout": "strided_sparse"
                if domain == "coefficient"
                else "contiguous",
                "ring_dimension": self.engine.config.N,
                "unique_count": extent,
                "batch_shape": tuple(source.shape[:-1]),
                "shape": (*source.shape[:-1], len(ids), extent),
                "dtype": str(resources.rns_context.dtype),
                "device": str(target),
            }
            typ = ckks.CompressedPlaintextType().with_state(
                {
                    key: cast(Attribute, state_attribute(value))
                    for key, value in state.items()
                }
            )
            implicit_type = core.MessageType().with_state(
                {
                    key: cast(Attribute, state_attribute(value))
                    for key, value in {
                        "shape": (*source.shape[:-1], len(ids), 1),
                        "dtype": str(resources.rns_context.dtype),
                        "device": str(target),
                    }.items()
                }
            )
            operation = ckks.PrepareCompressedPlaintextOp.create(
                operands=[source.value, *parameters],
                result_types=[typ, implicit_type]
                if domain == "coefficient"
                else [typ],
                attributes={
                    "depth": IntegerAttr(depth, 64),
                    "polynomial_domain": StringAttr(domain),
                    "scale": FloatAttr(scale, f64),
                    "ring_dimension": IntegerAttr(self.engine.config.N, 64),
                    "min_modulus": IntegerAttr(
                        min(self.engine.config.moduli[i] for i in ids), 64
                    ),
                },
            )
            c.block.add_op(operation)
            result = _Value(operation.results[0], c)
            if domain == "coefficient":
                implicit = _Value(operation.results[1], c)
                result._implicit_value = implicit.reshape(
                    *source.shape[:-1], len(ids)
                )
            return result
        if name == "encode":
            source = cast(_Value, c.substitute(p["message"]))
            target = torch.device(
                torch.get_default_device()
                if p["device"] is None
                else cast(torch.device | str, p["device"])
            )
            if torch.device(source.device) != target:
                source = c.numeric_call(
                    torch.Tensor.to, (source,), {"device": target}
                )
            resources = self.engine._dispatcher_for(target).device_resources
            depth = int(cast(int, p["depth"]))
            scale = float(
                self.engine.config.default_scale
                if p["scale"] is None
                else cast(float, p["scale"])
            )
            parameters = tuple(
                c.material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config, "encode_table", index=index
                    ),
                )
                for index, tensor in enumerate(resources.encode_operands())
            )
            shape = (*source.shape[:-1], self.engine.ring_dimension)
            typ = ckks.PlaintextType().with_state(
                {
                    key: state_attribute(value)
                    for key, value in {
                        "depth": depth,
                        "scale": scale,
                        "representation": "integer_coefficients",
                        "polynomial_domain": "coefficient",
                        "prime_ids": (),
                        "dtype": "torch.int64",
                        "device": str(target),
                        "shape": shape,
                        "batch_shape": shape[:-1],
                        "ring_dimension": self.engine.ring_dimension,
                    }.items()
                }
            )
            return c.emit(
                ckks.EncodeOp.create(
                    operands=(source.value, *parameters),
                    result_types=(typ,),
                    attributes={
                        "depth": IntegerAttr(depth, 64),
                        "scale": FloatAttr(scale, f64),
                    },
                )
            )
        if name in {
            "integer_coefficients_to_rns",
            "prepare_plaintext_for_addition",
            "prepare_plaintext_for_multiplication",
        }:
            source = _symbolic(p["plaintext"])
            basis = cast(ModulusBasis, p["modulus_basis"])
            if source.representation != "integer_coefficients":
                raise CaptureError(
                    "Plaintext preparation requires integer-coefficient input, as in Eager"
                )
            rows = depth_prime_ids(self.engine.config, int(source.depth), basis)
            resources = self.engine._dispatcher_for(
                torch.device(source.device)
            ).device_resources
            parameters = resources.rns_context.rns_parameters_for_prime_ids(
                rows
            )
            twice_modulus = c.material(
                parameters[0],
                core.MessageType(),
                parameter_description(
                    self.engine.config, "twice_modulus", prime_ids=list(rows)
                ),
            )
            typ = ckks.PlaintextType().with_state(
                {
                    **cast(OpenStateType, source.value.type).state.data,
                    "representation": StringAttr("rns"),
                    "basis": StringAttr(basis),
                    "residue_representation": StringAttr("standard"),
                    "prime_ids": state_attribute(rows),
                    "dtype": StringAttr(str(self.engine.dtype)),
                    "shape": state_attribute(
                        (
                            *source.shape[:-1],
                            len(rows),
                            self.engine.ring_dimension,
                        )
                    ),
                }
            )
            value = c.emit(
                ckks.IntegerCoefficientsToRnsOp.create(
                    operands=(source.value, twice_modulus),
                    result_types=(typ,),
                    attributes={
                        "depth": IntegerAttr(int(source.depth), 64),
                        "modulus_basis": StringAttr(basis),
                        "min_modulus": IntegerAttr(
                            min(self.engine.config.moduli[i] for i in rows), 64
                        ),
                    },
                )
            )
            if name == "integer_coefficients_to_rns":
                return value
            value = cast(
                _Value,
                self._emit(
                    "standard_residues_to_montgomery_residues", {"value": value}
                ),
            )
            if (
                name == "prepare_plaintext_for_multiplication"
                or p.get("polynomial_domain") == "ntt"
            ):
                value = cast(
                    _Value,
                    self._emit(
                        "coefficient_domain_to_ntt_domain", {"value": value}
                    ),
                )
            return value
        if name == "sum_ciphertext_batch":
            batch = _symbolic(p["batch"])
            dim = int(cast(int, p["dim"]))
            logical_dim = dim if dim >= 0 else dim + len(batch.batch_shape)
            if not 0 <= logical_dim < len(batch.batch_shape):
                raise CaptureError(
                    "sum_ciphertext_batch requires an existing batch axis"
                )
            if batch.batch_shape[logical_dim] == 1:
                return batch.select_batch(0, dim=logical_dim)
            return self._emit(
                "sum_batch", {"batch": batch, "axis": logical_dim}
            )
        if name == "rotate_many_with_keys":
            source = _symbolic(p["value"])
            keys = cast(Sequence[Any], p["keys"])
            domain = cast(PolynomialDomain, p["output_domain"])
            if not keys:
                return []
            if not p["use_hoisting"]:
                return [
                    self._rotate(
                        source, int(key.rotation_step), domain, key=key
                    )
                    for key in keys
                ]
            typ = cast(
                ckks.CiphertextType,
                _result_type(
                    source,
                    polynomial_domain=domain,
                    residue_representation=output_residues(domain),
                ),
            )
            parameters, attributes = self._key_parameters(source)
            attributes["rotation_steps"] = ArrayAttr(
                IntegerAttr(int(key.rotation_step), 64) for key in keys
            )
            operation = ckks.RotateManyOp(
                source.value,
                tuple(
                    c.key_reference(
                        key, role="rotation", step=int(key.rotation_step)
                    )
                    for key in keys
                ),
                tuple(typ for _ in keys),
                parameters=parameters,
                output_domain=domain,
                attributes=attributes,
            )
            c.block.add_op(operation)
            return [_Value(value, c) for value in operation.results]
        if name == "sum_ciphertexts":
            values = cast(Sequence[object], p["ciphertexts"])
            if not values:
                raise CaptureError(
                    "sum_ciphertexts requires at least one value"
                )
            result = self._mod_switch(
                _symbolic(values[0]), int(_symbolic(values[0]).depth)
            )
            for rhs in values[1:]:
                result = cast(
                    _Value, self._emit("add", {"lhs": result, "rhs": rhs})
                )
            return result
        if name == "rotate_many_by_steps":
            source = _symbolic(p["value"])
            steps = [
                RotationKey.normalize_step(
                    integer_index(step),
                    ring_dimension=self.engine.ring_dimension,
                )
                for step in cast(Sequence[int], p["rotation_steps"])
            ]
            if not p["use_hoisting"] or any(
                len(self.engine.rotation_key_steps(step)) > 1
                for step in steps
                if step
            ):
                return [
                    self._rotate(source, step, "coefficient") for step in steps
                ]
            nonzero = [step for step in steps if step]
            outputs: list[_Value] = []
            if nonzero:
                typ = cast(
                    ckks.CiphertextType,
                    _result_type(
                        source,
                        polynomial_domain="coefficient",
                        residue_representation=output_residues("coefficient"),
                    ),
                )
                parameters, attributes = self._key_parameters(source)
                attributes["rotation_steps"] = ArrayAttr(
                    IntegerAttr(step, 64) for step in nonzero
                )
                op = ckks.RotateManyOp(
                    source.value,
                    tuple(
                        c.key_reference(
                            self.engine._keys.rotation_keys.get(step),
                            role="rotation",
                            step=step,
                        )
                        for step in nonzero
                    ),
                    tuple(typ for _ in nonzero),
                    parameters=parameters,
                    attributes=attributes,
                )
                c.block.add_op(op)
                outputs = [_Value(result, c) for result in op.results]
            results = iter(outputs)
            return [
                next(results)
                if step
                else self._mod_switch(source, int(source.depth))
                for step in steps
            ]

        operands = [item for item in p.values() if isinstance(item, _Value)]
        if not operands:
            raise CaptureError(
                f"Engine.{name} requires a symbolic CKKS operand"
            )
        source = operands[0]
        typ = _result_type(source)
        attrs: dict[str, Attribute] = {}
        op_type: type[Operation]
        if name == "sum_batch":
            axis = int(cast(int, p["axis"]))
            batch_shape = source.batch_shape
            typ = _result_type(
                source,
                batch_shape=batch_shape[:axis] + batch_shape[axis + 1 :],
            )
            attrs["axis"] = IntegerAttr(axis, 64)
            op_type = ckks.SumBatchOp
        elif name in {"add", "subtract", "negate"}:
            op_type = {
                "add": ckks.AddOp,
                "subtract": ckks.SubtractOp,
                "negate": ckks.NegateOp,
            }[name]
            if name != "negate":
                self._matching_operands(source, _symbolic(p["rhs"]), scale=True)
        elif name in {"multiply", "multiply_plaintext", "add_plaintext"}:
            rhs = _symbolic(p["rhs"] if name == "multiply" else p["plaintext"])
            self._matching_operands(
                source,
                rhs,
                scale=name == "add_plaintext",
                plaintext=name != "multiply",
            )
            if isinstance(rhs.value.type, ckks.CompressedPlaintextType):
                if rhs.residue_representation != "montgomery":
                    raise CaptureError(
                        "Compressed plaintext arithmetic requires Montgomery residues"
                    )
                if (
                    name == "multiply_plaintext"
                    and rhs.polynomial_domain != "ntt"
                ):
                    raise CaptureError(
                        "Compressed plaintext multiplication requires NTT data"
                    )
            if name == "add_plaintext":
                if isinstance(
                    rhs.value.type, ckks.CompressedPlaintextType
                ) and (
                    rhs.polynomial_domain != source.polynomial_domain
                    or source.residue_representation
                    != (
                        "montgomery"
                        if source.polynomial_domain == "ntt"
                        else "standard"
                    )
                ):
                    raise CaptureError(
                        "Compressed addition requires matching domains and a Montgomery plaintext"
                    )
                op_type = ckks.AddPlaintextOp
                if isinstance(rhs.value.type, ckks.CompressedPlaintextType):
                    op_type = ckks.AddCompressedPlaintextOp
                    attrs["compression_layout"] = StringAttr(
                        str(rhs.compression_layout)
                    )
                    if rhs.implicit_data is not None:
                        operands.append(
                            rhs.implicit_data.reshape(
                                *rhs.implicit_data.shape, 1
                            )
                        )
                    attrs["polynomial_domain"] = StringAttr(
                        source.polynomial_domain
                    )
            else:
                op_type = (
                    ckks.MultiplyOp
                    if name == "multiply"
                    else ckks.MultiplyPlaintextOp
                )
                typ = _result_type(
                    source,
                    scale=product_scale(float(source.scale), float(rhs.scale)),
                    **({"components": 3} if name == "multiply" else {}),
                )
                if name == "multiply_plaintext" and isinstance(
                    rhs.value.type, ckks.CompressedPlaintextType
                ):
                    op_type = ckks.MultiplyCompressedPlaintextOp
                    if rhs.implicit_data is not None:
                        operands.append(
                            rhs.implicit_data.reshape(
                                *rhs.implicit_data.shape, 1
                            )
                        )
                    attrs["compression_layout"] = StringAttr(
                        str(rhs.compression_layout)
                    )
        elif name in {
            "add_scalar",
            "multiply_scalar",
            "multiply_integer_scalar",
        }:
            if name == "multiply_integer_scalar":
                op_type = ckks.MultiplyIntegerScalarOp
                attrs["scalar"] = IntegerAttr(
                    integer_index(cast(int, p["scalar"])), 64
                )
            else:
                scalar = _float(p["scalar"])
                scale = coerce_scale(
                    source.scale
                    if p["scalar_scale"] is None
                    else p["scalar_scale"],
                    value_name="scalar scale",
                )
                attrs.update(
                    scalar=FloatAttr(scalar, f64),
                    scalar_scale=FloatAttr(scale, f64),
                )
                op_type = (
                    ckks.AddScalarOp
                    if name == "add_scalar"
                    else ckks.MultiplyScalarOp
                )
                if name == "multiply_scalar":
                    typ = _result_type(
                        source,
                        scale=coerce_scale(
                            product_scale(float(source.scale), scale),
                            value_name="multiply_scalar result",
                        ),
                    )
        elif name in {
            "coefficient_domain_to_ntt_domain",
            "ntt_domain_to_coefficient_domain",
        }:
            forward = name == "coefficient_domain_to_ntt_domain"
            op_type = ckks.ToNttOp if forward else ckks.FromNttOp
            domain, residues = transform_state(
                "forward" if forward else "inverse",
                ciphertext=isinstance(source.value.type, ckks.CiphertextType),
            )
            typ = _result_type(
                source,
                polynomial_domain=domain,
                residue_representation=residues,
            )
        elif name in {
            "standard_residues_to_montgomery_residues",
            "montgomery_residues_to_standard_residues",
        }:
            to_montgomery = name == "standard_residues_to_montgomery_residues"
            op_type = (
                ckks.ToMontgomeryResiduesOp
                if to_montgomery
                else ckks.ToStandardResiduesOp
            )
            typ = _result_type(
                source,
                residue_representation="montgomery"
                if to_montgomery
                else "standard",
            )
        elif name in {"mod_switch_to_depth", "mod_switch_to_next_depth"}:
            target = (
                integer_index(cast(int, p["target_depth"]))
                if name == "mod_switch_to_depth"
                else int(source.depth) + 1
            )
            return self._mod_switch(source, target)
        elif name == "rescale_to_next_depth":
            config = self.engine.config
            depth = int(source.depth)
            rows = depth_prime_ids(
                config, depth + 1, cast(ModulusBasis, source.modulus_basis)
            )
            typ = _result_type(
                source,
                depth=depth + 1,
                prime_ids=rows,
                scale=quotient_scale(
                    float(source.scale), config.rescale_divisor(depth)
                ),
            )
            dispatcher = self.engine._dispatcher_for(
                torch.device(source.device)
            )
            tensors, facts = dispatcher.device_resources.rescale_operands(
                len(source.prime_ids),
                len(config.q_depth_groups[depth]),
                include_p=source.modulus_basis == "QP",
                input_domain=source.polynomial_domain,
            )
            attrs = {
                name: attribute
                for name, value in {
                    **facts,
                    "parameter_names": tuple(tensors),
                }.items()
                if (attribute := state_attribute(value)) is not None
            }
            return c.emit(
                ckks.RescaleOp(
                    source.value,
                    typ,
                    parameters=tuple(
                        c.material(
                            tensor,
                            core.MessageType(),
                            parameter_description(
                                config,
                                "rescale_table",
                                depth=depth,
                                basis=source.modulus_basis,
                                input_domain=source.polynomial_domain,
                                ntt_backend=facts.get("ntt_backend"),
                                name=name,
                            ),
                        )
                        for name, tensor in tensors.items()
                    ),
                    attributes=attrs,
                    rounding=cast(str, p["rounding"]),
                    polynomial_domain=cast(
                        PolynomialDomain, source.polynomial_domain
                    ),
                )
            )
        elif name == "reinterpret_at_scale":
            scale = reinterpreted_scale(
                float(source.scale),
                _float(p["target_scale"]),
                cast(float | None, p["max_relative_change"]),
            )
            op_type = ckks.ReinterpretScaleOp
            attrs["scale"] = FloatAttr(scale, f64)
            typ = _result_type(source, scale=scale)
        elif name in {"relinearize", "conjugate", "switch_key"}:
            key = p["key"]
            if key is None:
                key = getattr(
                    self.engine._keys,
                    "relinearization_key"
                    if name == "relinearize"
                    else "conjugation_key",
                    None,
                )
            domain = cast(PolynomialDomain, p["output_domain"])
            typ = _result_type(
                source,
                polynomial_domain=domain,
                residue_representation=output_residues(domain),
                components=2,
            )
            op_type = {
                "relinearize": ckks.RelinearizeOp,
                "conjugate": ckks.ConjugateOp,
                "switch_key": ckks.SwitchKeyOp,
            }[name]
            parameters, attrs = self._key_parameters(source)
            attrs["output_domain"] = StringAttr(domain)
            key_value = c.key_reference(key, role=name)
            return c.emit(
                op_type.create(
                    operands=(source.value, key_value, *parameters),
                    result_types=(typ,),
                    attributes=attrs,
                )
            )
        elif name in {"rotate_by_step", "rotate_with_key"}:
            if name == "rotate_with_key":
                key = p["key"]
                if isinstance(key, RotationKey):
                    step = key.rotation_step
                elif isinstance(key, _Value) and isinstance(
                    key.value.type, ckks.EvaluationKeyType
                ):
                    step = int(key.rotation_step)
                else:
                    raise CaptureError(
                        "rotate_with_key requires a rotation key"
                    )
                return self._rotate(
                    source,
                    step,
                    cast(PolynomialDomain, p["output_domain"]),
                    key=key,
                )
            step = RotationKey.normalize_step(
                integer_index(cast(int, p["rotation_step"])),
                ring_dimension=self.engine.ring_dimension,
            )
            return self._rotate(source, step, "coefficient")
        else:
            raise CaptureError(f"Unsupported Engine method {name}")
        tensor_operands = [value.value for value in operands]
        if op_type in {
            ckks.AddOp,
            ckks.SumBatchOp,
            ckks.SubtractOp,
            ckks.NegateOp,
            ckks.MultiplyOp,
            ckks.AddPlaintextOp,
            ckks.MultiplyPlaintextOp,
            ckks.MultiplyCompressedPlaintextOp,
            ckks.AddCompressedPlaintextOp,
            ckks.ToMontgomeryResiduesOp,
            ckks.ToStandardResiduesOp,
        }:
            source = operands[0]
            context = self.engine._dispatcher_for(
                torch.device(source.device)
            ).rns_context
            parameters = context.rns_parameters_for_prime_ids(source.prime_ids)
            tensor_operands.append(
                c.material(
                    parameters,
                    rns.RnsParametersType(),
                    parameter_description(
                        self.engine.config,
                        "rns_parameters",
                        prime_ids=list(source.prime_ids),
                    ),
                )
            )
        elif op_type in {
            ckks.AddScalarOp,
            ckks.MultiplyScalarOp,
            ckks.MultiplyIntegerScalarOp,
        }:
            from fhelium.backend.ckks.scalar import scalar_operands

            source = operands[0]
            dispatcher = self.engine._dispatcher_for(
                torch.device(source.device)
            )
            integer = op_type is ckks.MultiplyIntegerScalarOp
            tensors, facts = scalar_operands(
                dispatcher.rns_context,
                source.prime_ids,
                cast(int | float, p["scalar"]),
                None
                if integer
                else cast(FloatAttr, attrs["scalar_scale"]).value.data,
                None if integer else dispatcher.rng.rounding_state,
            )
            tensor_operands.extend(
                c.material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config,
                        "rns_parameters",
                        prime_ids=list(source.prime_ids),
                    )
                    if index == 0
                    else parameter_description(
                        self.engine.config,
                        "integer_scalar",
                        prime_ids=list(source.prime_ids),
                        scalar=p["scalar"],
                    )
                    if integer
                    else parameter_description(
                        self.engine.config, "rounding_state"
                    ),
                )
                for index, tensor in enumerate(tensors)
            )
            attrs.update(
                {
                    name: attribute
                    for name, value in facts.items()
                    if (attribute := state_attribute(value)) is not None
                }
            )
        elif op_type in {ckks.ToNttOp, ckks.FromNttOp}:
            source = operands[0]
            context = self.engine._dispatcher_for(
                torch.device(source.device)
            ).ntt_context
            tables = context.tensor_operands(
                source.prime_ids, inverse=op_type is ckks.FromNttOp
            )
            tensor_operands.extend(
                c.material(
                    table,
                    rns.RnsParametersType()
                    if index == 0
                    else core.MessageType(),
                    parameter_description(
                        self.engine.config,
                        "rns_parameters",
                        prime_ids=list(source.prime_ids),
                    )
                    if index == 0
                    else parameter_description(
                        self.engine.config,
                        "ntt_table",
                        prime_ids=list(source.prime_ids),
                        direction="inverse"
                        if op_type is ckks.FromNttOp
                        else "forward",
                        index=index,
                        ntt_backend=context.ntt_backend_name,
                    ),
                )
                for index, table in enumerate(tables)
            )
            from fhelium.backend.ntt._selection import table_layout

            attrs["ntt_table_layout"] = StringAttr(
                table_layout(context.ntt_policy)
            )
            if self.engine._ntt_backend is not None:
                attrs["ntt_backend"] = StringAttr(context.ntt_backend_name)
        return c.emit(
            op_type.create(
                operands=tensor_operands,
                result_types=[typ],
                attributes=attrs,
            )
        )

    @staticmethod
    def _matching_operands(
        lhs: _Value, rhs: _Value, *, scale: bool, plaintext: bool = False
    ) -> None:
        fields = ("depth", "prime_ids", "modulus_basis")
        if not plaintext:
            fields += (
                "polynomial_domain",
                "residue_representation",
                "component_count",
            )
        if scale:
            fields += ("scale",)
        for field in fields:
            if getattr(lhs, field) != getattr(rhs, field):
                raise CaptureError(
                    f"Operands disagree on {field}; capture does not insert alignment"
                )

    def _mod_switch(self, source: _Value, depth: int) -> _Value:
        if depth < int(source.depth):
            raise CaptureError("ModSwitch cannot restore removed Q rows")
        rows = depth_prime_ids(
            self.engine.config, depth, cast(ModulusBasis, source.modulus_basis)
        )
        typ = _result_type(source, depth=depth, prime_ids=rows)
        return self.capture.emit(
            ckks.ModSwitchOp(
                source.value,
                typ,
                attributes={"target_depth": IntegerAttr(depth, 64)},
            )
        )

    def _key_parameters(
        self, source: _Value
    ) -> tuple[tuple[SSAValue, ...], dict[str, Attribute]]:
        dispatcher = self.engine._dispatcher_for(torch.device(source.device))
        tensors, facts = dispatcher.device_resources.key_switch_operands(
            int(source.depth)
        )
        attributes = {
            name: attribute
            for name, value in {
                **facts,
                "parameter_names": tuple(tensors),
            }.items()
            if (attribute := state_attribute(value)) is not None
        }
        from fhelium.backend.rns._preparation import (
            table_description,
        )

        operands = []
        for name, tensor in tensors.items():
            description = table_description(
                self.engine.config, int(source.depth), name, facts
            )
            kind = cast(str, description.pop("kind"))
            operands.append(
                self.capture.material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config, kind, **description
                    ),
                )
            )
        return tuple(operands), attributes

    def _rotate(
        self,
        source: _Value,
        step: int,
        domain: PolynomialDomain,
        *,
        key: object = None,
    ) -> _Value:
        if step == 0:
            return self._mod_switch(source, int(source.depth))
        if key is None:
            path = self.engine.rotation_key_steps(step)
            if len(path) > 1:
                for index, part in enumerate(path):
                    source = self._rotate(
                        source,
                        part,
                        domain if index == len(path) - 1 else "coefficient",
                    )
                return source
            key = self.engine._keys.rotation_keys.get(step)
        typ = _result_type(
            source,
            polynomial_domain=domain,
            residue_representation=output_residues(domain),
        )
        parameters, attributes = self._key_parameters(source)
        attributes.update(
            rotation_step=IntegerAttr(step, 64),
            input_domain=StringAttr(cast(str, source.polynomial_domain)),
            output_domain=StringAttr(domain),
        )
        return self.capture.emit(
            ckks.RotateOp(
                source.value,
                self.capture.key_reference(key, role="rotation", step=step),
                typ,
                parameters=parameters,
                attributes=attributes,
            )
        )


class _Capture:
    def __init__(
        self,
        workspace: CompileWorkspace,
        block: Block,
        material_names: Mapping[str, object] | None = None,
    ) -> None:
        self.workspace = workspace
        self.block = block
        self.engines: dict[int, _EngineCalls] = {}
        self.functions: dict[int, types.FunctionType] = {}
        self.materials: dict[str, torch.Tensor] = {}
        self.descriptions: dict[str, dict[str, object]] = {}
        self.names = named_material_tensors(material_names)
        self.tensor_symbols: dict[int, str] = {}
        self.material_refs: dict[tuple[int, Attribute], SSAValue] = {}

    def material(
        self,
        tensor: torch.Tensor,
        typ: OpenStateType,
        description: Mapping[str, object] | None = None,
    ) -> SSAValue:
        typ = typ.with_state(state_attributes(tensor))
        identity = (id(tensor), typ)
        symbol = self.tensor_symbols.get(id(tensor))
        if symbol is None:
            symbol = self.names.get(id(tensor), f"tensor{len(self.materials)}")
            reserved = (
                set(self.names.values())
                if id(tensor) not in self.names
                else set()
            )
            while symbol in self.materials or symbol in reserved:
                symbol += "_"
            self.tensor_symbols[id(tensor)] = symbol
            self.materials[symbol] = tensor
        prior = self.descriptions.get(symbol)
        if prior is None or prior.get("kind") == "Tensor":
            self.descriptions[symbol] = {
                "kind": "Tensor",
                **dict(description or {}),
                **(
                    {"label": self.names[id(tensor)]}
                    if id(tensor) in self.names
                    else {}
                ),
            }
        if identity in self.material_refs:
            return self.material_refs[identity]
        reference = core.MaterialRefOp(typ, symbol=symbol)
        self.block.add_op(reference)
        self.material_refs[identity] = reference.value
        return reference.value

    def emit(self, operation: Operation) -> _Value:
        self.block.add_op(operation)
        return _Value(operation.results[0], self)

    def key_reference(
        self, key: object, *, role: str, step: int | None = None
    ) -> SSAValue:
        if isinstance(key, _Value) and isinstance(
            key.value.type, ckks.EvaluationKeyType
        ):
            return key.value
        if isinstance(key, KeySwitchKey):
            return self.material(
                key.data,
                cast(OpenStateType, value_type(key)),
                key_description(key),
            )
        if key is not None:
            raise CaptureError(f"{role} requires an evaluation key")
        typ = ckks.EvaluationKeyType()
        if step is not None:
            typ = typ.with_state({"rotation_step": IntegerAttr(step, 64)})
        symbol = f"{role}_key_{len(tuple(self.block.ops))}"
        reference = core.MaterialRefOp(typ, symbol=symbol)
        self.block.add_op(reference)
        self.descriptions[symbol] = {
            "kind": {
                "rotation": "RotationKey",
                "relinearization": "RelinearizationKey",
                "conjugation": "ConjugationKey",
                "switch": "KeySwitchKey",
            }.get(role, role),
            **({"rotation_step": step} if step is not None else {}),
        }
        return reference.value

    def substitute(self, value: object) -> object:
        from .._callable import CompiledCallable

        if isinstance(value, CompiledCallable):
            if not isinstance(value._source, types.FunctionType):
                raise CaptureError(
                    "Only compiled helpers backed by a Python function can be source-inlined; "
                    "compose Program-backed helpers at the Program level"
                )
            return self.function(value._source)
        if isinstance(value, Engine):
            if id(value) not in self.engines:
                self.engines[id(value)] = _EngineCalls(value, self)
            return self.engines[id(value)]
        if isinstance(value, types.MethodType) and isinstance(
            value.__self__, Engine
        ):
            return getattr(self.substitute(value.__self__), value.__name__)
        if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
            if value.data is None:
                raise CaptureError(
                    "Prepare plaintext Tensor data before capturing its use"
                )
            result = _Value(self.material(value.data, value_type(value)), self)
            if (
                isinstance(value, CompressedPlaintext)
                and value.implicit_data is not None
            ):
                result._implicit_value = _Value(
                    self.material(value.implicit_data, core.MessageType()), self
                )
            return result
        if isinstance(value, torch.Tensor):
            return _Value(self.material(value, core.MessageType()), self)
        if isinstance(value, types.FunctionType):
            if (value.__module__ or "").split(".")[0] in {
                "torch",
                "numpy",
                "fhelium",
                "xdsl",
            }:
                return value
            return self.function(value)
        if isinstance(value, tuple):
            return tuple(self.substitute(item) for item in value)
        if isinstance(value, list):
            return [self.substitute(item) for item in value]
        if isinstance(value, dict):
            return {key: self.substitute(item) for key, item in value.items()}
        return value

    def numeric_call(self, function, args, kwargs):
        from ._tensor_capture import numeric_call

        return numeric_call(self, function, args, kwargs)

    def function(self, function: types.FunctionType) -> types.FunctionType:
        if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(
            function
        ):
            raise CaptureError(
                "Engine capture requires a synchronous Python function"
            )
        if id(function) in self.functions:
            return self.functions[id(function)]
        globals_ = dict(function.__globals__)
        closure = tuple(
            types.CellType(cell.cell_contents)
            for cell in function.__closure__ or ()
        )
        copied = types.FunctionType(
            function.__code__,
            globals_,
            function.__name__,
            None,
            closure or None,
        )
        self.functions[id(function)] = copied
        copied.__defaults__ = (
            tuple(self.substitute(value) for value in function.__defaults__)
            if function.__defaults__ is not None
            else None
        )
        copied.__kwdefaults__ = (
            {
                name: self.substitute(value)
                for name, value in function.__kwdefaults__.items()
            }
            if function.__kwdefaults__ is not None
            else None
        )
        global_names = {
            instruction.argval
            for instruction in dis.get_instructions(function)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        }
        for name in global_names:
            if name in globals_:
                globals_[name] = self.substitute(globals_[name])
        for cell in closure:
            cell.cell_contents = self.substitute(cell.cell_contents)
        return copied


def capture_eager(
    function: Callable[..., object],
    *,
    arguments: Mapping[str, object],
    workspace: CompileWorkspace | None = None,
    material_names: Mapping[str, object] | None = None,
) -> Compilation:
    """Capture a Python function as mixed public Tensor and encrypted dataflow.

    Runtime Tensor, ciphertext, plaintext and key arguments become SSA inputs.
    Python scalars are static parameters. Ordinary Tensor arithmetic remains
    ordinary Torch calls; encrypted numerical operators become semantic FHE
    operations. Supported Engine calls retain their concrete CKKS transitions
    and actual key/table Tensor operands. Capture does not encrypt public inputs
    or generate keys. Missing evaluation keys remain material placeholders.

    Fixed Tensor/value references, including nested list/tuple/dict containers,
    become live material bindings. Their contents are not constant-folded.
    Numerical data is never executed during capture: ordinary Tensor metadata
    is propagated with FakeTensor kernels. Explicit Engine codec preparation
    becomes runtime operations with advancing rounding state.

    The original function is retained as the source reference. Function-backed
    compiled helpers are source-inlined under the outer compilation choices.
    Static Python control flow and structured outputs are supported. Data-based
    Python branching, in-place operations and arbitrary callable objects require
    other frontend or manual Program support; no capture fallback is attempted.
    """
    if not isinstance(function, types.FunctionType):
        raise CaptureError(
            "Callable capture requires a Python function; use a Program for callable objects"
        )
    signature = inspect.signature(function)
    if any(
        parameter.kind
        in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in signature.parameters.values()
    ):
        raise CaptureError(
            "Captured functions cannot declare *args or **kwargs"
        )
    unknown = set(arguments) - set(signature.parameters)
    if unknown:
        raise CaptureError(f"Unknown callable arguments: {sorted(unknown)}")
    bound = signature.bind_partial()
    bound.arguments.update(arguments)
    bound.apply_defaults()
    missing = set(signature.parameters) - set(bound.arguments)
    if missing:
        raise CaptureError(f"Missing callable arguments: {sorted(missing)}")
    ordered = {name: bound.arguments[name] for name in signature.parameters}
    runtime = {
        name: value
        for name, value in ordered.items()
        if isinstance(
            value,
            (
                Ciphertext,
                Plaintext,
                CompressedPlaintext,
                torch.Tensor,
                KeySwitchKey,
            ),
        )
    }
    specs: dict[str, InputSpec] = {}
    for name, value in ordered.items():
        if isinstance(value, Ciphertext):
            specs[name] = encrypted(
                depth=value.depth,
                scale=value.scale,
                batch_mode="any" if value.batch_shape else "none",
                polynomial_domain=value.polynomial_domain,
                residue_representation=value.residue_representation,
            )
        elif isinstance(value, Plaintext):
            if value.representation != "rns":
                raise CaptureError(
                    "Prepare slot plaintexts outside the compiled computation"
                )
            specs[name] = plaintext()
        elif isinstance(value, CompressedPlaintext):
            specs[name] = plaintext()
        elif isinstance(value, (torch.Tensor, KeySwitchKey)):
            specs[name] = message()
        elif type(value) in {int, float, complex, bool, str, type(None)}:
            specs[name] = static(cast(StaticValue, value))
        else:
            raise CaptureError(
                f"Unsupported argument {name!r}; use the FX frontend for public Tensor computations"
            )
    physical = dict(runtime)
    input_fields = {}
    implicit_names = {}
    for name, value in runtime.items():
        if (
            isinstance(value, CompressedPlaintext)
            and value.implicit_data is not None
        ):
            extra = name + "_implicit_data"
            while extra in ordered or extra in physical:
                extra += "_"
            physical[extra] = value.implicit_data
            input_fields[extra] = [name, "implicit_data"]
            implicit_names[name] = extra
    block = Block(arg_types=[value_type(value) for value in physical.values()])
    work = CompileWorkspace() if workspace is None else workspace
    capture = _Capture(work, block, material_names)
    symbolic = {}
    for name, argument in zip(physical, block.args, strict=True):
        argument.name_hint = name
        symbolic[name] = _Value(argument, capture)
    for name in runtime:
        value = symbolic[name]
        if name in implicit_names:
            value._implicit_value = symbolic[implicit_names[name]]
        bound.arguments[name] = value
    try:
        output = capture.function(function)(*bound.args, **bound.kwargs)
    except CaptureError:
        raise
    except Exception as error:
        raise CaptureError(
            f"Engine callable capture failed: {error}"
        ) from error
    results: list[SSAValue] = []

    def describe(value: object) -> object:
        if isinstance(value, _Value):
            if (
                isinstance(value.value.type, ckks.CompressedPlaintextType)
                and value.implicit_data is not None
            ):
                fields = {
                    name: getattr(value, name)
                    for name in (
                        "depth",
                        "scale",
                        "ring_dimension",
                        "compression_layout",
                        "polynomial_domain",
                        "modulus_basis",
                        "residue_representation",
                        "prime_ids",
                    )
                }
                data = describe(value.data)
                implicit = describe(value.implicit_data)
                return {
                    "kind": "compressed_plaintext",
                    "data": data,
                    "implicit": implicit,
                    "fields": fields,
                }
            results.append(value.value)
            return {"kind": "ssa", "result": len(results) - 1}
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [describe(item) for item in value],
            }
        if isinstance(value, dict):
            return {
                "kind": "mapping",
                "entries": [
                    [encode_literal(key), describe(item)]
                    for key, item in value.items()
                ],
            }
        if type(value) in {int, float, complex, bool, str, type(None)}:
            return {"kind": "literal", "value": encode_literal(value)}
        raise CaptureError(
            "Captured outputs must be CKKS values, scalar literals, or tuple/list/dict containers"
        )

    descriptor = describe(output)
    block.add_op(ReturnOp(*results))
    program = Program.from_function(
        block,
        [value.type for value in results],
        module_attributes={
            "fhelium.frontend": StringAttr("eager-callable"),
            "fhelium.input_names": ArrayAttr(
                StringAttr(name) for name in physical
            ),
            "fhelium.input_fields": StringAttr(json.dumps(input_fields)),
            "fhelium.output_structure": StringAttr(json.dumps(descriptor)),
        },
    )
    work[CapturedCallable] = CapturedCallable(
        function, signature, MappingProxyType(specs), ""
    )
    for symbol, description in capture.descriptions.items():
        program.set_material_description(symbol, description)
    return Compilation(program, work, material_bindings=capture.materials)


__all__ = ["capture_eager"]
