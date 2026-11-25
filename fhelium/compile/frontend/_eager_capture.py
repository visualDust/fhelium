"""Capture mixed Tensor expressions and Engine calls into one Program.

Symbolic values preserve their numerical roles. Engine adapters expose actual
Tensor dataflow without executing numerical kernels or generating keys.
"""

from __future__ import annotations

import dis
import dataclasses
import sys
import inspect
import json
import math
import operator
import types
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from operator import index as integer_index
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

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
from fhelium.eager._engine import _KeyLayout, weighted_sum_terms
from fhelium.backend.ckks._key_switch_preparation import (
    key_switch_runtime_attributes,
)
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE, Program, value_role
from fhelium.ir.ckks_state import (
    allocation_state,
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
    key_placeholder,
    named_material_tensors,
    parameter_description,
)
from .._workspace import CompileWorkspace, WorkspaceKey
from ._captured_callable import CapturedCallable
from fhelium.ir.dialects._common import state_attribute

from ._eager_values import state_attributes, value_type
from ._pytorch_encoding import encode_literal
from ._specs import (
    InputSpec,
    StaticValue,
    encrypted,
    message,
    plaintext,
    static,
)


from ._objects import StaticMapping, StaticObject, python_callable, unwrap

if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.config import CkksConfig

    from ._preparation_capture import PreparationRecorder


class _Value:
    """Expose represented metadata while withholding a numerical Tensor payload."""

    _implicit_value: _Value | None

    def __init__(self, value: SSAValue, capture: _Capture) -> None:
        self.value = value
        self.capture = capture

    def __copy__(self) -> _Value:
        # A shallow Ciphertext copy shares its payload; the copy names the same SSA value.
        result = _Value(self.value, self.capture)
        result.__dict__.update(self.__dict__)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'scale':
            # Assigning a Ciphertext scale reinterprets its residues at that scale.
            engine = next(iter(self.capture.engines.values()), None)
            if engine is None:
                raise CaptureError(
                    'Ciphertext scale assignment requires an Engine in the captured computation'
                )
            value = cast(_Value, engine.reinterpret_at_scale(self, value)).value
            name = 'value'
        object.__setattr__(self, name, value)

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

    def _python_class(self):
        """Return the represented class for capture-controlled Python checks."""
        typ = self.value.type
        return (
            Ciphertext
            if isinstance(typ, ckks.CiphertextType)
            else CompressedPlaintext
            if isinstance(typ, ckks.CompressedPlaintextType)
            else Plaintext
            if isinstance(typ, ckks.PlaintextType)
            else torch.Tensor
        )

    def __getattribute__(self, name: str) -> Any:
        if name == '__class__':
            cls = _Value._python_class(self)
            # Native Tensor parsing must see the actual proxy type: this object
            # has no TensorImpl and enters Torch through __torch_function__.
            return type(self) if cls is torch.Tensor else cls
        return object.__getattribute__(self, name)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def view(self, *shape):
        return self.capture.numeric_call(torch.Tensor.view, (self, *shape), {})

    def neg(self):
        return -self

    def remainder(self, other):
        return self.capture.numeric_call(torch.remainder, (self, other), {})

    def assert_state(self, **kwargs):
        return Ciphertext.assert_state(cast(Any, self), **kwargs)

    def with_data(self, data):
        return self.replace(data=data)

    def with_storage(self, data, implicit_data):
        return self.replace(data=data, implicit_data=implicit_data)

    def replace(self, **fields):
        return self._replace_storage(fields, controlled=False)

    def _replace_storage(self, fields, *, controlled):
        from ._eager_values import replacement_state

        data = self.capture.substitute(fields.pop('data', self.data))
        implicit = self.capture.substitute(
            fields.pop('implicit_data', self.implicit_data)
        )
        state = replacement_state(
            self, data, implicit, fields, controlled=controlled
        )
        typ = cast(OpenStateType, self.value.type)
        operation, result = UnrealizedConversionCastOp.cast_one(
            _symbolic(data).value, type(typ)().with_state(state)
        )
        self.capture.block.add_op(operation)
        output = _Value(result, self.capture)
        output._implicit_value = cast(_Value | None, implicit)
        return output

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
        if state.get('dtype') in (
            StringAttr('torch.int32'),
            StringAttr('torch.int64'),
        ):
            state.setdefault('requires_grad', IntegerAttr(0, 1))
        return state

    def _with_resident_tensors(self, tensors):
        return self._replace_storage(
            {
                'data': tensors[0],
                'implicit_data': tensors[1] if len(tensors) == 2 else None,
            },
            controlled=True,
        )

    def __bool__(self) -> bool:
        raise CaptureError(
            "Captured computation cannot branch on Tensor or ciphertext data"
        )

    def __getattr__(self, name: str) -> Any:
        capture = self.capture
        if name == 'rotation_step':
            step = capture.key_steps.get(self.value)
            if step is None:
                raise CaptureError(
                    'Captured value does not represent a rotation key'
                )
            return step
        if name in _VALUE_METADATA:
            result = self._metadata(name)
        else:
            # Other fields depend only on the value type; repeated reads of one type share a result.
            typ = self.value.type
            found = capture.metadata.get((id(typ), name))
            if found is None:
                found = capture.metadata[(id(typ), name)] = (
                    typ,
                    self._metadata(name),
                )
            result = found[1]
        return result

    def _metadata(self, name: str) -> Any:
        if name == 'layout':
            return torch.strided
        if name == 'requires_grad':
            return bool(
                cast(
                    IntegerAttr,
                    self._tensor_state().get(
                        'requires_grad', IntegerAttr(0, 1)
                    ),
                ).value.data
            )
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
            if name == 'device':
                return torch.device(attribute.data)
            if name == 'dtype':
                return getattr(torch, attribute.data.removeprefix('torch.'))
            return attribute.data
        if isinstance(attribute, ArrayAttr):
            if name == "digit_prime_ids":
                return tuple(
                    tuple(item.value.data for item in rows)
                    for rows in attribute
                )
            return tuple(
                item.value.data
                for item in attribute
                if isinstance(item, IntegerAttr)
            )
        raise CaptureError(
            f"Captured CKKS values do not expose {name!r}; use Engine operations "
            "rather than inspecting or mutating their Tensor payloads"
        )


_VALUE_METADATA = frozenset({"data", "implicit_data", "_resident_tensors"})


def _result_type(source: _Value, **updates: object) -> OpenStateType:
    typ = cast(OpenStateType, source.value.type)
    # Output allocation can change physical strides. Mathematical batch shape,
    # dtype, device, and ring metadata remain usable by later passes.
    fields = allocation_state(dict(typ.state.data))
    for name, value in updates.items():
        attribute = (
            value if isinstance(value, Attribute) else state_attribute(value)
        )
        if attribute is not None:
            fields[name] = attribute
    return type(typ)().with_state(fields)


def _stacked_terms(values: object) -> bool:
    """Return whether a batched plaintext supplies weighted-sum terms as one operand."""
    if not isinstance(values, (Plaintext, CompressedPlaintext)):
        return False
    return values.is_batched and not (
        isinstance(values, CompressedPlaintext)
        and values.compression_layout == "strided_sparse"
    )


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


_WEIGHTED_SUMS = frozenset(
    {"sum_plaintext_products", "sum_plaintext_product_groups"}
)

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
        "_multiply_scalar",
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
        "iter_rotations_with_keys",
        "sum_rotations",
        "modulus_raise",
        "encapsulated_modulus_raise",
        "encode",
        "integer_coefficients_to_rns",
        "prepare_plaintext_for_addition",
        "prepare_plaintext_for_multiplication",
    }
)


# Engine methods implemented only through other captured Engine methods.
_COMPOSITES = frozenset({"multiply_constant", "multiply_by_monomial"})

# Engine methods whose emission is a function of operand types and static parameters.
_PLANNED = frozenset(
    {
        "add",
        "subtract",
        "negate",
        "multiply",
        "add_plaintext",
        "multiply_plaintext",
        "sum_batch",
        "coefficient_domain_to_ntt_domain",
        "ntt_domain_to_coefficient_domain",
        "standard_residues_to_montgomery_residues",
        "montgomery_residues_to_standard_residues",
        "reinterpret_at_scale",
    }
)
_SIGNATURES: dict[str, inspect.Signature] = {}


@dataclasses.dataclass(frozen=True)
class _EmissionPlan:
    """Operation, result type, attributes and generated operands decided by one emission."""

    op_type: type[Operation]
    result_type: Attribute
    attributes: dict[str, Attribute]
    properties: dict[str, Attribute]
    generated: tuple[SSAValue, ...]


class _EngineCalls:
    def __init__(self, engine: Engine, capture: _Capture) -> None:
        self.engine = engine
        self.capture = capture
        self.plans: dict[tuple[object, ...], _EmissionPlan] = {}
        self.config_attribute: StringAttr | None = None
        # Key-switch operands and runtime attributes per input depth, device, key layout and options.
        self.key_operands: dict[
            tuple[object, ...],
            tuple[tuple[SSAValue, ...], dict[str, Attribute]],
        ] = {}

    def __hash__(self):
        return hash(self.engine)

    def __eq__(self, other):
        return self.engine is (
            other.engine if isinstance(other, _EngineCalls) else other
        )

    def __getattr__(self, name: str) -> Any:
        if name == 'validate_key_switch_key':
            return self.engine.validate_key_switch_key
        if name == 'validate_ciphertext':

            def validate(value):
                from torch._subclasses.fake_tensor import FakeTensorMode

                recorder = self.capture.preparation
                with (
                    recorder.pause() if recorder is not None else nullcontext(),
                    FakeTensorMode(),
                ):
                    data = torch.empty(
                        value.shape, dtype=value.dtype, device=value.device
                    )
                    envelope = Ciphertext(
                        data,
                        value.depth,
                        value.scale,
                        value.prime_ids,
                        value.polynomial_domain,
                        value.modulus_basis,
                        value.residue_representation,
                    )
                    self.engine.validate_ciphertext(envelope)

            return validate
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
        if name in _COMPOSITES:
            # Compositions of captured Engine methods run with this proxy as self.
            return types.MethodType(getattr(Engine, name), self)
        if name not in _METHODS:
            raise CaptureError(
                f"Engine.{name} is outside the callable frontend's operation set; "
                "keep lifecycle work outside the computation or supply a Program"
            )
        method = getattr(Engine, name)

        def emit(*args: object, **kwargs: object) -> object:
            signature = _SIGNATURES.get(name)
            if signature is None:
                signature = _SIGNATURES[name] = inspect.signature(method)
            bound = signature.bind(self.engine, *args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            parameters.pop("self")
            if name in _WEIGHTED_SUMS:
                # Batched operands enumerate terms through views traced like select_batch.
                parameters["ciphertexts"] = weighted_sum_terms(
                    cast(Any, parameters["ciphertexts"])
                )
                if name == "sum_plaintext_product_groups":
                    parameters["plaintext_groups"] = tuple(
                        weighted_sum_terms(row)
                        for row in cast(Any, parameters["plaintext_groups"])
                    )
                elif not _stacked_terms(parameters["plaintexts"]):
                    parameters["plaintexts"] = weighted_sum_terms(
                        cast(Any, parameters["plaintexts"])
                    )
            recorder = self.capture.preparation
            if recorder is not None:
                from ._preparation_capture import (
                    PREPARATION_METHODS,
                    contains_proxy,
                )

                if name in PREPARATION_METHODS and not contains_proxy(
                    parameters
                ):
                    if parameters.get("inplace", False):
                        raise CaptureError(
                            "Plaintext preparation requires out-of-place Engine operations"
                        )
                    return recorder.engine(self.engine, name, parameters)
            parameters = {
                key: self.capture.runtime(value)
                for key, value in parameters.items()
            }
            if parameters.pop("inplace", False):
                raise CaptureError(
                    "Callable capture requires out-of-place Engine operations"
                )
            before = self.capture.block.last_op
            prior_config = self.capture.active_config
            self.capture.active_config = self.engine.config
            try:
                with (
                    recorder.pause() if recorder is not None else nullcontext()
                ):
                    result = self._emit(name, parameters)
            finally:
                self.capture.active_config = prior_config
            operation = (
                self.capture.block.first_op
                if before is None
                else before.next_op
            )
            configuration = self.config_attribute
            if configuration is None:
                configuration = self.config_attribute = StringAttr(
                    json.dumps(self.engine.config.dumps(), sort_keys=True)
                )
            while operation is not None:
                if operation.name.startswith("fhelium_ckks."):
                    operation.attributes["ckks_config"] = configuration
                operation = operation.next_op
            return result

        return emit

    def _weighted_sum(self, p, *, grouped):
        c = self.capture
        ciphertexts = tuple(_symbolic(x) for x in p["ciphertexts"])
        # One batched plaintext stays one operand whose leading axis holds the terms.
        stacked = (
            None
            if grouped or isinstance(p["plaintexts"], (tuple, list))
            else _symbolic(p["plaintexts"])
        )
        rows = (
            ((stacked,) * stacked.batch_shape[0],)
            if stacked is not None
            else p["plaintext_groups"]
            if grouped
            else (p["plaintexts"],)
        )
        rows = tuple(tuple(_symbolic(x) for x in row) for row in rows)
        if (
            not ciphertexts
            or not rows
            or any(len(row) != len(ciphertexts) for row in rows)
        ):
            raise CaptureError("Weighted sums require nonempty matching rows")
        source = ciphertexts[0]
        plaintexts = tuple(x for row in rows for x in row)
        if stacked is not None:
            plaintexts = (stacked,)
        for value in ciphertexts[1:]:
            self._matching_operands(source, value, scale=True)
        for value in plaintexts:
            self._matching_operands(source, value, scale=False, plaintext=True)
            if value.scale != plaintexts[0].scale:
                raise CaptureError("Weighted-sum plaintext scales must match")
        layouts = tuple(
            x.compression_layout
            if isinstance(x.value.type, ckks.CompressedPlaintextType)
            else "dense"
            for x in plaintexts
        )
        parameters = self.engine._dispatcher_for(
            source.device
        ).rns_context.rns_parameters_for_prime_ids(source.prime_ids)
        parameter = c.generated_material(
            parameters,
            rns.RnsParametersType(),
            parameter_description(
                self.engine.config,
                "rns_parameters",
                prime_ids=list(source.prime_ids),
            ),
        )

        def bundle(value, owner=None):
            state = value._tensor_state()
            if owner is not None:
                state = {
                    **state,
                    **{
                        name: owner.value.type.state.data[name]
                        for name in (
                            "polynomial_domain",
                            "residue_representation",
                            "prime_ids",
                            "basis",
                        )
                    },
                }
            typ = rns.RnsBundleType().with_state(state)
            # Rotation results feed many sums; each value is cast once.
            key = (value.value, typ)
            result = c.bundles.get(key)
            if result is None:
                operation, result = UnrealizedConversionCastOp.cast_one(
                    value.value, typ
                )
                c.block.add_op(operation)
                c.bundles[key] = result
            return result

        implicit = tuple(
            bundle(x.implicit_data, x)
            for x in plaintexts
            if isinstance(x.value.type, ckks.CompressedPlaintextType)
            and x.compression_layout == "strided_sparse"
        )
        batch = ((len(rows),) if grouped else ()) + source.batch_shape
        result_type = _result_type(
            source,
            batch_shape=batch,
            scale=product_scale(source.scale, plaintexts[0].scale),
        )
        bundle_type = rns.RnsBundleType().with_state(result_type.state.data)
        kwargs: dict[str, Any] = dict(
            plaintext_layouts=layouts,
            implicit_plaintexts=implicit,
            term_chunk_size=p.get("term_chunk_size"),
        )
        if stacked is not None:
            kwargs["plaintext_term_axis"] = 0
        if grouped:
            kwargs["group_count"] = len(rows)
        operation = (
            rns.MontgomeryWeightedSumsOp
            if grouped
            else rns.MontgomeryWeightedSumOp
        )(
            parameter,
            tuple(bundle(x) for x in ciphertexts),
            tuple(bundle(x) for x in plaintexts),
            bundle_type,
            **kwargs,
        )
        if p.get("implementation") is not None:
            # A requested implementation is fixed; otherwise Compile selects one.
            operation.attributes[EXECUTION_IMPLEMENTATION_ATTRIBUTE] = (
                StringAttr(
                    self.engine.plaintext_product_implementation(
                        source.device,
                        compressed=any(x != "dense" for x in layouts),
                        implementation=p.get("implementation"),
                    )
                )
            )
        c.block.add_op(operation)
        cast_op, _ = UnrealizedConversionCastOp.cast_one(
            operation.result, result_type
        )
        return c.emit(cast_op)

    def _plan_key(
        self, name: str, p: dict[str, object]
    ) -> tuple[object, ...] | None:
        if name not in _PLANNED:
            return None
        key: list[object] = [name]
        for field, value in p.items():
            if type(value) is _Value:
                if value.__dict__.get("_implicit_value") is not None:
                    return None
                key.append((field, self.capture.type_key(value.value.type)))
            elif value is None or type(value) in (bool, int, float, str):
                key.append((field, type(value).__name__, repr(value)))
            else:
                return None
        return tuple(key)

    def _record_plan(
        self,
        p: dict[str, object],
        result: object,
        before: Operation | None,
    ) -> _EmissionPlan | None:
        """Describe an emission that added generated material references and one operation."""
        block = self.capture.block
        if type(result) is not _Value:
            return None
        operation = result.value.owner
        if (
            not isinstance(operation, Operation)
            or operation is not block.last_op
            or len(operation.results) != 1
        ):
            return None
        added = block.first_op if before is None else before.next_op
        while added is not operation:
            if not isinstance(added, core.MaterialRefOp):
                return None
            added = added.next_op
        values = [item.value for item in p.values() if type(item) is _Value]
        operands = tuple(operation.operands)
        if len(operands) < len(values) or any(
            operand is not value for operand, value in zip(operands, values)
        ):
            return None
        generated = operands[len(values) :]
        if any(
            not isinstance(value.owner, core.MaterialRefOp)
            for value in generated
        ):
            return None
        return _EmissionPlan(
            type(operation),
            operation.results[0].type,
            dict(operation.attributes),
            dict(operation.properties),
            generated,
        )

    def _emit(self, name: str, p: dict[str, object]) -> object:
        key = self._plan_key(name, p)
        if key is None:
            return self._emit_operation(name, p)
        c = self.capture
        plan = self.plans.get(key)
        if plan is None:
            before = c.block.last_op
            result = self._emit_operation(name, p)
            plan = self._record_plan(p, result, before)
            if plan is not None:
                self.plans[key] = plan
            return result
        operation = plan.op_type.create(
            operands=(
                *(item.value for item in p.values() if type(item) is _Value),
                *plan.generated,
            ),
            result_types=(plan.result_type,),
            attributes=dict(plan.attributes),
            properties=dict(plan.properties),
        )
        c.block.add_op(operation)
        return _Value(operation.results[0], c)

    def _emit_operation(self, name: str, p: dict[str, object]) -> object:
        c = self.capture
        if name == "plaintext_product_implementation":
            return self.engine.plaintext_product_implementation(
                cast(torch.device | str, p["device"]),
                compressed=cast(bool, p["compressed"]),
                implementation=cast(str | None, p.get("implementation")),
            )
        if name in {
            "sum_plaintext_products",
            "sum_plaintext_product_groups",
        } and (
            p.get("implementation") is not None
            or p.get("term_chunk_size") is not None
        ):
            return self._weighted_sum(
                p, grouped=name == "sum_plaintext_product_groups"
            )
        if name == "zero_plaintext_like":
            source = cast(_Value, p["plaintext"])
            return source._with_resident_tensors(
                tuple(
                    c.numeric_call(torch.zeros_like, (t,), {})
                    for t in source._resident_tensors
                )
            )
        if name == "sum_plaintext_products":
            return self._weighted_sum(p, grouped=False)
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
                            "basis_conversion": p.get(
                                "basis_conversion", "exact"
                            ),
                            "moddown_rounding": p.get(
                                "moddown_rounding", "nearest"
                            ),
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
            parameters = c.generated_material(
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
                self.engine.config.scale_at(depth)
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
                c.generated_material(
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
                self.engine.config.scale_at(depth)
                if p["scale"] is None
                else cast(float, p["scale"])
            )
            parameters = tuple(
                c.generated_material(
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
            twice_modulus = c.generated_material(
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
        if name == "iter_rotations_with_keys":
            keys = tuple(cast(Sequence[Any], p["keys"]))
            direct = iter(
                cast(
                    Sequence[_Value],
                    self._emit(
                        "rotate_many_with_keys",
                        {
                            "value": p["value"],
                            "keys": [key for key in keys if key is not None],
                            "use_hoisting": True,
                            "output_domain": p["output_domain"],
                            "basis_conversion": p.get(
                                "basis_conversion", "exact"
                            ),
                            "moddown_rounding": p.get(
                                "moddown_rounding", "nearest"
                            ),
                        },
                    ),
                )
            )
            identity = p["value"]
            if p["output_domain"] == "ntt" and any(key is None for key in keys):
                identity = self._emit(
                    "coefficient_domain_to_ntt_domain", {"value": identity}
                )
            return iter(
                [
                    _symbolic(identity).clone() if key is None else next(direct)
                    for key in keys
                ]
            )
        if name in {"modulus_raise", "encapsulated_modulus_raise"}:
            source = _symbolic(p["value"])
            depth = int(cast(int, p["target_depth"]))
            typ = _result_type(
                source,
                depth=depth,
                prime_ids=depth_prime_ids(self.engine.config, depth, "Q"),
                polynomial_domain="coefficient",
                residue_representation="standard",
            )
            resources = self.engine._dispatcher_for(
                torch.device(source.device)
            ).device_resources
            details: dict[str, object] = {
                "depth": int(source.depth),
                "target_depth": depth,
            }
            if name == "modulus_raise":
                tensors, facts = resources.modulus_raise_operands(
                    int(source.depth), depth
                )
                kind = "centered_basis_extension_table"
            else:
                low_p = tuple(cast(Sequence[int], p["low_p_moduli"]))
                guard = int(cast(int, p["return_prime"]))
                return_p = tuple(cast(Sequence[int], p["return_p_moduli"]))
                rounding = str(p["moddown_rounding"])
                tensors, facts = resources.encapsulated_modulus_raise_operands(
                    int(source.depth),
                    depth,
                    low_p,
                    guard,
                    return_p,
                    rounding=rounding,
                )
                kind = "encapsulated_modulus_raise_table"
                details.update(
                    low_p_moduli=list(low_p),
                    return_prime=guard,
                    return_p_moduli=list(return_p),
                    rounding=rounding,
                    ntt_backend=facts["ntt_backend"],
                )
            attributes = {
                key: attribute
                for key, value in {
                    **facts,
                    "parameter_names": tuple(tensors),
                }.items()
                if (attribute := state_attribute(value)) is not None
            }
            operands = tuple(
                c.generated_material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config, kind, name=key, **details
                    ),
                )
                for key, tensor in tensors.items()
            )
            if name == "modulus_raise":
                operation = ckks.ModulusRaiseOp(
                    source.value,
                    typ,
                    target_depth=depth,
                    parameters=operands,
                    attributes=attributes,
                )
            else:
                low_key = c.key_reference(p["low_key"], role="switch")
                return_key = c.key_reference(p["return_key"], role="switch")
                operation = ckks.EncapsulatedModulusRaiseOp(
                    source.value,
                    low_key,
                    return_key,
                    typ,
                    key_seeds=c.key_seeds((low_key, return_key)),
                    target_depth=depth,
                    low_p_moduli=low_p,
                    return_prime=guard,
                    return_p_moduli=return_p,
                    rounding=rounding,
                    parameters=operands,
                    attributes=attributes,
                )
            return c.emit(operation)
        if name == "sum_rotations":
            terms = tuple(cast(Any, p["terms"]))
            if not terms:
                raise CaptureError("Rotation sum requires at least one term")
            sources = tuple(_symbolic(value) for value, _ in terms)
            source = sources[0]
            # Results of earlier operations carry no allocation layout facts.
            state = allocation_state(
                dict(cast(OpenStateType, source.value.type).state.data)
            )
            if any(
                allocation_state(
                    dict(cast(OpenStateType, value.value.type).state.data)
                )
                != state
                for value in sources
            ):
                raise CaptureError(
                    "Rotation-sum terms require matching states and shapes"
                )
            rescale = bool(p["rescale"])
            depth = int(source.depth) + int(rescale)
            scale = (
                quotient_scale(
                    float(source.scale),
                    self.engine.config.rescale_divisor(int(source.depth)),
                )
                if rescale
                else float(source.scale)
            )
            typ = _result_type(
                source,
                depth=depth,
                scale=scale,
                prime_ids=depth_prime_ids(self.engine.config, depth, "Q"),
                polynomial_domain="coefficient",
                residue_representation="standard",
            )
            rounding = str(p.get("moddown_rounding", "nearest"))
            parameters, attributes = self._key_parameters(
                source,
                rounding,
                keys=tuple(key for _, key in terms if key is not None),
                basis_conversion=str(p.get("basis_conversion", "exact")),
            )
            layout = self._key_layout(
                tuple(key for _, key in terms if key is not None)
            )
            if layout is not None:
                tables = self.engine._dispatcher_for(
                    source.device
                ).device_resources.rotation_sum_tables(
                    int(source.depth),
                    rescale=rescale,
                    rounding=rounding,
                    basis_conversion=str(p.get("basis_conversion", "exact")),
                    **layout,
                )
                extras = {
                    name: tensor
                    for name, tensor in tables.tensor_operands().items()
                    if name.startswith('sum_')
                }
                if source.polynomial_domain == 'ntt':
                    from fhelium.backend.ntt.automorphism import (
                        ntt_galois_indices,
                    )
                    from fhelium.backend.ckks.rotation._galois import (
                        rotation_galois_element,
                    )

                    for _, key in terms:
                        step = 0 if key is None else int(key.rotation_step)
                        if step:
                            extras[f'sum_galois_{step}'] = ntt_galois_indices(
                                self.engine.config.N,
                                rotation_galois_element(
                                    self.engine.config.N,
                                    step,
                                    self.engine.config.galois_generator,
                                ),
                                source.device,
                            )
                parameters += tuple(
                    c.generated_material(
                        tensor,
                        core.MessageType(),
                        parameter_description(
                            self.engine.config,
                            'rotation_sum_table',
                            depth=int(source.depth),
                            name=name,
                            rescale=rescale,
                            rounding=rounding,
                            basis_conversion=str(
                                p.get("basis_conversion", "exact")
                            ),
                            **layout,
                        ),
                    )
                    for name, tensor in extras.items()
                )
                attributes.update(
                    {
                        'parameter_names': ArrayAttr(
                            (
                                *cast(
                                    ArrayAttr[StringAttr],
                                    attributes.get(
                                        'parameter_names', ArrayAttr(())
                                    ),
                                ),
                                *(StringAttr(name) for name in extras),
                            )
                        ),
                        'sum_remaining_q_count': IntegerAttr(
                            tables.remaining_q_count, 64
                        ),
                        'sum_dropped_q_count': IntegerAttr(
                            tables.dropped_q_count, 64
                        ),
                        'sum_divisor': StringAttr(str(tables.divisor)),
                        'sum_rescale_divisor': StringAttr(
                            str(tables.rescale_divisor)
                        ),
                    }
                )
            steps = tuple(
                0 if key is None else int(key.rotation_step) for _, key in terms
            )
            keys = tuple(
                c.key_reference(key, role="rotation", step=step)
                for (_, key), step in zip(terms, steps, strict=True)
                if step
            )
            return c.emit(
                ckks.RotationSumOp(
                    tuple(value.value for value in sources),
                    keys,
                    steps,
                    typ,
                    key_seeds=c.key_seeds(keys),
                    rescale=rescale,
                    input_domain=str(source.polynomial_domain),
                    rounding=rounding,
                    parameters=parameters,
                    attributes=attributes,
                )
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
                        source,
                        int(key.rotation_step),
                        domain,
                        key=key,
                        moddown_rounding=str(
                            p.get("moddown_rounding", "nearest")
                        ),
                        basis_conversion=str(
                            p.get("basis_conversion", "exact")
                        ),
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
            parameters, attributes = self._key_parameters(
                source,
                str(p.get("moddown_rounding", "nearest")),
                keys=keys,
                basis_conversion=str(p.get("basis_conversion", "exact")),
            )
            attributes["rotation_steps"] = ArrayAttr(
                IntegerAttr(int(key.rotation_step), 64) for key in keys
            )
            key_values = tuple(
                c.key_reference(
                    key, role="rotation", step=int(key.rotation_step)
                )
                for key in keys
            )
            operation = ckks.RotateManyOp(
                source.value,
                key_values,
                tuple(typ for _ in keys),
                parameters=parameters,
                key_seeds=c.key_seeds(key_values),
                output_domain=domain,
                attributes=attributes,
            )
            c.emit(operation)
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
                    self._rotate(
                        source,
                        step,
                        "coefficient",
                        moddown_rounding=str(
                            p.get("moddown_rounding", "nearest")
                        ),
                        basis_conversion=str(
                            p.get("basis_conversion", "exact")
                        ),
                    )
                    for step in steps
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
                parameters, attributes = self._key_parameters(
                    source,
                    str(p.get("moddown_rounding", "nearest")),
                    keys=tuple(
                        self.engine._keys.rotation_keys.get(step)
                        for step in nonzero
                    ),
                    basis_conversion=str(p.get("basis_conversion", "exact")),
                )
                attributes["rotation_steps"] = ArrayAttr(
                    IntegerAttr(step, 64) for step in nonzero
                )
                key_values = tuple(
                    c.key_reference(
                        self.engine._keys.rotation_keys.get(step),
                        role="rotation",
                        step=step,
                    )
                    for step in nonzero
                )
                op = ckks.RotateManyOp(
                    source.value,
                    key_values,
                    tuple(typ for _ in nonzero),
                    parameters=parameters,
                    key_seeds=c.key_seeds(key_values),
                    attributes=attributes,
                )
                c.emit(op)
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
                rhs = _symbolic(p["rhs"])
                self._matching_operands(
                    source, rhs, scale=True, components=name == "subtract"
                )
                typ = _result_type(
                    source,
                    components=max(
                        int(source.component_count), int(rhs.component_count)
                    ),
                )
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
                        operands.append(c.append_unit_axis(rhs.implicit_data))
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
                        operands.append(c.append_unit_axis(rhs.implicit_data))
                    attrs["compression_layout"] = StringAttr(
                        str(rhs.compression_layout)
                    )
        elif name in {
            "add_scalar",
            "_multiply_scalar",
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
                if name == "_multiply_scalar":
                    typ = _result_type(
                        source,
                        scale=coerce_scale(
                            product_scale(float(source.scale), scale),
                            value_name="multiply_constant result",
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
            return self._mod_switch(
                source, target, relative=name == "mod_switch_to_next_depth"
            )
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
                tuple(source.prime_ids),
                rows,
                rounding=cast(str, p["rounding"]),
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
                        c.generated_material(
                            tensor,
                            core.MessageType(),
                            parameter_description(
                                config,
                                "rescale_table",
                                source_prime_ids=list(source.prime_ids),
                                target_prime_ids=list(rows),
                                rounding=cast(str, p["rounding"]),
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
            key_op_type = {
                "relinearize": ckks.RelinearizeOp,
                "conjugate": ckks.ConjugateOp,
                "switch_key": ckks.SwitchKeyOp,
            }[name]
            rescale = name == "relinearize" and bool(p["rescale"])
            rescale_rounding = str(p.get("rescale_rounding", "nearest"))
            parameters, attrs = self._key_parameters(
                source,
                str(p.get("moddown_rounding", "nearest")),
                keys=(key,),
                basis_conversion=str(p.get("basis_conversion", "exact")),
                rescale=rescale,
                rescale_rounding=rescale_rounding,
                output_domain=domain,
            )
            if name == "relinearize":
                attrs["rescale"] = IntegerAttr(int(rescale), 64)
                attrs["rescale_rounding"] = StringAttr(rescale_rounding)
            if rescale:
                if source.depth is None:
                    typ = type(typ)().with_state(
                        {
                            name: attribute
                            for name, attribute in typ.state.data.items()
                            if name not in {"depth", "prime_ids", "scale"}
                        }
                    )
                else:
                    depth = int(source.depth)
                    divisor = self.engine.config.rescale_divisor(depth)
                    typ = _result_type(
                        source,
                        components=2,
                        depth=depth + 1,
                        prime_ids=depth_prime_ids(
                            self.engine.config, depth + 1, "Q"
                        ),
                        scale=quotient_scale(source.scale, divisor)
                        if source.scale is not None
                        else None,
                        polynomial_domain=domain,
                        residue_representation=output_residues(domain),
                    )
            attrs["output_domain"] = StringAttr(domain)
            key_value = c.key_reference(key, role=name)
            return c.emit(
                key_op_type(
                    source.value,
                    key_value,
                    typ,
                    parameters=tuple(parameters),
                    key_seeds=c.key_seeds((key_value,)),
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
                    moddown_rounding=str(p.get("moddown_rounding", "nearest")),
                    basis_conversion=str(p.get("basis_conversion", "exact")),
                )
            step = RotationKey.normalize_step(
                integer_index(cast(int, p["rotation_step"])),
                ring_dimension=self.engine.ring_dimension,
            )
            return self._rotate(
                source,
                step,
                "coefficient",
                moddown_rounding=str(p.get("moddown_rounding", "nearest")),
                basis_conversion=str(p.get("basis_conversion", "exact")),
            )
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
                c.generated_material(
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
                None if integer else dispatcher.rng.seed,
            )
            tensor_operands.extend(
                c.generated_material(
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
            generated = {
                name: attribute
                for name, value in facts.items()
                if (attribute := state_attribute(value)) is not None
            }
            attrs.update(generated)
        elif op_type in {ckks.ToNttOp, ckks.FromNttOp}:
            source = operands[0]
            context = self.engine._dispatcher_for(
                torch.device(source.device)
            ).ntt_context
            tables = context.tensor_operands(
                source.prime_ids, inverse=op_type is ckks.FromNttOp
            )
            tensor_operands.extend(
                c.generated_material(
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
        operation = op_type.create(
            operands=tensor_operands,
            result_types=[typ],
            attributes=attrs,
        )
        return c.emit(operation)

    @staticmethod
    def _matching_operands(
        lhs: _Value,
        rhs: _Value,
        *,
        scale: bool,
        plaintext: bool = False,
        components: bool = True,
    ) -> None:
        fields = ("depth", "prime_ids", "modulus_basis")
        if not plaintext:
            fields += ("polynomial_domain", "residue_representation")
            if components:
                fields += ("component_count",)
        if scale:
            fields += ("scale",)
        for field in fields:
            if getattr(lhs, field) != getattr(rhs, field):
                raise CaptureError(
                    f"Operands disagree on {field}; capture does not insert alignment"
                )

    def _mod_switch(
        self, source: _Value, depth: int, *, relative: bool = True
    ) -> _Value:
        if depth < int(source.depth):
            raise CaptureError("ModSwitch cannot restore removed Q rows")
        config = self.engine.config
        rows = depth_prime_ids(
            config, depth, cast(ModulusBasis, source.modulus_basis)
        )
        typ = _result_type(
            source,
            depth=depth,
            prime_ids=rows,
            scale=source.scale
            if config.is_subbasis(int(source.depth), depth)
            else config.scale_at(depth),
        )
        operation = ckks.ModSwitchOp(
            source.value,
            typ,
            attributes={"target_depth": IntegerAttr(depth, 64)},
        )
        return self.capture.emit(operation)

    def _key_layout(self, keys: Sequence[object]) -> _KeyLayout | None:
        layouts = []
        for key in keys:
            if key is None:
                return None
            try:
                layouts.append(
                    (
                        tuple(cast(KeySwitchKey, key).prime_ids),
                        tuple(
                            tuple(rows)
                            for rows in cast(KeySwitchKey, key).digit_prime_ids
                        ),
                    )
                )
            except (AttributeError, CaptureError):
                return None
        if not layouts:
            return {}
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise CaptureError(
                "A shared key-switch execution group requires matching key rows and digit supports"
            )
        return {
            "key_prime_ids": layouts[0][0],
            "digit_prime_ids": layouts[0][1],
        }

    def _key_parameters(
        self,
        source: _Value,
        moddown_rounding: str = "nearest",
        basis_conversion: str = "exact",
        rescale: bool = False,
        rescale_rounding: str = "nearest",
        output_domain: str = "coefficient",
        *,
        keys: Sequence[object] = (),
    ) -> tuple[tuple[SSAValue, ...], dict[str, Attribute]]:
        if moddown_rounding not in {"floor", "nearest"}:
            raise CaptureError("moddown_rounding must be 'floor' or 'nearest'")
        if basis_conversion not in {"exact", "crt", "normalized_crt"}:
            raise CaptureError(
                "basis_conversion must be exact, crt, or normalized_crt"
            )
        layout = self._key_layout(keys)
        if keys and layout is None:
            return (), {
                "moddown_rounding": StringAttr(moddown_rounding),
                "basis_conversion": StringAttr(basis_conversion),
            }
        memo_key = (
            str(source.device),
            int(source.depth),
            moddown_rounding,
            basis_conversion,
            rescale,
            rescale_rounding,
            output_domain,
            None if layout is None else tuple(layout.items()),
        )
        found = self.key_operands.get(memo_key)
        if found is not None:
            # The mapping is extended by callers with per-operation attributes.
            return found[0], dict(found[1])
        dispatcher = self.engine._dispatcher_for(torch.device(source.device))
        resources = dispatcher.device_resources
        self.capture.resources[id(resources)] = resources
        tensors, facts = dispatcher.device_resources.key_switch_operands(
            int(source.depth),
            moddown_rounding=moddown_rounding,
            basis_conversion=basis_conversion,
            rescale=rescale,
            rescale_rounding=rescale_rounding,
            output_domain=output_domain,
            **(layout or {}),
        )
        attributes = {
            name: attribute
            for name, value in {
                **key_switch_runtime_attributes(facts),
                "moddown_rounding": moddown_rounding,
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
                self.capture.generated_material(
                    tensor,
                    core.MessageType(),
                    parameter_description(
                        self.engine.config, kind, **description
                    ),
                )
            )
        self.key_operands[memo_key] = (tuple(operands), attributes)
        return tuple(operands), dict(attributes)

    def _rotate(
        self,
        source: _Value,
        step: int,
        domain: PolynomialDomain,
        *,
        key: object = None,
        moddown_rounding: str = "nearest",
        basis_conversion: str = "exact",
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
                        moddown_rounding=moddown_rounding,
                        basis_conversion=basis_conversion,
                    )
                return source
            key = self.engine._keys.rotation_keys.get(step)
        typ = _result_type(
            source,
            polynomial_domain=domain,
            residue_representation=output_residues(domain),
        )
        parameters, attributes = self._key_parameters(
            source,
            moddown_rounding,
            keys=(key,),
            basis_conversion=basis_conversion,
        )
        attributes.update(
            rotation_step=IntegerAttr(step, 64),
            input_domain=StringAttr(cast(str, source.polynomial_domain)),
            output_domain=StringAttr(domain),
        )
        key_value = self.capture.key_reference(key, role="rotation", step=step)
        return self.capture.emit(
            ckks.RotateOp(
                source.value,
                key_value,
                typ,
                parameters=parameters,
                key_seeds=self.capture.key_seeds((key_value,)),
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
        self.active_config: CkksConfig | None = None
        self.preparation: PreparationRecorder | None = None
        self.runtime_inputs = {}
        self.block = block
        self.engines: dict[int, _EngineCalls] = {}
        self.functions: dict[types.FunctionType, types.FunctionType] = {}
        self.objects: dict[int, StaticObject] = {}
        self.materials: dict[str, torch.Tensor] = {}
        self.descriptions: dict[str, dict[str, object]] = {}
        self.names = named_material_tensors(material_names)
        self.tensor_symbols: dict[int, str] = {}
        self.material_refs: dict[
            tuple[int, bool], list[tuple[Attribute, SSAValue]]
        ] = {}
        self.metadata: dict[tuple[int, str], tuple[Attribute, object]] = {}
        self.bundles: dict[tuple[SSAValue, Attribute], SSAValue] = {}
        # Rotation steps of captured key values; key types record only layout.
        self.key_steps: dict[SSAValue, int] = {}
        # Seed-row operands of seeded key values.
        self.seeds: dict[SSAValue, SSAValue] = {}
        # Device resources of key-switching captures provide lowering tables.
        self.resources: dict[int, CkksDeviceResources] = {}
        self.type_keys: dict[int, tuple[Attribute, int]] = {}
        self.type_values: dict[Attribute, int] = {}

    def type_key(self, typ: Attribute) -> int:
        """Number equal types alike; each type object is hashed by value once."""
        found = self.type_keys.get(id(typ))
        if found is None:
            number = self.type_values.setdefault(typ, len(self.type_values))
            # The entry retains the type so its identity is not reused.
            found = self.type_keys[id(typ)] = (typ, number)
        return found[1]

    def generated_material(self, tensor, typ, description=None):
        return self.material(tensor, typ, description, generated=True)

    def material(
        self,
        tensor: torch.Tensor,
        typ: OpenStateType,
        description: Mapping[str, object] | None = None,
        *,
        generated: bool = False,
    ) -> SSAValue:
        typ = typ.with_state(state_attributes(tensor))
        references = self.material_refs.setdefault((id(tensor), generated), [])
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
        for prior_type, value in references:
            if prior_type == typ:
                return value
        reference = core.MaterialRefOp(typ, symbol=symbol)
        self.block.add_op(reference)
        if self.preparation is not None and symbol in self.preparation.symbols:
            view = self.preparation.views.get(
                symbol,
                {
                    "owner": symbol,
                    "offset": 0,
                    "shape": list(tensor.shape),
                    "strides": list(tensor.stride()),
                },
            )
            reference.attributes["fhelium.material_view"] = StringAttr(
                json.dumps(view)
            )
        references.append((typ, reference.value))
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
            value = self.material(
                key.data,
                cast(OpenStateType, value_type(key)),
                key_description(key),
            )
            seed = key.generated_seed
            if seed is not None and value not in self.seeds:
                key_symbol = cast(core.MaterialRefOp, value.owner).symbol
                self.seeds[value] = self.material(
                    seed,
                    cast(OpenStateType, value_type(seed)),
                    {
                        "kind": "KeySeed",
                        "key": cast(StringAttr, key_symbol).data,
                    },
                )
            return value
        if key is not None:
            raise CaptureError(f"{role} requires an evaluation key")
        symbol = f"{role}_key_{len(tuple(self.block.ops))}"
        reference = key_placeholder(
            symbol,
            {
                "kind": {
                    "rotation": "RotationKey",
                    "relinearization": "RelinearizationKey",
                    "conjugation": "ConjugationKey",
                    "switch": "KeySwitchKey",
                }.get(role, role),
                **({"rotation_step": step} if step is not None else {}),
            },
            self.descriptions,
        )
        self.block.add_op(reference)
        return reference.value

    def key_seeds(self, keys: Sequence[SSAValue]) -> tuple[SSAValue, ...]:
        """Return the seed operands of ``keys``: one per compact key, in order."""
        return tuple(self.seeds[key] for key in keys if key in self.seeds)

    def unwrapped_key(self, value):
        if isinstance(value, _EngineCalls):
            return value.engine
        if isinstance(value, tuple):
            return tuple(self.unwrapped_key(item) for item in value)
        return unwrap(value)

    def external(self, value, getter):
        if self.preparation is not None and isinstance(
            value, (torch.Tensor, Plaintext, CompressedPlaintext)
        ):
            return self.preparation.leaf(value, getter)
        return self.substitute(value)

    def runtime(self, value):
        """Export real preparation values when ciphertext dataflow consumes them."""
        if self.preparation is not None and isinstance(
            value, (torch.Tensor, Plaintext, CompressedPlaintext)
        ):
            original = self.runtime_inputs.get(id(value))
            if original is not None:
                return original
            tensor = value if isinstance(value, torch.Tensor) else value.data
            if tensor is None:
                raise CaptureError(
                    "Encode slot plaintexts before ciphertext consumption"
                )
            self.preparation.leaf(value)
            if not self.preparation.placeholder(tensor):
                self.preparation.export(tensor)
            result = _Value(self.material(tensor, value_type(value)), self)
            if (
                isinstance(value, CompressedPlaintext)
                and value.implicit_data is not None
            ):
                if not self.preparation.placeholder(value.implicit_data):
                    self.preparation.export(value.implicit_data)
                result._implicit_value = _Value(
                    self.material(value.implicit_data, core.MessageType()), self
                )
            return result
        if isinstance(value, (tuple, list)):
            return type(value)(self.runtime(item) for item in value)
        if isinstance(value, dict):
            return {key: self.runtime(item) for key, item in value.items()}
        return self.substitute(value)

    def substitute(self, value: object) -> object:
        from .._callable import CompiledCallable

        if isinstance(value, (_Value, StaticObject, _EngineCalls)):
            return value
        if self.preparation is not None:
            if value is torch:
                return self.preparation.torch
            if value is torch.tensor or value is torch.as_tensor:
                name = "tensor" if value is torch.tensor else "as_tensor"
                recorder = self.preparation
                return lambda *args, **kwargs: recorder.factory(
                    name, args, kwargs
                )
            if isinstance(
                value, (torch.Tensor, Plaintext, CompressedPlaintext)
            ):
                return self.preparation.leaf(value)
        if isinstance(value, CompiledCallable):
            if isinstance(value._source, Compilation):
                raise CaptureError(
                    'Compose Program-backed helpers at the Program level'
                )
            return self.substitute(value._source)
        if isinstance(value, Engine):
            if id(value) not in self.engines:
                self.engines[id(value)] = _EngineCalls(value, self)
            return self.engines[id(value)]
        if isinstance(value, types.MethodType):
            if isinstance(value.__self__, Engine):
                return getattr(self.substitute(value.__self__), value.__name__)
            return types.MethodType(
                self.function(value.__func__),
                self.object(value.__self__)
                if isinstance(value.__self__, type)
                else self.substitute(value.__self__),
            )
        if isinstance(value, (Ciphertext, Plaintext, CompressedPlaintext)):
            if value.data is None:
                raise CaptureError(
                    'Prepare plaintext Tensor data before capturing its use'
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
        if isinstance(value, KeySwitchKey):
            return value
        if isinstance(value, torch.Tensor):
            return _Value(self.material(value, core.MessageType()), self)
        if value is dataclasses.replace:

            def replace_value(obj, **fields):
                if isinstance(obj, _Value):
                    return obj.replace(**fields)
                return self.substitute(
                    dataclasses.replace(unwrap(obj), **fields)
                )

            return replace_value
        if value is id:
            return lambda obj: id(
                obj.engine if isinstance(obj, _EngineCalls) else unwrap(obj)
            )
        if value is isinstance:

            def static_isinstance(obj, classes):
                obj = (
                    obj.engine if isinstance(obj, _EngineCalls) else unwrap(obj)
                )
                return isinstance(obj, classes) or (
                    isinstance(obj, _Value)
                    and issubclass(obj._python_class(), classes)
                )

            return static_isinstance
        if value is type:

            def static_type(*args, **kwargs):
                if len(args) == 1 and not kwargs:
                    obj = args[0]
                    if isinstance(obj, _Value):
                        return obj._python_class()
                    return type(
                        obj.engine
                        if isinstance(obj, _EngineCalls)
                        else unwrap(obj)
                    )
                return type(*args, **kwargs)

            return static_type
        if isinstance(value, types.FunctionType):
            module = (value.__module__ or '').split('.')[0]
            if (
                module in {'torch', 'numpy', 'xdsl'}
                or module in sys.stdlib_module_names
            ):
                return value
            return self.function(value)
        if isinstance(value, tuple):
            return tuple(self.substitute(item) for item in value)
        if isinstance(value, list):
            return [self.substitute(item) for item in value]
        if isinstance(value, Mapping):
            return self.object(value)
        module = type(value).__module__.split('.')[0]
        if (
            not isinstance(value, (type, types.ModuleType))
            and module not in {'builtins', 'torch', 'numpy', 'xdsl'}
            and module not in sys.stdlib_module_names
            and (
                hasattr(value, '__dict__') or hasattr(type(value), '__slots__')
            )
        ):
            return self.object(value)
        return value

    def object(self, value):
        if id(value) not in self.objects:
            getter = type(value).__getattribute__
            expected = (
                type.__getattribute__
                if isinstance(value, type)
                else object.__getattribute__
            )
            if getter is not expected and type(value) not in (
                dict,
                types.MappingProxyType,
            ):
                raise CaptureError(
                    'Callable capture does not support custom __getattribute__'
                )
            self.objects[id(value)] = (
                StaticMapping(value, self)
                if isinstance(value, Mapping)
                else StaticObject(value, self)
            )
        return self.objects[id(value)]

    def numeric_call(self, function, args, kwargs):
        from ._tensor_capture import numeric_call

        args, kwargs = self.runtime(args), self.runtime(kwargs)
        with (
            self.preparation.pause()
            if self.preparation is not None
            else nullcontext()
        ):
            result = numeric_call(self, function, args, kwargs)
        return result

    def append_unit_axis(self, value):
        result = value.reshape(*value.shape, 1)
        return result

    def function(self, function: types.FunctionType) -> types.FunctionType:
        if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(
            function
        ):
            raise CaptureError(
                "Engine capture requires a synchronous Python function"
            )
        if function in self.functions:
            return self.functions[function]
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
        self.functions[function] = copied
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

        def global_reads(code):
            names = {
                instruction.argval
                for instruction in dis.get_instructions(code)
                if instruction.opname in {'LOAD_GLOBAL', 'LOAD_NAME'}
            }
            for constant in code.co_consts:
                if isinstance(constant, types.CodeType):
                    names.update(global_reads(constant))
            return names

        global_names = global_reads(function.__code__)
        for name in global_names:
            if name in {'id', 'type', 'isinstance'} and name not in globals_:
                globals_[name] = self.substitute(
                    {'id': id, 'type': type, 'isinstance': isinstance}[name]
                )
                continue
            if name in globals_:
                globals_[name] = self.external(
                    globals_[name], lambda name=name: function.__globals__[name]
                )
        for original, cell in zip(
            function.__closure__ or (), closure, strict=True
        ):
            cell.cell_contents = self.external(
                cell.cell_contents, lambda cell=original: cell.cell_contents
            )
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

    Python functions, bound methods and callable objects share source capture. Bound object members are read lazily; writes to those objects are rejected. Fixed Tensor/value references, including nested list/tuple/dict containers, become live material bindings. Their contents are not constant-folded. Numerical data is never executed during capture: ordinary Tensor metadata is propagated with FakeTensor kernels. Engine codec preparation calls become runtime operations with advancing rounding state.

    The original function is retained as the source reference. Function-backed
    compiled helpers are source-inlined under the outer compilation choices.
    Static Python control flow and structured outputs are supported. Data-based
    Python branching and in-place Tensor operations require other frontend or
    manual Program support; no capture fallback is attempted.
    """
    function = python_callable(function)
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
    seed_names = {}
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
        if isinstance(value, KeySwitchKey) and value.generated_seed is not None:
            extra = name + "_seed"
            while extra in ordered or extra in physical:
                extra += "_"
            physical[extra] = value.generated_seed
            input_fields[extra] = [name, "seed"]
            seed_names[name] = extra
    block = Block(arg_types=[value_type(value) for value in physical.values()])
    work = CompileWorkspace() if workspace is None else workspace
    capture = _Capture(work, block, material_names)
    if work.get("binding_time_split", True) and any(
        isinstance(value, Ciphertext) for value in ordered.values()
    ):
        from ._preparation_capture import PreparationRecorder

        capture.preparation = PreparationRecorder(capture, ordered)
    symbolic = {}
    for name, argument in zip(physical, block.args, strict=True):
        argument.name_hint = name
        symbolic[name] = _Value(argument, capture)
        physical_value = physical[name]
        if isinstance(physical_value, RotationKey):
            capture.key_steps[argument] = physical_value.rotation_step
    for name in runtime:
        value = symbolic[name]
        capture.runtime_inputs[id(runtime[name])] = value
        if name in implicit_names:
            value._implicit_value = symbolic[implicit_names[name]]
        if name in seed_names:
            capture.seeds[value.value] = symbolic[seed_names[name]].value
        if capture.preparation is None or not isinstance(
            runtime[name], (torch.Tensor, Plaintext, CompressedPlaintext)
        ):
            bound.arguments[name] = value
    try:
        if capture.preparation is None:
            output = cast(Callable[..., object], capture.substitute(function))(
                *bound.args, **bound.kwargs
            )
        else:
            output_holder = []

            def evaluate():
                output_holder.append(
                    capture.runtime(
                        cast(
                            Callable[..., object], capture.substitute(function)
                        )(*bound.args, **bound.kwargs)
                    )
                )

            capture.preparation.run(evaluate)
            output = output_holder[0]
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
                descriptor = {
                    "kind": "compressed_plaintext",
                    "data": data,
                    "implicit": implicit,
                    "fields": fields,
                }
                return descriptor
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

    if capture.preparation is not None:
        first = block.first_op
        declarations = {
            *capture.preparation.symbols,
            *(
                source["symbol"]
                for source in capture.preparation.inputs
                if source["kind"] == "material"
            ),
        }
        for operation in tuple(block.ops):
            if (
                isinstance(operation, core.MaterialRefOp)
                and operation.symbol is not None
                and operation.symbol.data in declarations
                and operation is not first
            ):
                assert first is not None
                block.detach_op(operation)
                block.insert_op_before(operation, first)
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
    if capture.descriptions:
        program.set_material_descriptions(capture.descriptions)
    if capture.resources:
        work[CAPTURED_RESOURCES] = tuple(capture.resources.values())
    compilation = Compilation(
        program, work, material_bindings=capture.materials
    )
    # The Compilation owns the bindings. Capture adapters must not retain an
    # earlier material generation through their Python closure cycles.
    capture.materials = {}
    capture.functions.clear()
    capture.objects.clear()
    capture.engines.clear()
    capture.runtime_inputs.clear()
    capture.preparation = None
    return compilation


CAPTURED_RESOURCES: WorkspaceKey[tuple[CkksDeviceResources, ...]] = (
    WorkspaceKey("captured_resources")
)


__all__ = ["CAPTURED_RESOURCES", "capture_eager"]
