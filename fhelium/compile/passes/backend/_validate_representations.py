"""Validate fixed polynomial-representation ABIs before Backend selection."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue

from fhelium.ir import Program
from fhelium.ir.dialects import ckks, ntt, rns

from ..._pipeline import PassResult
from ._operations import executable_operations

_Representation = tuple[str, str]
_COEFFICIENT_STANDARD: _Representation = ("coefficient", "standard")
_COEFFICIENT_MONTGOMERY: _Representation = ("coefficient", "montgomery")
_NTT_MONTGOMERY: _Representation = ("ntt", "montgomery")


def _representation(value: SSAValue, *, label: str) -> _Representation:
    state = getattr(getattr(value.type, "state", None), "data", {})
    domain = state.get("polynomial_domain")
    residues = state.get("residue_representation")
    if (
        not isinstance(domain, StringAttr)
        or domain.data == "unknown"
        or not isinstance(residues, StringAttr)
        or residues.data == "unknown"
    ):
        raise ValueError(
            f"{label} requires concrete polynomial-domain and residue "
            "representation state"
        )
    return domain.data, residues.data


def _require(
    value: SSAValue,
    allowed: Collection[_Representation],
    *,
    label: str,
) -> None:
    representation = _representation(value, label=label)
    if representation not in allowed:
        raise ValueError(
            f"{label} has representation {representation!r}; "
            f"expected one of {tuple(sorted(allowed))!r}"
        )


def _require_values(
    operation: Operation,
    values: tuple[SSAValue, ...],
    allowed: Collection[_Representation],
) -> None:
    for index, value in enumerate(values):
        _require(value, allowed, label=f"{operation.name} value {index}")


def _validate_ckks(operation: Operation) -> bool:
    if isinstance(operation, ckks.ToNttOp):
        _require(
            operation.value,
            {_COEFFICIENT_STANDARD, _COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_NTT_MONTGOMERY},
            label=f"{operation.name} result",
        )
    elif isinstance(operation, ckks.FromNttOp):
        _require(
            operation.value,
            {_NTT_MONTGOMERY},
            label=f"{operation.name} input",
        )
        results = (
            {_COEFFICIENT_STANDARD}
            if isinstance(operation.result.type, ckks.CiphertextType)
            else {_COEFFICIENT_STANDARD, _COEFFICIENT_MONTGOMERY}
        )
        _require(operation.result, results, label=f"{operation.name} result")
    elif isinstance(operation, ckks.ToMontgomeryResiduesOp):
        _require(
            operation.value,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} result",
        )
    elif isinstance(operation, ckks.ToStandardResiduesOp):
        _require(
            operation.value,
            {_COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} result",
        )
    elif isinstance(operation, (ckks.MultiplyOp, ckks.MultiplyPlaintextOp)):
        _require_values(
            operation,
            (*operation.operands, operation.results[0]),
            {_NTT_MONTGOMERY},
        )
    elif isinstance(operation, ckks.EncryptOp):
        results = (
            {_NTT_MONTGOMERY}
            if operation.output_domain.data == "ntt"
            else {_COEFFICIENT_STANDARD}
        )
        _require(operation.result, results, label=f"{operation.name} result")
    elif isinstance(operation, ckks.RelinearizeOp):
        _require(
            operation.value,
            {_NTT_MONTGOMERY},
            label=f"{operation.name} input",
        )
        results = (
            {_NTT_MONTGOMERY}
            if operation.output_domain.data == "ntt"
            else {_COEFFICIENT_STANDARD}
        )
        _require(operation.result, results, label=f"{operation.name} result")
    elif isinstance(operation, ckks.RescaleOp):
        domain = operation.input_domain.data
        if operation.output_domain.data != domain:
            raise ValueError(f"{operation.name} must preserve its domain")
        values = (
            {_NTT_MONTGOMERY} if domain == "ntt" else {_COEFFICIENT_STANDARD}
        )
        _require_values(operation, (operation.value, operation.result), values)
    elif isinstance(operation, (ckks.SwitchKeyOp, ckks.ConjugateOp)):
        _require(
            operation.operands[0],
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} input",
        )
        results = (
            {_NTT_MONTGOMERY}
            if operation.output_domain.data == "ntt"
            else {_COEFFICIENT_STANDARD}
        )
        _require(
            operation.results[0], results, label=f"{operation.name} result"
        )
    elif isinstance(operation, ckks.AddPlaintextOp):
        ciphertext = _representation(
            operation.ciphertext, label=f"{operation.name} ciphertext"
        )
        if ciphertext not in {_COEFFICIENT_STANDARD, _NTT_MONTGOMERY}:
            raise ValueError(
                f"{operation.name} has unsupported ciphertext representation"
            )
        plaintext = (
            _COEFFICIENT_MONTGOMERY
            if ciphertext == _COEFFICIENT_STANDARD
            else _NTT_MONTGOMERY
        )
        _require(
            operation.plaintext,
            {plaintext},
            label=f"{operation.name} plaintext",
        )
        _require(
            operation.result, {ciphertext}, label=f"{operation.name} result"
        )
    elif isinstance(operation, ckks.GroupedRotationWeightedSumOp):
        _require(
            operation.value,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} input",
        )
        _require_values(
            operation,
            (*operation.plaintexts, operation.result),
            {_NTT_MONTGOMERY},
        )
    elif isinstance(operation, (ckks.RotateOp, ckks.RotateManyOp)):
        input_domain = (
            operation.input_domain.data
            if isinstance(operation, ckks.RotateOp)
            else "coefficient"
        )
        _require(
            operation.operands[0],
            {
                _NTT_MONTGOMERY
                if input_domain == "ntt"
                else _COEFFICIENT_STANDARD
            },
            label=f"{operation.name} input",
        )
        domain = operation.output_domain.data
        if domain not in {"coefficient", "ntt"}:
            raise ValueError(
                f"{operation.name} has unsupported output_domain {domain!r}"
            )
        results = (
            {_COEFFICIENT_STANDARD}
            if domain == "coefficient"
            else {_NTT_MONTGOMERY}
        )
        _require_values(operation, tuple(operation.results), results)
    elif isinstance(operation, ckks.DecryptOp):
        domain = operation.input_domain.data
        if domain not in {"coefficient", "ntt"}:
            raise ValueError(
                f"{operation.name} has unsupported input_domain {domain!r}"
            )
        inputs = (
            {_COEFFICIENT_STANDARD}
            if domain == "coefficient"
            else {_NTT_MONTGOMERY}
        )
        _require(operation.ciphertext, inputs, label=f"{operation.name} input")
    else:
        return False
    return True


def _validate_ntt(operation: Operation) -> bool:
    expectations: dict[
        type[Operation], tuple[_Representation, _Representation]
    ] = {
        ntt.CoefficientStandardToNttMontgomeryOp: (
            _COEFFICIENT_STANDARD,
            _NTT_MONTGOMERY,
        ),
        ntt.CoefficientMontgomeryToNttMontgomeryOp: (
            _COEFFICIENT_MONTGOMERY,
            _NTT_MONTGOMERY,
        ),
        ntt.NttMontgomeryToCoefficientStandardOp: (
            _NTT_MONTGOMERY,
            _COEFFICIENT_STANDARD,
        ),
        ntt.InverseMontgomeryOp: (
            _NTT_MONTGOMERY,
            _COEFFICIENT_MONTGOMERY,
        ),
    }
    expected = expectations.get(type(operation))
    if expected is None:
        return False
    _require(
        operation.operands[0], {expected[0]}, label=f"{operation.name} input"
    )
    _require(
        operation.results[0], {expected[1]}, label=f"{operation.name} result"
    )
    return True


def _validate_rns(operation: Operation) -> bool:
    fixed_same: dict[type[Operation], tuple[int, _Representation]] = {
        rns.MultiplyPlaintextOp: (2, _NTT_MONTGOMERY),
        rns.MontgomeryMultiplyOp: (2, _NTT_MONTGOMERY),
        rns.KeySwitchDigitProductOp: (1, _NTT_MONTGOMERY),
        rns.AddMontgomeryLazyOp: (2, _NTT_MONTGOMERY),
        rns.ModDownNttQpToQOp: (1, _NTT_MONTGOMERY),
        rns.ModDownQpToQOp: (1, _COEFFICIENT_STANDARD),
        rns.CoefficientAutomorphismOp: (1, _COEFFICIENT_STANDARD),
    }
    fixed = fixed_same.get(type(operation))
    if isinstance(
        operation,
        (rns.MontgomeryWeightedSumOp, rns.MontgomeryWeightedSumsOp),
    ):
        _require_values(
            operation,
            (*operation.terms, *operation.results),
            {_NTT_MONTGOMERY},
        )
    elif fixed is not None:
        value_operands, representation = fixed
        _require_values(
            operation,
            (*operation.operands[:value_operands], *operation.results),
            {representation},
        )
    elif isinstance(operation, rns.AddPlaintextOp):
        domain = operation.polynomial_domain.data
        ciphertext = (
            _NTT_MONTGOMERY if domain == "ntt" else _COEFFICIENT_STANDARD
        )
        plaintext = (
            _NTT_MONTGOMERY if domain == "ntt" else _COEFFICIENT_MONTGOMERY
        )
        _require(
            operation.ciphertext,
            {ciphertext},
            label=f"{operation.name} ciphertext",
        )
        _require(
            operation.plaintext,
            {plaintext},
            label=f"{operation.name} plaintext",
        )
        _require(
            operation.result, {ciphertext}, label=f"{operation.name} result"
        )
    elif isinstance(operation, rns.RescaleDropLeadingPrimesOp):
        domain = operation.input_domain.data
        if operation.output_domain.data != domain:
            raise ValueError(f"{operation.name} must preserve its domain")
        representation = (
            _NTT_MONTGOMERY if domain == "ntt" else _COEFFICIENT_STANDARD
        )
        _require_values(
            operation,
            (operation.value, operation.result),
            {representation},
        )
    elif isinstance(operation, rns.HybridModUpDigitOp):
        _require(
            operation.source,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} result",
        )
    elif isinstance(operation, rns.StandardToMontgomeryOp):
        _require(
            operation.value,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} result",
        )
    elif isinstance(operation, rns.MontgomeryToStandardOp):
        _require(
            operation.value,
            {_COEFFICIENT_MONTGOMERY},
            label=f"{operation.name} input",
        )
        _require(
            operation.result,
            {_COEFFICIENT_STANDARD},
            label=f"{operation.name} result",
        )
    else:
        return False
    return True


@dataclass(frozen=True)
class ValidateExecutionRepresentationsPass:
    """Require concrete legal representations at fixed executable ABIs."""

    name: str = "validate-execution-representations"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        del workspace
        matched = 0
        for operation in executable_operations(program):
            matched += int(
                _validate_ckks(operation)
                or _validate_ntt(operation)
                or _validate_rns(operation)
            )
        return PassResult.unchanged(program, matched=matched)


__all__ = ["ValidateExecutionRepresentationsPass"]
