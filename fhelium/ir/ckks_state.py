"""Metadata equations shared by CKKS execution and callable capture.

These functions derive scale, prime-row, and representation metadata from operation inputs.
"""

from __future__ import annotations

from typing import Literal

from fhelium.config import CkksConfig
from fhelium.values._scale import coerce_scale
from fhelium.values.state import (
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)


def product_scale(lhs: float, rhs: float) -> float:
    r"""Return $\Delta_{out}=\Delta_{lhs}\Delta_{rhs}$ without rescaling."""

    return lhs * rhs


def quotient_scale(scale: float, divisor: int) -> float:
    r"""Return $\Delta_{out}=\Delta/M$ for the complete dropped-group product."""

    return scale / float(divisor)


def output_residues(domain: PolynomialDomain) -> ResidueRepresentation:
    """Return the ciphertext residue representation for a selected domain."""

    if domain not in {"coefficient", "ntt"}:
        raise ValueError("output_domain must be 'coefficient' or 'ntt'")
    return "montgomery" if domain == "ntt" else "standard"


def transform_state(
    direction: Literal["forward", "inverse"], *, ciphertext: bool
) -> tuple[PolynomialDomain, ResidueRepresentation]:
    """Return the public state after a forward or inverse negacyclic NTT.

    Forward transforms produce NTT/Montgomery values. Inverse transforms produce
    coefficient/standard ciphertexts or coefficient/Montgomery plaintexts.
    """

    if direction == "forward":
        return "ntt", "montgomery"
    return "coefficient", "standard" if ciphertext else "montgomery"


def depth_prime_ids(
    config: CkksConfig, depth: int, basis: ModulusBasis
) -> tuple[int, ...]:
    """Select the Q suffix at a depth and retain P rows for a QP value."""

    stop = config.total_num_primes if basis == "QP" else config.num_q_primes
    return tuple(range(config.q_row_start(depth), stop))


def reinterpreted_scale(
    current: float, target: float, max_relative_change: float | None = None
) -> float:
    """Validate the caller's metadata-only scale change and return its target."""

    from fhelium.errors import ScaleMismatchError

    scale = coerce_scale(target, value_name="target_scale")
    if max_relative_change is not None:
        limit = float(max_relative_change)
        if limit < 0:
            raise ValueError("max_relative_change must be non-negative")
        change = max(current / scale, scale / current) - 1.0
        if change > limit:
            raise ScaleMismatchError(
                operation="reinterpret_at_scale",
                lhs_name="current",
                lhs_scale=current,
                rhs_name="target",
                rhs_scale=scale,
            )
    return scale


__all__ = [
    "depth_prime_ids",
    "output_residues",
    "product_scale",
    "quotient_scale",
    "reinterpreted_scale",
    "transform_state",
]


def allocation_state(
    state: dict[str, object], **updates: object
) -> dict[str, object]:
    """Preserve mathematical metadata while dropping an allocation's old layout."""
    return {
        **{
            name: value
            for name, value in state.items()
            if name not in {"shape", "strides", "requires_grad"}
        },
        **updates,
    }


def arithmetic_state(
    operation: str,
    source: dict[str, object],
    *,
    rhs: dict[str, object] | None = None,
    config: CkksConfig | None = None,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Apply one CKKS metadata equation to ordinary represented facts.

    Depth is the consumed-depth index. Multiplication preserves depth and multiplies actual scales; quotient rescaling removes one complete configured group. The returned state describes a fresh allocation, whose physical strides remain unspecified.
    """
    options = options or {}
    state = allocation_state(source)
    if operation in {
        "add",
        "subtract",
        "multiply",
        "add_plaintext",
        "multiply_plaintext",
        "add_compressed_plaintext",
        "multiply_compressed_plaintext",
    }:
        assert rhs is not None
        plaintext = "plaintext" in operation
        fields = ("depth", "prime_ids", "basis")
        if not plaintext:
            fields += (
                "polynomial_domain",
                "residue_representation",
                "components",
            )
        if not operation.startswith("multiply"):
            fields += ("scale",)
        for name in fields:
            if source.get(name) != rhs.get(name):
                raise ValueError(f"{operation} operands disagree on {name}")
        if operation.startswith("multiply"):
            state["scale"] = product_scale(source["scale"], rhs["scale"])
        if operation == "multiply":
            state["components"] = 3
    elif operation in {"to_ntt", "from_ntt"}:
        domain, residues = transform_state(
            "forward" if operation == "to_ntt" else "inverse",
            ciphertext="components" in source,
        )
        state.update(polynomial_domain=domain, residue_representation=residues)
    elif operation in {"to_montgomery", "to_standard"}:
        state["residue_representation"] = (
            "montgomery" if operation == "to_montgomery" else "standard"
        )
    elif operation in {"add_scalar", "multiply_scalar"}:
        scale = (
            source["scale"]
            if options.get("scalar_scale_from_input")
            else options["scalar_scale"]
        )
        if operation == "multiply_scalar":
            state["scale"] = product_scale(source["scale"], scale)
    elif operation == "reinterpret_scale":
        state["scale"] = reinterpreted_scale(
            source["scale"],
            options["scale"],
            options.get("max_relative_change"),
        )
    elif operation == "sum_batch":
        batch = tuple(source["batch_shape"])
        axis = options["axis"]
        state["batch_shape"] = batch[:axis] + batch[axis + 1 :]
    elif operation == "select_batch":
        state["batch_shape"] = tuple(source["batch_shape"])[1:]
    elif operation in {
        "rotate",
        "rotate_many",
        "relinearize",
        "switch_key",
        "conjugate",
        "rotation_sum",
    }:
        domain = options.get("output_domain", "coefficient")
        state.update(
            polynomial_domain=domain,
            residue_representation=output_residues(domain),
        )
        if operation in {"relinearize", "switch_key", "conjugate"}:
            state["components"] = 2
    if operation == "rescale" or (
        operation in {"relinearize", "rotation_sum"} and options.get("rescale")
    ):
        assert config is not None
        depth = source["depth"]
        if depth >= config.max_depth:
            raise ValueError(
                "Rescale leaves the configured public modulus chain"
            )
        state.update(
            depth=depth + 1,
            prime_ids=depth_prime_ids(config, depth + 1, source["basis"]),
            scale=quotient_scale(
                source["scale"], config.rescale_divisor(depth)
            ),
        )
    elif operation in {
        "mod_switch",
        "modulus_raise",
        "encapsulated_modulus_raise",
    }:
        assert config is not None
        offset = options.get("depth_offset")
        depth = (
            source["depth"] + offset
            if offset is not None
            else options["target_depth"]
        )
        if operation == "mod_switch" and depth < source["depth"]:
            raise ValueError("ModSwitch cannot restore removed Q rows")
        if operation != "mod_switch" and depth > source["depth"]:
            raise ValueError(
                "Modulus raise target must precede or equal source depth"
            )
        if not 0 <= depth <= config.max_depth:
            raise ValueError(
                "Target depth is outside the configured modulus chain"
            )
        basis = source["basis"] if operation == "mod_switch" else "Q"
        state.update(
            depth=depth, prime_ids=depth_prime_ids(config, depth, basis)
        )
        if operation != "mod_switch":
            state.update(
                polynomial_domain="coefficient",
                residue_representation="standard",
            )
    return state
