r"""Residue structure of RNS operations: operand roles, result layout and component bodies.

An RNS operation that a code generator may implement inside a fused region registers an :class:`OperationStructure`. Its operands have roles:

- ``data``: a residue bundle read at the result's coefficient, prime row and batch position;
- ``compact``: a compressed plaintext row read through its compression layout;
- ``key``: a two-component evaluation-key digit read at the result's QP rows;
- ``parameters`` and ``table``: per-row modulus constants and transform tables;
- ``index``: a per-coefficient index or sign table shared by all rows, components and batch entries.

The result layout states how result extents follow from the data operands. ``same`` keeps the first data operand's layout; ``select`` removes the component axis by selecting component ``component``; ``pack`` stacks single-component operands on a new component axis; ``key_product`` forms two components from one digit row; ``component_product`` forms three components from two two-component operands; ``transform`` is a polynomial transform along the coefficient axis, whose stages a Backend schedules. ``permute`` keeps the data layout and reads each result coefficient from the coefficient selected by the ``index`` operands. ``reduce_rows`` computes every result row at one coefficient from all prime rows of the data operand at that coefficient; the result row set is a subset of the input rows, and the per-row constants come from the ``parameters`` operands.

``reduction`` declares an ordered accumulation axis. The result is $a_{T-1}$ with $a_0=f(\text{term}_0)$ and $a_t=a_{t-1}+f(\text{term}_t)$ in standard residues, where term $t$ reads the operands at ``reduction.terms[t]`` and $f$ is the ``reduction.term`` structure applied to those operands followed by the operation's parameters. ``body`` states the same fold with every term written out, so a code generator may either loop over the axis or emit the expanded expression. When ``reduction.stacked`` names an operand whose leading axis holds the terms, ``body`` is None and the reduction is the only description.

``body`` is None when the operation's arithmetic is not expressed as scalar component expressions; a code generator then needs an implementation specific to the layout. The body gives every result component as a :class:`Scalar` expression of operand components at one coefficient, row and batch position. Arithmetic is modulo the row's prime $q_i$: ``add`` and ``subtract`` give standard residues in $[0,q_i)$, ``add_lazy`` adds without the final reduction, ``negate`` maps $x\mapsto (q_i-x)\bmod q_i$, ``multiply`` is the Montgomery product $xyR^{-1}$, and ``to_montgomery``/``from_montgomery`` multiply by $R$ and $R^{-1}$. A ``compact`` node reads the compressed plaintext entry of the current coefficient; its first datum selects the compression layout (cyclic, contiguous or strided sparse).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

Role: TypeAlias = Literal[
    "data", "compact", "key", "parameters", "table", "index", "seed"
]
Layout: TypeAlias = Literal[
    "same",
    "select",
    "pack",
    "key_product",
    "component_product",
    "transform",
    "permute",
    "reduce_rows",
]
COMPRESSION_LAYOUTS = ("cyclic", "contiguous", "strided_sparse")


@dataclass(frozen=True)
class Scalar:
    """One residue expression; ``load`` leaves carry ``(operand, component)``."""

    op: str
    args: tuple[Scalar, ...] = ()
    data: tuple[int, ...] = ()


def load(operand: int, component: int = 0) -> Scalar:
    """Read component ``component`` of operand ``operand`` at the current position."""
    return Scalar("load", data=(operand, component))


StructureDescription: TypeAlias = Callable[[int, Mapping[str, object]], "OperationStructure"]
"""Map an operand count and Python attribute values to an operation's structure."""

ComponentBody: TypeAlias = Callable[[tuple[int, ...]], tuple[Scalar, ...]]
"""Map the component count of every operand position (0 for non-data roles) to the result components."""


@dataclass(frozen=True)
class Reduction:
    """An ordered sum over operand groups that share one term structure.

    When ``stacked`` names an operand, that operand appears in every group and its leading axis enumerates the terms: term $t$ reads row $t$.
    """

    terms: tuple[tuple[int, ...], ...]
    term: OperationStructure
    stacked: int | None = None


@dataclass(frozen=True)
class OperationStructure:
    """Operand roles, result layout and component body of one operation instance.

    ``plaintext`` marks operations whose second ``data`` operand is a prepared plaintext row without a component axis. ``accepted`` is False when this instance's attributes select semantics the body does not express.
    """

    roles: tuple[Role, ...]
    layout: Layout
    body: ComponentBody | None = None
    component: int | None = None
    accepted: bool = True
    plaintext: bool = False
    reduction: Reduction | None = None

    def operands(self, *roles: Role) -> tuple[int, ...]:
        """Return operand positions with one of ``roles``."""
        return tuple(i for i, role in enumerate(self.roles) if role in roles)


def _broadcast(operand: int, component: int, counts: tuple[int, ...]) -> Scalar:
    return load(operand, component if counts[operand] > 1 else 0)


def elementwise(code: str, arity: int) -> OperationStructure:
    """``arity`` residue operands and parameters; every component applies ``code``.

    Single-component operands broadcast over the first operand's components.
    """

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        return tuple(
            Scalar(code, tuple(_broadcast(i, c, counts) for i in range(arity)))
            for c in range(counts[0])
        )

    return OperationStructure(("data",) * arity + ("parameters",), "same", body)


def plaintext_product() -> OperationStructure:
    """Multiply every ciphertext component by plaintext row $p$: $c'_j=c_j p R^{-1}$."""

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        return tuple(
            Scalar("multiply", (load(0, c), load(1))) for c in range(counts[0])
        )

    return OperationStructure(
        ("data", "data", "parameters"), "same", body, plaintext=True
    )


def plaintext_sum(ntt: bool) -> OperationStructure:
    """Add plaintext row $p$ to component zero, $c'_0=c_0+p$, keeping the lazy range."""

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        return (
            Scalar("add_lazy", (load(0), load(1))),
            *(load(0, c) for c in range(1, counts[0])),
        )

    return OperationStructure(
        ("data", "data", "parameters"), "same", body, accepted=ntt, plaintext=True
    )


def compressed(
    operation: Literal["add", "multiply"],
    operand_count: int,
    layout: str,
    ntt_plaintext: bool,
) -> OperationStructure:
    r"""Combine a ciphertext with a compressed plaintext row $u$.

    Multiplication gives $c'_j=c_j u R^{-1}$. Addition gives $c'_0=c_0+u$ for an NTT/Montgomery plaintext and $c'_0=c_0+uR^{-1}$ otherwise. The strided-sparse layout reads its implicit entries from one additional compact operand.
    """
    compact_operands = tuple(range(1, operand_count - 1))
    index = COMPRESSION_LAYOUTS.index(layout)
    accepted = (layout == "strided_sparse") == (len(compact_operands) == 2)
    expanded = Scalar(
        "compact", tuple(load(i) for i in compact_operands), (index,)
    )

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        if operation == "multiply":
            return tuple(
                Scalar("multiply", (load(0, c), expanded))
                for c in range(counts[0])
            )
        term = expanded if ntt_plaintext else Scalar("from_montgomery", (expanded,))
        return (
            Scalar("add", (load(0), term)),
            *(load(0, c) for c in range(1, counts[0])),
        )

    return OperationStructure(
        ("data",) + ("compact",) * len(compact_operands) + ("parameters",),
        "same",
        body,
        accepted=accepted,
        plaintext=True,
    )


def key_product(seeded: bool = False) -> OperationStructure:
    """Multiply digit row $d$ by key digit $(k_0,k_1)$: $(d k_0 R^{-1}, d k_1 R^{-1})$.

    A seeded key adds its seed-row operand, from which code generation derives $k_1$.
    """

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        return tuple(Scalar("multiply", (load(0), load(1, c))) for c in range(2))

    roles: tuple[Role, ...] = ("data", "key", "parameters")
    return OperationStructure(
        roles + (("seed",) if seeded else ()), "key_product", body
    )


def tensor_product() -> OperationStructure:
    r"""Two-component product $(a_0b_0,\ a_0b_1+a_1b_0,\ a_1b_1)$ with Montgomery products and a lazy middle sum."""

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        def product(i: int, j: int) -> Scalar:
            return Scalar("multiply", (load(0, i), load(1, j)))

        return (
            product(0, 0),
            Scalar("add_lazy", (product(0, 1), product(1, 0))),
            product(1, 1),
        )

    return OperationStructure(
        ("data", "data", "parameters"), "component_product", body
    )


def select_component(component: int) -> OperationStructure:
    """Select one component of a bundle."""
    return OperationStructure(
        ("data",), "select", lambda counts: (load(0, component),), component
    )


def pack(count: int) -> OperationStructure:
    """Stack ``count`` single-component bundles as components."""
    return OperationStructure(
        ("data",) * count, "pack", lambda counts: tuple(load(i) for i in range(count))
    )


def transform(table_count: int) -> OperationStructure:
    """A coefficient-axis polynomial transform with parameters and ``table_count`` tables."""
    return OperationStructure(
        ("data", "parameters") + ("table",) * table_count, "transform"
    )


def permute(index_count: int) -> OperationStructure:
    r"""Coefficient permutation with sign: result coefficient $j$ is $s_j x_{\pi(j)} \bmod q_i$ for every row and component.

    Operands are the data bundle, its per-row parameters, and ``index_count`` per-coefficient tables ($\pi$, then $s\in\{\pm1\}$).
    """
    return OperationStructure(
        ("data", "parameters") + ("index",) * index_count, "permute"
    )


def reduce_rows(
    operand_count: int, attributes: Mapping[str, object]
) -> OperationStructure:
    """Row-mixing quotient at each coefficient over residue bundles followed by per-row constant tables.

    ``parameter_names`` counts the trailing constant operands; without it, only the first operand is a residue bundle.
    """
    names = attributes.get("parameter_names")
    parameter_count = len(names) if isinstance(names, tuple) else operand_count - 1
    data_count = operand_count - parameter_count
    return OperationStructure(
        ("data",) * data_count + ("parameters",) * parameter_count,
        "reduce_rows",
    )


def weighted_sum(
    operand_count: int, attributes: Mapping[str, object]
) -> OperationStructure:
    r"""Ordered sum $\sum_t c_t \widetilde p_t R^{-1}$ of ``term_count`` ciphertext/plaintext products.

    Operands are $T$ ciphertexts, $T$ plaintexts and the parameters; each plaintext is a dense row or a cyclic/contiguous compact row. Strided-sparse plaintexts carry implicit operands and are not described. With ``plaintext_term_axis`` one plaintext operand stacks the $T$ rows on its leading axis; the structure then declares the reduction only, with no expanded body.
    """
    count = int(cast(int, attributes["term_count"]))
    if "plaintext_term_axis" in attributes:
        (layout,) = cast(tuple[str, ...], attributes.get("plaintext_layouts") or ("dense",))
        accepted = (
            attributes["plaintext_term_axis"] == 0
            and operand_count == count + 2
            and layout != "strided_sparse"
        )
        term = (
            plaintext_product()
            if layout == "dense"
            else compressed("multiply", 3, layout, False)
        )
        return OperationStructure(
            ("data",) * count
            + ("data" if layout == "dense" else "compact", "parameters"),
            "same",
            accepted=accepted,
            reduction=Reduction(
                tuple((t, count) for t in range(count)), term, stacked=count
            )
            if accepted
            else None,
        )
    layouts = attributes.get("plaintext_layouts") or ("dense",) * count
    layouts = cast(tuple[str, ...], layouts)
    accepted = operand_count == 2 * count + 1 and "strided_sparse" not in layouts
    terms = tuple(
        plaintext_product()
        if layout == "dense"
        else compressed("multiply", 3, layout, False)
        for layout in layouts
    )
    groups = tuple((t, count + t) for t in range(count))

    def body(counts: tuple[int, ...]) -> tuple[Scalar, ...]:
        total: tuple[Scalar, ...] = ()
        for (ciphertext, plaintext), term in zip(groups, terms, strict=True):
            positions = (ciphertext, plaintext, operand_count - 1)

            def rebase(item: Scalar) -> Scalar:
                if item.op == "load":
                    return load(positions[item.data[0]], item.data[1])
                return Scalar(item.op, tuple(rebase(a) for a in item.args), item.data)

            product = tuple(
                rebase(item)
                for item in cast(ComponentBody, term.body)(
                    (counts[ciphertext], counts[plaintext], 0)
                )
            )
            total = (
                product
                if not total
                else tuple(
                    Scalar("add", (a, b)) for a, b in zip(total, product, strict=True)
                )
            )
        return total

    return OperationStructure(
        ("data",) * count
        + tuple("data" if layout == "dense" else "compact" for layout in layouts)
        + ("parameters",),
        "same",
        body,
        accepted=accepted,
        reduction=Reduction(groups, terms[0])
        if accepted and len(set(layouts)) == 1
        else None,
    )


def attribute_values(operation) -> dict[str, object]:
    """Return string, integer and array-of-those attributes of an operation as Python values; arrays become tuples."""
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    def convert(value):
        if isinstance(value, StringAttr):
            return value.data
        if isinstance(value, IntegerAttr):
            return int(value.value.data)
        if isinstance(value, ArrayAttr):
            items = tuple(convert(item) for item in value)
            return None if None in items else items
        return None

    values: dict[str, object] = {}
    for name, value in operation.attributes.items():
        converted = convert(value)
        if converted is not None:
            values[name] = converted
    return values


_STRUCTURES: dict[object, OperationStructure | None] = {}
STRUCTURE_LOOKUPS = {"hits": 0, "misses": 0}


def operation_structure(operation) -> OperationStructure | None:
    """Resolve the registered residue structure of an IR operation instance.

    Results are memoized by operation type, operand count and attributes. Operations without a registered structure resolve to None before their attributes are hashed.
    """
    from ._operation_specs import DEFAULT_OPERATION_SPECS

    spec = DEFAULT_OPERATION_SPECS.get(operation.name)
    if (
        spec is None
        or spec.structure is None
        or spec.operation_type is not type(operation)
        or spec.effect != "pure"
    ):
        return None
    key = (type(operation), len(operation.operands), tuple(operation.attributes.items()))
    structure = _STRUCTURES.get(key, _STRUCTURES)
    if structure is not _STRUCTURES:
        STRUCTURE_LOOKUPS["hits"] += 1
        return structure
    STRUCTURE_LOOKUPS["misses"] += 1
    structure = spec.structure(len(operation.operands), attribute_values(operation))
    _STRUCTURES[key] = structure
    return structure


__all__ = [
    "COMPRESSION_LAYOUTS",
    "OperationStructure",
    "StructureDescription",
    "attribute_values",
    "Scalar",
    "compressed",
    "elementwise",
    "key_product",
    "load",
    "operation_structure",
    "pack",
    "permute",
    "plaintext_product",
    "plaintext_sum",
    "Reduction",
    "reduce_rows",
    "select_component",
    "transform",
    "weighted_sum",
]
