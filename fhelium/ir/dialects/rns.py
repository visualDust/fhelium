r"""Logical residue-number-system operations used below CKKS lowering.

A residue number system (RNS) stores a polynomial row modulo each active
ciphertext prime in Q.  The QP basis appends special P primes used by hybrid
key switching.  A hybrid digit is one selected group of Q rows lifted together
during that procedure.  Montgomery representation stores residue
$x_i$ as $x_iR_i\bmod q_i$; number-theoretic-transform (NTT) form
stores the negacyclic transform of each polynomial row.

Each operation defines a transformation of an RNS bundle and its representation state. Parameter and table operands carry numerical Tensors supplied by Eager or bound to a Program before execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from xdsl.dialects.builtin import (
    ArrayAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    AttrSizedOperandSegments,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import (
    OperationSpec,
    flat_with_attributes,
    flat_without_attributes,
    registered_operation_spec,
)
from ._common import OpenStateType
from .._dependencies import (
    DependencyKind,
    OperationDependencies,
    ValueDependency,
    operand_dependencies,
    operand_relations,
    string_fact,
)


@irdl_attr_definition
class RnsBundleType(OpenStateType):
    """RNS components with partial arithmetic, layout, and placement state."""

    name = "fhelium_rns.bundle"


@irdl_attr_definition
class RnsParametersType(OpenStateType):
    """CKKS-parameter-specific RNS moduli and Montgomery parameters."""

    name = "fhelium_rns.parameters"


@irdl_op_definition
class AddStandardOp(IRDLOperation):
    r"""Add two standard-residue polynomial bundles row by row.

    An RNS (residue number system) row stores one polynomial modulo an active
    prime $q_i$.  For every component, coefficient, and row, this operation
    computes $z_i=(x_i+y_i)\bmod q_i$.  Both operands use the same Q or QP
    prime rows and store values in the standard range $[0,q_i)$.  Component
    shape, depth, basis, polynomial domain, Montgomery factor, and scale are
    unchanged."""

    name = "fhelium_rns.add_standard"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class SubtractStandardOp(IRDLOperation):
    r"""Subtract standard-residue polynomial bundles row by row.

    For each active prime $q_i$, component, and coefficient, the result is
    $z_i=(x_i-y_i)\bmod q_i$.  The operation preserves component shape, prime
    rows, depth, modulus basis, polynomial domain, and scale."""

    name = "fhelium_rns.subtract_standard"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class NegateStandardOp(IRDLOperation):
    r"""Negate a standard-residue polynomial bundle row by row.

    For every active prime $q_i$, component, and coefficient, the result is
    $z_i=(-x_i)\bmod q_i$.  Component shape and all represented CKKS state are
    preserved."""

    name = "fhelium_rns.negate_standard"
    value = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                },
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class SumStandardBatchOp(IRDLOperation):
    r"""Reduce one leading batch axis of a standard-residue bundle.

    The operand is $[*outer, n, *inner, \text{limb}, \text{coefficient}]$
    with standard residues $x^{(k)}_{j,i}\in[0,q_i)$.  ``dim`` selects the
    reduced axis of length $n$; the result keeps the remaining axes and
    computes
    $z_{j,i}=\sum_{k<n}x^{(k)}_{j,i}\bmod q_i$
    with every partial sum reduced to $[0,q_i)$.  The reduction is the
    modular sum of the selected axis: component, limb, coefficient, depth,
    scale, polynomial domain, and residue representation are preserved while
    the selected axis leaves the batch shape."""

    name = "fhelium_rns.sum_standard_batch"
    value = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    dim = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'mixing',
                },
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        dim: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "dim": (IntegerAttr(dim, 64) if isinstance(dim, int) else dim)
            },
        )


class _CiphertextPlaintextOp(IRDLOperation):
    ciphertext = operand_def(RnsBundleType)
    plaintext = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        ciphertext: SSAValue | Operation,
        plaintext: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[ciphertext, plaintext, parameters],
            result_types=[result_type],
            attributes=attributes,
        )


@irdl_op_definition
class AddPlaintextOp(_CiphertextPlaintextOp):
    r"""Add a prepared plaintext polynomial to ciphertext component zero.

    For ciphertext $(c_0,\ldots,c_{k-1})$ and plaintext polynomial $p$, each
    active-prime row computes $c'_0=c_0+p\pmod {q_i}$ and
    $c'_j=c_j$ for $j>0$. With coefficient/standard ciphertext input, the
    prepared plaintext uses coefficient-domain Montgomery residues and the
    operation returns standard residues. With NTT/Montgomery input, both
    operands and the result remain NTT/Montgomery. Depth, component count, and
    ciphertext scale are preserved."""

    name = "fhelium_rns.add_plaintext"
    polynomial_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'reindexed',
                },
            },
        )


@irdl_op_definition
class MultiplyPlaintextOp(_CiphertextPlaintextOp):
    r"""Multiply every ciphertext component by a prepared plaintext.

    The ciphertext and plaintext are NTT-domain Montgomery bundles on matching
    prime rows.  For each component $j$, prime $q_i$, and transform index
    $k$, Montgomery multiplication computes
    $c'_{j,i,k}=c_{j,i,k}p_{i,k}R_i^{-1}\bmod q_i$.  Thus the polynomial
    meaning is $c'_j=c_jp$ in $R_{q_i}$.  Component count and depth remain;
    the CKKS result scale is the product of operand scales."""

    name = "fhelium_rns.multiply_plaintext"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'reindexed',
                },
            },
        )


def _weighted_layout_attributes(layouts, chunk_size):
    attributes = {}
    if layouts:
        attributes["plaintext_layouts"] = ArrayAttr(
            StringAttr(x) for x in layouts
        )
    if chunk_size is not None:
        attributes["term_chunk_size"] = IntegerAttr(chunk_size, 64)
    return attributes


def _verify_weighted_layouts(operation, terms, groups):
    if terms <= 0 or groups <= 0:
        raise VerifyException(
            "Weighted sums require positive term and group counts"
        )
    layouts = operation.plaintext_layouts
    names = tuple(x.data for x in layouts) if layouts is not None else ()
    if names and (
        len(names) != terms * groups
        or any(
            name not in {"dense", "cyclic", "contiguous", "strided_sparse"}
            for name in names
        )
    ):
        raise VerifyException(
            "Weighted-sum layouts must describe every plaintext"
        )
    if len(operation.terms) != terms * (groups + 1) + names.count(
        "strided_sparse"
    ):
        raise VerifyException(
            "Weighted-sum operands must include each sparse implicit row"
        )
    chunk = operation.term_chunk_size
    if chunk is not None and chunk.value.data <= 0:
        raise VerifyException("Weighted-sum term chunk size must be positive")


@irdl_op_definition
class MontgomeryWeightedSumOp(IRDLOperation):
    r"""Sum ciphertext/plaintext products in NTT/Montgomery representation.

    For $T$ ciphertext bundles $c_t$ followed by $T$ plaintext bundles $p_t$,
    each component, batch element, active-prime row, and NTT index computes

    $$
    y=\sum_{t=0}^{T-1}c_t\widetilde p_t R_i^{-1}\pmod {q_i}.
    $$

    Here $c_t$ and $\widetilde p_t$ are stored Montgomery residues;
    $\widetilde p_t$ denotes the logical expansion of its plaintext layout.
    The operands are ciphertexts, plaintexts, one implicit row Tensor per
    strided-sparse plaintext in plaintext order, then RNS parameters.
    ``plaintext_layouts`` selects dense, cyclic, contiguous or strided-sparse
    indexing; omitting it means dense. Sparse rows have shape ``[*batch, limb]``.
    ``term_chunk_size`` optionally bounds generated expressions.

    Every term uses the same component and batch shape, depth, prime rows,
    ciphertext scale, and plaintext scale.  ``term_count`` separates the two
    equal operand ranges.  The result keeps the ciphertext shape, depth, and
    NTT/Montgomery representation; its CKKS scale is the product of the common
    ciphertext and plaintext scales.
    """

    name = "fhelium_rns.montgomery_weighted_sum"
    terms = var_operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    term_count = attr_def(IntegerAttr)
    plaintext_layouts = opt_attr_def(ArrayAttr[StringAttr])
    term_chunk_size = opt_attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Map dense/compact plaintext coordinates and shared ciphertext components."""
        count = int(self.term_count.value.data)
        layouts = (
            tuple(x.data for x in self.plaintext_layouts)
            if self.plaintext_layouts is not None
            else ()
        )
        relations = []
        for index in range(len(self.operands) - 1):
            plaintext = index - count
            compact = (
                plaintext >= 0
                and bool(layouts)
                and (plaintext >= len(layouts) or layouts[plaintext] != "dense")
            )
            relations.append(
                ValueDependency(
                    0,
                    index,
                    {
                        "coefficient": "reindexed" if compact else "element",
                        "limb": "element",
                        "component": "element"
                        if index < count
                        else "reindexed",
                        "batch": "reindexed",
                    },
                )
            )
        return OperationDependencies(tuple(relations))

    def __init__(
        self,
        parameters: SSAValue | Operation,
        ciphertexts: tuple[SSAValue | Operation, ...],
        plaintexts: tuple[SSAValue | Operation, ...],
        result_type: Attribute,
        *,
        plaintext_layouts: Sequence[str] = (),
        implicit_plaintexts: tuple[SSAValue | Operation, ...] = (),
        term_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            operands=(
                (*ciphertexts, *plaintexts, *implicit_plaintexts),
                parameters,
            ),
            result_types=[result_type],
            attributes={
                "term_count": IntegerAttr(len(ciphertexts), 64),
                **_weighted_layout_attributes(
                    plaintext_layouts, term_chunk_size
                ),
            },
        )

    def verify_(self) -> None:
        _verify_weighted_layouts(self, int(self.term_count.value.data), 1)


@irdl_op_definition
class MontgomeryWeightedSumsOp(IRDLOperation):
    r"""Apply a represented plaintext matrix to shared ciphertext terms.

    For $T$ ciphertext bundles $c_t$ and $G T$ plaintext bundles $p_{g,t}$,
    the operation returns $G$ independent bundles

    $$
    y_g=\sum_{t=0}^{T-1}c_t\widetilde p_{g,t}R_i^{-1}\pmod {q_i}.
    $$

    The inputs and output are stored Montgomery residues, and
    $\widetilde p_{g,t}$ is the logically expanded plaintext. Each sparse
    plaintext contributes one implicit row Tensor of shape ``[*batch, limb]``
    after the plaintext matrix and before parameters, in group-major order.
    ``plaintext_layouts`` describes the matrix entries; omission means dense.
    ``term_chunk_size`` optionally bounds generated expressions.

    Operands before ``parameters`` contain the $T$ ciphertexts followed by the
    group-major plaintext matrix. Every ciphertext has one NTT/Montgomery
    component, batch, depth, and prime-row layout; every plaintext has the
    matching row and batch layout and one common scale. The results keep the
    ciphertext state and have scale $\Delta_c\Delta_p$. ``term_count`` and
    ``group_count`` describe the supplied matrix.
    """

    name = "fhelium_rns.montgomery_weighted_sums"
    terms = var_operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    term_count = attr_def(IntegerAttr)
    plaintext_layouts = opt_attr_def(ArrayAttr[StringAttr])
    term_chunk_size = opt_attr_def(IntegerAttr)
    group_count = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Map dense/compact plaintext coordinates and shared ciphertext components."""
        count = int(self.term_count.value.data)
        layouts = (
            tuple(x.data for x in self.plaintext_layouts)
            if self.plaintext_layouts is not None
            else ()
        )
        relations = []
        for index in range(len(self.operands) - 1):
            plaintext = index - count
            compact = (
                plaintext >= 0
                and bool(layouts)
                and (plaintext >= len(layouts) or layouts[plaintext] != "dense")
            )
            relations.append(
                ValueDependency(
                    0,
                    index,
                    {
                        "coefficient": "reindexed" if compact else "element",
                        "limb": "element",
                        "component": "element"
                        if index < count
                        else "reindexed",
                        "batch": "reindexed",
                    },
                )
            )
        return OperationDependencies(tuple(relations))

    def __init__(
        self,
        parameters: SSAValue | Operation,
        ciphertexts: tuple[SSAValue | Operation, ...],
        plaintexts: tuple[SSAValue | Operation, ...],
        result_type: Attribute,
        *,
        group_count: int,
        plaintext_layouts: Sequence[str] = (),
        implicit_plaintexts: tuple[SSAValue | Operation, ...] = (),
        term_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            operands=(
                (*ciphertexts, *plaintexts, *implicit_plaintexts),
                parameters,
            ),
            result_types=[result_type],
            attributes={
                "term_count": IntegerAttr(len(ciphertexts), 64),
                "group_count": IntegerAttr(group_count, 64),
                **_weighted_layout_attributes(
                    plaintext_layouts, term_chunk_size
                ),
            },
        )

    def verify_(self) -> None:
        _verify_weighted_layouts(
            self,
            int(self.term_count.value.data),
            int(self.group_count.value.data),
        )


@irdl_op_definition
class RescaleDropLeadingPrimesOp(IRDLOperation):
    r"""Divide-round by the product of a leading prime group and remove its rows.

    For the first ``drop_count`` primes, let $M$ be their product and
    $r=x\bmod M$ in $[0,M)$. Floor quotient uses $t=r$; nearest quotient
    uses $t=r-M$ when $r>M/2$, otherwise $t=r$. Each surviving row is
    $y_i=(x_i-t)M^{-1}\bmod q_i$. Component count and polynomial domain
    are preserved. NTT/Montgomery input may retain surviving evaluations
    while constructing the rounded coefficient correction from dropped rows.
    The caller supplies resulting CKKS depth and scale; this RNS operation
    describes a row-group quotient, not a count of CKKS transitions."""

    name = "fhelium_rns.rescale_drop_leading_primes"
    value = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    drop_count = attr_def(IntegerAttr, default_value=IntegerAttr(1, 64))
    rounding = opt_attr_def(StringAttr)
    input_domain = attr_def(StringAttr, default_value=StringAttr("coefficient"))
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe the rounded quotient in its represented polynomial domain."""
        domain = string_fact(
            self, "input_domain", state_name="polynomial_domain"
        )
        position: DependencyKind = (
            "element"
            if domain == "coefficient"
            else "mixing"
            if domain == "ntt"
            else "unknown"
        )
        return operand_dependencies(
            self,
            (0,),
            {
                "coefficient": position,
                "limb": "mixing",
                "component": "element",
                "batch": "element",
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        drop_count: int = 1,
        rounding: str | StringAttr = "nearest",
        polynomial_domain: Literal["coefficient", "ntt"] = "coefficient",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["drop_count"] = IntegerAttr(drop_count, 64)
        attrs["rounding"] = (
            StringAttr(rounding) if isinstance(rounding, str) else rounding
        )
        attrs["input_domain"] = StringAttr(polynomial_domain)
        attrs["output_domain"] = StringAttr(polynomial_domain)
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        """Accept only the represented CKKS quotient-rounding rules."""

        if self.drop_count.value.data < 1:
            raise VerifyException("RNS rescale drop_count must be positive")
        if self.rounding is not None and self.rounding.data not in {
            "nearest",
            "floor",
        }:
            raise VerifyException(
                "RNS rescale rounding must be 'nearest' or 'floor'"
            )
        if (
            self.input_domain.data not in {"coefficient", "ntt"}
            or self.output_domain != self.input_domain
        ):
            raise VerifyException(
                "RNS rescale must preserve coefficient or NTT domain"
            )


@irdl_op_definition
class ExtractComponentOp(IRDLOperation):
    r"""Select one ciphertext-component polynomial without arithmetic.

    For an RNS bundle $(c_0,\ldots,c_{k-1})$, ``component=j`` returns $c_j$
    with the same coefficient values, active prime rows, polynomial domain,
    Montgomery state, depth, and scale."""

    name = "fhelium_rns.extract_component"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    component = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        component: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={
                "component": (
                    IntegerAttr(component, 64)
                    if isinstance(component, int)
                    else component
                )
            },
        )

    def verify_(self) -> None:
        if int(self.component.value.data) < 0:
            raise VerifyException("RNS component index must be nonnegative")


@irdl_op_definition
class PackTwoComponentsOp(IRDLOperation):
    r"""Stack two polynomial bundles as $(c_0,c_1)$.

    No residue arithmetic is performed.  Both inputs retain their active prime
    rows and representation; a new leading component axis forms a two-component
    ciphertext-shaped bundle."""

    name = "fhelium_rns.pack_two_components"
    component0 = operand_def(RnsBundleType)
    component1 = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
            },
        )

    def __init__(
        self,
        component0: SSAValue | Operation,
        component1: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[component0, component1],
            result_types=[result_type],
        )


@irdl_op_definition
class PackThreeComponentsOp(IRDLOperation):
    r"""Stack three polynomial bundles as $(c_0,c_1,c_2)$.

    No residue arithmetic is performed.  Matching prime rows and representation
    are preserved while a new leading component axis forms a three-component
    ciphertext-shaped bundle."""

    name = "fhelium_rns.pack_three_components"
    component0 = operand_def(RnsBundleType)
    component1 = operand_def(RnsBundleType)
    component2 = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                2: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
            },
        )

    def __init__(
        self,
        component0: SSAValue | Operation,
        component1: SSAValue | Operation,
        component2: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[component0, component1, component2],
            result_types=[result_type],
        )


@irdl_op_definition
class MontgomeryMultiplyOp(IRDLOperation):
    r"""Multiply NTT/Montgomery polynomial bundles pointwise.

    For active prime $q_i$, transform index $k$, and Montgomery radix $R_i$,
    the operation computes $z_{i,k}=x_{i,k}y_{i,k}R_i^{-1}\bmod q_i$.
    The result remains NTT/Montgomery and represents polynomial multiplication in
    $\mathbb{Z}_{q_i}[X]/(X^N+1)$.  Prime rows and structural axes are
    preserved."""

    name = "fhelium_rns.montgomery_multiply"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                2: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
            },
        )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class HybridModUpDigitOp(IRDLOperation):
    r"""Extend one hybrid-RNS digit from active Q rows to active QP rows.

    A hybrid digit is a selected group of Q-prime residues used in key switching.
    For each coefficient, the selected rows are reconstructed as one integer
    class $d$ modulo their product and basis-extended so every active Q and
    special P row stores $dR_i\bmod q_i$ or $dR_i\bmod p_i$.  The result is
    coefficient-domain Montgomery data; it does not change the source ciphertext
    scale."""

    name = "fhelium_rns.hybrid_modup_digit"
    source = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    digit_index = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'mixing',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        source: SSAValue | Operation,
        parameters: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        attributes: Mapping[str, Attribute] | None = None,
        digit_index: int = 0,
    ) -> None:
        super().__init__(
            operands=[source, parameters],
            result_types=[result_type],
            attributes={
                **(attributes or {}),
                "digit_index": IntegerAttr(digit_index, 64),
            },
        )

    def verify_(self) -> None:
        if int(self.digit_index.value.data) < 0:
            raise VerifyException(
                "Hybrid ModUp digit index must be nonnegative"
            )


@irdl_op_definition
class KeySwitchDigitProductOp(IRDLOperation):
    r"""Multiply one lifted digit by its evaluation-key pair.

    For NTT/Montgomery digit $d$ and key digit $(k_{d,0},k_{d,1})$, the result
    has two components
    $(d k_{d,0},d k_{d,1})$ in every active QP prime row.  Products use
    Montgomery pointwise multiplication and remain NTT/Montgomery.  The operation
    forms one summand of hybrid key switching and leaves CKKS depth and scale
    unchanged."""

    name = "fhelium_rns.key_switch_digit_product"
    digit = operand_def(RnsBundleType)
    key = operand_def()
    parameters = operand_def()
    result = result_def(RnsBundleType)
    key_digit_index = attr_def(IntegerAttr)
    key_role = opt_attr_def(StringAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'reindexed',
                    'batch': 'element',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'reindexed',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )

    def __init__(
        self,
        digit: SSAValue | Operation,
        key: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        key_digit_index: int,
        key_row_start: int = 0,
    ) -> None:
        super().__init__(
            operands=[digit, key, parameters],
            result_types=[result_type],
            attributes={
                "key_digit_index": IntegerAttr(key_digit_index, 64),
                "key_row_start": IntegerAttr(key_row_start, 64),
            },
        )

    def verify_(self) -> None:
        if int(self.key_digit_index.value.data) < 0:
            raise VerifyException(
                "Key-switch key digit index must be nonnegative"
            )


@irdl_op_definition
class AddMontgomeryLazyOp(IRDLOperation):
    r"""Accumulate two Montgomery bundles without full standard reduction.

    For each active prime $q_i$ or $p_i$, the represented result is
    $z=x+y\pmod {q_i}$ while storage may remain in the implementation's lazy
    range, currently bounded modulo $2q_i$.  Polynomial domain, Montgomery
    factor, prime rows, component axes, depth, and scale are preserved."""

    name = "fhelium_rns.add_montgomery_lazy"
    lhs = operand_def(RnsBundleType)
    rhs = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
                1: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'reindexed',
                },
            },
        )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class ModDownQpToQOp(IRDLOperation):
    r"""Remove the special-prime product P from a key-switch accumulator.

    Let $P=\prod_j p_j$ for the special P rows.  For each coefficient of each
    accumulator component, one P prime $p_d$ is removed at a time.  If
    $d=x\bmod p_d$, every surviving row becomes
    $(x_i-d)p_d^{-1}\bmod q_i$ or
    $(x_i-d)p_d^{-1}\bmod p_i$.  Repeating this residue-algebra division removes
    all P rows and returns active-Q residues. Nearest adds $H=(P-1)/2$
    before division; floor adds zero. The quotient is
    $\lfloor(x+H)/P\rfloor$ or $\lfloor x/P\rfloor$ respectively.
    This is the ModDown stage of hybrid
    key switching; key construction supplies the factor P, so CKKS message scale
    and depth do not change."""

    name = "fhelium_rns.moddown_qp_to_q"
    value = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    rounding = opt_attr_def(StringAttr, default_value=StringAttr("nearest"))
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'mixing',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        rounding: str | StringAttr = "nearest",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.setdefault(
            "rounding",
            StringAttr(rounding) if isinstance(rounding, str) else rounding,
        )
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "ModDown rounding must be 'floor' or 'nearest'"
            )


@irdl_op_definition
class ModDownAddRescaleOp(IRDLOperation):
    r"""Divide-round $v+Pa$ by auxiliary P and one complete Q group D.

    For $H_P=0$ or $(P-1)/2$ and $H_D=0$ or $(D-1)/2$, compute $\lfloor(v+Pa+H_P+PH_D)/(PD)\rfloor$ modulo the surviving Q primes. Inputs v and a are NTT/Montgomery; v contains the accumulator alone, and a supplies the corresponding prefix of component polynomials in Q. Actual ordered rows and both rounding offsets are supplied by numerical Tensor operands. The result contains only Q minus D in the requested polynomial domain and residue representation.
    """

    name = "fhelium_rns.moddown_add_rescale"
    value = operand_def(RnsBundleType)
    base = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    output_domain = attr_def(
        StringAttr, default_value=StringAttr("coefficient")
    )
    moddown_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    rescale_rounding = opt_attr_def(
        StringAttr, default_value=StringAttr("nearest")
    )
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        base: SSAValue | Operation,
        parameters: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        output_domain: Literal["coefficient", "ntt"] = "coefficient",
        moddown_rounding: str = "nearest",
        rescale_rounding: str = "nearest",
        attributes: Mapping[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=[value, base, parameters],
            result_types=[result_type],
            attributes={
                **(attributes or {}),
                "output_domain": StringAttr(output_domain),
                "moddown_rounding": StringAttr(moddown_rounding),
                "rescale_rounding": StringAttr(rescale_rounding),
            },
        )

    def dependencies(self) -> OperationDependencies:
        """Read matching components and batches, with coefficient/prime mixing."""
        return operand_relations(
            self,
            {
                index: {
                    "component": "element",
                    "limb": "mixing",
                    "batch": "element",
                    "coefficient": "mixing",
                }
                for index in (0, 1)
            },
        )

    def verify_(self) -> None:
        if self.output_domain.data not in {"coefficient", "ntt"}:
            raise VerifyException(
                "Joint quotient output_domain must be coefficient or ntt"
            )
        if self.moddown_rounding.data not in {
            "nearest",
            "floor",
        } or self.rescale_rounding.data not in {"nearest", "floor"}:
            raise VerifyException(
                "Joint quotient rounding must be nearest or floor"
            )


@irdl_op_definition
class ModDownNttQpToQOp(IRDLOperation):
    r"""Remove P from two QP accumulators and retain NTT/Montgomery Q rows.

    The input has shape ``[2, *batch, |Q|+|P|, N]``; the result has shape
    ``[2, *batch, |Q|, N]``. Parameter and transform-table operands supply
    numerical data for the same active RNS bases.

    Let $r$ be the coefficient remainder, in $[0,P)$ for floor or
    $[-(P-1)/2,(P-1)/2]$ for nearest, reconstructed from
    $\operatorname{INTT}_P(\widehat{x}_P)$. The result is
    $\widehat{x}_Q P^{-1}+\operatorname{NTT}_Q(-rP^{-1})\bmod q_i$.
    This equals coefficient-domain ModDown followed by forward NTT, without
    inverting the Q rows. Output drops P, preserves active Q rows and N,
    and remains in Montgomery representation. It does not rescale CKKS values.
    An optional coefficient/standard Q polynomial ``coefficient_c0`` adds
    $\operatorname{NTT}_Q(c_0)$ to component zero by adding $c_0$ to its
    coefficient correction before the forward transform.
    """

    name = "fhelium_rns.moddown_ntt_qp_to_q"
    value = operand_def(RnsBundleType)
    coefficient_c0 = opt_operand_def(RnsBundleType)
    parameters = var_operand_def()
    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    result = result_def(RnsBundleType)
    rounding = opt_attr_def(StringAttr, default_value=StringAttr("nearest"))
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        relations: dict[int, dict[str, DependencyKind]] = {
            0: {
                'coefficient': 'mixing',
                'limb': 'mixing',
                'component': 'element',
                'batch': 'element',
            }
        }
        if self.coefficient_c0 is not None:
            relations[1] = {
                'coefficient': 'mixing',
                'limb': 'element',
                'component': 'reindexed',
                'batch': 'element',
            }
        return operand_relations(self, relations)

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: Sequence[SSAValue | Operation],
        result_type: Attribute,
        *,
        coefficient_c0: SSAValue | Operation | None = None,
        rounding: str | StringAttr = "nearest",
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.setdefault(
            "rounding",
            StringAttr(rounding) if isinstance(rounding, str) else rounding,
        )
        super().__init__(
            operands=[value, coefficient_c0, parameters],
            result_types=[result_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.rounding.data not in {"floor", "nearest"}:
            raise VerifyException(
                "ModDown rounding must be 'floor' or 'nearest'"
            )


@irdl_op_definition
class CoefficientAutomorphismOp(IRDLOperation):
    r"""Apply the negacyclic ring automorphism $\sigma_g$.

    For odd ``galois_element`` $g$, $\sigma_g$ substitutes
    $X\mapsto X^g$ in $R_q=\mathbb{Z}_q[X]/(X^N+1)$.  A coefficient
    $a_jX^j$ moves to index $gj\bmod N$ and changes sign when reduction of
    $gj$ modulo $2N$ crosses $N$.  The same permutation is applied to
    every component and prime row.  Depth, scale, basis, and residue form remain
    unchanged."""

    name = "fhelium_rns.coefficient_automorphism"
    value = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    galois_element = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'reindexed',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
        *,
        galois_element: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                "galois_element": (
                    IntegerAttr(galois_element, 64)
                    if isinstance(galois_element, int)
                    else galois_element
                )
            },
        )

    def verify_(self) -> None:
        if int(self.galois_element.value.data) % 2 == 0:
            raise VerifyException("RNS Galois element must be odd")


class _ResidueConversionOp(IRDLOperation):
    value = operand_def(RnsBundleType)
    parameters = operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        parameters: SSAValue | Operation,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
        )


@irdl_op_definition
class StandardToMontgomeryOp(_ResidueConversionOp):
    r"""Convert coefficient residues to Montgomery representation.

    For every active prime $q_i$ and stored residue $x_i$, the result is
    $x_iR_i\bmod q_i$, where $R_i$ is the Montgomery radix.  Polynomial
    coefficients, prime rows, depth, basis, and CKKS scale otherwise do not
    change."""

    name = "fhelium_rns.standard_to_montgomery"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class MontgomeryToStandardOp(_ResidueConversionOp):
    r"""Remove the Montgomery factor from coefficient residues.

    For every active prime $q_i$, the operation maps stored
    $x_iR_i\bmod q_i$ to $x_i\bmod q_i$ by Montgomery reduction.  Polynomial
    coefficients, prime rows, depth, basis, and CKKS scale are preserved."""

    name = "fhelium_rns.montgomery_to_standard"

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )


@irdl_op_definition
class RestrictDepthOp(IRDLOperation):
    r"""Select the suffix of prime rows belonging to a later depth.

    If depth $\ell$ uses active rows $(q_\ell,\ldots,q_L)$, restriction to
    $t\ge\ell$ discards $(q_\ell,\ldots,q_{t-1})$ and copies the remaining
    rows.  No division or rounding occurs, so the actual scale and every surviving
    residue are unchanged.  P rows remain when the modulus basis is QP."""

    name = "fhelium_rns.restrict_depth"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    target_depth = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'reindexed',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        target_depth: int | IntegerAttr,
    ) -> None:
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={
                "target_depth": (
                    IntegerAttr(target_depth, 64)
                    if isinstance(target_depth, int)
                    else target_depth
                )
            },
        )


@irdl_op_definition
class ReinterpretScaleOp(IRDLOperation):
    r"""Replace scale metadata without changing an RNS payload.

    The result refers to the same residue classes and polynomial representation
    but records the supplied actual scale $\Delta'$.  Consequently decoding
    interprets the represented coefficients as $x/\Delta'$ rather than
    $x/\Delta$; no modular arithmetic, row selection, or depth transition is
    performed."""

    name = "fhelium_rns.reinterpret_scale"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    scale = attr_def(FloatAttr)
    traits = traits_def(Pure())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return operand_relations(
            self,
            {
                0: {
                    'coefficient': 'element',
                    'limb': 'element',
                    'component': 'element',
                    'batch': 'element',
                }
            },
        )

    def __init__(
        self,
        value: SSAValue | Operation,
        result_type: Attribute,
        *,
        scale: float | FloatAttr,
    ) -> None:
        scale_attribute: Attribute
        if isinstance(scale, FloatAttr):
            scale_attribute = scale
        else:
            scale_attribute = FloatAttr(float(scale), Float64Type())
        super().__init__(
            operands=[value],
            result_types=[result_type],
            attributes={"scale": scale_attribute},
        )


@irdl_op_definition
class CrtBasisExtendOp(IRDLOperation):
    r"""Evaluate a supplied CRT lift matrix in target Montgomery form.

    Source a_i is coefficient/standard. For D=product(q_i), D_i=D/q_i,
    and u_i=D_i^{-1} mod q_i, raw lift chooses sum(a_i D_i u_i).
    A nonempty source-normalizer operand first applies a_i*u_i mod q_i,
    and its matrix chooses sum((a_i*u_i mod q_i) D_i). Neither lift
    promises the least nonnegative representative modulo D.

    Numerical operands are source parameters, target parameters, the matrix,
    target reciprocals, source normalizers and source reciprocals. The matrix
    contains one target Montgomery factor; output is canonical Montgomery
    [*batch, target row, coefficient] with fresh storage. Empty normalizers
    and reciprocals select the raw matrix. All source and table operands are
    live and read-only. Ordered prime rows and their parameter data belong
    to the supplied operands, not to an implicit key-switch configuration.
    """

    name = "fhelium_rns.crt_basis_extend"
    value = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(self, value, parameters, result_type):
        super().__init__(
            operands=(value, parameters), result_types=(result_type,)
        )

    def verify_(self) -> None:
        if len(self.parameters) != 6:
            raise VerifyException(
                "CRT lift requires six numerical Tensor operands"
            )

    def dependencies(self) -> OperationDependencies:
        return operand_dependencies(
            self,
            (0,),
            {
                "coefficient": "element",
                "limb": "mixing",
                "component": "element",
                "batch": "element",
            },
        )


@irdl_op_definition
class CenteredBasisExtendOp(IRDLOperation):
    r"""Extend each centered CRT coefficient to supplied target prime rows.

    Numerical operands specify ordered source/target bases, Garner factors,
    midpoint digits and source-product correction. Standard residues and all
    leading component/batch axes are preserved. Output owns new storage.
    """

    name = "fhelium_rns.centered_basis_extend"
    value = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    traits = traits_def(Pure())

    def __init__(self, value, parameters, result_type, *, attributes=None):
        super().__init__(
            operands=(value, parameters),
            result_types=(result_type,),
            attributes=attributes or {},
        )


@irdl_op_definition
class SliceRowsOp(IRDLOperation):
    r"""Return a contiguous interval of prime-row positions as a view.

    Result row i aliases source row start+i; coefficients, representations,
    component and batch axes are unchanged. The result type names those rows.
    """

    name = "fhelium_rns.slice_rows"
    value = operand_def(RnsBundleType)
    result = result_def(RnsBundleType)
    start = attr_def(IntegerAttr)
    count = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(self, value, result_type, *, start, count):
        super().__init__(
            operands=(value,),
            result_types=(result_type,),
            attributes={
                "start": IntegerAttr(start, 64),
                "count": IntegerAttr(count, 64),
            },
        )


@irdl_op_definition
class ScaleAndPadRowsOp(IRDLOperation):
    r"""Multiply rows by supplied Montgomery scalars and append zero rows.

    For input x_i and multiplier m_i R, the retained rows are x_i m_i
    in the input representation. Additional rows contain zero.
    """

    name = "fhelium_rns.scale_and_pad_rows"
    value = operand_def(RnsBundleType)
    row_scales = operand_def()
    parameters = operand_def()
    result = result_def(RnsBundleType)
    pad_rows = attr_def(IntegerAttr)
    traits = traits_def(Pure())

    def __init__(self, value, row_scales, parameters, result_type, *, pad_rows):
        super().__init__(
            operands=[value, row_scales, parameters],
            result_types=[result_type],
            attributes={"pad_rows": IntegerAttr(pad_rows, 64)},
        )


@irdl_op_definition
class RoundedTrailingQuotientOp(IRDLOperation):
    r"""Reorder coefficient rows and divide by their trailing-prime product D.

    Supplied row order, inverse and half-offset tensors define the quotient
    floor((x+h)/D), with h=0 or (D-1)/2. Output contains the surviving rows;
    its caller-assigned scale accounts for any combined CKKS rescaling.
    """

    name = "fhelium_rns.rounded_trailing_quotient"
    value = operand_def(RnsBundleType)
    parameters = var_operand_def()
    result = result_def(RnsBundleType)
    drop_count = attr_def(IntegerAttr)
    rounding = attr_def(StringAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value,
        parameters,
        result_type,
        *,
        drop_count,
        rounding,
        attributes=None,
    ):
        super().__init__(
            operands=[value, parameters],
            result_types=[result_type],
            attributes={
                **(attributes or {}),
                "drop_count": IntegerAttr(drop_count, 64),
                "rounding": StringAttr(rounding),
            },
        )


_RNS_OPERATION_TYPES = (
    CrtBasisExtendOp,
    CenteredBasisExtendOp,
    SliceRowsOp,
    ScaleAndPadRowsOp,
    RoundedTrailingQuotientOp,
    AddStandardOp,
    SubtractStandardOp,
    NegateStandardOp,
    SumStandardBatchOp,
    AddPlaintextOp,
    MultiplyPlaintextOp,
    MontgomeryWeightedSumOp,
    MontgomeryWeightedSumsOp,
    RescaleDropLeadingPrimesOp,
    ExtractComponentOp,
    PackTwoComponentsOp,
    PackThreeComponentsOp,
    MontgomeryMultiplyOp,
    HybridModUpDigitOp,
    KeySwitchDigitProductOp,
    AddMontgomeryLazyOp,
    ModDownQpToQOp,
    ModDownNttQpToQOp,
    ModDownAddRescaleOp,
    CoefficientAutomorphismOp,
    StandardToMontgomeryOp,
    MontgomeryToStandardOp,
    RestrictDepthOp,
    ReinterpretScaleOp,
)

_RNS_ATTRIBUTES: dict[type[Operation], tuple[str, ...]] = {
    ScaleAndPadRowsOp: ("pad_rows",),
    RoundedTrailingQuotientOp: ("drop_count", "rounding"),
    AddPlaintextOp: ("polynomial_domain",),
    SumStandardBatchOp: ("dim",),
    MontgomeryWeightedSumOp: (
        "term_count",
        "plaintext_layouts",
        "term_chunk_size",
    ),
    MontgomeryWeightedSumsOp: (
        "term_count",
        "group_count",
        "plaintext_layouts",
        "term_chunk_size",
    ),
    RescaleDropLeadingPrimesOp: (
        "drop_count",
        "rounding",
        "input_domain",
        "output_domain",
    ),
    ExtractComponentOp: ("component",),
    HybridModUpDigitOp: ("digit_index",),
    KeySwitchDigitProductOp: ("key_digit_index", "key_role"),
    ModDownQpToQOp: ("rounding",),
    ModDownNttQpToQOp: ("rounding",),
    ModDownAddRescaleOp: (
        "output_domain",
        "moddown_rounding",
        "rescale_rounding",
        "joint_drop_count",
        "joint_drop_is_prefix",
        "joint_drop_prime_ids",
        "ckks_config",
        "ntt_algorithm",
        "ntt_group_width",
        "ntt_radix",
        "ntt_table_layout",
    ),
    CoefficientAutomorphismOp: ("galois_element",),
    RestrictDepthOp: ("target_depth",),
    ReinterpretScaleOp: ("scale",),
}

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    registered_operation_spec(
        operation_type,
        "rns",
        validator=(
            flat_with_attributes(*_RNS_ATTRIBUTES[operation_type])
            if operation_type in _RNS_ATTRIBUTES
            else flat_without_attributes
        ),
    )
    for operation_type in _RNS_OPERATION_TYPES
)
"""Semantic specifications owned by the RNS dialect."""


FHEliumRns = Dialect(
    "fhelium_rns",
    [
        *_RNS_OPERATION_TYPES,
    ],
    [
        RnsBundleType,
        RnsParametersType,
    ],
)
"""Logical RNS operations and resource types."""


__all__ = [
    "CrtBasisExtendOp",
    "CenteredBasisExtendOp",
    "SliceRowsOp",
    "ScaleAndPadRowsOp",
    "RoundedTrailingQuotientOp",
    "AddStandardOp",
    "AddPlaintextOp",
    "SumStandardBatchOp",
    "AddMontgomeryLazyOp",
    "CoefficientAutomorphismOp",
    "ExtractComponentOp",
    "NegateStandardOp",
    "HybridModUpDigitOp",
    "KeySwitchDigitProductOp",
    "ModDownQpToQOp",
    "ModDownNttQpToQOp",
    "ModDownAddRescaleOp",
    "MontgomeryToStandardOp",
    "MontgomeryMultiplyOp",
    "MontgomeryWeightedSumOp",
    "MontgomeryWeightedSumsOp",
    "MultiplyPlaintextOp",
    "OPERATION_SPECS",
    "PackTwoComponentsOp",
    "PackThreeComponentsOp",
    "SubtractStandardOp",
    "FHEliumRns",
    "RescaleDropLeadingPrimesOp",
    "RestrictDepthOp",
    "ReinterpretScaleOp",
    "RnsBundleType",
    "RnsParametersType",
    "StandardToMontgomeryOp",
]
