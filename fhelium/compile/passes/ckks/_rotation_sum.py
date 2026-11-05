"""Recognize rotation sums and opt into one shared rounded quotient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from xdsl.dialects.builtin import IntegerAttr, StringAttr, FloatAttr, ArrayAttr
from xdsl.rewriter import Rewriter

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks
from ..._pipeline import DecisionRecord, PassResult, PassStats
from ..program._purity import known_pure, known_region_blocks
from ._rotation_hoisting import RotationHoistingPass


def _unassigned(operation):
    return EXECUTION_IMPLEMENTATION_ATTRIBUTE not in operation.attributes


@dataclass(frozen=True)
class AggregateRotationSumsPass:
    r"""Replace compatible rotation/add trees with a shared QP quotient.

    Including this pass is the caller's choice to change intermediate CKKS rounding. Matching uses SSA values and represented state. Independent intermediate users are preserved. Unknown effects, caller assignments, mismatched tables/state or partial required facts leave the region unchanged.

    ``combine_rescale`` also recognizes uniformly rescaled inputs and divides
    the aggregate by P times the complete dropped Q group. Otherwise each
    original input rescale remains before the aggregate key switching.

    Measured on RTX PRO 6000 Blackwell Server Edition: 128x128 @ 128x8
    BSGS matrices, baby step 8, N=16384, Q=8, P=1, int64 residues, scale 2^40.
    Latencies include device synchronization, using resident inputs; values
    are medians of three process medians, with 30 samples per process.

    | Execution / implementation | Matrix | Baseline (ms) | Aggregate (ms) |
    | --- | --- | ---: | ---: |
    | CUDA Graph / default fusion | PT x CT | 9.947 | 8.539 |
    | CUDA Graph / default fusion | CT x CT | 17.901 | 16.444 |
    | No Graph / whole operation | PT x CT | 12.702 | 11.835 |
    | No Graph / whole operation | CT x CT | 21.132 | 25.383 |

    Both matrix workloads improved with Graph replay. Without Graph, PT x CT
    improved, but CT x CT became 20.1% slower.
    """

    combine_rescale: bool = True
    name: str = 'aggregate-rotation-sums'

    def run(self, compilation):
        decisions = []
        matched = transformed = 0

        def visit(block):
            nonlocal matched, transformed
            for operation in tuple(block.ops):
                if _unassigned(operation):
                    for child in known_region_blocks(operation):
                        visit(child)
            # Process maximal output sums before their child additions.
            for root in reversed(tuple(block.ops)):
                if (
                    root.parent_block() is not block
                    or type(root) is not ckks.AddOp
                    or not _unassigned(root)
                ):
                    continue
                matched += 1
                leaves = []
                tree = []

                def flatten(value, at_root=False):
                    owner = value.owner
                    if (
                        type(owner) is ckks.AddOp
                        and owner.parent_block() is block
                        and _unassigned(owner)
                        and (at_root or len(tuple(value.uses)) == 1)
                    ):
                        tree.append(owner)
                        for operand in (owner.lhs, owner.rhs):
                            flatten(operand)
                    else:
                        leaves.append(value)

                flatten(root.result, True)
                sources = []
                keys = []
                steps = []
                rotations = []
                incompatible = False
                for leaf in leaves:
                    owner = leaf.owner
                    if (
                        type(owner) is ckks.RotateOp
                        and owner.parent_block() is block
                        and _unassigned(owner)
                    ):
                        step = RotationHoistingPass._step(owner)
                        if (
                            step is None
                            or owner.output_domain.data != 'coefficient'
                        ):
                            incompatible = True
                            break
                        rotations.append(owner)
                        sources.append(owner.value)
                        steps.append(step)
                        if step:
                            keys.append(owner.key)
                    else:
                        sources.append(leaf)
                        steps.append(0)
                if incompatible or len(rotations) < 2:
                    continue
                first = rotations[0]
                rounding = first.moddown_rounding.data
                if any(
                    r.modup_method != first.modup_method
                    or r.moddown_rounding.data != rounding
                    or RotationHoistingPass._key_layout(r)
                    != RotationHoistingPass._key_layout(first)
                    or tuple(r.parameters) != tuple(first.parameters)
                    or r.attributes.get('parameter_names')
                    != first.attributes.get('parameter_names')
                    or r.attributes.get('ckks_config')
                    != first.attributes.get('ckks_config')
                    for r in rotations
                ):
                    continue
                involved = [*tree, *rotations]
                rescale = False
                lifted_rescale = False
                anchor = root
                if self.combine_rescale and all(
                    type(v.owner) is ckks.RescaleOp and _unassigned(v.owner)
                    for v in sources
                ):
                    rescales = [v.owner for v in sources]
                    if all(
                        r.rounding is not None and r.rounding.data == rounding
                        for r in rescales
                    ):
                        sources = [r.value for r in rescales]
                        involved.extend(rescales)
                        rescale = True
                        lifted_rescale = True
                if self.combine_rescale and not rescale:
                    users = tuple(root.result.uses)
                    if len(users) == 1:
                        consumer = users[0].operation
                        if (
                            type(consumer) is ckks.RescaleOp
                            and _unassigned(consumer)
                            and consumer.rounding is not None
                            and consumer.rounding.data == rounding
                            and consumer.parent_block() is block
                        ):
                            anchor = consumer
                            rescale = True
                            involved.append(consumer)
                # Remove a common exact representation round-trip, retaining
                # the input Tensor operands rather than synthesizing values.
                if all(
                    type(v.owner) is ckks.FromNttOp and _unassigned(v.owner)
                    for v in sources
                ):
                    conversions = [v.owner for v in sources]
                    sources = [o.operands[0] for o in conversions]
                    involved.extend(conversions)
                source_type = sources[0].type
                if not isinstance(source_type, ckks.CiphertextType) or any(
                    v.type != source_type for v in sources
                ):
                    continue
                state = source_type.state.data
                if state.get('components') != IntegerAttr(2, 64) or state.get(
                    'basis'
                ) != StringAttr('Q'):
                    continue
                if not (
                    isinstance(state.get('depth'), IntegerAttr)
                    and isinstance(state.get('scale'), FloatAttr)
                    and isinstance(state.get('prime_ids'), ArrayAttr)
                    and isinstance(state.get('dtype'), StringAttr)
                    and isinstance(state.get('device'), StringAttr)
                ):
                    continue
                domain = state.get('polynomial_domain')
                residues = state.get('residue_representation')
                if (domain, residues) not in [
                    (StringAttr('coefficient'), StringAttr('standard')),
                    (StringAttr('ntt'), StringAttr('montgomery')),
                ]:
                    continue
                result_state = anchor.result.type.state.data
                if result_state.get('polynomial_domain') != StringAttr(
                    'coefficient'
                ):
                    continue
                if any(o.parent_block() is not block for o in involved):
                    continue
                positions = {o: i for i, o in enumerate(block.ops)}
                start = min(positions[o] for o in involved)
                stop = positions[anchor]
                interval = tuple(block.ops)[start : stop + 1]
                # Independent pure operations keep their position and selected
                # implementations. Only the matched, unassigned nodes change.
                if any(not known_pure(o) for o in interval):
                    continue
                # A high-Q rewrite requests different numerical operands;
                # retain existing symbols unchanged and declare new high-Q
                # tables through ordinary preparation. Keys must already cover
                # the higher Q rows. No new keys are synthesized.
                if lifted_rescale:
                    ids = cast(ArrayAttr, state['prime_ids'])
                    if any(
                        not isinstance(
                            k.type.state.data.get('prime_ids'), ArrayAttr
                        )
                        or not set(ids).issubset(
                            set(cast(ArrayAttr, k.type.state.data['prime_ids']))
                        )
                        for k in keys
                    ):
                        continue
                    attrs = {
                        k: v
                        for k, v in first.attributes.items()
                        if k in {'ckks_config', 'ntt_backend', 'modup_method'}
                    }
                    parameters = ()
                else:
                    attrs = {
                        k: v
                        for k, v in first.attributes.items()
                        if k
                        not in {
                            'rotation_step',
                            'input_domain',
                            'output_domain',
                        }
                    }
                    parameters = tuple(first.parameters)
                aggregate = ckks.RotationSumOp(
                    tuple(sources),
                    tuple(keys),
                    tuple(steps),
                    anchor.result.type,
                    rescale=rescale,
                    input_domain=cast(StringAttr, domain).data,
                    rounding=rounding,
                    modup_method=first.modup_method.data,
                    parameters=parameters,
                    attributes=attrs,
                )
                Rewriter.replace_op(anchor, aggregate)
                transformed += 1
                decisions.append(
                    DecisionRecord(
                        aggregate.name,
                        selected='shared-qp-moddown-rescale'
                        if rescale
                        else 'shared-qp-moddown',
                        details=(
                            f'contributions={len(sources)}',
                            f'key_products={len(keys)}',
                            'caller-selected rounding reassociation',
                        ),
                    )
                )
                # Delete only now-dead pure nodes in this matched region.
                # An intermediate used outside the tree remains executable.
                for operation in sorted(
                    set(involved) - {anchor},
                    key=lambda o: positions[o],
                    reverse=True,
                ):
                    if operation.parent_block() is block and not any(
                        r.uses for r in operation.results
                    ):
                        Rewriter.erase_op(operation)

        for function in compilation.program.functions:
            for block in function.body.blocks:
                visit(block)
        return PassResult(
            compilation.program,
            PassStats(matched=matched, transformed=transformed),
            decisions=tuple(decisions),
        )
