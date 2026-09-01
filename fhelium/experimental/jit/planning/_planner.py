"""Select backend assignments and form adjacent straight-line regions."""

from __future__ import annotations

from collections.abc import Sequence

from ._records import ProposalDecision, ProposalSelection, RegionProposal


def form_adjacent_regions(
    proposals: Sequence[RegionProposal],
) -> tuple[RegionProposal, ...]:
    """Form execution regions from adjacent identical backend assignments."""

    merged: list[RegionProposal] = []
    for proposal in sorted(proposals, key=lambda item: (item.start, item.stop)):
        if not merged:
            merged.append(proposal)
            continue
        previous = merged[-1]
        if (
            previous.stop != proposal.start
            or previous.provider != proposal.provider
            or previous.implementation != proposal.implementation
            or previous.metadata.get("merge_adjacent", True) is False
            or proposal.metadata.get("merge_adjacent", True) is False
        ):
            merged.append(proposal)
            continue
        merged[-1] = RegionProposal(
            previous.provider,
            previous.implementation,
            (*previous.operation_ids, *proposal.operation_ids),
            (*previous.operation_indices, *proposal.operation_indices),
            input_requirements=(
                *previous.input_requirements,
                *proposal.input_requirements,
            ),
            output_requirements=(
                *previous.output_requirements,
                *proposal.output_requirements,
            ),
            required_bindings=tuple(
                dict.fromkeys(
                    (*previous.required_bindings, *proposal.required_bindings)
                )
            ),
            lowering_depth=previous.lowering_depth,
            priority=max(previous.priority, proposal.priority),
            diagnostics=(*previous.diagnostics, *proposal.diagnostics),
            metadata={
                "region_formation": "adjacent-backend-assignments",
                "operation_count": len(previous.operation_indices)
                + len(proposal.operation_indices),
            },
        )
    return tuple(merged)


def select_region_proposals(
    proposals: Sequence[RegionProposal],
    *,
    provider_order: Sequence[str],
) -> ProposalSelection:
    """Select supported assignments under caller provider order.

    Provider order is the primary policy decision. Within one provider, higher
    priority and then larger candidates win. Remaining ties use source order
    and implementation identity. Adjacent operations selected for the same
    implementation form one execution region.
    """

    providers = tuple(provider_order)
    if not providers:
        raise ValueError("provider_order must enable at least one provider")
    if len(set(providers)) != len(providers) or any(
        not item for item in providers
    ):
        raise ValueError("provider_order must contain distinct non-empty names")
    rank = {provider: index for index, provider in enumerate(providers)}
    ordered = tuple(
        sorted(
            proposals,
            key=lambda item: (
                rank.get(item.provider, len(rank)),
                -item.priority,
                -len(item.operation_indices),
                item.start,
                item.stop,
                item.implementation,
                item.operation_ids,
            ),
        )
    )

    occupied: dict[int, RegionProposal] = {}
    selected: list[RegionProposal] = []
    decisions: list[ProposalDecision] = []
    for proposal in ordered:
        if proposal.provider not in rank:
            decisions.append(
                ProposalDecision(
                    proposal,
                    False,
                    "provider is not enabled by the caller policy",
                )
            )
            continue
        if not proposal.supported:
            decisions.append(
                ProposalDecision(
                    proposal,
                    False,
                    "provider rejected the assignment during coverage",
                )
            )
            continue
        conflicts: list[RegionProposal] = []
        for index in proposal.operation_indices:
            conflict = occupied.get(index)
            if conflict is not None and all(
                conflict is not candidate for candidate in conflicts
            ):
                conflicts.append(conflict)
        if conflicts:
            conflict = min(
                conflicts,
                key=lambda item: (
                    item.start,
                    item.provider,
                    item.implementation,
                ),
            )
            decisions.append(
                ProposalDecision(
                    proposal,
                    False,
                    "operations already assigned to "
                    f"{conflict.provider}/{conflict.implementation}",
                )
            )
            continue
        selected.append(proposal)
        for index in proposal.operation_indices:
            occupied[index] = proposal
        decisions.append(ProposalDecision(proposal, True, "selected by policy"))

    return ProposalSelection(
        form_adjacent_regions(selected),
        tuple(decisions),
    )


__all__ = ["form_adjacent_regions", "select_region_proposals"]
