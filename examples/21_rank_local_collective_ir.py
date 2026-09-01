#!/usr/bin/env python3

"""Build rank-local collective IR and expose its generic combine region.

A rank-local Program receives one ciphertext and one launch-bound process
group. The source Program preserves a specialized ciphertext-add all-reduce;
a caller-selected Compile pass may instead lower it to generic all-reduce with
a visible CKKS-add combine region. Neither representation proves cross-rank
ordering, uniform control flow, associativity, or deadlock freedom.
"""

from __future__ import annotations

from common import print_table
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import compile as fh_compile
from fhelium import ir
from fhelium.compile.passes.distributed import (
    LowerSpecializedCollectivesPass,
)
from fhelium.ir.dialects import ckks, distributed


def rank_local_program() -> ir.Program:
    """Return one rank-local broadcast and ciphertext-reduction Program."""

    ciphertext_type = ckks.CiphertextType()
    group_type = distributed.GroupType()
    block = Block(arg_types=(ciphertext_type, group_type))
    rank = distributed.RankOp(block.args[1])
    group_size = distributed.GroupSizeOp(block.args[1])
    broadcast = distributed.BroadcastOp(
        block.args[0],
        block.args[1],
        root=0,
    )
    reduced = distributed.AllReduceAddCiphertextOp(
        broadcast.result,
        block.args[1],
    )
    block.add_ops(
        (
            rank,
            group_size,
            broadcast,
            reduced,
            ReturnOp(reduced.result),
        )
    )
    return ir.Program.from_function(block, (ciphertext_type,))


def _decision_rows(
    label: str,
    result: fh_compile.Compilation,
) -> list[list[object]]:
    """Return printable transformation choices from one Compile result."""

    return [
        [
            label,
            decision.subject,
            decision.selected,
            decision.candidates,
            decision.details,
        ]
        for report in result.reports
        for decision in report.decisions
    ]


def main() -> None:
    source = rank_local_program()
    preserved = fh_compile.compile(
        source,
        pipeline=fh_compile.Pipeline(
            (LowerSpecializedCollectivesPass(lower_ciphertext_add=False),)
        ),
    )
    lowered = fh_compile.compile(
        source,
        pipeline=fh_compile.Pipeline((LowerSpecializedCollectivesPass(),)),
    )

    source_inventory = ir.inventory_program(source)
    preserved_inventory = ir.inventory_program(preserved.program)
    lowered_inventory = ir.inventory_program(lowered.program)
    rows = _decision_rows("preserve", preserved) + _decision_rows(
        "lower", lowered
    )

    print_table(
        ["Program", "operations", "collective form"],
        [
            [
                "source",
                sum(source_inventory.operation_counts.values()),
                "fhelium_dist.all_reduce_add_ciphertext",
            ],
            [
                "preserved",
                sum(preserved_inventory.operation_counts.values()),
                "fhelium_dist.all_reduce_add_ciphertext",
            ],
            [
                "lowered",
                sum(lowered_inventory.operation_counts.values()),
                "fhelium_dist.all_reduce + visible fhelium_ckks.add",
            ],
        ],
    )
    print()
    print_table(
        ["request", "subject", "selected", "candidates", "details"],
        rows,
    )
    print()
    print("--- specialized rank-local Program ---")
    print(ir.format_program(source))
    print()
    print("--- generic combine-region Program ---")
    print(ir.format_program(lowered.program))


if __name__ == "__main__":
    main()
