"""Bound static generated-kernel complexity before Triton compilation."""

from __future__ import annotations

# Montgomery calls, pointer arguments, and generated Python source lines.
DEFAULT_KERNEL_BUDGET = (768, 1536, 12000)
DEFAULT_POINTWISE_BUDGET = (128, 1536, 12000)
KernelBudget = tuple[int, int, int] | None


def kernel_cost(source: str) -> tuple[int, int, int]:
    """Count static arithmetic sites, launch pointers, and source lines.

    A runtime loop contributes its body once. Layout scalars and constexpr
    parameters carry annotations; data pointers are the unannotated arguments.
    These counts describe compiler input rather than runtime work or GPU limits.
    """
    signature = source[source.index("(") + 1 : source.index("):")]
    return (
        source.count("montgomery_mul(") + source.count("montgomery_reduce("),
        sum(bool(p.strip()) and ":" not in p for p in signature.split(",")),
        source.count("\n"),
    )


def within_budget(source: str, budget: KernelBudget) -> bool:
    return budget is None or all(
        cost <= limit
        for cost, limit in zip(kernel_cost(source), budget, strict=True)
    )


def indivisible_kernel(source: str, budget: KernelBudget) -> ValueError:
    return ValueError(
        f"One output or NTT endpoint exceeds kernel_budget: "
        f"cost={kernel_cost(source)}, budget={budget}; split its computation "
        "or select a larger TritonFusionImplementation kernel budget"
    )
