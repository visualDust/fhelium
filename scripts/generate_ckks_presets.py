#!/usr/bin/env python3
r"""Generate the built-in CKKS parameter presets of libfhelium.

Every preset uses NTT primes below $2^{30}$ drawn from two pools: main primes
taken downward from $2^{30}$ and terminal primes taken around $2^{25}$, using the terminal count among 2, 4,
6 and 8 that yields the greatest depth. The Q
order is ``reverse(terminal) ++ main``. Depth state $(m, t)$ selects the first
$m$ main and first $t$ terminal primes. The maximum-depth basis is the smallest state with at least
``scale + 9.5`` bits. Each earlier depth adds the state whose rescale ratio
$D/R$ is nearest to $2^{\mathrm{scale}}$, preferring fewer rows; the depth-0
basis is the union of all Q primes when its first ratio lies between the
scale target and 12 bits above it. Every later ratio stays within one bit of the target (1.5 bits for
$N=2^{17}$, whose NTT primes below $2^{30}$ are sparser). The P primes are
the next primes below $2^{30}$ outside Q. The default scale of the maximum
depth is $2^{\mathrm{scale}}$, and every earlier depth uses
$\Delta_d=\sqrt{\Delta_{d+1}\rho_d}$ for the rescale ratio $\rho_d$. The largest
depth whose complete Q and P product meets the 128-bit ternary budget is used.

The output is ``csrc/parameters/ckks_preset_data.inc``. The rendering is
deterministic, and the write is atomic.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_prime_catalog import is_prime_64  # noqa: E402

_BUDGET = {14: 430, 15: 868, 16: 1747, 17: 3523}
_P_COUNT = {14: 3, 15: 5, 16: 10, 17: 16}
_TERMINAL_COUNTS = (2, 4, 6, 8)
_SCALES = (30, 40, 50)
_HEADROOM_BITS = 10
_MAIN_POOL = 240
_RATIO_TOLERANCE_BITS = {14: 1.0, 15: 1.0, 16: 1.0, 17: 1.5}


def _descending_primes(order: int, below: int, count: int, exclude=()):
    excluded = set(exclude)
    result = []
    candidate = (below - 1) // order * order + 1
    while len(result) < count and candidate > order:
        if candidate not in excluded and is_prime_64(candidate):
            result.append(candidate)
        candidate -= order
    return result


def _primes_around(order: int, bits: float, count: int, exclude=()):
    excluded = set(exclude)
    result = []
    up = math.floor(2.0**bits) // order * order + 1
    down = up - order
    while len(result) < count:
        for candidate in (down, up):
            if (
                len(result) < count
                and candidate not in excluded
                and is_prime_64(candidate)
            ):
                result.append(candidate)
        up += order
        down -= order
    return result


def _bits(values) -> float:
    return sum(math.log2(value) for value in values)


def _fill_budget(order: int, q, p, budget: float) -> tuple[int, ...]:
    """Append the largest NTT prime that fits the budget left by Q and P."""
    remaining = budget - _bits(q) - _bits(p)
    if remaining < 1:
        return tuple(p)
    extra = _descending_primes(
        order, math.floor(2.0 ** min(30.0, remaining)), 1, exclude=(*q, *p)
    )
    return (*p, *extra)


def _balanced_scales(q_moduli, bases, final_scale: float) -> tuple[float, ...]:
    scales = [float(final_scale)]
    for depth in range(len(bases) - 2, -1, -1):
        source, target = set(bases[depth]), set(bases[depth + 1])
        ratio = Fraction(
            math.prod(q_moduli[i] for i in source - target),
            math.prod(q_moduli[i] for i in target - source),
        )
        scales.append(math.sqrt(scales[-1] * float(ratio)))
    return tuple(reversed(scales))


@dataclass(frozen=True)
class _Chain:
    states: tuple[tuple[int, int], ...]
    main: tuple[int, ...]
    terminal: tuple[int, ...]

    def basis(self, state):
        m, t = state
        return set(self.main[:m]) | set(self.terminal[:t])

    def ratio_bits(self, depth):
        source, target = self.basis(self.states[depth]), self.basis(
            self.states[depth + 1]
        )
        return _bits(source - target) - _bits(target - source)


def _base_state(scale: int, main, terminal, headroom=_HEADROOM_BITS):
    candidates = [
        (m, t)
        for m in range(0, 4)
        for t in range(0, len(terminal) + 1)
        if m + t > 0
        and _bits(main[:m]) + _bits(terminal[:t]) >= scale + headroom - 0.5
    ]
    return min(
        candidates,
        key=lambda s: (s[0] + s[1], _bits(main[: s[0]]) + _bits(terminal[: s[1]])),
    )


def _extend(states, scale, main, terminal):
    m, t = states[-1]
    best = None
    for m2 in range(len(main) + 1):
        for t2 in range(len(terminal) + 1):
            if m2 + t2 <= m + t:
                continue
            added = main[m:m2] + terminal[t:t2]
            removed = main[m2:m] + terminal[t2:t]
            ratio = _bits(added) - _bits(removed)
            key = (abs(ratio - scale), m2 + t2, len(removed))
            if best is None or key < best[0]:
                best = (key, (m2, t2))
    return states + [best[1]]


def _chain(
    depth: int, scale: int, main, terminal, headroom=_HEADROOM_BITS
) -> _Chain:
    states = [_base_state(scale, main, terminal, headroom)]
    for _ in range(depth - 1):
        states = _extend(states, scale, main, terminal)
    union = (max(s[0] for s in states), max(s[1] for s in states))
    if depth > 0 and union not in states:
        candidate = _Chain(
            tuple(reversed(states + [union])), tuple(main), tuple(terminal)
        )
        if candidate.ratio_bits(0) >= scale - 0.5:
            return candidate
    if depth > 0:
        states = _extend(states, scale, main, terminal)
    return _Chain(tuple(reversed(states)), tuple(main), tuple(terminal))


@dataclass(frozen=True)
class Preset:
    name: str
    log_n: int
    q_moduli: tuple[int, ...]
    bases: tuple[tuple[int, ...], ...]
    scales: tuple[float, ...]
    p_moduli: tuple[int, ...]


def _interval_preset(name, log_n, scale, q_order, intervals, p) -> Preset:
    bases = tuple(tuple(range(a, b)) for a, b in intervals)
    return Preset(
        name=name,
        log_n=log_n,
        q_moduli=q_order,
        bases=bases,
        scales=_balanced_scales(q_order, bases, 2.0**scale),
        p_moduli=p,
    )


def _preset_with(log_n: int, scale: int, terminal_count: int) -> Preset:
    order = 2 << log_n
    main = _descending_primes(order, 1 << 30, _MAIN_POOL)
    terminal = _primes_around(order, 25, terminal_count, exclude=main)
    p_pool = _descending_primes(order, 1 << 30, _MAIN_POOL + _P_COUNT[log_n] + 8)
    best = None
    failures = 0
    for depth in range(1, 200):
        if failures >= 4:
            break
        chain = _chain(depth, scale, main, terminal)
        m_used = max(s[0] for s in chain.states)
        t_used = max(s[1] for s in chain.states)
        q_main, q_terminal = main[:m_used], terminal[:t_used]
        p = tuple(x for x in p_pool if x not in q_main)[: _P_COUNT[log_n]]
        ratios = [chain.ratio_bits(d) for d in range(depth)]
        balanced = all(
            abs(r - scale) <= _RATIO_TOLERANCE_BITS[log_n] for r in ratios[1:]
        ) and scale - _RATIO_TOLERANCE_BITS[log_n] <= ratios[0] <= scale + _HEADROOM_BITS + 2
        if not balanced or (
            _bits(q_main) + _bits(q_terminal) + _bits(p) > _BUDGET[log_n]
        ):
            failures += 1
            continue
        failures = 0
        best = (depth, chain, q_main, q_terminal, p)
    assert best is not None
    depth, chain, q_main, q_terminal, p = best
    t_used = len(q_terminal)
    q_order = tuple(reversed(q_terminal)) + tuple(q_main)
    bases = tuple((t_used - t, t_used + m) for m, t in chain.states)
    p = _fill_budget(order, q_order, p, _BUDGET[log_n])
    return _interval_preset(
        f"slots{1 << (log_n - 1)}-scale{scale}-depth{depth}",
        log_n, scale, q_order, bases, p,
    )


def _preset(log_n: int, scale: int) -> Preset:
    candidates = [
        _preset_with(log_n, scale, count) for count in _TERMINAL_COUNTS
    ]
    return max(candidates, key=lambda p: (len(p.bases), -len(p.q_moduli)))


# Prime widths of the 16 dropped groups of the "3/38" bootstrap segment:
# four CoeffsToSlots stages, nine EvalMod depths (six for the degree-38 seed,
# three double angles) and three SlotsToCoeffs stages. "t" reuses a terminal
# prime outside the first application basis.
_BOOT_GROUPS = (
    ("t", 27.8),
    ("t", 27.8),
    ("t", 27.8),
    ("t", 27.8, 27.8),
    (27.8, 27.8),
    *((29.75, 29.75),) * 8,
    (27.7,),
    (27.7,),
    (27.7, 27.7),
)
# (log N, scale bits, application levels) of the bootstrap presets.
_BOOT_TIERS = ((16, 40, 4), (16, 40, 8), (16, 40, 13), (16, 50, 8), (16, 50, 12))


def _bootstrap_preset(log_n: int, scale: int, levels: int) -> Preset:
    order = 2 << log_n
    main = _descending_primes(order, 1 << 30, _MAIN_POOL)
    terminal = _primes_around(order, 25, 4, exclude=main)
    states = [_base_state(scale, main, terminal)]
    for _ in range(levels):
        states = _extend(states, scale, main, terminal)
    chain = _Chain(tuple(reversed(states)), tuple(main), tuple(terminal))
    application = [chain.basis(state) for state in chain.states]
    used = set().union(*application)
    reusable = [p for p in terminal if p in used and p not in application[0]]
    # Main primes of deeper application bases outside the first one fill the
    # leading 29.75-bit EvalMod slots, so depth 0 holds every Q prime.
    reusable_main = [
        p for p in main if p in used and p not in application[0]
    ]
    groups = []
    for widths in _BOOT_GROUPS:
        group = []
        for width in widths:
            if width == "t" and reusable:
                prime = reusable.pop(0)
            elif width == 29.75 and reusable_main:
                prime = reusable_main.pop(0)
            else:
                bits = 25 if width == "t" else width
                prime = _primes_around(order, bits, 1, exclude=used)[0]
            used.add(prime)
            group.append(prime)
        groups.append(group)
    boot = [prime for group in groups for prime in group]
    q_order = tuple(dict.fromkeys(boot + sorted(used - set(boot))))
    index = {prime: i for i, prime in enumerate(q_order)}
    bases = [
        set().union(*map(set, groups[d:])) | application[0]
        for d in range(len(groups))
    ] + application
    remaining = _BUDGET[log_n] - _bits(q_order)
    p = []
    for prime in _descending_primes(order, 1 << 30, _MAIN_POOL + 64, used):
        if _bits(p) + math.log2(prime) > remaining:
            break
        p.append(prime)
    p = list(_fill_budget(order, q_order, p, _BUDGET[log_n]))
    bases = tuple(tuple(sorted(index[x] for x in basis)) for basis in bases)
    return Preset(
        name=f"slots{1 << (log_n - 1)}-scale{scale}-depth{len(bases) - 1}"
        f"-boot{levels}",
        log_n=log_n,
        q_moduli=q_order,
        bases=bases,
        scales=_balanced_scales(q_order, bases, 2.0**scale),
        p_moduli=tuple(p),
    )


def presets() -> list[Preset]:
    return [
        _preset(log_n, scale) for log_n in sorted(_BUDGET) for scale in _SCALES
    ] + [_bootstrap_preset(*tier) for tier in _BOOT_TIERS]


def _render(items: list[Preset]) -> str:
    lines = [
        "// Generated by scripts/generate_ckks_presets.py; do not edit.",
        "",
        "struct PresetData {",
        "  const char* name;",
        "  int64_t log_n;",
        "  int64_t q_first;",
        "  int64_t q_count;",
        "  int64_t base_first;",
        "  int64_t base_count;",
        "  int64_t row_first;",
        "  int64_t p_first;",
        "  int64_t p_count;",
        "};",
        "",
    ]
    primes: list[int] = []
    basis_rows: list[int] = []
    sizes: list[int] = []
    scales: list[float] = []
    rows = []
    for item in items:
        q_first = len(primes)
        primes.extend(item.q_moduli)
        p_first = len(primes)
        primes.extend(item.p_moduli)
        base_first = len(sizes)
        row_first = len(basis_rows)
        for basis in item.bases:
            basis_rows.extend(basis)
            sizes.append(len(basis))
        scales.extend(item.scales)
        rows.append(
            f'    {{"{item.name}", {item.log_n}, {q_first}, '
            f"{len(item.q_moduli)}, {base_first}, {len(item.bases)}, "
            f"{row_first}, {p_first}, {len(item.p_moduli)}}},"
        )
    lines.append("constexpr int64_t kPresetPrimes[] = {")
    for index in range(0, len(primes), 6):
        lines.append(
            "    " + ", ".join(str(v) for v in primes[index : index + 6]) + ","
        )
    lines.append("};")
    lines.append("")
    for label, values, width, text in (
        ("int64_t kPresetBasisRows", basis_rows, 16, str),
        ("int64_t kPresetBasisSizes", sizes, 16, str),
        ("double kPresetScales", scales, 3, float.hex),
    ):
        lines.append(f"constexpr {label}[] = {{")
        for index in range(0, len(values), width):
            lines.append(
                "    "
                + ", ".join(text(v) for v in values[index : index + width])
                + ","
            )
        lines.append("};")
        lines.append("")
    lines.append("")
    lines.append("constexpr PresetData kPresets[] = {")
    lines.extend(rows)
    lines.append("};")
    return "\n".join(lines) + "\n"


def main() -> None:
    items = presets()
    for item in items:
        q_bits = _bits(item.q_moduli)
        p_bits = _bits(item.p_moduli)
        bases = [{item.q_moduli[i] for i in basis} for basis in item.bases]
        ratios = [
            _bits(bases[d] - bases[d + 1]) - _bits(bases[d + 1] - bases[d])
            for d in range(len(bases) - 1)
        ]
        largest = max(len(bases[d] - bases[d + 1]) for d in range(len(ratios)))
        print(
            f"{item.name}: Q rows {len(item.q_moduli)}, logQ {q_bits:.1f}, "
            f"P {len(item.p_moduli)} logP {p_bits:.1f}, "
            f"logQP {q_bits + p_bits:.1f}/{_BUDGET[item.log_n]}, "
            f"ratio bits {ratios[0]:.2f} then [{min(ratios[1:] or ratios):.2f}, "
            f"{max(ratios[1:] or ratios):.2f}], max |D| {largest}, "
            f"base bits {_bits(bases[-1]):.1f}"
        )
    target = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "parameters"
        / "ckks_preset_data.inc"
    )
    text = _render(items)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".presets-")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(temporary, target)


if __name__ == "__main__":
    main()
