"""Integer modular arithmetic shared by generated Triton kernels."""

import linecache

import triton
import triton.language as tl


@triton.jit
def montgomery_mul(a, b, q, k, radix_bits: tl.constexpr):
    """Return native-compatible REDC: canonical for R32, lazy for R62."""
    if radix_bits == 32:
        # With k = -q^{-1} mod 2^32, the low word of a*b + m*q vanishes and
        # carries exactly when the low word of a*b is nonzero.
        a = a.to(tl.uint32)
        b = b.to(tl.uint32)
        q = q.to(tl.uint32)
        low = a * b
        m = low * k.to(tl.uint32)
        u = tl.umulhi(a, b) + tl.umulhi(m, q) + (low != 0).to(tl.uint32)
        return tl.where(u >= q, u - q, u).to(tl.int64)
    else:
        a = a.to(tl.uint64)
        b = b.to(tl.uint64)
        q = q.to(tl.uint64)
        k = k.to(tl.uint64)
        low = a * b
        high = tl.umulhi(a, b)
        m = (low * k) & 0x3FFFFFFFFFFFFFFF
        mq_low = m * q
        mq_high = tl.umulhi(m, q)
        total_low = low + mq_low
        carry = (total_low < low).to(tl.uint64)
        total_high = high + mq_high + carry
        return ((total_low >> 62) | (total_high << 2)).to(tl.int64)


@triton.jit
def montgomery_reduce(a, q, k, radix_bits: tl.constexpr):
    one = tl.full((), 1, tl.uint64)
    return montgomery_mul(a, one, q, k, radix_bits)


@triton.jit
def add_lazy(a, b, q):
    value = a + b
    return tl.where(value >= 2 * q, value - 2 * q, value)


@triton.jit
def subtract_lazy(a, b, q):
    value = a - b
    return tl.where(value < 0, value + 2 * q, value)


@triton.jit
def canonicalize(a, q):
    return tl.where(a >= q, a - q, a)


@triton.jit
def shoup32_mul(value, root, quotient, q):
    """Return value*root modulo q in [0,2q) for value<2q and q<2^30.

    quotient carries floor(root*2^32/q) as an unsigned word. The low-word
    difference equals the nonnegative remainder because its value is below 2q.
    A standard root preserves the input's Montgomery representation.
    """
    value = value.to(tl.uint32)
    root = root.to(tl.uint32)
    quotient = quotient.to(tl.uint32)
    q = q.to(tl.uint32)
    return (value * root - tl.umulhi(value, quotient) * q).to(tl.int64)


@triton.jit
def butterfly_sum(u, v, q, radix_bits: tl.constexpr):
    """Return (u + v, u - v) in [0, 2q) for u, v in [0, 2q)."""
    if radix_bits == 32:
        # Residues and 2q fit 31 bits; 32-bit lanes halve register use.
        u = u.to(tl.uint32)
        v = v.to(tl.uint32)
        twice = 2 * q.to(tl.uint32)
    else:
        twice = 2 * q
    total = u + v
    difference = u + twice - v
    total = tl.where(total < twice, total, total - twice)
    difference = tl.where(difference < twice, difference, difference - twice)
    return total.to(tl.int64), difference.to(tl.int64)


@triton.jit
def narrow(value, radix_bits: tl.constexpr):
    """Return residues in the narrowest lane type of the arithmetic radix."""
    if radix_bits == 32:
        value = value.to(tl.uint32)
    return value


@triton.jit
def _rotl(x, n: tl.constexpr):
    return (x << n) | (x >> (32 - n))


@triton.jit
def _quarter(a, b, c, d):
    a = a + b
    d = _rotl(d ^ a, 16)
    c = c + d
    b = _rotl(b ^ c, 12)
    a = a + b
    d = _rotl(d ^ a, 8)
    c = c + d
    b = _rotl(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _low_word(value):
    return (value.to(tl.uint64) & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def _high_word(value):
    return (value.to(tl.uint64) >> 32).to(tl.uint32)


@triton.jit
def _seeded_block(seed, row, segment, attempt, rounds: tl.constexpr):
    """Return the 16 words of the seeded-key ChaCha block of ``segment`` and
    ``attempt`` at stored ``row`` (vectors of equal shape)."""
    w0 = tl.load(seed)
    w1 = tl.load(seed + 1)
    w2 = tl.load(seed + 2)
    w3 = tl.load(seed + 3)
    nonce = tl.load(seed + 4)
    digit = tl.load(seed + 5)
    zero = segment.to(tl.uint32) * 0
    s0 = zero + 0x61707865
    s1 = zero + 0x3320646E
    s2 = zero + 0x79622D32
    s3 = zero + 0x6B206574
    s4 = zero + _low_word(w0)
    s5 = zero + _high_word(w0)
    s6 = zero + _low_word(w1)
    s7 = zero + _high_word(w1)
    s8 = zero + _low_word(w2)
    s9 = zero + _high_word(w2)
    s10 = zero + _low_word(w3)
    s11 = zero + _high_word(w3)
    s12 = (segment.to(tl.int64) | (attempt << 24)).to(tl.uint32)
    s13 = (row.to(tl.int64) | (digit << 16)).to(tl.uint32)
    s14 = zero + _low_word(nonce)
    s15 = zero + _high_word(nonce)
    x0, x1, x2, x3, x4, x5, x6, x7 = s0, s1, s2, s3, s4, s5, s6, s7
    x8, x9, x10, x11, x12, x13, x14, x15 = s8, s9, s10, s11, s12, s13, s14, s15
    for _ in tl.static_range(rounds // 2):
        x0, x4, x8, x12 = _quarter(x0, x4, x8, x12)
        x1, x5, x9, x13 = _quarter(x1, x5, x9, x13)
        x2, x6, x10, x14 = _quarter(x2, x6, x10, x14)
        x3, x7, x11, x15 = _quarter(x3, x7, x11, x15)
        x0, x5, x10, x15 = _quarter(x0, x5, x10, x15)
        x1, x6, x11, x12 = _quarter(x1, x6, x11, x12)
        x2, x7, x8, x13 = _quarter(x2, x7, x8, x13)
        x3, x4, x9, x14 = _quarter(x3, x4, x9, x14)
    return (
        x0 + s0, x1 + s1, x2 + s2, x3 + s3, x4 + s4, x5 + s5, x6 + s6, x7 + s7,
        x8 + s8, x9 + s9, x10 + s10, x11 + s11, x12 + s12, x13 + s13, x14 + s14,
        x15 + s15,
    )


@triton.jit
def _stack4(a, b, c, d):
    return tl.join(tl.join(a, c), tl.join(b, d))


@triton.jit
def _stack8(x0, x1, x2, x3, x4, x5, x6, x7):
    """Stack vectors [S] into [S, 8] with column i holding ``x_i``."""
    stacked = tl.join(_stack4(x0, x2, x4, x6), _stack4(x1, x3, x5, x7))
    return tl.reshape(stacked, (x0.numel, 8))


def _segment_source(name: str, count: int, wide: bool) -> str:
    """Return the source of a seeded-key segment sampler.

    The sampler returns ``count`` vectors: the first accepted candidates of
    each segment in order. The first block places candidate i only in slots
    [i - spare, i], which covers every segment it completes; programs with an
    incomplete segment resample all of their segments with the general
    placement over successive attempts.
    """
    candidates = 8 if wide else 16
    spare = candidates - count
    slots = [f"o{j}" for j in range(count)]

    def candidate(i: int, block: str, prefix: str) -> list[str]:
        if wide:
            return [
                f"{prefix}w{i} = {block}[{2 * i}].to(tl.uint64) | ({block}[{2 * i + 1}].to(tl.uint64) << 32)",
                f"{prefix}a{i} = {prefix}w{i} * modulus >= threshold",
                f"{prefix}v{i} = tl.umulhi({prefix}w{i}, modulus)",
            ]
        return [
            f"{prefix}w{i} = {block}[{i}].to(tl.uint64) * modulus",
            f"{prefix}a{i} = ({prefix}w{i} & 0xFFFFFFFF) >= threshold",
            f"{prefix}v{i} = {prefix}w{i} >> 32",
        ]

    threshold = (
        "(modulus * 0 - modulus) % modulus"
        if wide
        else "((q.to(tl.uint32) * 0 - q.to(tl.uint32)) % q.to(tl.uint32)).to(tl.uint64)"
    )
    lines = [
        f"def {name}(seed, row, segment, q, rounds: tl.constexpr):",
        "    modulus = q.to(tl.uint64)",
        f"    threshold = {threshold}",
        "    filled = segment.to(tl.int32) * 0",
        *(f"    {slot} = modulus * 0" for slot in slots),
        "    block = _seeded_block(seed, row, segment, 0, rounds)",
    ]
    for i in range(candidates):
        lines += ["    " + line for line in candidate(i, "block", "")]
        for j in range(max(0, i - spare), min(i, count - 1) + 1):
            lines.append(f"    o{j} = tl.where(a{i} & (filled == {j}), v{i}, o{j})")
        lines.append(f"    filled += a{i}.to(tl.int32)")
    lines += [
        f"    if tl.min(filled) < {count}:",
        "        filled = filled * 0",
        "        attempt = 0",
        f"        while (tl.min(filled) < {count}) & (attempt < 256):",
        "            again = _seeded_block(seed, row, segment, attempt, rounds)",
    ]
    for i in range(candidates):
        lines += ["            " + line for line in candidate(i, "again", "r")]
        for j in range(count):
            lines.append(
                f"            o{j} = tl.where(ra{i} & (filled == {j}), rv{i}, o{j})"
            )
        lines.append(f"            filled += ra{i}.to(tl.int32)")
    lines += ["            attempt += 1", f"    return {', '.join(slots)}"]
    return "\n".join(lines) + "\n"


def _compile_segment(name: str, count: int, wide: bool):
    source = _segment_source(name, count, wide)
    filename = f"<fhelium-seeded-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"tl": tl, "_seeded_block": _seeded_block, "__name__": __name__}
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    return triton.jit(namespace[name])


# Rows with q < 2^32 sample segments of eight coefficients from 32-bit
# candidates; rows with q >= 2^32 sample four from 64-bit candidates.
_segments8 = _compile_segment("_segments8", 8, wide=False)
_segments4 = _compile_segment("_segments4", 4, wide=True)


@triton.jit
def _select8(o, j):
    return tl.where(
        j < 4,
        tl.where(j < 2, tl.where(j == 0, o[0], o[1]), tl.where(j == 2, o[2], o[3])),
        tl.where(j < 6, tl.where(j == 4, o[4], o[5]), tl.where(j == 6, o[6], o[7])),
    )


@triton.jit
def _select4(o, j):
    return tl.where(j < 2, tl.where(j == 0, o[0], o[1]), tl.where(j == 2, o[2], o[3]))


@triton.jit
def seeded_key_residue(seed, row, coefficient, q, rounds: tl.constexpr, RADIX: tl.constexpr):
    r"""Return the seeded-key residue at stored ``row`` and ``coefficient``.

    ``seed`` points to the digit's int64 seed row $(w_0,\ldots,w_3,\mathrm{id},d)$.
    Rows with $q < 2^{32}$ take coefficients $8s$ to $8s+7$ from the first
    eight 32-bit candidates $w$ of segment $s$ with
    $w q \bmod 2^{32} \ge 2^{32} \bmod q$, each yielding
    $\lfloor w q / 2^{32}\rfloor$; rows with $q \ge 2^{32}$ take four
    coefficients per segment from 64-bit word pairs. Candidates are read in
    order from the ChaCha blocks with key $w$, nonce $\mathrm{id}$ and
    counter $((r \mathbin{|} d\cdot 2^{16})\cdot 2^{32}) + s + 2^{24} t$ for
    attempts $t = 0, 1, \ldots$, as in the native ``seeded_key_residues``.
    Each element samples its own segment.
    """
    c = coefficient.to(tl.int64)
    rows = row.to(tl.int64) + c * 0
    moduli = q.to(tl.int64) + c * 0
    narrow = tl.where(moduli < (1 << 32), moduli, 3)
    value = _select8(_segments8(seed, rows, c // 8, narrow, rounds), c % 8)
    if RADIX == 64:
        if tl.max(moduli) >= (1 << 32):
            wide = _select4(_segments4(seed, rows, c // 4, moduli, rounds), c % 4)
            value = tl.where(moduli >= (1 << 32), wide, value)
    return value.to(tl.int64)


@triton.jit
def _group_first(value, count: tl.constexpr):
    """Return the minimum of each group of ``count`` consecutive elements."""
    return tl.min(tl.reshape(value, (value.numel // count, count)), axis=1)


@triton.jit
def seeded_key_residue8(seed, row, coefficient, q, rounds: tl.constexpr, RADIX: tl.constexpr):
    """Return ``seeded_key_residue`` for a 1-D ``coefficient`` vector whose
    elements $8k$ to $8k+7$ hold coefficients $8m$ to $8m+7$ of one row.

    Each segment is sampled once.
    """
    c = coefficient.to(tl.int64)
    rows = row.to(tl.int64) + c * 0
    moduli = q.to(tl.int64) + c * 0
    narrow = tl.where(moduli < (1 << 32), moduli, 3)
    o = _segments8(
        seed, _group_first(rows, 8), _group_first(c, 8) // 8, _group_first(narrow, 8), rounds
    )
    value = tl.reshape(_stack8(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]), (c.numel,))
    if RADIX == 64:
        if tl.max(moduli) >= (1 << 32):
            o4 = _segments4(
                seed, _group_first(rows, 4), _group_first(c, 4) // 4, _group_first(moduli, 4), rounds
            )
            wide = tl.reshape(_stack4(o4[0], o4[1], o4[2], o4[3]), (c.numel,))
            value = tl.where(moduli >= (1 << 32), wide, value)
    return value.to(tl.int64)
