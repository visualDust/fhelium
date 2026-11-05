"""Integer modular arithmetic shared by generated Triton kernels."""

import triton
import triton.language as tl


@triton.jit
def montgomery_mul(a, b, q, k, radix_bits: tl.constexpr):
    """Return native-compatible REDC: canonical for R32, lazy for R62."""
    a = a.to(tl.uint64)
    b = b.to(tl.uint64)
    q = q.to(tl.uint64)
    k = k.to(tl.uint64)
    low = a * b
    if radix_bits == 32:
        m = (low * k) & 0xFFFFFFFF
        u = (low + m * q) >> 32
        return tl.where(u >= q, u - q, u).to(tl.int64)
    else:
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
