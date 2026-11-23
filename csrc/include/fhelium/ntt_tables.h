#pragma once

//! Construction of negacyclic NTT roots, twiddle tables, and schedules for
//! the transforms in `fhelium/ntt.h`.
//!
//! For a modulus $q \equiv 1 \pmod{2N}$ with $N = 2^{\mathtt{log\_n}}$,
//! $\psi_q$ denotes the primitive $2N$-th root of unity returned by
//! `negacyclic_root`, and $\operatorname{br}_k$ reverses the low $k$ bits of
//! an index. Twiddle tables hold standard residues in the integral `dtype`,
//! with one row per modulus in the order of `moduli`; convert them to
//! Montgomery form with the `rns_params` of the same moduli before use. The
//! functions compute on the host and return new CPU Tensors.

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <tuple>

#include <fhelium/export.h>

namespace fhelium::ntt {

/// Returns $\psi_q = c^{(q-1)/2N} \bmod q$ for the smallest integer
/// $c \ge 2$ with $\psi_q^N \ne 1$, which makes $\psi_q$ a primitive
/// $2N$-th root of unity.
FHELIUM_API int64_t negacyclic_root(int64_t modulus, int64_t log_n);

/// Returns the bit-reversed powers of $\psi_q$ and $\psi_q^{-1}$.
///
/// Column $k$ of the forward table holds $\psi_q^{\operatorname{br}_{\log
/// N}(k)}$ and column $k$ of the inverse table holds
/// $\psi_q^{-\operatorname{br}_{\log N}(k)}$, both of shape `[modulus, N]`.
/// When `shoup32` is true, `dtype` is int32, every $q < 2^{30}$, and each
/// table has shape `[modulus, N, 2]` holding the pair
/// $(w, \lfloor w 2^{32} / q \rfloor)$ with the quotient stored as the bit
/// pattern of a uint32; Shoup pairs are used without Montgomery conversion.
FHELIUM_API std::tuple<at::Tensor, at::Tensor> compact_twiddles(
    at::IntArrayRef moduli,
    int64_t log_n,
    at::ScalarType dtype,
    bool shoup32 = false);

/// Returns the tables of the fixed power-of-two radix transform with radix
/// $r = 2^b \in \{4, 8, 16\}$, where $b$ divides `log_n`.
///
/// Each digit of the transform is a twisted cyclic $r$-point NTT. The
/// forward and inverse outer tables have shape `[modulus, N - 1]`: for every
/// digit group they hold $\beta^1, \dots, \beta^{r-1}$, where $\beta =
/// \psi_q^{\operatorname{br}_{\log N}(c)}$ for the group's compact index $c$
/// and the inverse tables use $\psi_q^{-1}$. The root tables of shape
/// `[modulus, r]` hold the powers $\psi_q^{2Nk/r}$ and
/// $\psi_q^{-2Nk/r}$ for $0 \le k < r$. The results are, in order, forward
/// outer, inverse outer, forward roots, and inverse roots.
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
power_of_two_radix_twiddles(at::IntArrayRef moduli,
                            int64_t log_n,
                            int64_t radix,
                            at::ScalarType dtype);

/// Returns the schedules and stage twiddles of the indexed radix-2
/// transform.
///
/// The forward transform is decimation in frequency: stage $s$ has $2^s$
/// groups of span $h = N / 2^{s+1}$, and butterfly $(e, e + h)$ of group $g$
/// uses compact twiddle column $2^s + g$. The inverse transform is
/// decimation in time with the spans in reverse order. The results are, in
/// order, the forward indices (int32, shape `[2, log_n, N/2]`: even then odd
/// positions), the forward twiddles (shape `[modulus, log_n, N/2]`), the
/// inverse indices, and the inverse twiddles; twiddles are gathered from the
/// `compact_twiddles` tables.
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
indexed_radix2_plan(at::IntArrayRef moduli,
                    int64_t log_n,
                    at::ScalarType dtype);

/// Returns the forward indexed radix-2 plan of the subring transform of
/// length $U = 2^{\mathtt{log\_extent}} < N$ that evaluates with the root
/// $\psi_q^{N/U}$ of the full ring.
///
/// The results are the even and odd indices (int32, shape
/// `[log_extent, U/2]`) and the stage twiddles (shape
/// `[modulus, log_extent, U/2]`), with the schedule of
/// `indexed_radix2_plan` for length $U$.
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor>
subring_indexed_radix2_plan(at::IntArrayRef moduli,
                            int64_t log_n,
                            int64_t log_extent,
                            at::ScalarType dtype);

/// Returns the int32 gather indices of shape `[N]` that apply the Galois
/// automorphism $X \mapsto X^g$ to bit-reversed NTT evaluations: position
/// $j$ of the result reads stored position `indices[j]`.
FHELIUM_API at::Tensor galois_indices(int64_t log_n, int64_t galois_element);

}  // namespace fhelium::ntt
