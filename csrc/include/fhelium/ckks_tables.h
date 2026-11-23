#pragma once

//! Construction of the CKKS slot-embedding and Galois index tables.
//!
//! The ring is $\mathbb{Z}[X]/(X^N + 1)$ with $N = 2^{\mathtt{log\_n}}$ and
//! $N/2$ complex slots. A rotation by `step` follows
//! `torch.roll(slots, shifts=step)`. The slot generator $\gamma \in \{3, 5\}$
//! orders the slots by the powers of $\gamma$ modulo $2N$. Index tables are
//! computed on the host and returned as new CPU Tensors.

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <tuple>

#include <fhelium/export.h>

namespace fhelium::ckks {

/// Returns the Galois element $g = \gamma^{e} \bmod 2N$ of the automorphism
/// $X \mapsto X^g$ that rotates the slots by `step`, where $e$ is $-$`step`
/// for $\gamma = 5$ and `step` for $\gamma = 3$, reduced modulo $N$.
FHELIUM_API int64_t rotation_galois_element(int64_t log_n,
                                            int64_t step,
                                            int64_t generator);

/// Returns the gather that applies $X \mapsto X^g$ to coefficient-domain
/// residues: destination coefficient $j$ reads source coefficient
/// `indices[j]` and multiplies it by `signs[j]` $\in \{-1, 1\}$. The results
/// are int32 and int8 Tensors of shape `[N]`.
FHELIUM_API std::tuple<at::Tensor, at::Tensor> coefficient_galois_indices(
    int64_t log_n,
    int64_t galois_element);

/// Returns the int64 slot permutations of the canonical embedding.
///
/// Encoding scatters slot $i$ to FFT position `pre[i]` (shape `[N/2]`), and
/// decoding gathers FFT position `post[k]` into slot position $k$ of the
/// conjugate-extended vector (shape `[N]`).
FHELIUM_API std::tuple<at::Tensor, at::Tensor> slot_permutations(
    int64_t log_n,
    int64_t generator);

/// Returns the encoded length $U$ of a message that repeats with period
/// `period` slots: $U = \max(2\,\mathtt{period}, 4)$ for $\gamma = 3$ and
/// $\max(2\,\mathtt{period}, 2)$ for $\gamma = 5$. `period` is a power of
/// two dividing $N/2$, and $U < N$.
FHELIUM_API int64_t periodic_extent(int64_t period,
                                    int64_t log_n,
                                    int64_t generator);

/// Returns the int64 slot scatter of shape `[U/2]` for a periodic message of
/// encoded length $U$: position $i$ is $(\sigma^i \bmod 2U - 1)/2$ for
/// $\sigma = 5$ when $\gamma = 5$ and $\sigma = 3^{-1} \bmod 2U$ when
/// $\gamma = 3$.
FHELIUM_API at::Tensor periodic_slot_positions(int64_t extent,
                                               int64_t generator);

/// Returns the complex128 twist $e^{-i\pi k/N}$ (`inverse` false) or
/// $e^{i\pi k/N}$ (`inverse` true) for $0 \le k < N$ on `device`.
///
/// Encoding multiplies the FFT of the conjugate-extended slots by the
/// forward twist; decoding multiplies the coefficients by the inverse twist
/// before the inverse FFT.
FHELIUM_API at::Tensor slot_twist(int64_t length,
                                  bool inverse,
                                  c10::Device device);

}  // namespace fhelium::ckks
