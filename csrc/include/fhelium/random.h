#pragma once

//! Cryptographically secure sampling for CKKS secrets, masks, and errors.
//!
//! Every sampler reads the ChaCha20 key stream selected by a `seed` Tensor:
//! an int64 Tensor of shape `[5]` on the device that receives the samples.
//! Elements 0 to 3 hold the 256-bit key as little-endian 64-bit words, and
//! element 4 holds the 64-bit nonce. Block $b$ of the stream is the ChaCha20
//! block (RFC 8439, 20 rounds) whose 64-bit block counter is $b$ and whose
//! nonce occupies state words 14 and 15. Under one key, two calls with the
//! same nonce read the same stream, so a caller assigns a distinct nonce to
//! every sampling call. Because the seed is read from device memory, a
//! captured CUDA Graph samples fresh values on replay when the captured
//! work updates the nonce in place.
//!
//! Sample $i$ of a call reads the consecutive stream words starting at word
//! $iw$ for a sampler-specific word count $w$, so each value depends only on
//! the seed and its flat index. Results are therefore identical on CPU and
//! GPU and independent of launch configuration. A `shape` argument is a
//! coefficient shape `[*batch, coefficient]`. A sampler given `rns_params`
//! returns the residues of its integer samples modulo every prime $q_j$ in
//! the reduced range $[0, q_j)$, with shape `[*batch, limb, coefficient]` and
//! the dtype of `rns_params` (row layout in `fhelium/rns_parameters.h`);
//! without `rns_params` it returns the centered int64 samples with shape
//! `shape`. The integers do not depend on `rns_params`, so the same seed
//! yields the same polynomial in every RNS basis.

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <optional>

#include <fhelium/export.h>

namespace fhelium::random {

/// Returns a seed whose key comes from the operating system's
/// cryptographic random source and whose nonce is zero.
///
/// The result is a CPU int64 Tensor of shape `[5]`; move it to the sampling
/// device before use.
FHELIUM_API at::Tensor generate_seed();

/// Returns `block_count` ChaCha20 key-stream blocks starting at block
/// `first_block`.
///
/// The result is a uint32 Tensor of shape `[block_count, 16]` on the device
/// of `seed`; row $r$ holds the 16 words of block `first_block` $+ r$.
FHELIUM_API at::Tensor chacha20_keystream(const at::Tensor& seed,
                                          int64_t first_block,
                                          int64_t block_count);

/// Samples residues uniformly and independently modulo every prime.
///
/// Each residue reads four words as a 128-bit integer $u$ and returns
/// $\lfloor u q_j / 2^{128} \rfloor$, whose distribution differs from the
/// uniform distribution on $[0, q_j)$ by at most $q_j / 2^{128}$ in
/// statistical distance. The result has shape
/// `[*shape[:-1], limb, shape[-1]]` and the dtype of `rns_params`.
FHELIUM_API at::Tensor sample_uniform_residues(const at::Tensor& seed,
                                               at::IntArrayRef shape,
                                               const at::Tensor& rns_params);

/// Returns the cumulative distribution table of the centered discrete
/// Gaussian distribution with parameter $\sigma$.
///
/// The distribution assigns $x \in \mathbb{Z}$ probability proportional to
/// $\exp(-x^2 / 2\sigma^2)$. The table is a CPU int64 Tensor of shape
/// `[M, 2]` for the magnitude bound
/// $M = \lceil \sigma \sqrt{256 \ln 2} \rceil + 1$, beyond which the
/// distribution has mass below $2^{-128}$. Row $k$ holds
/// $\Pr[|x| \le k]$ as a 127-bit fixed-point fraction split into its high
/// and low 64-bit words. Probabilities are computed in double-double
/// arithmetic and carry an absolute error below $2^{-100}$.
FHELIUM_API at::Tensor gaussian_cdt_table(double sigma);

/// Samples the centered discrete Gaussian distribution described by
/// `cdt_table`.
///
/// Each sample reads four words as a 128-bit integer. Its lowest bit selects
/// the sign, and the magnitude is the number of table rows not exceeding
/// the remaining 127 bits. The comparison visits every row, so the work per
/// sample does not depend on the sampled value. `cdt_table` comes from
/// `gaussian_cdt_table` and resides on the device of `seed`.
FHELIUM_API at::Tensor sample_discrete_gaussian(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    const at::Tensor& cdt_table,
    const std::optional<at::Tensor>& rns_params);

/// Adds discrete Gaussian samples to residues in place.
///
/// `residues` is a contiguous Tensor of shape `[*batch, limb, coefficient]`
/// with the dtype and limbs of `rns_params` and entries in $[0, 2q_j)$. The
/// samples are those of `sample_discrete_gaussian` for the shape
/// `[*batch, coefficient]` and the same seed, and every entry becomes
/// $(x + e) \bmod q_j$ in $[0, q_j)$ for the sample $e$ of its coefficient.
FHELIUM_API void add_discrete_gaussian_(at::Tensor& residues,
                                        const at::Tensor& seed,
                                        const at::Tensor& cdt_table,
                                        const at::Tensor& rns_params);

/// Samples the centered binomial distribution with parameter $k$.
///
/// Each sample reads two words $a, b$ and returns
/// $\mathrm{popcount}(a \bmod 2^k) - \mathrm{popcount}(b \bmod 2^k)$ for
/// $1 \le k \le 32$. The distribution has mean zero and standard deviation
/// $\sqrt{k/2}$ and is supported on $[-k, k]$.
FHELIUM_API at::Tensor sample_centered_binomial(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    int64_t k,
    const std::optional<at::Tensor>& rns_params);

/// Samples independent ternary coefficients with
/// $\Pr[\pm 1] = \rho / 2$ and $\Pr[0] = 1 - \rho$.
///
/// Each sample reads two words as a 64-bit integer $u$. The coefficient is
/// nonzero when $\lfloor u / 2 \rfloor < \lfloor \rho \, 2^{63} \rfloor$,
/// and the lowest bit of $u$ selects its sign. $\rho = 1/2$ gives the
/// $\mathcal{ZO}(1/2)$ encryption mask exactly; $\rho = 2/3$ gives the
/// uniform ternary distribution within statistical distance $2^{-62}$.
/// Requires $0 \le \rho \le 1$.
FHELIUM_API at::Tensor sample_ternary(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    double nonzero_probability,
    const std::optional<at::Tensor>& rns_params);

/// Samples ternary coefficients with exactly `weight` nonzero entries in
/// each row of the final axis.
///
/// Coefficient $i$ reads four words: a 127-bit sorting label and a sign
/// bit. The `weight` coefficients with the smallest labels in a row receive
/// $\pm 1$ by their sign bits and the others receive zero, so the support is
/// a uniform `weight`-subset unless two labels collide, which has
/// probability below $n^2 / 2^{128}$ for $n$ coefficients per row. Requires
/// $0 \le$ `weight` $\le$ `shape[-1]`.
FHELIUM_API at::Tensor sample_fixed_weight_ternary(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    int64_t weight,
    const std::optional<at::Tensor>& rns_params);

/// Rounds float64 values to integers randomly with expectation equal to
/// the value.
///
/// Value $x$ reads one word $u$ and rounds to
/// $\mathrm{sign}(x) (\lfloor |x| \rfloor + [u < \lfloor 2^{32}
/// (|x| - \lfloor |x| \rfloor) \rfloor])$, so the rounding probability is
/// quantized to $2^{-32}$. Value $i$ of the flattened `values` reads word
/// $i \cdot$ `word_stride`, so a compact array of every `word_stride`-th
/// coefficient rounds with the words of the corresponding full array. The
/// result is an int64 Tensor with the shape of `values`, which reside on the
/// device of `seed`. Requires `word_stride` $\ge 1$.
FHELIUM_API at::Tensor stochastic_round(const at::Tensor& seed,
                                        const at::Tensor& values,
                                        int64_t word_stride = 1);

}  // namespace fhelium::random
