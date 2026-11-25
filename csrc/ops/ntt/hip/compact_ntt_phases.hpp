#pragma once

#include <torch/torch.h>

#include <type_traits>

#include "../../common/hip/kernel_support.hpp"
#include "../../common/montgomery.hpp"
#include "../../common/rns_parameters.h"

// Compact radix-2 transforms in two passes over global memory.
//
// For N = 2^L the DIF forward transform runs stages [0, A) in a column pass
// and stages [A, L) in a contiguous pass, with A = floor(L / 2). A column
// tile holds 2^A coefficients spaced 2^(L-A) apart for W adjacent columns;
// a contiguous tile holds 2^(L-A) adjacent coefficients. The inverse DIT
// transform runs stages [0, L-A) contiguously and [L-A, L) by columns. Each
// pass executes its stages in rounds of three in registers, exchanging the
// tile through shared memory between rounds. Butterflies, twiddle indices and
// lazy intervals match the per-stage compact schedule, so results are
// identical for both table encodings.

namespace fhelium::ntt::compact {

constexpr int kRoundBits = 3;
constexpr int kRoundSize = 1 << kRoundBits;
constexpr int kPassThreads = 256;

template <typename T>
struct Modulus {
  T twice, lo, hi, neg_lo, neg_hi;

  __device__ __forceinline__ Modulus(const T* params, int64_t stride, int row)
      : twice(params[RNS_PARAM_TWICE_MODULUS * stride + row]),
        lo(params[RNS_PARAM_MODULUS_LO * stride + row]),
        hi(params[RNS_PARAM_MODULUS_HI * stride + row]),
        neg_lo(params[RNS_PARAM_NEG_INV_MODULUS_LO * stride + row]),
        neg_hi(params[RNS_PARAM_NEG_INV_MODULUS_HI * stride + row]) {}

  __device__ __forceinline__ T mul(T a, T b) const {
    return montgomery_mul(a, b, lo, hi, neg_lo, neg_hi);
  }
};

// Loads C consecutive values from a C-aligned position of a 16-byte aligned
// row; the caller guarantees alignment of the row start.
template <typename T, int C>
__device__ __forceinline__ void load_run(T (&out)[C], const T* p) {
  constexpr int bytes = C * static_cast<int>(sizeof(T));
  if constexpr (bytes >= 16) {
#pragma unroll
    for (int i = 0; i < C; i += 16 / static_cast<int>(sizeof(T))) {
      const int4 v = __ldg(reinterpret_cast<const int4*>(p + i));
      const T* vt = reinterpret_cast<const T*>(&v);
#pragma unroll
      for (int k = 0; k < 16 / static_cast<int>(sizeof(T)); ++k) out[i + k] = vt[k];
    }
  } else if constexpr (bytes == 8) {
    const int2 v = __ldg(reinterpret_cast<const int2*>(p));
    const T* vt = reinterpret_cast<const T*>(&v);
#pragma unroll
    for (int k = 0; k < C; ++k) out[k] = vt[k];
  } else {
#pragma unroll
    for (int k = 0; k < C; ++k) out[k] = __ldg(p + k);
  }
}

template <typename T, int C>
__device__ __forceinline__ void store_run(T* p, const T (&in)[C]) {
  static_assert(C * sizeof(T) % 16 == 0, "vector store width");
#pragma unroll
  for (int i = 0; i < C; i += 16 / static_cast<int>(sizeof(T))) {
    *reinterpret_cast<int4*>(p + i) = *reinterpret_cast<const int4*>(&in[i]);
  }
}

// Twiddle words read for C consecutive table entries.
template <typename T, int WORDS>
struct TwiddleRun {
  T w[WORDS];
};

// Montgomery-encoded [prime, N] table.
template <typename T>
struct MontgomeryTwiddles {
  const T* data;
  int64_t row_stride;

  struct Row {
    const T* t;
    template <int C>
    using Run = TwiddleRun<T, C>;
    template <int C>
    __device__ __forceinline__ void load(Run<C>& run, int index) const {
      load_run<T, C>(run.w, t + index);
    }
    template <int C>
    __device__ __forceinline__ T mul(const Run<C>& run, int c, T value,
                                     const Modulus<T>& m) const {
      return m.mul(run.w[c], value);
    }
  };
  __device__ __forceinline__ Row row(int r) const { return {data + r * row_stride}; }
};

// [prime, N, 2] pairs of a standard root w < q and floor(w 2^32 / q).
struct Shoup32Twiddles {
  const int32_t* data;
  int64_t row_stride;

  struct Row {
    const int32_t* t;
    template <int C>
    using Run = TwiddleRun<int32_t, 2 * C>;
    template <int C>
    __device__ __forceinline__ void load(Run<C>& run, int index) const {
      load_run<int32_t, 2 * C>(run.w, t + 2 * index);
    }
    template <int C>
    __device__ __forceinline__ int32_t mul(const Run<C>& run, int c, int32_t value,
                                           const Modulus<int32_t>& m) const {
      const uint32_t q = static_cast<uint32_t>(m.lo) |
                         (static_cast<uint32_t>(m.hi) << 16);
      const uint32_t x = static_cast<uint32_t>(value);
      // q < 2^30, w < q and x < 2q: the remainder lies in [0, 2q).
      return static_cast<int32_t>(
          x * static_cast<uint32_t>(run.w[2 * c]) -
          __umulhi(x, static_cast<uint32_t>(run.w[2 * c + 1])) * q);
    }
  };
  __device__ __forceinline__ Row row(int r) const { return {data + r * row_stride}; }
};

struct PassOperand {
  int64_t batch_stride, row_stride;
};

// Coefficient-wise transitions applied when a pass loads its operand.
enum class Load { Plain, ToMontgomery };

// Final-pass outputs. StoreValues writes the transform to the pass
// destination; the other outputs bind one (batch, row) and receive each value.
struct StoreValues {};

// Inverse outputs: Montgomery N^{-1} normalization, then optionally the
// standard lazy representation, the standard representation, or the centered
// representative of the standard residue.
enum class InverseOutput { Montgomery, StandardLazy, Standard, Centered };

template <typename T, InverseOutput OUT>
struct StoreInverse {
  T* data;
  int64_t batch_stride, row_stride;
  struct Bound {
    T* p;
    T n_inverse;
    __device__ __forceinline__ T convert(T v, const Modulus<T>& m) const {
      v = m.mul(v, n_inverse);
      if constexpr (OUT != InverseOutput::Montgomery) {
        v = montgomery_reduce(v, m.lo, m.hi, m.neg_lo, m.neg_hi);
        if constexpr (OUT != InverseOutput::StandardLazy) {
          v = reduce_lazy_residue(v, m.twice);
          if constexpr (OUT == InverseOutput::Centered) v = center_residue(v, m.twice);
        }
      }
      return v;
    }
    template <int E>
    __device__ __forceinline__ void store_run(int c, T (&x)[E], const Modulus<T>& m) const {
#pragma unroll
      for (int e = 0; e < E; ++e) x[e] = convert(x[e], m);
      compact::store_run<T, E>(p + c, x);
    }
    __device__ __forceinline__ void store(int c, T v, const Modulus<T>& m) const {
      p[c] = convert(v, m);
    }
  };
  __device__ __forceinline__ Bound bind(int batch, int row, const T* params,
                                        int64_t param_stride) const {
    return {data + batch * batch_stride + row * row_stride,
            params[RNS_PARAM_N_INV_MONTGOMERY * param_stride + row]};
  }
};

// Forward output NTT(x) + multiplier[row] * addend, for the NTT-domain ModDown.
template <typename T>
struct StoreAddScaled {
  T* data;
  int64_t batch_stride, row_stride;
  const T* addend;
  int64_t addend_batch_stride, addend_row_stride;
  const T* multiplier;
  int64_t multiplier_stride;
  struct Bound {
    T* p;
    const T* a;
    T multiplier;
    __device__ __forceinline__ void store(int c, T v, const Modulus<T>& m) const {
      p[c] = add_lazy_residues(v, m.mul(a[c], multiplier), m.twice);
    }
    template <int E>
    __device__ __forceinline__ void store_run(int c, T (&x)[E], const Modulus<T>& m) const {
      T addends[E];
      load_run<T, E>(addends, a + c);
#pragma unroll
      for (int e = 0; e < E; ++e)
        x[e] = add_lazy_residues(x[e], m.mul(addends[e], multiplier), m.twice);
      compact::store_run<T, E>(p + c, x);
    }
  };
  __device__ __forceinline__ Bound bind(int batch, int row, const T*, int64_t) const {
    return {data + batch * batch_stride + row * row_stride,
            addend + batch * addend_batch_stride + row * addend_row_stride,
            multiplier[row * multiplier_stride]};
  }
};

// Accumulates NTT(x) times the two key components into two accumulators; the
// transformed digit itself is not stored. Row r of the operand reads key row
// key_row_start + r for Q rows and key_p_row_start + (r - q_rows) for P rows.
template <typename T>
struct AccumulateKeyProducts {
  const T* key;
  int64_t key_component_stride, key_row_stride;
  T* accumulator0;
  PassOperand layout0;
  T* accumulator1;
  PassOperand layout1;
  int key_row_start, key_q_rows, key_p_row_start;
  struct Bound {
    const T* k0;
    const T* k1;
    T* a0;
    T* a1;
    __device__ __forceinline__ void store(int c, T v, const Modulus<T>& m) const {
      a0[c] = add_lazy_residues(a0[c], m.mul(v, k0[c]), m.twice);
      a1[c] = add_lazy_residues(a1[c], m.mul(v, k1[c]), m.twice);
    }
    template <int E>
    __device__ __forceinline__ void store_run(int c, T (&x)[E], const Modulus<T>& m) const {
      T key0[E], key1[E], sum0[E], sum1[E];
      load_run<T, E>(key0, k0 + c);
      load_run<T, E>(key1, k1 + c);
      load_run<T, E>(sum0, a0 + c);
      load_run<T, E>(sum1, a1 + c);
#pragma unroll
      for (int e = 0; e < E; ++e) {
        sum0[e] = add_lazy_residues(sum0[e], m.mul(x[e], key0[e]), m.twice);
        sum1[e] = add_lazy_residues(sum1[e], m.mul(x[e], key1[e]), m.twice);
      }
      compact::store_run<T, E>(a0 + c, sum0);
      compact::store_run<T, E>(a1 + c, sum1);
    }
  };
  __device__ __forceinline__ Bound bind(int batch, int row, const T*, int64_t) const {
    const int key_row = row < key_q_rows ? key_row_start + row
                                         : key_p_row_start + (row - key_q_rows);
    const T* k = key + key_row * key_row_stride;
    return {k, k + key_component_stride,
            accumulator0 + batch * layout0.batch_stride + row * layout0.row_stride,
            accumulator1 + batch * layout1.batch_stride + row * layout1.row_stride};
  }
};

__device__ __forceinline__ int padded(int i) { return i + (i >> 5); }

// One radix-2 stage acting on register bit BIT of a 2^K register window.
// Pairs that differ only below BIT share a twiddle; the C distinct twiddles
// are consecutive table entries starting at `index`.
template <typename T, bool INVERSE, int K, int BIT, typename Row>
__device__ __forceinline__ void window_stage_at(T (&x)[1 << K], const Row& tw, int index,
                                             const Modulus<T>& m) {
  constexpr int E = 1 << K;
  constexpr int C = E >> (BIT + 1);
  typename Row::template Run<C> run;
  tw.template load<C>(run, index);
#pragma unroll
  for (int e = 0; e < E; ++e) {
    if (e & (1 << BIT)) continue;
    const int high = e | (1 << BIT);
    const T u = x[e];
    if constexpr (!INVERSE) {
      const T v = tw.template mul<C>(run, e >> (BIT + 1), x[high], m);
      x[e] = add_lazy_residues(u, v, m.twice);
      x[high] = sub_lazy_residues(u, v, m.twice);
    } else {
      const T v = x[high];
      x[high] = tw.template mul<C>(run, e >> (BIT + 1),
                                   sub_lazy_residues(u, v, m.twice), m);
      x[e] = add_lazy_residues(u, v, m.twice);
    }
  }
}

template <typename T, bool INVERSE, int K, typename Row>
__device__ __forceinline__ void window_stage(T (&x)[1 << K], const Row& tw, int bit,
                                             int index, const Modulus<T>& m) {
  static_assert(K == 3, "stage dispatch covers three register bits");
  if (bit == 0) window_stage_at<T, INVERSE, K, 0>(x, tw, index, m);
  else if (bit == 1) window_stage_at<T, INVERSE, K, 1>(x, tw, index, m);
  else window_stage_at<T, INVERSE, K, 2>(x, tw, index, m);
}

// One pass of M stages over tiles of 2^M coefficients. A tile is indexed by
// r in [0, 2^M): r * 2^(L-M) + column for a column pass, tile * 2^M + r
// otherwise. Forward local stage l acts on r bit M-1-l and global stage
// first_stage + l; inverse local stage l acts on r bit l. LOAD applies to the
// values read from `source`; `output` receives the values of a final pass,
// and other passes store to `destination`.
template <typename T, typename Twiddles, bool INVERSE, int M, bool COLUMN,
          Load LOAD, typename Output>
__global__ void __launch_bounds__(kPassThreads) compact_ntt_pass_kernel(
    const T* __restrict__ source, PassOperand source_layout, T* destination,
    PassOperand destination_layout, Output output, const Twiddles twiddles,
    const T* __restrict__ params, int64_t param_stride, int log_n, int first_stage,
    int log_columns) {
  constexpr int K = kRoundBits;
  constexpr int E = kRoundSize;
  constexpr int ROUNDS = (M + K - 1) / K;
  extern __shared__ unsigned char shared_raw[];
  T* shared = reinterpret_cast<T*>(shared_raw);
  const int row = blockIdx.z;
  const int batch = blockIdx.y;
  const int N = 1 << log_n;
  const Modulus<T> m(params, param_stride, row);
  const auto tw = twiddles.row(row);
  const int columns = 1 << log_columns;
  const int tid = threadIdx.x;
  int lane, rest;
  if constexpr (COLUMN) {
    lane = tid & (columns - 1);
    rest = tid >> log_columns;
  } else {
    rest = tid & ((1 << (M - K)) - 1);
    lane = tid >> (M - K);
  }
  const int group = blockIdx.x * columns + lane;
  auto global_index = [&](int r) {
    return COLUMN ? (r << (log_n - M)) + group : (group << M) + r;
  };
  auto shared_index = [&](int r) {
    return padded(COLUMN ? (r << log_columns) + lane : (lane << M) + r);
  };
  const T* in = source + batch * source_layout.batch_stride + row * source_layout.row_stride;
  T* out = destination + batch * destination_layout.batch_stride +
           row * destination_layout.row_stride;
  T x[E];
#pragma unroll
  for (int round = 0; round < ROUNDS; ++round) {
    const int bits = round == ROUNDS - 1 ? M - round * K : K;
    // Window [low, low + K) of r held in registers during this round.
    int low;
    if constexpr (!INVERSE) {
      low = M - (round + 1) * K;
      low = low > 0 ? low : 0;
    } else {
      low = round * K < M - K ? round * K : M - K;
    }
    const int base = (rest & ((1 << low) - 1)) | ((rest >> low) << (low + K));
    if (round == 0) {
      if (!COLUMN && low == 0) {
        load_run<T, E>(x, in + global_index(base));
      } else {
#pragma unroll
        for (int e = 0; e < E; ++e) x[e] = in[global_index(base | (e << low))];
      }
      if constexpr (LOAD == Load::ToMontgomery) {
        const T r2 = params[RNS_PARAM_R2 * param_stride + row];
#pragma unroll
        for (int e = 0; e < E; ++e) x[e] = m.mul(x[e], r2);
      }
    } else {
#pragma unroll
      for (int e = 0; e < E; ++e) x[e] = shared[shared_index(base | (e << low))];
    }
    const int first = global_index(base);
#pragma unroll
    for (int j = 0; j < K; ++j) {
      if (j >= bits) break;
      const int local = round * K + j;
      const int stage = first_stage + local;
      const int bit = (INVERSE ? local : M - 1 - local) - low;
      const int index = INVERSE ? (N >> (stage + 1)) + (first >> (stage + 1))
                                : (1 << stage) + (first >> (log_n - stage));
      window_stage<T, INVERSE, K>(x, tw, bit, index, m);
    }
    if (round == ROUNDS - 1) {
      if constexpr (std::is_same_v<Output, StoreValues>) {
        if (!COLUMN && low == 0) {
          store_run<T, E>(out + global_index(base), x);
        } else {
#pragma unroll
          for (int e = 0; e < E; ++e) out[global_index(base | (e << low))] = x[e];
        }
      } else {
        const auto bound = output.bind(batch, row, params, param_stride);
        if (!COLUMN && low == 0) {
          bound.template store_run<E>(global_index(base), x, m);
        } else {
#pragma unroll
          for (int e = 0; e < E; ++e) bound.store(global_index(base | (e << low)), x[e], m);
        }
      }
    } else {
      __syncthreads();
#pragma unroll
      for (int e = 0; e < E; ++e) shared[shared_index(base | (e << low))] = x[e];
      __syncthreads();
    }
  }
}

struct PassGeometry {
  int log_n;
  int column_bits;      // stages of the column pass
  int contiguous_bits;  // stages of the contiguous pass
};

inline PassGeometry pass_geometry(int64_t n) {
  int log_n = 0;
  while ((int64_t{1} << log_n) < n) ++log_n;
  return {log_n, log_n / 2, log_n - log_n / 2};
}

template <typename T, typename Twiddles, bool INVERSE, int M, bool COLUMN, Load LOAD,
          typename Output>
void launch_pass_m(const T* source, PassOperand source_layout, T* destination,
                   PassOperand destination_layout, const Output& output,
                   const Twiddles& twiddles, const T* params, int64_t param_stride,
                   int log_n, int first_stage, int rows, int batches,
                   hipStream_t stream) {
  constexpr int K = kRoundBits;
  const int lanes_per_block = kPassThreads >> (M - K);
  const int groups = 1 << (log_n - M);
  const int lanes = lanes_per_block < groups ? lanes_per_block : groups;
  int log_columns = 0;
  while ((1 << log_columns) < lanes) ++log_columns;
  const int tile = (1 << M) << log_columns;
  const size_t shared_bytes = static_cast<size_t>(tile + tile / 32 + 1) * sizeof(T);
  // Batches of one row run adjacently and share its twiddles in L2.
  const dim3 grid(groups >> log_columns, batches, rows);
  compact_ntt_pass_kernel<T, Twiddles, INVERSE, M, COLUMN, LOAD, Output>
      <<<grid, lanes << (M - K), shared_bytes, stream>>>(
          source, source_layout, destination, destination_layout, output, twiddles,
          params, param_stride, log_n, first_stage, log_columns);
}

template <typename T, typename Twiddles, bool INVERSE, bool COLUMN, Load LOAD,
          typename Output>
void launch_pass(int stages, const T* source, PassOperand source_layout, T* destination,
                 PassOperand destination_layout, const Output& output,
                 const Twiddles& twiddles, const T* params, int64_t param_stride,
                 int log_n, int first_stage, int rows, int batches,
                 hipStream_t stream) {
#define FHELIUM_COMPACT_PASS(M)                                                    \
  case M:                                                                         \
    launch_pass_m<T, Twiddles, INVERSE, M, COLUMN, LOAD, Output>(                 \
        source, source_layout, destination, destination_layout, output, twiddles, \
        params, param_stride, log_n, first_stage, rows, batches, stream);         \
    break;
  switch (stages) {
    FHELIUM_COMPACT_PASS(4)
    FHELIUM_COMPACT_PASS(5)
    FHELIUM_COMPACT_PASS(6)
    FHELIUM_COMPACT_PASS(7)
    FHELIUM_COMPACT_PASS(8)
    FHELIUM_COMPACT_PASS(9)
    FHELIUM_COMPACT_PASS(10)
    default:
      TORCH_CHECK(false, "Compact NTT supports N from 2^8 to 2^20");
  }
#undef FHELIUM_COMPACT_PASS
}

// Operand views: [batch, prime, coefficient] with unit coefficient stride.
template <typename T>
struct Operand {
  T* data;
  PassOperand layout;
  int rows, batches, n;
};

template <typename T>
Operand<T> operand(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 3 && tensor.stride(2) == 1,
              "Compact NTT operands require unit coefficient stride");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0 &&
                  tensor.stride(1) * sizeof(T) % 16 == 0 &&
                  tensor.stride(0) * sizeof(T) % 16 == 0,
              "Compact NTT operands require 16-byte aligned rows");
  return {tensor.data_ptr<T>(), {tensor.stride(0), tensor.stride(1)},
          static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0)),
          static_cast<int>(tensor.size(2))};
}

// Calls `function` with the twiddle reader for a Montgomery [prime, N] table
// or, for int32 operands, a Shoup32 [prime, N, 2] table.
template <typename T, typename Function>
void with_twiddles(const torch::Tensor& table, Function&& function) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(table.data_ptr()) % 16 == 0 &&
                  table.stride(0) * table.element_size() % 16 == 0 &&
                  table.stride(-1) == 1 && (table.dim() == 2 || table.stride(1) == 2),
              "Compact twiddle rows must be contiguous and 16-byte aligned");
  if constexpr (std::is_same_v<T, int32_t>) {
    if (table.dim() == 3) {
      function(Shoup32Twiddles{table.data_ptr<int32_t>(), table.stride(0)});
      return;
    }
  }
  function(MontgomeryTwiddles<T>{table.data_ptr<T>(), table.stride(0)});
}

// Forward transform of `source` into `destination` (which may alias it),
// with the final values delivered to `output`.
template <typename T, Load LOAD, typename Output>
void forward(const Operand<T>& source, const Operand<T>& destination,
             const Output& output, const torch::Tensor& twiddles,
             const torch::Tensor& params, hipStream_t stream) {
  TORCH_CHECK(params.stride(1) == 1, "RNS parameter rows must be contiguous");
  const auto g = pass_geometry(destination.n);
  const T* p = params.data_ptr<T>();
  const int64_t ps = params.stride(0);
  with_twiddles<T>(twiddles, [&](const auto& tw) {
    using Tw = std::decay_t<decltype(tw)>;
    launch_pass<T, Tw, false, true, LOAD>(
        g.column_bits, source.data, source.layout, destination.data,
        destination.layout, StoreValues{}, tw, p, ps, g.log_n, 0,
        destination.rows, destination.batches, stream);
    launch_pass<T, Tw, false, false, Load::Plain>(
        g.contiguous_bits, destination.data, destination.layout, destination.data,
        destination.layout, output, tw, p, ps, g.log_n, g.column_bits,
        destination.rows, destination.batches, stream);
  });
  check_hip_launch();
}

template <typename T, typename Output>
void inverse(const Operand<T>& values, const Output& output,
             const torch::Tensor& twiddles, const torch::Tensor& params,
             hipStream_t stream) {
  TORCH_CHECK(params.stride(1) == 1, "RNS parameter rows must be contiguous");
  const auto g = pass_geometry(values.n);
  const T* p = params.data_ptr<T>();
  const int64_t ps = params.stride(0);
  with_twiddles<T>(twiddles, [&](const auto& tw) {
    using Tw = std::decay_t<decltype(tw)>;
    launch_pass<T, Tw, true, false, Load::Plain>(
        g.contiguous_bits, values.data, values.layout, values.data, values.layout,
        StoreValues{}, tw, p, ps, g.log_n, 0, values.rows, values.batches, stream);
    launch_pass<T, Tw, true, true, Load::Plain>(
        g.column_bits, values.data, values.layout, values.data, values.layout, output,
        tw, p, ps, g.log_n, g.contiguous_bits, values.rows, values.batches, stream);
  });
  check_hip_launch();
}

}  // namespace fhelium::ntt::compact
