#pragma once

#include <c10/util/Exception.h>

#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__CUDACC__) && !defined(__HIPCC__)
#include <intrin.h>
#endif

// ChaCha20 block function and per-sample transforms shared by the CPU, CUDA,
// and HIP samplers. Sample i of a call consumes stream words [i*w, i*w + w)
// for the sampler's word count w, which divides the 16-word block.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FHELIUM_RANDOM_HD __host__ __device__ __forceinline__
#else
#define FHELIUM_RANDOM_HD inline
#endif

namespace fhelium::random::detail {

constexpr int kBlockWords = 16;
constexpr int kSeedWords = 5;

struct StreamKey {
  uint32_t key[8];
  uint32_t nonce[2];
};

FHELIUM_RANDOM_HD StreamKey stream_key(const int64_t* seed) {
  StreamKey key;
  for (int i = 0; i < 4; ++i) {
    const auto word = static_cast<uint64_t>(seed[i]);
    key.key[2 * i] = static_cast<uint32_t>(word);
    key.key[2 * i + 1] = static_cast<uint32_t>(word >> 32);
  }
  const auto nonce = static_cast<uint64_t>(seed[4]);
  key.nonce[0] = static_cast<uint32_t>(nonce);
  key.nonce[1] = static_cast<uint32_t>(nonce >> 32);
  return key;
}

FHELIUM_RANDOM_HD uint32_t rotl32(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

#define FHELIUM_CHACHA_QUARTER_ROUND(x, a, b, c, d) \
  x[a] += x[b];                                     \
  x[d] = rotl32(x[d] ^ x[a], 16);                   \
  x[c] += x[d];                                     \
  x[b] = rotl32(x[b] ^ x[c], 12);                   \
  x[a] += x[b];                                     \
  x[d] = rotl32(x[d] ^ x[a], 8);                    \
  x[c] += x[d];                                     \
  x[b] = rotl32(x[b] ^ x[c], 7);

// ChaCha block with `Rounds` rounds (RFC 8439 for Rounds = 20): key words
// 4..11, the 64-bit block counter in words 12..13 and the nonce in words
// 14..15.
template <int Rounds>
FHELIUM_RANDOM_HD void chacha_block(const StreamKey& key,
                                    uint64_t counter,
                                    uint32_t out[kBlockWords]) {
  static_assert(Rounds % 2 == 0, "ChaCha uses double rounds");
  uint32_t state[kBlockWords] = {0x61707865u,
                                 0x3320646eu,
                                 0x79622d32u,
                                 0x6b206574u,
                                 key.key[0],
                                 key.key[1],
                                 key.key[2],
                                 key.key[3],
                                 key.key[4],
                                 key.key[5],
                                 key.key[6],
                                 key.key[7],
                                 static_cast<uint32_t>(counter),
                                 static_cast<uint32_t>(counter >> 32),
                                 key.nonce[0],
                                 key.nonce[1]};
  uint32_t x[kBlockWords];
  for (int i = 0; i < kBlockWords; ++i) {
    x[i] = state[i];
  }
  for (int round = 0; round < Rounds / 2; ++round) {
    FHELIUM_CHACHA_QUARTER_ROUND(x, 0, 4, 8, 12)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 1, 5, 9, 13)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 2, 6, 10, 14)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 3, 7, 11, 15)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 0, 5, 10, 15)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 1, 6, 11, 12)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 2, 7, 8, 13)
    FHELIUM_CHACHA_QUARTER_ROUND(x, 3, 4, 9, 14)
  }
  for (int i = 0; i < kBlockWords; ++i) {
    out[i] = x[i] + state[i];
  }
}

FHELIUM_RANDOM_HD void chacha20_block(const StreamKey& key,
                                      uint64_t counter,
                                      uint32_t out[kBlockWords]) {
  chacha_block<20>(key, counter, out);
}

FHELIUM_RANDOM_HD uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __umul64hi(a, b);
#elif defined(__SIZEOF_INT128__)
  return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
#else
  return __umulh(a, b);
#endif
}

FHELIUM_RANDOM_HD int popcount32(uint32_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __popc(x);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(x);
#else
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return static_cast<int>((((x + (x >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
#endif
}

FHELIUM_RANDOM_HD uint64_t word64(const uint32_t* words) {
  return static_cast<uint64_t>(words[0]) |
         (static_cast<uint64_t>(words[1]) << 32);
}

// floor(u * q / 2^128) for the 128-bit u = words[0..3].
FHELIUM_RANDOM_HD uint64_t scale128(const uint32_t* words, uint64_t q) {
  const uint64_t low = word64(words);
  const uint64_t high = word64(words + 2);
  const uint64_t product_low = high * q;
  const uint64_t sum = product_low + mulhi64(low, q);
  return mulhi64(high, q) + (sum < product_low ? 1u : 0u);
}

// Uniform residue modulo q_j, where j is the limb of output element i.
template <typename scalar_t>
struct UniformResidueSampler {
  static constexpr int kWords = 4;
  const scalar_t* twice_modulus;
  int64_t twice_modulus_stride;
  int64_t limbs;
  int64_t coefficients;

  FHELIUM_RANDOM_HD int64_t operator()(const uint32_t* words, int64_t i) const {
    const int64_t limb = (i / coefficients) % limbs;
    const auto q =
        static_cast<uint64_t>(twice_modulus[limb * twice_modulus_stride]) >> 1;
    return static_cast<int64_t>(scale128(words, q));
  }
};

// Thresholds are 127-bit fractions stored as (high, low) int64 pairs.
struct GaussianSampler {
  static constexpr int kWords = 4;
  const int64_t* table;
  int64_t rows;

  FHELIUM_RANDOM_HD int64_t operator()(const uint32_t* words, int64_t) const {
    const uint64_t low = word64(words);
    const uint64_t high = word64(words + 2);
    const uint64_t value_high = high >> 1;
    const uint64_t value_low = (low >> 1) | (high << 63);
    int64_t magnitude = 0;
    for (int64_t row = 0; row < rows; ++row) {
      const auto threshold_high = static_cast<uint64_t>(table[2 * row]);
      const auto threshold_low = static_cast<uint64_t>(table[2 * row + 1]);
      magnitude += static_cast<int64_t>(
          (value_high > threshold_high) |
          ((value_high == threshold_high) & (value_low >= threshold_low)));
    }
    return (low & 1u) ? -magnitude : magnitude;
  }
};

struct CenteredBinomialSampler {
  static constexpr int kWords = 2;
  uint32_t mask;

  FHELIUM_RANDOM_HD int64_t operator()(const uint32_t* words, int64_t) const {
    return popcount32(words[0] & mask) - popcount32(words[1] & mask);
  }
};

struct TernarySampler {
  static constexpr int kWords = 2;
  uint64_t threshold;

  FHELIUM_RANDOM_HD int64_t operator()(const uint32_t* words, int64_t) const {
    const uint64_t value = word64(words);
    const int64_t nonzero = (value >> 1) < threshold ? 1 : 0;
    return (value & 1u) ? -nonzero : nonzero;
  }
};

// Rounds value i / stride with word i; indices that are not multiples of
// `stride` select no value and return zero.
struct StochasticRoundSampler {
  static constexpr int kWords = 1;
  const double* values;
  int64_t stride;

  FHELIUM_RANDOM_HD int64_t operator()(const uint32_t* words, int64_t i) const {
    if (i % stride != 0) {
      return 0;
    }
    const double value = values[i / stride];
    const double magnitude = value < 0.0 ? -value : value;
    const double base = static_cast<double>(static_cast<int64_t>(magnitude));
    const auto threshold =
        static_cast<uint32_t>((magnitude - base) * 4294967296.0);
    const int64_t rounded =
        static_cast<int64_t>(base) + (words[0] < threshold ? 1 : 0);
    return value < 0.0 ? -rounded : rounded;
  }
};

// Output writers store sample i into a range of output limbs. A centered or
// direct output has a single limb.
template <typename T>
struct DirectOutput {
  T* data;

  FHELIUM_RANDOM_HD int64_t limbs() const { return 1; }

  FHELIUM_RANDOM_HD void operator()(int64_t i,
                                    int64_t value,
                                    int64_t,
                                    int64_t) const {
    data[i] = static_cast<T>(value);
  }
};

// Stores sample i at index i / stride when i is a multiple of `stride`.
template <typename T>
struct StridedOutput {
  T* data;
  int64_t stride;

  FHELIUM_RANDOM_HD int64_t limbs() const { return 1; }

  FHELIUM_RANDOM_HD void operator()(int64_t i,
                                    int64_t value,
                                    int64_t,
                                    int64_t) const {
    if (i % stride == 0) {
      data[i / stride] = static_cast<T>(value);
    }
  }
};

// Writes centered sample i of a [batch, coefficient] grid as reduced
// residues into limbs [limb_begin, limb_end) of a [batch, limb, coefficient]
// output.
template <typename scalar_t>
struct ResidueOutput {
  scalar_t* data;
  const scalar_t* twice_modulus;
  int64_t twice_modulus_stride;
  int64_t limb_count;
  int64_t coefficients;

  FHELIUM_RANDOM_HD int64_t limbs() const { return limb_count; }

  FHELIUM_RANDOM_HD void operator()(int64_t i,
                                    int64_t value,
                                    int64_t limb_begin,
                                    int64_t limb_end) const {
    const int64_t batch = i / coefficients;
    scalar_t* column = data + batch * (limb_count - 1) * coefficients + i;
    for (int64_t limb = limb_begin; limb < limb_end; ++limb) {
      const int64_t q =
          static_cast<int64_t>(twice_modulus[limb * twice_modulus_stride]) >> 1;
      column[limb * coefficients] =
          static_cast<scalar_t>(value < 0 ? value + q : value);
    }
  }
};

// Adds centered sample i of a [batch, coefficient] grid to the residues of
// limbs [limb_begin, limb_end) of a [batch, limb, coefficient] Tensor. Inputs
// lie in [0, 2q_j); sums are reduced to [0, q_j).
template <typename scalar_t>
struct AddResidueOutput {
  scalar_t* data;
  const scalar_t* twice_modulus;
  int64_t twice_modulus_stride;
  int64_t limb_count;
  int64_t coefficients;

  FHELIUM_RANDOM_HD int64_t limbs() const { return limb_count; }

  FHELIUM_RANDOM_HD void operator()(int64_t i,
                                    int64_t value,
                                    int64_t limb_begin,
                                    int64_t limb_end) const {
    const int64_t batch = i / coefficients;
    scalar_t* column = data + batch * (limb_count - 1) * coefficients + i;
    for (int64_t limb = limb_begin; limb < limb_end; ++limb) {
      const auto q =
          static_cast<uint64_t>(twice_modulus[limb * twice_modulus_stride]) >>
          1;
      const uint64_t error = static_cast<uint64_t>(
          value < 0 ? value + static_cast<int64_t>(q) : value);
      uint64_t sum = static_cast<uint64_t>(column[limb * coefficients]) + error;
      sum = sum >= q ? sum - q : sum;
      sum = sum >= q ? sum - q : sum;
      column[limb * coefficients] = static_cast<scalar_t>(sum);
    }
  }
};

// Emits every limb of the samples whose words lie in stream block `block`.
template <typename Sampler, typename Output>
FHELIUM_RANDOM_HD void emit_block(const Sampler& sampler,
                                  const Output& output,
                                  const uint32_t* words,
                                  int64_t block,
                                  int64_t count) {
  constexpr int per_block = kBlockWords / Sampler::kWords;
  for (int t = 0; t < per_block; ++t) {
    const int64_t i = block * per_block + t;
    if (i >= count) {
      break;
    }
    output(i, sampler(words + t * Sampler::kWords, i), 0, output.limbs());
  }
}

template <typename Sampler>
constexpr int64_t blocks_for(int64_t count) {
  constexpr int per_block = kBlockWords / Sampler::kWords;
  return (count + per_block - 1) / per_block;
}

// Seeded key-switching keys. Digit d of a seeded key carries the seed row
// [k0, k1, k2, k3, key_id, d] (int64): the 256-bit key, the 64-bit key
// identifier and the stored digit index. Its uniform component a[d, r, c] at
// stored row r is sampled by segments. Segment s of a row with q_r < 2^32
// holds coefficients 8s .. 8s + 7; with q_r >= 2^32 it holds 4s .. 4s + 3.
// The segment consumes candidates in order from the ChaCha<Rounds> blocks
// with nonce key_id and block counter ((r | d << 16) << 32) | (s + t << 24)
// for attempts t = 0, 1, ...: the 16 words of a block (q_r < 2^32) or its 8
// little-endian word pairs (q_r >= 2^32). With W = 32 or 64, a candidate w is
// accepted when (w q_r) mod 2^W >= 2^W mod q_r and then yields
// floor(w q_r / 2^W) (Lemire's rejection method), which is uniform on
// [0, q_r). The coefficients of the segment are its first accepted values.
// The value is the stored residue itself (NTT domain, Montgomery
// representation); no conversion follows. Rows and digits are below 2^16 and
// segments below 2^24, so distinct (key_id, d, r, s, t) select distinct blocks.
constexpr int kSeededKeyRowWords = 6;
// Coefficients that one seeded-key sampling call produces; a multiple of
// both segment lengths.
constexpr int kSeededKeyCoefficientsPerCall = 8;
constexpr int64_t kSeededKeyIndexLimit = int64_t{1} << 16;
constexpr int64_t kSeededKeySegmentLimit = int64_t{1} << 24;
// A candidate is rejected with probability (2^W mod q) / 2^W < 1/2, so fewer
// than Count acceptances among the candidates of this many blocks has
// probability below 2^-1000; the bound keeps block counters distinct.
constexpr int kSeededKeyAttemptLimit = 256;

template <int Rounds>
FHELIUM_RANDOM_HD void seeded_key_block(const int64_t* seed_row,
                                        int64_t row,
                                        int64_t segment,
                                        int attempt,
                                        uint32_t out[kBlockWords]) {
  const StreamKey key = stream_key(seed_row);
  const uint64_t high =
      static_cast<uint64_t>(row) | (static_cast<uint64_t>(seed_row[5]) << 16);
  const uint64_t counter = (high << 32) | static_cast<uint64_t>(segment) |
                           (static_cast<uint64_t>(attempt) << 24);
  chacha_block<Rounds>(key, counter, out);
}

// Fills `out[0 .. Count)` with the segment's values for Candidates candidates
// of `Width` bits per block. The first block places candidate i only in
// slots [i - (Candidates - Count), i], which covers every segment that it
// completes; other segments restart with the general placement.
template <int Rounds, int Width, int Count, typename Out>
FHELIUM_RANDOM_HD void seeded_key_segment(const int64_t* seed_row,
                                          int64_t row,
                                          int64_t segment,
                                          uint64_t q,
                                          Out out[Count]) {
  constexpr int kCandidates = Width == 32 ? kBlockWords : kBlockWords / 2;
  constexpr int kSpare = kCandidates - Count;
  // 2^W mod q, evaluated in W-bit arithmetic.
  const uint64_t threshold =
      Width == 32 ? (0u - static_cast<uint32_t>(q)) % static_cast<uint32_t>(q)
                  : (uint64_t{0} - q) % q;
  const auto candidate = [&](const uint32_t* block, int i, bool& accepted) -> uint64_t {
    if constexpr (Width == 32) {
      const uint64_t product = static_cast<uint64_t>(block[i]) * q;
      accepted = static_cast<uint32_t>(product) >= threshold;
      return product >> 32;
    } else {
      const uint64_t w = word64(block + 2 * i);
      accepted = w * q >= threshold;
      return mulhi64(w, q);
    }
  };
  uint32_t block[kBlockWords];
  seeded_key_block<Rounds>(seed_row, row, segment, 0, block);
  int filled = 0;
#pragma unroll
  for (int i = 0; i < kCandidates; ++i) {
    bool accepted;
    const uint64_t value = candidate(block, i, accepted);
#pragma unroll
    for (int j = (i > kSpare ? i - kSpare : 0); j <= (i < Count - 1 ? i : Count - 1); ++j)
      if (accepted && filled == j) out[j] = static_cast<Out>(value);
    filled += accepted ? 1 : 0;
  }
  if (filled >= Count) return;
  filled = 0;
  for (int attempt = 0; attempt < kSeededKeyAttemptLimit && filled < Count; ++attempt) {
    seeded_key_block<Rounds>(seed_row, row, segment, attempt, block);
#pragma unroll
    for (int i = 0; i < kCandidates; ++i) {
      bool accepted;
      const uint64_t value = candidate(block, i, accepted);
#pragma unroll
      for (int j = 0; j < Count; ++j)
        if (accepted && filled == j) out[j] = static_cast<Out>(value);
      filled += accepted ? 1 : 0;
    }
  }
}

// Seeded-key residues of coefficients first .. first + 7 of stored row `row`
// with modulus q; `first` is a multiple of kSeededKeyCoefficientsPerCall.
// Out is uint32_t only when q < 2^32 holds for every row (int32 residues).
template <int Rounds, typename Out>
FHELIUM_RANDOM_HD void seeded_key_residues(const int64_t* seed_row,
                                           int64_t row,
                                           int64_t first,
                                           uint64_t q,
                                           Out out[kSeededKeyCoefficientsPerCall]) {
  if constexpr (sizeof(Out) == 4) {
    seeded_key_segment<Rounds, 32, 8>(seed_row, row, first / 8, q, out);
  } else if (q < (uint64_t{1} << 32)) {
    seeded_key_segment<Rounds, 32, 8>(seed_row, row, first / 8, q, out);
  } else {
    seeded_key_segment<Rounds, 64, 4>(seed_row, row, first / 4, q, out);
    seeded_key_segment<Rounds, 64, 4>(seed_row, row, first / 4 + 1, q, out + 4);
  }
}

// Calls `body(rounds_constant)` with the compile-time ChaCha round count of
// a seeded-key specification. Adding a supported round count adds one case.
template <typename Body>
inline void dispatch_seeded_key_rounds(int64_t rounds, Body&& body) {
  switch (rounds) {
    case 12:
      body(std::integral_constant<int, 12>{});
      return;
    default:
      TORCH_CHECK_VALUE(false, "Unsupported seeded-key ChaCha round count ", rounds);
  }
}

}  // namespace fhelium::random::detail
