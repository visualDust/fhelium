#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#include "../backends.h"
#include "../random_core.h"

namespace fhelium::random::cpu {

namespace {

using ::fhelium::random::detail::kBlockWords;
using ::fhelium::random::detail::rotl32;
using ::fhelium::random::detail::StreamKey;

// Blocks evaluated together; the lane loops compile to SIMD instructions.
constexpr int kLanes = 8;

#define FHELIUM_CHACHA_LANE_QUARTER_ROUND(a, b, c, d) \
  for (int l = 0; l < kLanes; ++l) {                  \
    x[a][l] += x[b][l];                               \
    x[d][l] = rotl32(x[d][l] ^ x[a][l], 16);          \
    x[c][l] += x[d][l];                               \
    x[b][l] = rotl32(x[b][l] ^ x[c][l], 12);          \
    x[a][l] += x[b][l];                               \
    x[d][l] = rotl32(x[d][l] ^ x[a][l], 8);           \
    x[c][l] += x[d][l];                               \
    x[b][l] = rotl32(x[b][l] ^ x[c][l], 7);           \
  }

// Writes blocks counter, ..., counter + kLanes - 1 to out[lane][word].
void chacha20_lanes(const StreamKey& key,
                    uint64_t counter,
                    uint32_t out[kLanes][kBlockWords]) {
  uint32_t state[kBlockWords][kLanes];
  for (int l = 0; l < kLanes; ++l) {
    const uint64_t block = counter + static_cast<uint64_t>(l);
    state[0][l] = 0x61707865u;
    state[1][l] = 0x3320646eu;
    state[2][l] = 0x79622d32u;
    state[3][l] = 0x6b206574u;
    for (int i = 0; i < 8; ++i) {
      state[4 + i][l] = key.key[i];
    }
    state[12][l] = static_cast<uint32_t>(block);
    state[13][l] = static_cast<uint32_t>(block >> 32);
    state[14][l] = key.nonce[0];
    state[15][l] = key.nonce[1];
  }
  uint32_t x[kBlockWords][kLanes];
  std::memcpy(x, state, sizeof(x));
  for (int round = 0; round < 10; ++round) {
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(0, 4, 8, 12)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(1, 5, 9, 13)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(2, 6, 10, 14)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(3, 7, 11, 15)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(0, 5, 10, 15)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(1, 6, 11, 12)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(2, 7, 8, 13)
    FHELIUM_CHACHA_LANE_QUARTER_ROUND(3, 4, 9, 14)
  }
  for (int l = 0; l < kLanes; ++l) {
    for (int i = 0; i < kBlockWords; ++i) {
      out[l][i] = x[i][l] + state[i][l];
    }
  }
}

#undef FHELIUM_CHACHA_LANE_QUARTER_ROUND

StreamKey cpu_stream_key(const at::Tensor& seed) {
  TORCH_CHECK(seed.is_cpu(), "CPU sampling requires a CPU seed");
  return ::fhelium::random::detail::stream_key(seed.const_data_ptr<int64_t>());
}

// Calls body(first_block, words[kLanes][16]) for every lane group covering
// blocks [first, first + block_count).
template <typename Body>
void for_each_lane_group(const StreamKey& key,
                         uint64_t first,
                         int64_t block_count,
                         const Body& body) {
  const int64_t groups = (block_count + kLanes - 1) / kLanes;
  at::parallel_for(0, groups, 64, [&](int64_t begin, int64_t end) {
    alignas(64) uint32_t words[kLanes][kBlockWords];
    for (int64_t group = begin; group < end; ++group) {
      const int64_t block = group * kLanes;
      chacha20_lanes(key, first + static_cast<uint64_t>(block), words);
      body(block, words);
    }
  });
}

}  // namespace

template <typename Sampler, typename Output>
void run(const at::Tensor& seed,
         const Sampler& sampler,
         const Output& output,
         int64_t count) {
  const int64_t blocks = ::fhelium::random::detail::blocks_for<Sampler>(count);
  for_each_lane_group(
      cpu_stream_key(seed),
      0,
      blocks,
      [&](int64_t block, const uint32_t (&words)[kLanes][kBlockWords]) {
        const int lanes =
            static_cast<int>(std::min<int64_t>(kLanes, blocks - block));
        for (int l = 0; l < lanes; ++l) {
          ::fhelium::random::detail::emit_block(
              sampler, output, words[l], block + l, count);
        }
      });
}

void keystream_into(const at::Tensor& seed,
                    uint64_t first_block,
                    int64_t block_count,
                    uint32_t* out) {
  for_each_lane_group(
      cpu_stream_key(seed),
      first_block,
      block_count,
      [&](int64_t block, const uint32_t (&words)[kLanes][kBlockWords]) {
        const int64_t lanes = std::min<int64_t>(kLanes, block_count - block);
        std::memcpy(out + block * kBlockWords,
                    words,
                    static_cast<size_t>(lanes) * sizeof(words[0]));
      });
}

#include "../random_ops.inl"

}  // namespace fhelium::random::cpu
