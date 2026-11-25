// GPU samplers shared by the CUDA and HIP backends. Included inside
// namespace fhelium::random::gpu after the backend defines
// FHELIUM_RANDOM_GPU_STREAM(device_index), which yields the current stream,
// FHELIUM_RANDOM_GPU_SM_COUNT(device_index), which yields the multiprocessor
// count, and FHELIUM_RANDOM_GPU_CHECK_LAUNCH(), which reports launch errors.

namespace detail_gpu {

using ::fhelium::random::detail::kBlockWords;
using ::fhelium::random::detail::StreamKey;

constexpr int kThreads = 128;

// Each thread evaluates one stream block and stages its samples in shared
// memory; the block then stores the tile for the limbs selected by
// blockIdx.y, so consecutive threads write consecutive coefficients.
template <typename Sampler, typename Output>
__global__ void __launch_bounds__(kThreads)
    sample_kernel(const int64_t* seed,
                  Sampler sampler,
                  Output output,
                  int64_t count,
                  int64_t limbs_per_group) {
  constexpr int per_block = kBlockWords / Sampler::kWords;
  constexpr int tile = kThreads * per_block;
  __shared__ int64_t values[tile];
  const int64_t tile_begin = static_cast<int64_t>(blockIdx.x) * tile;
  const int local = threadIdx.x * per_block;
  if (tile_begin + local < count) {
    const StreamKey key = ::fhelium::random::detail::stream_key(seed);
    uint32_t words[kBlockWords];
    ::fhelium::random::detail::chacha20_block(
        key, static_cast<uint64_t>(blockIdx.x) * kThreads + threadIdx.x, words);
    for (int t = 0; t < per_block; ++t) {
      values[local + t] =
          sampler(words + t * Sampler::kWords, tile_begin + local + t);
    }
  }
  __syncthreads();
  const int64_t limb_begin = blockIdx.y * limbs_per_group;
  const int64_t limb_end = limb_begin + limbs_per_group < output.limbs()
                               ? limb_begin + limbs_per_group
                               : output.limbs();
  const int64_t tile_count =
      count - tile_begin < tile ? count - tile_begin : tile;
  for (int64_t index = threadIdx.x; index < tile_count; index += kThreads) {
    output(tile_begin + index, values[index], limb_begin, limb_end);
  }
}

__global__ void keystream_kernel(const int64_t* seed,
                                 uint64_t first_block,
                                 int64_t blocks,
                                 uint32_t* out) {
  const int64_t block =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= blocks) {
    return;
  }
  const StreamKey key = ::fhelium::random::detail::stream_key(seed);
  uint32_t words[kBlockWords];
  ::fhelium::random::detail::chacha20_block(
      key, first_block + static_cast<uint64_t>(block), words);
  auto* row = reinterpret_cast<uint4*>(out + block * kBlockWords);
  for (int i = 0; i < kBlockWords / 4; ++i) {
    row[i] = make_uint4(
        words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
  }
}

// One thread per (digit, row, eight-coefficient call) of a seeded key.
template <int Rounds, typename scalar_t>
__global__ void __launch_bounds__(kThreads)
    seeded_key_kernel(const int64_t* seed_rows,
                      const scalar_t* moduli,
                      int64_t stride,
                      int64_t rows,
                      int64_t calls,
                      scalar_t* out) {
  namespace core = ::fhelium::random::detail;
  constexpr int kCall = core::kSeededKeyCoefficientsPerCall;
  const int64_t call = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (call >= calls) {
    return;
  }
  const int64_t row = blockIdx.y;
  const int64_t digit = blockIdx.z;
  const auto q = static_cast<uint64_t>(moduli[row * stride]);
  std::make_unsigned_t<scalar_t> values[kCall];
  core::seeded_key_residues<Rounds>(
      seed_rows + digit * core::kSeededKeyRowWords, row, call * kCall, q, values);
  scalar_t* target = out + ((digit * rows + row) * calls + call) * kCall;
  for (int j = 0; j < kCall; ++j) {
    target[j] = static_cast<scalar_t>(values[j]);
  }
}

inline unsigned grid_for(int64_t blocks) {
  return static_cast<unsigned>((blocks + kThreads - 1) / kThreads);
}

}  // namespace detail_gpu

template <typename Sampler, typename Output>
void run(const at::Tensor& seed,
         const Sampler& sampler,
         const Output& output,
         int64_t count) {
  TORCH_CHECK(seed.is_cuda(), "GPU sampling requires a GPU seed");
  if (count == 0) {
    return;
  }
  constexpr int64_t tile =
      detail_gpu::kThreads * (detail_gpu::kBlockWords / Sampler::kWords);
  const int64_t tiles = (count + tile - 1) / tile;
  // Split limbs across blockIdx.y until the grid has about four blocks per
  // multiprocessor; each split recomputes the tile's stream blocks.
  const int device = seed.device().index();
  const int64_t target = 4 * FHELIUM_RANDOM_GPU_SM_COUNT(device);
  const int64_t limbs = output.limbs();
  const int64_t groups = std::min<int64_t>(
      limbs, std::max<int64_t>(1, (target + tiles - 1) / tiles));
  const int64_t limbs_per_group = (limbs + groups - 1) / groups;
  const dim3 grid(
      static_cast<unsigned>(tiles),
      static_cast<unsigned>((limbs + limbs_per_group - 1) / limbs_per_group));
  const c10::DeviceGuard guard(seed.device());
  detail_gpu::sample_kernel<<<grid,
                              detail_gpu::kThreads,
                              0,
                              FHELIUM_RANDOM_GPU_STREAM(device)>>>(
      seed.const_data_ptr<int64_t>(), sampler, output, count, limbs_per_group);
  FHELIUM_RANDOM_GPU_CHECK_LAUNCH();
}

void keystream_into(const at::Tensor& seed,
                    uint64_t first_block,
                    int64_t block_count,
                    uint32_t* out) {
  TORCH_CHECK(seed.is_cuda(), "GPU sampling requires a GPU seed");
  if (block_count == 0) {
    return;
  }
  const c10::DeviceGuard guard(seed.device());
  detail_gpu::
      keystream_kernel<<<detail_gpu::grid_for(block_count),
                         detail_gpu::kThreads,
                         0,
                         FHELIUM_RANDOM_GPU_STREAM(seed.device().index())>>>(
          seed.const_data_ptr<int64_t>(), first_block, block_count, out);
  FHELIUM_RANDOM_GPU_CHECK_LAUNCH();
}

template <int Rounds, typename scalar_t>
void seeded_key_into(const int64_t* seed_rows,
                     int64_t digits,
                     const scalar_t* moduli,
                     int64_t stride,
                     int64_t rows,
                     int64_t ring_dimension,
                     scalar_t* out,
                     const at::Tensor& like) {
  TORCH_CHECK(like.is_cuda(), "GPU seeded-key expansion requires GPU seed rows");
  if (digits == 0 || rows == 0) {
    return;
  }
  const int64_t calls =
      ring_dimension / ::fhelium::random::detail::kSeededKeyCoefficientsPerCall;
  const c10::DeviceGuard guard(like.device());
  const dim3 grid(detail_gpu::grid_for(calls),
                  static_cast<unsigned>(rows),
                  static_cast<unsigned>(digits));
  detail_gpu::seeded_key_kernel<Rounds, scalar_t>
      <<<grid, detail_gpu::kThreads, 0,
         FHELIUM_RANDOM_GPU_STREAM(like.device().index())>>>(
          seed_rows, moduli, stride, rows, calls, out);
  FHELIUM_RANDOM_GPU_CHECK_LAUNCH();
}

#include "random_ops.inl"
