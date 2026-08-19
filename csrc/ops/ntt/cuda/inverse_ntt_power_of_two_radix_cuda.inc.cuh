#pragma once

// Power-of-two radix INTT entry points share one transform body and select only
// the explicit representation epilogue at compile time. Shared-memory coverage
// is native-owned and is overridable only through the separate diagnostic
// operator namespace.

enum class PowerOfTwoRadixInverseOutput {
  Montgomery,
  StandardLazy,
  Standard,
  Centered,
};

template <typename scalar_t, PowerOfTwoRadixInverseOutput OUTPUT>
void launch_inverse_ntt_power_of_two_radix_compact_cuda(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  const auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  launch_inverse_ntt_power_of_two_radix_cuda<scalar_t>(a,
                                                       outer_twiddles,
                                                       radix_root_powers,
                                                       rns_params,
                                                       radix,
                                                       shared_memory_log_n,
                                                       stream);

  const int rows = static_cast<int>(a.size(1));
  const int N = static_cast<int>(a.size(2));
  const dim3 grid(rows, N / kCudaBlockSize, a.size(0));
  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  if constexpr (OUTPUT == PowerOfTwoRadixInverseOutput::Montgomery) {
    inverse_ntt_normalize_montgomery_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
  } else if constexpr (OUTPUT == PowerOfTwoRadixInverseOutput::StandardLazy) {
    inverse_ntt_normalize_to_standard_lazy_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
  } else if constexpr (OUTPUT == PowerOfTwoRadixInverseOutput::Standard) {
    inverse_ntt_normalize_to_standard_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
  } else {
    inverse_ntt_normalize_to_centered_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
  }
}

#define DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY(ENTRY_NAME, OUTPUT_KIND)   \
  void ENTRY_NAME(torch::Tensor a,                                         \
                  const torch::Tensor outer_twiddles,                      \
                  const torch::Tensor radix_root_powers,                   \
                  const torch::Tensor rns_params,                          \
                  const int64_t radix,                                     \
                  const int64_t shared_memory_log_n) {                     \
    auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");         \
    AT_DISPATCH_INTEGRAL_TYPES(                                            \
        a.scalar_type(), #ENTRY_NAME, ([&] {                               \
          launch_inverse_ntt_power_of_two_radix_compact_cuda<scalar_t,     \
                                                             OUTPUT_KIND>( \
              batch_rows,                                                  \
              outer_twiddles,                                              \
              radix_root_powers,                                           \
              rns_params,                                                  \
              static_cast<int>(radix),                                     \
              static_cast<int>(shared_memory_log_n));                      \
        }));                                                               \
  }

DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY(
    inverse_ntt_montgomery_power_of_two_radix_compact_inplace_cuda,
    PowerOfTwoRadixInverseOutput::Montgomery)
DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY(
    inverse_ntt_to_standard_lazy_power_of_two_radix_compact_inplace_cuda,
    PowerOfTwoRadixInverseOutput::StandardLazy)
DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY(
    inverse_ntt_to_standard_power_of_two_radix_compact_inplace_cuda,
    PowerOfTwoRadixInverseOutput::Standard)
DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY(
    inverse_ntt_to_centered_power_of_two_radix_compact_inplace_cuda,
    PowerOfTwoRadixInverseOutput::Centered)

#undef DEFINE_POWER_OF_TWO_RADIX_INVERSE_ENTRY
