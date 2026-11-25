#include <torch/torch.h>

#include "../backends.h"
#include "../ntt_validation.h"
#include "forward_ntt_cuda.h"

namespace fhelium::ntt::gpu {

// CUDA indexed execution uses one launch per stage and provides the validation
// baseline for compact CUDA policies. CPU indexed execution is the CPU
// production backend.
void forward_ntt_montgomery_indexed_(at::Tensor& montgomery_residues,
                                     const at::Tensor& even_indices,
                                     const at::Tensor& odd_indices,
                                     const at::Tensor& forward_twiddles,
                                     const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_montgomery_indexed_inplace_cuda(montgomery_residues,
                                              even_indices,
                                              odd_indices,
                                              forward_twiddles,
                                              rns_params);
}

void forward_ntt_to_montgomery_indexed_(at::Tensor& standard_residues,
                                        const at::Tensor& even_indices,
                                        const at::Tensor& odd_indices,
                                        const at::Tensor& forward_twiddles,
                                        const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_to_montgomery_indexed_inplace_cuda(standard_residues,
                                                 even_indices,
                                                 odd_indices,
                                                 forward_twiddles,
                                                 rns_params);
}

at::Tensor forward_ntt_to_montgomery_indexed(
    const at::Tensor& standard_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  return forward_ntt_to_montgomery_indexed_cuda(standard_residues,
                                                even_indices,
                                                odd_indices,
                                                forward_twiddles,
                                                rns_params);
}

void forward_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_to_montgomery_power_of_two_radix_compact_(
    at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_cuda(
      standard_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

at::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  return forward_ntt_to_montgomery_power_of_two_radix_compact_cuda(
      standard_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t shared_memory_log_n) {
  fhelium::ntt::validate_shared_memory_log_n(shared_memory_log_n);
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      shared_memory_log_n);
}

}  // namespace fhelium::ntt::gpu

namespace fhelium::ntt::diagnostic::gpu {

void forward_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n) {
  ::fhelium::ntt::gpu::
      forward_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
          montgomery_residues,
          outer_twiddles,
          radix_root_powers,
          rns_params,
          shared_memory_log_n);
}

}  // namespace fhelium::ntt::diagnostic::gpu
