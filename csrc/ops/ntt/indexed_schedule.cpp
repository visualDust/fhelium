#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

// Each stage partitions the N coefficients into disjoint even/odd pairs.
// Accelerator schedules are read through temporary CPU transfers during
// preparation; validation neither changes nor retains the original tensors.
void validate_indexed_schedule(const torch::Tensor even_indices,
                               const torch::Tensor odd_indices,
                               const int64_t ring_dimension) {
  TORCH_CHECK(ring_dimension >= 2 &&
                  (ring_dimension & (ring_dimension - 1)) == 0,
              "Indexed NTT ring dimension must be a power of two >= 2");
  TORCH_CHECK(even_indices.device() == odd_indices.device(),
              "Indexed NTT schedules must share a device");
  TORCH_CHECK(even_indices.scalar_type() == at::kInt &&
                  odd_indices.scalar_type() == at::kInt,
              "Indexed NTT schedules must use torch.int32");
  TORCH_CHECK(even_indices.dim() == 2 && odd_indices.dim() == 2 &&
                  even_indices.sizes() == odd_indices.sizes(),
              "Indexed NTT schedules must have matching [stage, butterfly] shapes");
  int64_t expected_stage_count = 0;
  for (int64_t n = ring_dimension; n > 1; n >>= 1) {
    ++expected_stage_count;
  }
  TORCH_CHECK(even_indices.size(0) == expected_stage_count &&
                  even_indices.size(1) == ring_dimension / 2,
              "Indexed NTT schedules must have shape [log2(N), N/2]");

  const auto even_cpu = even_indices.device().is_cpu()
      ? even_indices : even_indices.cpu();
  const auto odd_cpu = odd_indices.device().is_cpu()
      ? odd_indices : odd_indices.cpu();
  const int64_t coefficient_count = ring_dimension;
  const auto even = even_cpu.accessor<int32_t, 2>();
  const auto odd = odd_cpu.accessor<int32_t, 2>();
  std::vector<uint8_t> seen(static_cast<size_t>(coefficient_count));
  for (int64_t stage = 0; stage < expected_stage_count; ++stage) {
    std::fill(seen.begin(), seen.end(), uint8_t{0});
    for (int64_t butterfly = 0; butterfly < even_indices.size(1); ++butterfly) {
      const int64_t even_index = even[stage][butterfly];
      const int64_t odd_index = odd[stage][butterfly];
      TORCH_CHECK(even_index >= 0 && even_index < coefficient_count &&
                      odd_index >= 0 && odd_index < coefficient_count,
                  "Indexed NTT schedule contains an out-of-range coefficient "
                  "index at stage ",
                  stage,
                  ", butterfly ",
                  butterfly);
      TORCH_CHECK(!seen[static_cast<size_t>(even_index)] &&
                      !seen[static_cast<size_t>(odd_index)] &&
                      even_index != odd_index,
                  "Indexed NTT schedule must partition every coefficient "
                  "exactly once per stage; duplicate at stage ",
                  stage,
                  ", butterfly ",
                  butterfly);
      seen[static_cast<size_t>(even_index)] = uint8_t{1};
      seen[static_cast<size_t>(odd_index)] = uint8_t{1};
    }
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def("validate_indexed_schedule(Tensor even_indices, Tensor odd_indices, "
        "int ring_dimension) -> ()");
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeExplicitAutograd, m) {
  m.impl("validate_indexed_schedule", &validate_indexed_schedule);
}
