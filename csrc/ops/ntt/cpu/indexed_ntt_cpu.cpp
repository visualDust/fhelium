#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>
#include <vector>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"
#include "../ntt_validation.h"

namespace {

enum class CpuForwardInput : int { kMontgomery, kStandard };
enum class CpuInverseOutput : int {
  kMontgomery,
  kStandardLazy,
  kStandard,
  kCentered,
};

using fhelium::cpu::adaptive_grain;
using fhelium::cpu::add_lazy;
using fhelium::cpu::canonicalize;
using fhelium::cpu::center;
using fhelium::cpu::load_constants;
using fhelium::cpu::MontgomeryConstants;
using fhelium::cpu::multiply;
using fhelium::cpu::reduce;
using fhelium::cpu::subtract_lazy;

template <typename scalar_t, CpuForwardInput input>
void forward_indexed_rows(torch::Tensor residues,
                          const torch::Tensor even_indices,
                          const torch::Tensor odd_indices,
                          const torch::Tensor forward_twiddles,
                          const torch::Tensor rns_params) {
  scalar_t* values = residues.data_ptr<scalar_t>();
  const int32_t* even = even_indices.data_ptr<int32_t>();
  const int32_t* odd = odd_indices.data_ptr<int32_t>();
  const scalar_t* twiddles = forward_twiddles.data_ptr<scalar_t>();
  const scalar_t* parameters = rns_params.data_ptr<scalar_t>();
  const int64_t batch_count = residues.size(0);
  const int64_t limb_count = residues.size(1);
  const int64_t coefficient_count = residues.size(2);
  const int64_t stage_count = even_indices.size(0);
  const int64_t butterfly_count = even_indices.size(1);
  const int64_t row_count = batch_count * limb_count;
  const int64_t parameter_row_stride = rns_params.stride(0);
  const int64_t parameter_limb_stride = rns_params.stride(1);
  const int64_t even_stage_stride = even_indices.stride(0);
  const int64_t even_butterfly_stride = even_indices.stride(1);
  const int64_t odd_stage_stride = odd_indices.stride(0);
  const int64_t odd_butterfly_stride = odd_indices.stride(1);
  const int64_t twiddle_limb_stride = forward_twiddles.stride(0);
  const int64_t twiddle_stage_stride = forward_twiddles.stride(1);
  const int64_t twiddle_butterfly_stride = forward_twiddles.stride(2);
  const int64_t value_stride0 = residues.stride(0);
  const int64_t value_stride1 = residues.stride(1);
  const int64_t value_stride2 = residues.stride(2);

  if constexpr (input == CpuForwardInput::kStandard) {
    const int64_t elements = row_count * coefficient_count;
    at::parallel_for(
        0,
        elements,
        adaptive_grain(elements),
        [&](const int64_t begin, const int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t row = index / coefficient_count;
            const int64_t limb = row % limb_count;
            const int64_t coefficient_begin = index - row * coefficient_count;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficient_count, end - row * coefficient_count);
            const auto constants = load_constants(
                parameters, parameter_row_stride, parameter_limb_stride, limb);
            const int64_t batch = row / limb_count;
            scalar_t* row_values =
                values + batch * value_stride0 + limb * value_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              row_values[coefficient * value_stride2] =
                  multiply(row_values[coefficient * value_stride2],
                           constants.r2,
                           constants);
            }
            index = row * coefficient_count + coefficient_end;
          }
        });
  }

  // One parallel_for per stage over the flat (row, butterfly) space. Stages
  // of one transform are dependent, so a barrier between stages is required;
  // the flat space keeps every worker busy even for a single-limb transform.
  // The repeated team-launch cost can regress latency for N below 4096; the
  // stage-flat policy targets conventional FHElium workloads with N >= 4096
  // while preserving numerical support for every accepted smaller power.
  const int64_t flat = row_count * butterfly_count;
  for (int64_t stage = 0; stage < stage_count; ++stage) {
    const int32_t* stage_even = even + stage * even_stage_stride;
    const int32_t* stage_odd = odd + stage * odd_stage_stride;
    at::parallel_for(
        0,
        flat,
        adaptive_grain(flat),
        [&](const int64_t begin, const int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t row = index / butterfly_count;
            const int64_t batch = row / limb_count;
            const int64_t limb = row - batch * limb_count;
            const int64_t butterfly_begin = index - row * butterfly_count;
            const int64_t butterfly_end =
                std::min<int64_t>(butterfly_count, end - row * butterfly_count);
            const auto constants = load_constants(
                parameters, parameter_row_stride, parameter_limb_stride, limb);
            scalar_t* row_values =
                values + batch * value_stride0 + limb * value_stride1;
            const scalar_t* stage_twiddles = twiddles +
                                             limb * twiddle_limb_stride +
                                             stage * twiddle_stage_stride;
            if (even_butterfly_stride == 1 && odd_butterfly_stride == 1 &&
                twiddle_butterfly_stride == 1) {
              for (int64_t butterfly = butterfly_begin;
                   butterfly < butterfly_end;
                   ++butterfly) {
                const int64_t even_index = stage_even[butterfly];
                const int64_t odd_index = stage_odd[butterfly];
                const scalar_t upper = row_values[even_index * value_stride2];
                const scalar_t lower = row_values[odd_index * value_stride2];
                const scalar_t product =
                    multiply(stage_twiddles[butterfly], lower, constants);
                row_values[even_index * value_stride2] =
                    add_lazy(upper, product, constants.twice_modulus);
                row_values[odd_index * value_stride2] =
                    subtract_lazy(upper, product, constants.twice_modulus);
              }
            } else {
              const int32_t* even_cursor =
                  stage_even + butterfly_begin * even_butterfly_stride;
              const int32_t* odd_cursor =
                  stage_odd + butterfly_begin * odd_butterfly_stride;
              const scalar_t* twiddle_cursor =
                  stage_twiddles + butterfly_begin * twiddle_butterfly_stride;
              for (int64_t butterfly = butterfly_begin;
                   butterfly < butterfly_end;
                   ++butterfly) {
                const int64_t even_index = *even_cursor;
                const int64_t odd_index = *odd_cursor;
                const scalar_t upper = row_values[even_index * value_stride2];
                const scalar_t lower = row_values[odd_index * value_stride2];
                const scalar_t product =
                    multiply(*twiddle_cursor, lower, constants);
                row_values[even_index * value_stride2] =
                    add_lazy(upper, product, constants.twice_modulus);
                row_values[odd_index * value_stride2] =
                    subtract_lazy(upper, product, constants.twice_modulus);
                even_cursor += even_butterfly_stride;
                odd_cursor += odd_butterfly_stride;
                twiddle_cursor += twiddle_butterfly_stride;
              }
            }
            index = row * butterfly_count + butterfly_end;
          }
        });
  }
}

template <typename scalar_t, CpuInverseOutput output>
void inverse_indexed_rows(torch::Tensor residues,
                          const torch::Tensor even_indices,
                          const torch::Tensor odd_indices,
                          const torch::Tensor inverse_twiddles,
                          const torch::Tensor rns_params) {
  scalar_t* values = residues.data_ptr<scalar_t>();
  const int32_t* even = even_indices.data_ptr<int32_t>();
  const int32_t* odd = odd_indices.data_ptr<int32_t>();
  const scalar_t* twiddles = inverse_twiddles.data_ptr<scalar_t>();
  const scalar_t* parameters = rns_params.data_ptr<scalar_t>();
  const int64_t batch_count = residues.size(0);
  const int64_t limb_count = residues.size(1);
  const int64_t coefficient_count = residues.size(2);
  const int64_t stage_count = even_indices.size(0);
  const int64_t butterfly_count = even_indices.size(1);
  const int64_t row_count = batch_count * limb_count;
  const int64_t parameter_row_stride = rns_params.stride(0);
  const int64_t parameter_limb_stride = rns_params.stride(1);
  const int64_t even_stage_stride = even_indices.stride(0);
  const int64_t even_butterfly_stride = even_indices.stride(1);
  const int64_t odd_stage_stride = odd_indices.stride(0);
  const int64_t odd_butterfly_stride = odd_indices.stride(1);
  const int64_t twiddle_limb_stride = inverse_twiddles.stride(0);
  const int64_t twiddle_stage_stride = inverse_twiddles.stride(1);
  const int64_t twiddle_butterfly_stride = inverse_twiddles.stride(2);
  const int64_t value_stride0 = residues.stride(0);
  const int64_t value_stride1 = residues.stride(1);
  const int64_t value_stride2 = residues.stride(2);

  const int64_t flat = row_count * butterfly_count;
  for (int64_t stage = 0; stage < stage_count; ++stage) {
    const int32_t* stage_even = even + stage * even_stage_stride;
    const int32_t* stage_odd = odd + stage * odd_stage_stride;
    at::parallel_for(
        0,
        flat,
        adaptive_grain(flat),
        [&](const int64_t begin, const int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t row = index / butterfly_count;
            const int64_t batch = row / limb_count;
            const int64_t limb = row - batch * limb_count;
            const int64_t butterfly_begin = index - row * butterfly_count;
            const int64_t butterfly_end =
                std::min<int64_t>(butterfly_count, end - row * butterfly_count);
            const auto constants = load_constants(
                parameters, parameter_row_stride, parameter_limb_stride, limb);
            scalar_t* row_values =
                values + batch * value_stride0 + limb * value_stride1;
            const scalar_t* stage_twiddles = twiddles +
                                             limb * twiddle_limb_stride +
                                             stage * twiddle_stage_stride;
            if (even_butterfly_stride == 1 && odd_butterfly_stride == 1 &&
                twiddle_butterfly_stride == 1) {
              for (int64_t butterfly = butterfly_begin;
                   butterfly < butterfly_end;
                   ++butterfly) {
                const int64_t even_index = stage_even[butterfly];
                const int64_t odd_index = stage_odd[butterfly];
                const scalar_t upper = row_values[even_index * value_stride2];
                const scalar_t lower = row_values[odd_index * value_stride2];
                const scalar_t difference =
                    subtract_lazy(upper, lower, constants.twice_modulus);
                row_values[even_index * value_stride2] =
                    add_lazy(upper, lower, constants.twice_modulus);
                row_values[odd_index * value_stride2] =
                    multiply(stage_twiddles[butterfly], difference, constants);
              }
            } else {
              const int32_t* even_cursor =
                  stage_even + butterfly_begin * even_butterfly_stride;
              const int32_t* odd_cursor =
                  stage_odd + butterfly_begin * odd_butterfly_stride;
              const scalar_t* twiddle_cursor =
                  stage_twiddles + butterfly_begin * twiddle_butterfly_stride;
              for (int64_t butterfly = butterfly_begin;
                   butterfly < butterfly_end;
                   ++butterfly) {
                const int64_t even_index = *even_cursor;
                const int64_t odd_index = *odd_cursor;
                const scalar_t upper = row_values[even_index * value_stride2];
                const scalar_t lower = row_values[odd_index * value_stride2];
                const scalar_t difference =
                    subtract_lazy(upper, lower, constants.twice_modulus);
                row_values[even_index * value_stride2] =
                    add_lazy(upper, lower, constants.twice_modulus);
                row_values[odd_index * value_stride2] =
                    multiply(*twiddle_cursor, difference, constants);
                even_cursor += even_butterfly_stride;
                odd_cursor += odd_butterfly_stride;
                twiddle_cursor += twiddle_butterfly_stride;
              }
            }
            index = row * butterfly_count + butterfly_end;
          }
        });
  }

  // Final per-coefficient normalization by N^{-1} with the requested
  // output representation; flat over rows so one-limb calls stay parallel.
  const int64_t elements = row_count * coefficient_count;
  at::parallel_for(
      0,
      elements,
      adaptive_grain(elements),
      [&](const int64_t begin, const int64_t end) {
        int64_t index = begin;
        while (index < end) {
          const int64_t row = index / coefficient_count;
          const int64_t limb = row % limb_count;
          const int64_t coefficient_begin = index - row * coefficient_count;
          const int64_t coefficient_end = std::min<int64_t>(
              coefficient_count, end - row * coefficient_count);
          const auto constants = load_constants(
              parameters, parameter_row_stride, parameter_limb_stride, limb);
          const int64_t batch = row / limb_count;
          scalar_t* row_values =
              values + batch * value_stride0 + limb * value_stride1;
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            scalar_t value = multiply(row_values[coefficient * value_stride2],
                                      constants.n_inverse_montgomery,
                                      constants);
            if constexpr (output != CpuInverseOutput::kMontgomery) {
              value = reduce(value, constants);
              if constexpr (output == CpuInverseOutput::kStandard) {
                value = canonicalize(value, constants.twice_modulus);
              } else if constexpr (output == CpuInverseOutput::kCentered) {
                value = center(canonicalize(value, constants.twice_modulus),
                               constants.twice_modulus);
              }
            }
            row_values[coefficient * value_stride2] = value;
          }
          index = row * coefficient_count + coefficient_end;
        }
      });
}

void validate_cpu_indexed(const torch::Tensor& residues,
                          const torch::Tensor& even_indices,
                          const torch::Tensor& odd_indices,
                          const torch::Tensor& twiddles,
                          const torch::Tensor& rns_params) {
  TORCH_CHECK(residues.device().is_cpu(),
              "indexed NTT CPU implementation requires CPU residues");
  fhelium::ntt::validate_indexed_tables(
      residues, even_indices, odd_indices, twiddles, rns_params);

  const int64_t coefficient_count = residues.size(-1);
  const int64_t expected_stage_count = even_indices.size(0);
  const auto even = even_indices.accessor<int32_t, 2>();
  const auto odd = odd_indices.accessor<int32_t, 2>();
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

template <CpuForwardInput input>
void forward_indexed_cpu(torch::Tensor residues,
                         const torch::Tensor even_indices,
                         const torch::Tensor odd_indices,
                         const torch::Tensor forward_twiddles,
                         const torch::Tensor rns_params) {
  validate_cpu_indexed(
      residues, even_indices, odd_indices, forward_twiddles, rns_params);
  auto batch_rows = view_rns_batch_3d(residues, "residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "forward_ntt_indexed_cpu", [&] {
        forward_indexed_rows<scalar_t, input>(batch_rows,
                                              even_indices,
                                              odd_indices,
                                              forward_twiddles,
                                              rns_params);
      });
}

template <CpuInverseOutput output>
void inverse_indexed_cpu(torch::Tensor residues,
                         const torch::Tensor even_indices,
                         const torch::Tensor odd_indices,
                         const torch::Tensor inverse_twiddles,
                         const torch::Tensor rns_params) {
  validate_cpu_indexed(
      residues, even_indices, odd_indices, inverse_twiddles, rns_params);
  auto batch_rows = view_rns_batch_3d(residues, "residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "inverse_ntt_indexed_cpu", [&] {
        inverse_indexed_rows<scalar_t, output>(batch_rows,
                                               even_indices,
                                               odd_indices,
                                               inverse_twiddles,
                                               rns_params);
      });
}

void forward_ntt_montgomery_indexed_cpu_(torch::Tensor residues,
                                         const torch::Tensor even_indices,
                                         const torch::Tensor odd_indices,
                                         const torch::Tensor forward_twiddles,
                                         const torch::Tensor rns_params) {
  forward_indexed_cpu<CpuForwardInput::kMontgomery>(
      residues, even_indices, odd_indices, forward_twiddles, rns_params);
}

void forward_ntt_to_montgomery_indexed_cpu_(
    torch::Tensor residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  forward_indexed_cpu<CpuForwardInput::kStandard>(
      residues, even_indices, odd_indices, forward_twiddles, rns_params);
}

torch::Tensor forward_ntt_to_montgomery_indexed_cpu(
    const torch::Tensor residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  auto out = residues.clone();
  forward_ntt_to_montgomery_indexed_cpu_(
      out, even_indices, odd_indices, forward_twiddles, rns_params);
  return out;
}

void inverse_ntt_montgomery_indexed_cpu_(torch::Tensor residues,
                                         const torch::Tensor even_indices,
                                         const torch::Tensor odd_indices,
                                         const torch::Tensor inverse_twiddles,
                                         const torch::Tensor rns_params) {
  inverse_indexed_cpu<CpuInverseOutput::kMontgomery>(
      residues, even_indices, odd_indices, inverse_twiddles, rns_params);
}

void inverse_ntt_to_standard_lazy_indexed_cpu_(
    torch::Tensor residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params) {
  inverse_indexed_cpu<CpuInverseOutput::kStandardLazy>(
      residues, even_indices, odd_indices, inverse_twiddles, rns_params);
}

void inverse_ntt_to_standard_indexed_cpu_(torch::Tensor residues,
                                          const torch::Tensor even_indices,
                                          const torch::Tensor odd_indices,
                                          const torch::Tensor inverse_twiddles,
                                          const torch::Tensor rns_params) {
  inverse_indexed_cpu<CpuInverseOutput::kStandard>(
      residues, even_indices, odd_indices, inverse_twiddles, rns_params);
}

void inverse_ntt_to_centered_indexed_cpu_(torch::Tensor residues,
                                          const torch::Tensor even_indices,
                                          const torch::Tensor odd_indices,
                                          const torch::Tensor inverse_twiddles,
                                          const torch::Tensor rns_params) {
  inverse_indexed_cpu<CpuInverseOutput::kCentered>(
      residues, even_indices, odd_indices, inverse_twiddles, rns_params);
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CPU, m) {
  m.impl("forward_ntt_montgomery_indexed_",
         &forward_ntt_montgomery_indexed_cpu_);
  m.impl("forward_ntt_to_montgomery_indexed_",
         &forward_ntt_to_montgomery_indexed_cpu_);
  m.impl("forward_ntt_to_montgomery_indexed",
         &forward_ntt_to_montgomery_indexed_cpu);
  m.impl("inverse_ntt_montgomery_indexed_",
         &inverse_ntt_montgomery_indexed_cpu_);
  m.impl("inverse_ntt_to_standard_lazy_indexed_",
         &inverse_ntt_to_standard_lazy_indexed_cpu_);
  m.impl("inverse_ntt_to_standard_indexed_",
         &inverse_ntt_to_standard_indexed_cpu_);
  m.impl("inverse_ntt_to_centered_indexed_",
         &inverse_ntt_to_centered_indexed_cpu_);
}
