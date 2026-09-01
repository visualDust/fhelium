#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "tensor_validation.h"

namespace {

using fhelium::ckks::cpu::check_cpu_peer;

template <typename scalar_t, bool nearest>
void rescale_loop(torch::Tensor out,
                  const torch::Tensor remaining,
                  const torch::Tensor inverse,
                  const torch::Tensor dropped,
                  const torch::Tensor params,
                  int64_t half_drop_prime) {
  scalar_t* output = out.data_ptr<scalar_t>();
  const scalar_t* input = remaining.data_ptr<scalar_t>();
  const scalar_t* inverse_values = inverse.data_ptr<scalar_t>();
  const scalar_t* dropped_values = dropped.data_ptr<scalar_t>();
  const scalar_t* parameter_rows = params.data_ptr<scalar_t>();
  const int64_t batch_count = remaining.size(0);
  const int64_t limbs = remaining.size(1);
  const int64_t coefficients = remaining.size(2);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t inverse_stride = inverse.stride(0);
  const int64_t input_stride0 = remaining.stride(0);
  const int64_t input_stride1 = remaining.stride(1);
  const int64_t input_stride2 = remaining.stride(2);
  const int64_t dropped_stride0 = dropped.stride(0);
  const int64_t dropped_stride1 = dropped.stride(1);
  const int64_t out_stride0 = out.stride(0);
  const int64_t out_stride1 = out.stride(1);
  const int64_t out_stride2 = out.stride(2);
  const int64_t elements = batch_count * limbs * coefficients;
  at::parallel_for(
      0,
      elements,
      fhelium::cpu::adaptive_grain(elements),
      [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        while (index < end) {
          const int64_t batch_limb = index / coefficients;
          const int64_t limb = batch_limb % limbs;
          const int64_t batch = batch_limb / limbs;
          const int64_t coefficient_begin = index - batch_limb * coefficients;
          const int64_t coefficient_end =
              std::min<int64_t>(coefficients, end - batch_limb * coefficients);
          const scalar_t* dropped_row =
              dropped_values + batch * dropped_stride0;
          const auto constants =
              fhelium::cpu::load_constants(parameter_rows,
                                           parameter_row_stride,
                                           parameter_limb_stride,
                                           limb);
          const scalar_t* input_row =
              input + batch * input_stride0 + limb * input_stride1;
          scalar_t* output_row =
              output + batch * out_stride0 + limb * out_stride1;
          const scalar_t inverse_value = inverse_values[limb * inverse_stride];
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            const scalar_t dropped_value =
                dropped_row[coefficient * dropped_stride1];
            scalar_t quotient = static_cast<scalar_t>(
                input_row[coefficient * input_stride2] - dropped_value);
            quotient = fhelium::cpu::multiply_split(
                quotient, inverse_value, constants);
            if constexpr (nearest) {
              quotient = static_cast<scalar_t>(
                  quotient + (dropped_value > half_drop_prime ? 1 : 0));
            }
            output_row[coefficient * out_stride2] =
                fhelium::cpu::reduce_to_standard(quotient,
                                                 constants.twice_modulus);
          }
          index = batch_limb * coefficients + coefficient_end;
        }
      });
}

template <bool nearest>
void rescale_cpu_(torch::Tensor remaining_residues,
                  const torch::Tensor inverse,
                  const torch::Tensor dropped_residue,
                  const torch::Tensor rns_params,
                  int64_t half_drop_prime,
                  const char* operation) {
  for (const auto& peer : {inverse, dropped_residue, rns_params}) {
    check_cpu_peer(remaining_residues, peer, operation, "peer tensor");
  }
  auto remaining = view_rns_batch_3d(remaining_residues, "remaining_residues");
  const auto dropped =
      view_coefficient_batch_2d(dropped_residue, "dropped_residue");
  check_rns_parameter_rows(remaining, rns_params, operation);
  check_rns_row_vector(inverse, remaining.size(1), operation, "inverse");
  TORCH_CHECK(dropped.size(0) == remaining.size(0) &&
                  dropped.size(1) == remaining.size(2),
              operation,
              " dropped residue shape mismatch");
  at::assert_no_internal_overlap(remaining);
  at::assert_no_overlap(remaining, inverse);
  at::assert_no_overlap(remaining, dropped);
  at::assert_no_overlap(remaining, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      remaining_residues.scalar_type(), "ckks_rescale_cpu", [&] {
        rescale_loop<scalar_t, nearest>(remaining,
                                        remaining,
                                        inverse,
                                        dropped,
                                        rns_params,
                                        half_drop_prime);
      });
}

template <bool nearest>
torch::Tensor rescale_cpu(const torch::Tensor remaining,
                          const torch::Tensor inverse,
                          const torch::Tensor dropped,
                          const torch::Tensor params,
                          int64_t half_drop_prime,
                          const char* operation) {
  auto out = remaining.clone();
  rescale_cpu_<nearest>(
      out, inverse, dropped, params, half_drop_prime, operation);
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  m.impl("rescale_drop_leading_prime_nearest",
         [](const torch::Tensor a,
            const torch::Tensor inv,
            const torch::Tensor dropped,
            const torch::Tensor p,
            int64_t half) {
           return rescale_cpu<true>(a,
                                    inv,
                                    dropped,
                                    p,
                                    half,
                                    "ckks_rescale_drop_leading_prime_nearest");
         });
  m.impl("rescale_drop_leading_prime_nearest_",
         [](torch::Tensor a,
            const torch::Tensor inv,
            const torch::Tensor dropped,
            const torch::Tensor p,
            int64_t half) {
           rescale_cpu_<true>(a,
                              inv,
                              dropped,
                              p,
                              half,
                              "ckks_rescale_drop_leading_prime_nearest");
         });
  m.impl(
      "rescale_drop_leading_prime_truncate",
      [](const torch::Tensor a,
         const torch::Tensor inv,
         const torch::Tensor dropped,
         const torch::Tensor p) {
        return rescale_cpu<false>(
            a, inv, dropped, p, 0, "ckks_rescale_drop_leading_prime_truncate");
      });
  m.impl(
      "rescale_drop_leading_prime_truncate_",
      [](torch::Tensor a,
         const torch::Tensor inv,
         const torch::Tensor dropped,
         const torch::Tensor p) {
        rescale_cpu_<false>(
            a, inv, dropped, p, 0, "ckks_rescale_drop_leading_prime_truncate");
      });
}
