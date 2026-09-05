#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "tensor_validation.h"

namespace {

using fhelium::ckks::cpu::check_cpu_peer;

torch::Tensor keyswitch_moddown_cpu(const torch::Tensor q_residues,
                                    const torch::Tensor p_residues,
                                    const torch::Tensor inverse,
                                    const torch::Tensor rns_params) {
  constexpr const char* operation = "keyswitch_moddown_qp_to_q";
  for (const auto& peer : {p_residues, inverse, rns_params}) {
    check_cpu_peer(q_residues, peer, operation, "peer tensor");
  }
  const auto q = view_rns_batch_3d(q_residues, "q_residues");
  const auto p = view_rns_batch_3d(p_residues, "p_residues");
  TORCH_CHECK(q.size(0) == p.size(0) && q.size(2) == p.size(2),
              operation,
              " Q/P batch and coefficient shapes differ");
  TORCH_CHECK(
      rns_params.dim() == 2 && rns_params.size(1) == q.size(1) + p.size(1),
      operation,
      " RNS parameter row count mismatch");
  TORCH_CHECK(inverse.dim() == 2 && inverse.size(0) == p.size(1) &&
                  inverse.size(1) >= q.size(1) + p.size(1) - 1,
              operation,
              " inverse table shape mismatch");
  auto out = torch::empty_like(q_residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(q_residues.scalar_type(), operation, [&] {
    const scalar_t* q_values = q.data_ptr<scalar_t>();
    const scalar_t* p_values = p.data_ptr<scalar_t>();
    const scalar_t* inverse_values = inverse.data_ptr<scalar_t>();
    const scalar_t* parameter_rows = rns_params.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int64_t batch_count = q.size(0);
    const int64_t q_count = q.size(1);
    const int64_t p_count = p.size(1);
    const int64_t coefficients = q.size(2);
    const int64_t parameter_row_stride = rns_params.stride(0);
    const int64_t parameter_limb_stride = rns_params.stride(1);
    const int64_t inverse_drop_stride = inverse.stride(0);
    const int64_t inverse_limb_stride = inverse.stride(1);
    const int64_t q_stride0 = q.stride(0);
    const int64_t q_stride1 = q.stride(1);
    const int64_t q_stride2 = q.stride(2);
    const int64_t p_stride0 = p.stride(0);
    const int64_t p_stride1 = p.stride(1);
    const int64_t p_stride2 = p.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    // After preparing the reverse mixed-radix P digits d_j, the sequential
    // divisions equal q/P - sum_j d_j/(p_0 ... p_j) modulo each Q prime.
    // These coefficients are shared by all coefficients and batch items.
    c10::SmallVector<scalar_t, 64> prefix_inverse(q_count * p_count);
    for (int64_t row = 0; row < q_count; ++row) {
      const auto constants = fhelium::cpu::load_constants(
          parameter_rows, parameter_row_stride, parameter_limb_stride, row);
      scalar_t product = inverse_values[(p_count - 1) * inverse_drop_stride +
                                        row * inverse_limb_stride];
      prefix_inverse[row * p_count] = product;
      for (int64_t j = 1; j < p_count; ++j) {
        product = fhelium::cpu::multiply(
            product,
            inverse_values[(p_count - j - 1) * inverse_drop_stride +
                           row * inverse_limb_stride],
            constants);
        prefix_inverse[row * p_count + j] = product;
      }
    }
    const int64_t elements = batch_count * coefficients;
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          c10::SmallVector<scalar_t, 8> p_chain(static_cast<size_t>(p_count));
          int64_t index = begin;
          while (index < end) {
            const int64_t batch = index / coefficients;
            const int64_t coefficient_begin = index - batch * coefficients;
            const int64_t coefficient_end =
                std::min<int64_t>(coefficients, end - batch * coefficients);
            const scalar_t* q_rows = q_values + batch * q_stride0;
            const scalar_t* p_rows = p_values + batch * p_stride0;
            scalar_t* result_rows = result + batch * result_stride0;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              for (int64_t row = 0; row < p_count; ++row) {
                p_chain[static_cast<size_t>(row)] =
                    p_rows[row * p_stride1 + coefficient * p_stride2];
              }
              for (int64_t row = p_count - 2; row >= 0; --row) {
                scalar_t value = p_chain[static_cast<size_t>(row)];
                const int64_t parameter_row = q_count + row;
                const auto constants =
                    fhelium::cpu::load_constants(parameter_rows,
                                                 parameter_row_stride,
                                                 parameter_limb_stride,
                                                 parameter_row);
                for (int64_t lower = p_count - 1; lower > row; --lower) {
                  const scalar_t difference = fhelium::cpu::subtract_lazy(
                      value,
                      p_chain[static_cast<size_t>(lower)],
                      constants.twice_modulus);
                  value = fhelium::cpu::multiply_split(
                      difference,
                      inverse_values[(p_count - lower - 1) *
                                         inverse_drop_stride +
                                     (row + q_count) * inverse_limb_stride],
                      constants);
                }
                p_chain[static_cast<size_t>(row)] = value;
              }

              for (int64_t row = 0; row < q_count; ++row) {
                const auto constants =
                    fhelium::cpu::load_constants(parameter_rows,
                                                 parameter_row_stride,
                                                 parameter_limb_stride,
                                                 row);
                scalar_t value = fhelium::cpu::multiply(
                    q_rows[row * q_stride1 + coefficient * q_stride2],
                    prefix_inverse[row * p_count + p_count - 1],
                    constants);
                for (int64_t p_row = 0; p_row < p_count; ++p_row) {
                  const scalar_t contribution = fhelium::cpu::multiply(
                      p_chain[static_cast<size_t>(p_row)],
                      prefix_inverse[row * p_count + p_row],
                      constants);
                  value = fhelium::cpu::subtract_lazy(
                      value, contribution, constants.twice_modulus);
                }
                result_rows[row * result_stride1 +
                            coefficient * result_stride2] =
                    fhelium::cpu::reduce_to_standard(value,
                                                     constants.twice_modulus);
              }
            }
            index = batch * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

void keyswitch_accumulate_cpu_(torch::Tensor accumulator0_qp,
                               torch::Tensor accumulator1_qp,
                               const torch::Tensor digit_qp,
                               const torch::Tensor key_digit,
                               const torch::Tensor rns_params,
                               const int64_t key_row_start,
                               const std::optional<torch::Tensor> source_indices) {
  constexpr const char* operation = "keyswitch_accumulate_digit_products";
  for (const auto& peer :
       {accumulator0_qp, accumulator1_qp, key_digit, rns_params}) {
    check_cpu_peer(digit_qp, peer, operation, "peer tensor");
  }
  auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  const auto digit = view_rns_batch_3d(digit_qp, "digit_qp");
  check_rns_binary_3d(accumulator0, digit, operation, false);
  check_rns_binary_3d(accumulator1, digit, operation, false);
  check_rns_parameter_rows(digit, rns_params, operation);
  const int32_t* permutation = nullptr;
  int64_t permutation_stride = 0;
  if (source_indices) {
    const auto& indices = *source_indices;
    TORCH_CHECK(indices.device() == digit_qp.device() &&
                    indices.scalar_type() == torch::kInt32 &&
                    indices.dim() == 1 && indices.size(0) == digit.size(2),
                operation, " requires int32 source indices matching N");
    at::assert_no_overlap(accumulator0, indices);
    at::assert_no_overlap(accumulator1, indices);
    permutation = indices.data_ptr<int32_t>();
    permutation_stride = indices.stride(0);
    for (int64_t i = 0; i < digit.size(2); ++i) {
      TORCH_CHECK(permutation[i * permutation_stride] >= 0 &&
                      permutation[i * permutation_stride] < digit.size(2),
                  operation, " source index is out of bounds");
    }
  }
  TORCH_CHECK(key_digit.dim() == 3 && key_digit.size(0) == 2 &&
                  key_row_start >= 0 &&
                  key_row_start + digit.size(1) <= key_digit.size(1) &&
                  key_digit.size(2) == digit.size(2),
              operation,
              " key digit shape or row interval mismatch");
  at::assert_no_internal_overlap(accumulator0);
  at::assert_no_internal_overlap(accumulator1);
  at::assert_no_overlap(accumulator0, accumulator1);
  for (const auto& read_only : {digit, key_digit, rns_params}) {
    at::assert_no_overlap(accumulator0, read_only);
    at::assert_no_overlap(accumulator1, read_only);
  }
  AT_DISPATCH_INTEGRAL_TYPES(digit_qp.scalar_type(), operation, [&] {
    scalar_t* out0 = accumulator0.data_ptr<scalar_t>();
    scalar_t* out1 = accumulator1.data_ptr<scalar_t>();
    const scalar_t* source = digit.data_ptr<scalar_t>();
    const scalar_t* key = key_digit.data_ptr<scalar_t>();
    const scalar_t* parameter_rows = rns_params.data_ptr<scalar_t>();
    const int64_t batch_count = digit.size(0);
    const int64_t limbs = digit.size(1);
    const int64_t coefficients = digit.size(2);
    const int64_t parameter_row_stride = rns_params.stride(0);
    const int64_t parameter_limb_stride = rns_params.stride(1);
    const int64_t digit_stride0 = digit.stride(0);
    const int64_t digit_stride1 = digit.stride(1);
    const int64_t digit_stride2 = digit.stride(2);
    const int64_t key_stride1 = key_digit.stride(1);
    const int64_t key_stride2 = key_digit.stride(2);
    const int64_t out0_stride0 = accumulator0.stride(0);
    const int64_t out0_stride1 = accumulator0.stride(1);
    const int64_t out0_stride2 = accumulator0.stride(2);
    const int64_t out1_stride0 = accumulator1.stride(0);
    const int64_t out1_stride1 = accumulator1.stride(1);
    const int64_t out1_stride2 = accumulator1.stride(2);
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
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const auto constants =
                fhelium::cpu::load_constants(parameter_rows,
                                             parameter_row_stride,
                                             parameter_limb_stride,
                                             limb);
            const scalar_t* source_row =
                source + batch * digit_stride0 + limb * digit_stride1;
            const int64_t key_row = key_row_start + limb;
            const int64_t key_component_stride = key_digit.stride(0);
            const scalar_t* key_row0 = key + key_row * key_stride1;
            const scalar_t* key_row1 =
                key + key_component_stride + key_row * key_stride1;
            scalar_t* out0_row =
                out0 + batch * out0_stride0 + limb * out0_stride1;
            scalar_t* out1_row =
                out1 + batch * out1_stride0 + limb * out1_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              const int64_t source_coefficient = permutation
                  ? permutation[coefficient * permutation_stride] : coefficient;
              const scalar_t value = source_row[source_coefficient * digit_stride2];
              const scalar_t product0 = fhelium::cpu::multiply(
                  value, key_row0[coefficient * key_stride2], constants);
              const scalar_t product1 = fhelium::cpu::multiply(
                  value, key_row1[coefficient * key_stride2], constants);
              out0_row[coefficient * out0_stride2] =
                  fhelium::cpu::add_lazy(out0_row[coefficient * out0_stride2],
                                         product0,
                                         constants.twice_modulus);
              out1_row[coefficient * out1_stride2] =
                  fhelium::cpu::add_lazy(out1_row[coefficient * out1_stride2],
                                         product1,
                                         constants.twice_modulus);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  m.impl("keyswitch_moddown_qp_to_q", &keyswitch_moddown_cpu);
  m.impl("keyswitch_accumulate_digit_products_", &keyswitch_accumulate_cpu_);
}
