#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>
#include <vector>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

enum class BinaryOperation : int { kMontgomeryMul, kAddLazy, kSubLazy };
enum class UnaryOperation : int {
  kToMontgomery,
  kFromMontgomery,
  kCanonicalize,
  kCenter,
  kShiftPositive,
};
enum class RepetitionLayout : int { kCyclic, kContiguous };

void check_cpu_peer(const torch::Tensor& reference,
                    const torch::Tensor& peer,
                    const char* operation,
                    const char* peer_name) {
  TORCH_CHECK(reference.device().is_cpu(), operation, " requires CPU tensors");
  TORCH_CHECK(peer.device() == reference.device(),
              operation,
              " requires ",
              peer_name,
              " on the operand CPU device");
  TORCH_CHECK(peer.scalar_type() == reference.scalar_type(),
              operation,
              " requires ",
              peer_name,
              " with the operand dtype");
}

void check_compressed_binary(const torch::Tensor& lhs,
                             const torch::Tensor& rhs,
                             const char* operation) {
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3,
              operation,
              " requires canonical rank-three views");
  TORCH_CHECK(
      lhs.size(1) == rhs.size(1), operation, " operand limb counts differ");
  TORCH_CHECK(rhs.size(0) == lhs.size(0) || rhs.size(0) == 1,
              operation,
              " operand batch counts differ");
  TORCH_CHECK(rhs.size(2) > 0 && rhs.size(2) < lhs.size(2) &&
                  (rhs.size(2) & (rhs.size(2) - 1)) == 0 &&
                  lhs.size(2) % rhs.size(2) == 0,
              operation,
              " requires a power-of-two compressed extent dividing N");
}

template <typename scalar_t, BinaryOperation operation>
void binary_loop(torch::Tensor out,
                 const torch::Tensor lhs,
                 const torch::Tensor rhs,
                 const torch::Tensor params) {
  scalar_t* output = out.data_ptr<scalar_t>();
  const scalar_t* left = lhs.data_ptr<scalar_t>();
  const scalar_t* right = rhs.data_ptr<scalar_t>();
  const scalar_t* parameters = params.data_ptr<scalar_t>();
  const int64_t batch_count = lhs.size(0);
  const int64_t limbs = lhs.size(1);
  const int64_t coefficients = lhs.size(2);
  const int64_t rhs_batch_count = rhs.size(0);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t lhs_stride0 = lhs.stride(0);
  const int64_t lhs_stride1 = lhs.stride(1);
  const int64_t lhs_stride2 = lhs.stride(2);
  const int64_t rhs_stride0 = rhs.stride(0);
  const int64_t rhs_stride1 = rhs.stride(1);
  const int64_t rhs_stride2 = rhs.stride(2);
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
          const int64_t rhs_batch = rhs_batch_count == batch_count ? batch : 0;
          const auto constants = fhelium::cpu::load_constants(
              parameters, parameter_row_stride, parameter_limb_stride, limb);
          const scalar_t* left_row =
              left + batch * lhs_stride0 + limb * lhs_stride1;
          const scalar_t* right_row =
              right + rhs_batch * rhs_stride0 + limb * rhs_stride1;
          scalar_t* output_row =
              output + batch * out_stride0 + limb * out_stride1;
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            const scalar_t a = left_row[coefficient * lhs_stride2];
            const scalar_t b = right_row[coefficient * rhs_stride2];
            if constexpr (operation == BinaryOperation::kMontgomeryMul) {
              output_row[coefficient * out_stride2] =
                  fhelium::cpu::multiply(a, b, constants);
            } else if constexpr (operation == BinaryOperation::kAddLazy) {
              output_row[coefficient * out_stride2] =
                  fhelium::cpu::add_lazy(a, b, constants.twice_modulus);
            } else {
              output_row[coefficient * out_stride2] =
                  fhelium::cpu::subtract_lazy(a, b, constants.twice_modulus);
            }
          }
          index = batch_limb * coefficients + coefficient_end;
        }
      });
}

template <BinaryOperation operation>
torch::Tensor binary_cpu(const torch::Tensor lhs,
                         const torch::Tensor rhs,
                         const torch::Tensor params,
                         const char* operation_name) {
  check_cpu_peer(lhs, rhs, operation_name, "rhs");
  check_cpu_peer(lhs, params, operation_name, "rns_params");
  const auto left = view_rns_batch_3d(lhs, "lhs");
  const auto right = view_rns_batch_3d(rhs, "rhs");
  check_rns_binary_3d(left, right, operation_name, true);
  check_rns_parameter_rows(left, params, operation_name);
  auto out = torch::empty_like(lhs);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_binary_cpu", [&] {
    binary_loop<scalar_t, operation>(output, left, right, params);
  });
  return out;
}

template <typename scalar_t, UnaryOperation operation>
void unary_loop(torch::Tensor residues, const torch::Tensor params) {
  scalar_t* values = residues.data_ptr<scalar_t>();
  const scalar_t* parameters = params.data_ptr<scalar_t>();
  const int64_t batch_count = residues.size(0);
  const int64_t limbs = residues.size(1);
  const int64_t coefficients = residues.size(2);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t stride0 = residues.stride(0);
  const int64_t stride1 = residues.stride(1);
  const int64_t stride2 = residues.stride(2);
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
          const auto constants = fhelium::cpu::load_constants(
              parameters, parameter_row_stride, parameter_limb_stride, limb);
          scalar_t* row_values = values + batch * stride0 + limb * stride1;
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            scalar_t value = row_values[coefficient * stride2];
            if constexpr (operation == UnaryOperation::kToMontgomery) {
              value = fhelium::cpu::multiply(value, constants.r2, constants);
            } else if constexpr (operation == UnaryOperation::kFromMontgomery) {
              value = fhelium::cpu::reduce(value, constants);
            } else if constexpr (operation == UnaryOperation::kCanonicalize) {
              value =
                  fhelium::cpu::canonicalize(value, constants.twice_modulus);
            } else if constexpr (operation == UnaryOperation::kCenter) {
              value = fhelium::cpu::center(value, constants.twice_modulus);
            } else {
              value =
                  fhelium::cpu::shift_positive(value, constants.twice_modulus);
            }
            row_values[coefficient * stride2] = value;
          }
          index = batch_limb * coefficients + coefficient_end;
        }
      });
}

template <UnaryOperation operation>
void unary_cpu(torch::Tensor residues,
               const torch::Tensor params,
               const char* operation_name) {
  check_cpu_peer(residues, params, operation_name, "rns_params");
  auto rows = view_rns_batch_3d(residues, "residues");
  check_rns_parameter_rows(rows, params, operation_name);
  at::assert_no_internal_overlap(rows);
  at::assert_no_overlap(rows, params);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), "rns_unary_cpu", [&] {
    unary_loop<scalar_t, operation>(rows, params);
  });
}

template <typename scalar_t, RepetitionLayout layout>
void compressed_loop(torch::Tensor out,
                     const torch::Tensor lhs,
                     const torch::Tensor rhs,
                     const torch::Tensor params) {
  scalar_t* output = out.data_ptr<scalar_t>();
  const scalar_t* left = lhs.data_ptr<scalar_t>();
  const scalar_t* right = rhs.data_ptr<scalar_t>();
  const scalar_t* parameters = params.data_ptr<scalar_t>();
  const int64_t batch_count = lhs.size(0);
  const int64_t limbs = lhs.size(1);
  const int64_t coefficients = lhs.size(2);
  const int64_t unique = rhs.size(2);
  const int64_t repeat = coefficients / unique;
  const int64_t rhs_batch_count = rhs.size(0);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t lhs_stride0 = lhs.stride(0);
  const int64_t lhs_stride1 = lhs.stride(1);
  const int64_t lhs_stride2 = lhs.stride(2);
  const int64_t rhs_stride0 = rhs.stride(0);
  const int64_t rhs_stride1 = rhs.stride(1);
  const int64_t rhs_stride2 = rhs.stride(2);
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
          const int64_t rhs_batch = rhs_batch_count == batch_count ? batch : 0;
          const auto constants = fhelium::cpu::load_constants(
              parameters, parameter_row_stride, parameter_limb_stride, limb);
          const scalar_t* left_row =
              left + batch * lhs_stride0 + limb * lhs_stride1;
          const scalar_t* right_row =
              right + rhs_batch * rhs_stride0 + limb * rhs_stride1;
          scalar_t* output_row =
              output + batch * out_stride0 + limb * out_stride1;
          for (int64_t coefficient = coefficient_begin;
               coefficient < coefficient_end;
               ++coefficient) {
            const int64_t rhs_index = layout == RepetitionLayout::kCyclic
                                          ? coefficient % unique
                                          : coefficient / repeat;
            output_row[coefficient * out_stride2] =
                fhelium::cpu::multiply(left_row[coefficient * lhs_stride2],
                                       right_row[rhs_index * rhs_stride2],
                                       constants);
          }
          index = batch_limb * coefficients + coefficient_end;
        }
      });
}

template <RepetitionLayout layout>
torch::Tensor compressed_cpu(const torch::Tensor lhs,
                             const torch::Tensor rhs,
                             const torch::Tensor params,
                             const char* operation_name) {
  check_cpu_peer(lhs, rhs, operation_name, "compressed_rhs");
  check_cpu_peer(lhs, params, operation_name, "rns_params");
  const auto left = view_rns_batch_3d(lhs, "lhs");
  const auto right = view_rns_batch_3d(rhs, "compressed_rhs");
  check_compressed_binary(left, right, operation_name);
  check_rns_parameter_rows(left, params, operation_name);
  auto out = torch::empty_like(lhs);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_compressed_cpu", [&] {
    compressed_loop<scalar_t, layout>(output, left, right, params);
  });
  return out;
}

void row_scalars_cpu_(torch::Tensor residues,
                      const torch::Tensor scalars,
                      const torch::Tensor params) {
  constexpr const char* operation = "rns_montgomery_mul_row_scalars";
  check_cpu_peer(residues, scalars, operation, "row_scalars");
  check_cpu_peer(residues, params, operation, "rns_params");
  auto rows = view_rns_batch_3d(residues, "residues");
  check_rns_parameter_rows(rows, params, operation);
  check_rns_row_vector(scalars, rows.size(1), operation, "row_scalars");
  at::assert_no_internal_overlap(rows);
  at::assert_no_overlap(rows, scalars);
  at::assert_no_overlap(rows, params);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    scalar_t* values = rows.data_ptr<scalar_t>();
    const scalar_t* row_scalars = scalars.data_ptr<scalar_t>();
    const scalar_t* parameters = params.data_ptr<scalar_t>();
    const int64_t batch_count = rows.size(0);
    const int64_t limbs = rows.size(1);
    const int64_t coefficients = rows.size(2);
    const int64_t parameter_row_stride = params.stride(0);
    const int64_t parameter_limb_stride = params.stride(1);
    const int64_t scalar_stride = scalars.stride(0);
    const int64_t stride0 = rows.stride(0);
    const int64_t stride1 = rows.stride(1);
    const int64_t stride2 = rows.stride(2);
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
            const auto constants = fhelium::cpu::load_constants(
                parameters, parameter_row_stride, parameter_limb_stride, limb);
            const scalar_t row_scalar = row_scalars[limb * scalar_stride];
            scalar_t* row_values = values + batch * stride0 + limb * stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              row_values[coefficient * stride2] = fhelium::cpu::multiply(
                  row_values[coefficient * stride2], row_scalar, constants);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
}

torch::Tensor add_lazy_twice_modulus_cpu(const torch::Tensor lhs,
                                         const torch::Tensor rhs,
                                         const torch::Tensor twice_modulus) {
  constexpr const char* operation = "rns_add_lazy_with_twice_modulus";
  check_cpu_peer(lhs, rhs, operation, "rhs");
  check_cpu_peer(lhs, twice_modulus, operation, "twice_modulus");
  const auto left = view_rns_batch_3d(lhs, "lhs");
  const auto right = view_rns_batch_3d(rhs, "rhs");
  check_rns_binary_3d(left, right, operation, true);
  check_rns_row_vector(twice_modulus, left.size(1), operation, "twice_modulus");
  auto out = torch::empty_like(lhs);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), operation, [&] {
    scalar_t* values = output.data_ptr<scalar_t>();
    const scalar_t* a = left.data_ptr<scalar_t>();
    const scalar_t* b = right.data_ptr<scalar_t>();
    const scalar_t* twice = twice_modulus.data_ptr<scalar_t>();
    const int64_t twice_stride = twice_modulus.stride(0);
    const int64_t batch_count = left.size(0);
    const int64_t limbs = left.size(1);
    const int64_t coefficients = left.size(2);
    const int64_t rhs_batch_count = right.size(0);
    const int64_t a_stride0 = left.stride(0);
    const int64_t a_stride1 = left.stride(1);
    const int64_t a_stride2 = left.stride(2);
    const int64_t b_stride0 = right.stride(0);
    const int64_t b_stride1 = right.stride(1);
    const int64_t b_stride2 = right.stride(2);
    const int64_t out_stride0 = output.stride(0);
    const int64_t out_stride1 = output.stride(1);
    const int64_t out_stride2 = output.stride(2);
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
            const int64_t rhs_batch =
                rhs_batch_count == batch_count ? batch : 0;
            const scalar_t twice_modulus_value = twice[limb * twice_stride];
            const scalar_t* a_row = a + batch * a_stride0 + limb * a_stride1;
            const scalar_t* b_row =
                b + rhs_batch * b_stride0 + limb * b_stride1;
            scalar_t* output_row =
                values + batch * out_stride0 + limb * out_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              output_row[coefficient * out_stride2] =
                  fhelium::cpu::add_lazy(a_row[coefficient * a_stride2],
                                         b_row[coefficient * b_stride2],
                                         twice_modulus_value);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

torch::Tensor lift_centered_cpu(const torch::Tensor coefficients,
                                const torch::Tensor twice_modulus) {
  constexpr const char* operation = "rns_lift_centered_coefficients";
  check_cpu_peer(coefficients, twice_modulus, operation, "twice_modulus");
  const auto input = view_coefficient_batch_2d(coefficients, "coefficients");
  check_rns_row_vector(
      twice_modulus, twice_modulus.size(0), operation, "twice_modulus");
  auto shape = coefficients.sizes().vec();
  shape.insert(shape.end() - 1, twice_modulus.size(0));
  auto out = coefficients.new_empty(shape);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(coefficients.scalar_type(), operation, [&] {
    scalar_t* values = output.data_ptr<scalar_t>();
    const scalar_t* source = input.data_ptr<scalar_t>();
    const scalar_t* twice = twice_modulus.data_ptr<scalar_t>();
    const int64_t twice_stride = twice_modulus.stride(0);
    const int64_t batch_count = output.size(0);
    const int64_t limbs = output.size(1);
    const int64_t coefficient_count = output.size(2);
    const int64_t source_stride0 = input.stride(0);
    const int64_t source_stride1 = input.stride(1);
    const int64_t out_stride0 = output.stride(0);
    const int64_t out_stride1 = output.stride(1);
    const int64_t out_stride2 = output.stride(2);
    const int64_t elements = batch_count * limbs * coefficient_count;
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficient_count;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin =
                index - batch_limb * coefficient_count;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficient_count, end - batch_limb * coefficient_count);
            const scalar_t twice_modulus_value = twice[limb * twice_stride];
            const scalar_t* source_row = source + batch * source_stride0;
            scalar_t* output_row =
                values + batch * out_stride0 + limb * out_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              output_row[coefficient * out_stride2] =
                  fhelium::cpu::shift_positive(
                      source_row[coefficient * source_stride1],
                      twice_modulus_value);
            }
            index = batch_limb * coefficient_count + coefficient_end;
          }
        });
  });
  return out;
}

using fhelium::cpu::MontgomeryConstants;

torch::Tensor mixed_radix_decompose_cpu(
    const torch::Tensor source_residues,
    const torch::Tensor normalizers,
    const torch::Tensor propagation,
    const torch::Tensor modulus_lo,
    const torch::Tensor modulus_hi,
    const torch::Tensor neg_inv_modulus_lo,
    const torch::Tensor neg_inv_modulus_hi) {
  constexpr const char* operation = "mixed_radix_decompose";
  const auto source = view_rns_batch_3d(source_residues, "source_residues");
  TORCH_CHECK(source.device().is_cpu(), operation, " requires CPU tensors");
  TORCH_CHECK(source.size(1) <= 8, operation, " supports at most 8 rows");
  check_rns_row_vector(
      normalizers, source.size(1) - 1, operation, "normalizers");
  TORCH_CHECK(propagation.dim() == 2 &&
                  propagation.size(0) == source.size(1) - 1 &&
                  propagation.size(1) == source.size(1),
              operation,
              " propagation table shape mismatch");
  for (const auto& table : {normalizers,
                            propagation,
                            modulus_lo,
                            modulus_hi,
                            neg_inv_modulus_lo,
                            neg_inv_modulus_hi}) {
    check_cpu_peer(source_residues, table, operation, "table");
  }
  auto out = torch::empty_like(source_residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(source_residues.scalar_type(), operation, [&] {
    const scalar_t* values = source.data_ptr<scalar_t>();
    scalar_t* digits_out = output.data_ptr<scalar_t>();
    const scalar_t* norm = normalizers.data_ptr<scalar_t>();
    const scalar_t* prop = propagation.data_ptr<scalar_t>();
    const scalar_t* lo = modulus_lo.data_ptr<scalar_t>();
    const scalar_t* hi = modulus_hi.data_ptr<scalar_t>();
    const scalar_t* inv_lo = neg_inv_modulus_lo.data_ptr<scalar_t>();
    const scalar_t* inv_hi = neg_inv_modulus_hi.data_ptr<scalar_t>();
    const int64_t normalizer_stride = normalizers.stride(0);
    const int64_t propagation_row_stride = propagation.stride(0);
    const int64_t propagation_limb_stride = propagation.stride(1);
    const int64_t modulus_lo_stride = modulus_lo.stride(0);
    const int64_t modulus_hi_stride = modulus_hi.stride(0);
    const int64_t neg_inv_modulus_lo_stride = neg_inv_modulus_lo.stride(0);
    const int64_t neg_inv_modulus_hi_stride = neg_inv_modulus_hi.stride(0);
    const int64_t row_count = source.size(1);
    const int64_t coefficient_count = source.size(2);
    const int64_t source_stride0 = source.stride(0);
    const int64_t source_stride1 = source.stride(1);
    const int64_t source_stride2 = source.stride(2);
    const int64_t out_stride0 = output.stride(0);
    const int64_t out_stride1 = output.stride(1);
    const int64_t out_stride2 = output.stride(2);
    const int64_t elements = source.size(0) * coefficient_count;
    const auto first_constants =
        fhelium::cpu::load_constants(lo,
                                     modulus_lo_stride,
                                     hi,
                                     modulus_hi_stride,
                                     inv_lo,
                                     neg_inv_modulus_lo_stride,
                                     inv_hi,
                                     neg_inv_modulus_hi_stride,
                                     0);
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          std::vector<scalar_t> digits(static_cast<size_t>(row_count));
          int64_t index = begin;
          while (index < end) {
            const int64_t batch = index / coefficient_count;
            const int64_t coefficient_begin = index - batch * coefficient_count;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficient_count, end - batch * coefficient_count);
            const scalar_t* source_rows = values + batch * source_stride0;
            scalar_t* output_rows = digits_out + batch * out_stride0;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              const scalar_t first_residue =
                  fhelium::cpu::canonicalize_lazy_operand(
                      source_rows[coefficient * source_stride2],
                      first_constants);
              std::fill(digits.begin(), digits.end(), first_residue);
              for (int64_t step = 0; step < row_count - 1; ++step) {
                const int64_t row = step + 1;
                const auto constants =
                    fhelium::cpu::load_constants(lo,
                                                 modulus_lo_stride,
                                                 hi,
                                                 modulus_hi_stride,
                                                 inv_lo,
                                                 neg_inv_modulus_lo_stride,
                                                 inv_hi,
                                                 neg_inv_modulus_hi_stride,
                                                 row);
                const scalar_t digit = fhelium::cpu::multiply_split(
                    static_cast<scalar_t>(
                        source_rows[row * source_stride1 +
                                    coefficient * source_stride2] -
                        digits[row]),
                    norm[step * normalizer_stride],
                    constants);
                digits[row] = digit;
                for (int64_t target = row + 1; target < row_count; ++target) {
                  digits[target] += fhelium::cpu::multiply(
                      digit,
                      prop[step * propagation_row_stride +
                           target * propagation_limb_stride],
                      fhelium::cpu::load_constants(lo,
                                                   modulus_lo_stride,
                                                   hi,
                                                   modulus_hi_stride,
                                                   inv_lo,
                                                   neg_inv_modulus_lo_stride,
                                                   inv_hi,
                                                   neg_inv_modulus_hi_stride,
                                                   target));
                }
              }
              for (int64_t row = 0; row < row_count; ++row) {
                output_rows[row * out_stride1 + coefficient * out_stride2] =
                    digits[row];
              }
            }
            index = batch * coefficient_count + coefficient_end;
          }
        });
  });
  return out;
}

torch::Tensor mixed_radix_extend_cpu(const torch::Tensor components,
                                     const torch::Tensor coefficients,
                                     const torch::Tensor params,
                                     int64_t destination_rows) {
  constexpr const char* operation = "mixed_radix_basis_extend_to_montgomery";
  TORCH_CHECK(destination_rows > 0, "destination_row_count must be positive");
  check_cpu_peer(components, coefficients, operation, "coefficients");
  check_cpu_peer(components, params, operation, "rns_params");
  const auto digits = view_rns_batch_3d(components, "components");
  TORCH_CHECK(
      coefficients.dim() == 2 && coefficients.size(0) == digits.size(1) - 1 &&
          (digits.size(1) == 1 || coefficients.size(1) == destination_rows),
      operation,
      " coefficient table shape mismatch");
  auto shape = components.sizes().vec();
  shape[shape.size() - 2] = destination_rows;
  auto out = torch::empty(shape, components.options());
  auto output = view_rns_batch_3d(out, "out");
  check_rns_parameter_rows(output, params, operation);
  AT_DISPATCH_INTEGRAL_TYPES(components.scalar_type(), operation, [&] {
    const scalar_t* source = digits.data_ptr<scalar_t>();
    const scalar_t* table = coefficients.data_ptr<scalar_t>();
    const scalar_t* parameters = params.data_ptr<scalar_t>();
    scalar_t* values = output.data_ptr<scalar_t>();
    const int64_t digit_count = digits.size(1);
    const int64_t coefficient_count = digits.size(2);
    const int64_t parameter_row_stride = params.stride(0);
    const int64_t parameter_limb_stride = params.stride(1);
    const int64_t coefficient_digit_stride = coefficients.stride(0);
    const int64_t coefficient_row_stride = coefficients.stride(1);
    const int64_t source_stride0 = digits.stride(0);
    const int64_t source_stride1 = digits.stride(1);
    const int64_t source_stride2 = digits.stride(2);
    const int64_t out_stride0 = output.stride(0);
    const int64_t out_stride1 = output.stride(1);
    const int64_t out_stride2 = output.stride(2);
    const int64_t elements =
        digits.size(0) * destination_rows * coefficient_count;
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_row = index / coefficient_count;
            const int64_t row = batch_row % destination_rows;
            const int64_t batch = batch_row / destination_rows;
            const int64_t coefficient_begin =
                index - batch_row * coefficient_count;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficient_count, end - batch_row * coefficient_count);
            const auto constants = fhelium::cpu::load_constants(
                parameters, parameter_row_stride, parameter_limb_stride, row);
            const scalar_t* source_rows = source + batch * source_stride0;
            scalar_t* output_row =
                values + batch * out_stride0 + row * out_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              scalar_t value = fhelium::cpu::multiply(
                  source_rows[coefficient * source_stride2],
                  constants.r2,
                  constants);
              for (int64_t digit = 1; digit < digit_count; ++digit) {
                value = fhelium::cpu::add_lazy(
                    value,
                    fhelium::cpu::multiply(
                        source_rows[digit * source_stride1 +
                                    coefficient * source_stride2],
                        table[(digit - 1) * coefficient_digit_stride +
                              row * coefficient_row_stride],
                        constants),
                    constants.twice_modulus);
              }
              output_row[coefficient * out_stride2] = value;
            }
            index = batch_row * coefficient_count + coefficient_end;
          }
        });
  });
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CPU, m) {
  m.impl(
      "montgomery_mul",
      [](const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return binary_cpu<BinaryOperation::kMontgomeryMul>(
            a, b, p, "rns_montgomery_mul");
      });
  m.impl(
      "montgomery_mul_cyclic_compressed",
      [](const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return compressed_cpu<RepetitionLayout::kCyclic>(
            a, b, p, "rns_montgomery_mul_cyclic_compressed");
      });
  m.impl(
      "montgomery_mul_contiguous_compressed",
      [](const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return compressed_cpu<RepetitionLayout::kContiguous>(
            a, b, p, "rns_montgomery_mul_contiguous_compressed");
      });
  m.impl("montgomery_mul_row_scalars_", &row_scalars_cpu_);
  m.impl("to_montgomery_", [](torch::Tensor a, const torch::Tensor p) {
    unary_cpu<UnaryOperation::kToMontgomery>(a, p, "rns_to_montgomery");
  });
  m.impl("from_montgomery_", [](torch::Tensor a, const torch::Tensor p) {
    unary_cpu<UnaryOperation::kFromMontgomery>(a, p, "rns_from_montgomery");
  });
  m.impl(
      "add_lazy",
      [](const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return binary_cpu<BinaryOperation::kAddLazy>(a, b, p, "rns_add_lazy");
      });
  m.impl("add_lazy_with_twice_modulus", &add_lazy_twice_modulus_cpu);
  m.impl(
      "sub_lazy",
      [](const torch::Tensor a, const torch::Tensor b, const torch::Tensor p) {
        return binary_cpu<BinaryOperation::kSubLazy>(a, b, p, "rns_sub_lazy");
      });
  m.impl("canonicalize_residues_", [](torch::Tensor a, const torch::Tensor p) {
    unary_cpu<UnaryOperation::kCanonicalize>(a, p, "rns_canonicalize");
  });
  m.impl("center_residues_", [](torch::Tensor a, const torch::Tensor p) {
    unary_cpu<UnaryOperation::kCenter>(a, p, "rns_center_residues");
  });
  m.impl(
      "shift_residues_positive_", [](torch::Tensor a, const torch::Tensor p) {
        unary_cpu<UnaryOperation::kShiftPositive>(a, p, "rns_shift_positive");
      });
  m.impl("lift_centered_coefficients", &lift_centered_cpu);
  m.impl("mixed_radix_decompose", &mixed_radix_decompose_cpu);
  m.impl("mixed_radix_basis_extend_to_montgomery", &mixed_radix_extend_cpu);
}
