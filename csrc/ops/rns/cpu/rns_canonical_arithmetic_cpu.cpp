#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

// CPU execution shares the CUDA operator schemas and uses PyTorch's intra-op
// pool to partition flattened batch/limb/coefficient work into contiguous
// chunks;
// each inner loop stops at an RNS-row end before selecting the next modulus.
enum class CanonicalBinaryOperation : int { kAdd, kSubtract };

template <typename scalar_t, CanonicalBinaryOperation operation>
void canonical_binary_loop(torch::Tensor out,
                           const torch::Tensor lhs,
                           const torch::Tensor rhs,
                           const torch::Tensor params) {
  scalar_t* out_values = out.data_ptr<scalar_t>();
  const scalar_t* lhs_values = lhs.data_ptr<scalar_t>();
  const scalar_t* rhs_values = rhs.data_ptr<scalar_t>();
  const scalar_t* parameter_values = params.data_ptr<scalar_t>();
  const int64_t batch_count = lhs.size(0);
  const int64_t rhs_batch_count = rhs.size(0);
  const int64_t limb_count = lhs.size(1);
  const int64_t coefficient_count = lhs.size(2);
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
  const int64_t element_count = batch_count * limb_count * coefficient_count;

  at::parallel_for(
      0,
      element_count,
      fhelium::cpu::adaptive_grain(element_count),
      [&](int64_t begin, int64_t end) {
        while (begin < end) {
          const int64_t coefficient = begin % coefficient_count;
          const int64_t batch_limb = begin / coefficient_count;
          const int64_t limb = batch_limb % limb_count;
          const int64_t batch = batch_limb / limb_count;
          const int64_t rhs_batch = rhs_batch_count == batch_count ? batch : 0;
          const int64_t row_end =
              std::min(end, begin + coefficient_count - coefficient);
          const int64_t twice_modulus = static_cast<int64_t>(
              parameter_values[RNS_PARAM_TWICE_MODULUS * parameter_row_stride +
                               limb * parameter_limb_stride]);
          const int64_t modulus = twice_modulus >> 1;
          const scalar_t* lhs_row =
              lhs_values + batch * lhs_stride0 + limb * lhs_stride1;
          const scalar_t* rhs_row =
              rhs_values + rhs_batch * rhs_stride0 + limb * rhs_stride1;
          scalar_t* out_row =
              out_values + batch * out_stride0 + limb * out_stride1;

          for (int64_t index = begin; index < row_end; ++index) {
            const int64_t offset = coefficient + index - begin;
            const int64_t a =
                static_cast<int64_t>(lhs_row[offset * lhs_stride2]);
            const int64_t b =
                static_cast<int64_t>(rhs_row[offset * rhs_stride2]);
            int64_t value;
            if constexpr (operation == CanonicalBinaryOperation::kAdd) {
              value = a + b;
              if (value >= twice_modulus) value -= twice_modulus;
            } else {
              value = a - b;
              if (value < 0) {
                value += twice_modulus;
              } else if (value >= twice_modulus) {
                value -= twice_modulus;
              }
            }
            out_row[offset * out_stride2] = static_cast<scalar_t>(
                value < modulus ? value : value - modulus);
          }
          begin = row_end;
        }
      });
}

void check_cpu_canonical_binary(const torch::Tensor& out,
                                const torch::Tensor& lhs,
                                const torch::Tensor& rhs,
                                const torch::Tensor& params,
                                const char* operation_name) {
  TORCH_CHECK(lhs.device().is_cpu(), operation_name, " requires CPU lhs");
  TORCH_CHECK(rhs.device() == lhs.device(),
              operation_name,
              " requires operands on the same CPU device");
  TORCH_CHECK(params.device() == lhs.device(),
              operation_name,
              " requires RNS parameters on the same CPU device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(),
              operation_name,
              " operand dtypes differ: ",
              lhs.scalar_type(),
              " vs ",
              rhs.scalar_type());
  TORCH_CHECK(lhs.scalar_type() == params.scalar_type(),
              operation_name,
              " requires RNS parameters with the operand dtype");
  TORCH_CHECK(params.size(0) > RNS_PARAM_TWICE_MODULUS,
              operation_name,
              " requires the twice-modulus parameter row");
  TORCH_CHECK(out.sizes() == lhs.sizes(),
              operation_name,
              " output shape must match lhs");
  check_mutable_rns_output(out, rhs, params);
}

template <CanonicalBinaryOperation operation>
void canonical_binary_into(torch::Tensor out,
                           const torch::Tensor lhs,
                           const torch::Tensor rhs,
                           const torch::Tensor params,
                           const char* operation_name) {
  auto out_rows = view_rns_batch_3d(out, "out");
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  check_rns_binary_3d(lhs_rows, rhs_rows, operation_name, true);
  check_rns_parameter_rows(lhs_rows, params, operation_name);
  check_cpu_canonical_binary(
      out_rows, lhs_rows, rhs_rows, params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(
      lhs.scalar_type(), "rns_canonical_binary_cpu", [&] {
        canonical_binary_loop<scalar_t, operation>(
            out_rows, lhs_rows, rhs_rows, params);
      });
}

torch::Tensor rns_add_canonical_cpu(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  canonical_binary_into<CanonicalBinaryOperation::kAdd>(
      out, lhs, rhs, rns_params, "rns_add_canonical");
  return out;
}

void rns_add_canonical_inplace_cpu(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params) {
  canonical_binary_into<CanonicalBinaryOperation::kAdd>(
      lhs, lhs, rhs, rns_params, "rns_add_canonical");
}

torch::Tensor rns_sub_canonical_cpu(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  canonical_binary_into<CanonicalBinaryOperation::kSubtract>(
      out, lhs, rhs, rns_params, "rns_sub_canonical");
  return out;
}

torch::Tensor rns_montgomery_mul_row_scalars_canonical_cpu(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  constexpr const char* operation = "rns_montgomery_mul_row_scalars_canonical";
  TORCH_CHECK(residues.device().is_cpu(), operation, " requires CPU residues");
  TORCH_CHECK(row_scalars.device() == residues.device() &&
                  rns_params.device() == residues.device(),
              operation,
              " requires all tensors on the operand CPU device");
  TORCH_CHECK(row_scalars.scalar_type() == residues.scalar_type() &&
                  rns_params.scalar_type() == residues.scalar_type(),
              operation,
              " requires all tensors with the operand dtype");
  const auto input = view_rns_batch_3d(residues, "residues");
  check_rns_parameter_rows(input, rns_params, operation);
  check_rns_row_vector(row_scalars, input.size(1), operation, "row_scalars");
  auto out = torch::empty_like(residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    const scalar_t* values = input.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const scalar_t* scalars = row_scalars.data_ptr<scalar_t>();
    const scalar_t* parameters = rns_params.data_ptr<scalar_t>();
    const int64_t limbs = input.size(1);
    const int64_t coefficients = input.size(2);
    const int64_t parameter_row_stride = rns_params.stride(0);
    const int64_t parameter_limb_stride = rns_params.stride(1);
    const int64_t scalar_stride = row_scalars.stride(0);
    const int64_t value_stride0 = input.stride(0);
    const int64_t value_stride1 = input.stride(1);
    const int64_t value_stride2 = input.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = input.size(0) * limbs * coefficients;
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
            const scalar_t row_scalar = scalars[limb * scalar_stride];
            const scalar_t* row_values =
                values + batch * value_stride0 + limb * value_stride1;
            scalar_t* row_result =
                result + batch * result_stride0 + limb * result_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              row_result[coefficient * result_stride2] =
                  fhelium::cpu::canonicalize(
                      fhelium::cpu::multiply(
                          row_values[coefficient * value_stride2],
                          row_scalar,
                          constants),
                      constants.twice_modulus);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

void rns_sub_canonical_inplace_cpu(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params) {
  canonical_binary_into<CanonicalBinaryOperation::kSubtract>(
      lhs, lhs, rhs, rns_params, "rns_sub_canonical");
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CPU, m) {
  m.impl("add_canonical", &rns_add_canonical_cpu);
  m.impl("add_canonical_", &rns_add_canonical_inplace_cpu);
  m.impl("sub_canonical", &rns_sub_canonical_cpu);
  m.impl("sub_canonical_", &rns_sub_canonical_inplace_cpu);
  m.impl("montgomery_mul_row_scalars_canonical",
         &rns_montgomery_mul_row_scalars_canonical_cpu);
}
