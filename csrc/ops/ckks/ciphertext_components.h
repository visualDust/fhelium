#pragma once

#include <torch/torch.h>

#include "../common/rns_batch.h"
#include "../common/rns_parameters.h"

namespace fhelium::ckks {

struct TwoComponentMultiplyInputs {
  torch::Tensor lhs0;
  torch::Tensor lhs1;
  torch::Tensor rhs0;
  torch::Tensor rhs1;
};

struct ThreeComponentMultiplyOutput {
  torch::Tensor packed;
  torch::Tensor d0;
  torch::Tensor d1;
  torch::Tensor d2;
};

inline TwoComponentMultiplyInputs check_two_component_multiply_inputs(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& params,
    const char* operation) {
  TORCH_CHECK(lhs.dim() >= 3,
              operation,
              " requires [2, *batch, limb, ntt_index] operands");
  TORCH_CHECK(lhs.size(0) == 2 && rhs.dim() == lhs.dim() && rhs.size(0) == 2,
              operation,
              " requires exactly two components in both operands");
  TORCH_CHECK(
      lhs.sizes() == rhs.sizes(), operation, " requires equal operand shapes");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kInt32 || lhs.scalar_type() == torch::kInt64,
      operation,
      " supports only int32 and int64 payloads");
  TORCH_CHECK(rhs.scalar_type() == lhs.scalar_type(),
              operation,
              " requires equal operand dtypes");
  TORCH_CHECK(rhs.device() == lhs.device(),
              operation,
              " requires operands on the same execution device");
  TORCH_CHECK(params.scalar_type() == lhs.scalar_type() &&
                  params.device() == lhs.device(),
              operation,
              " requires RNS parameters with the operand dtype and device");

  TwoComponentMultiplyInputs inputs{
      view_rns_batch_3d(lhs.select(0, 0), "lhs_components[0]"),
      view_rns_batch_3d(lhs.select(0, 1), "lhs_components[1]"),
      view_rns_batch_3d(rhs.select(0, 0), "rhs_components[0]"),
      view_rns_batch_3d(rhs.select(0, 1), "rhs_components[1]")};
  check_rns_binary_3d(inputs.lhs0, inputs.lhs1, operation, false);
  check_rns_binary_3d(inputs.lhs0, inputs.rhs0, operation, false);
  check_rns_binary_3d(inputs.lhs0, inputs.rhs1, operation, false);
  check_rns_parameter_rows(inputs.lhs0, params, operation);
  return inputs;
}

inline ThreeComponentMultiplyOutput allocate_three_component_output(
    const torch::Tensor& lhs) {
  auto sizes = lhs.sizes().vec();
  sizes[0] = 3;
  auto packed = torch::empty(sizes, lhs.options());
  return ThreeComponentMultiplyOutput{
      packed,
      view_rns_batch_3d(packed.select(0, 0), "out[0]"),
      view_rns_batch_3d(packed.select(0, 1), "out[1]"),
      view_rns_batch_3d(packed.select(0, 2), "out[2]")};
}

}  // namespace fhelium::ckks
