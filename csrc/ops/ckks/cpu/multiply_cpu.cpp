#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "../../common/cpu/montgomery.h"
#include "../ciphertext_components.h"

namespace {

constexpr const char* kOperation = "multiply_two_component_ntt_montgomery";

template <typename scalar_t>
void multiply_two_component_loop(
    const fhelium::ckks::TwoComponentMultiplyInputs& inputs,
    const fhelium::ckks::ThreeComponentMultiplyOutput& output,
    const torch::Tensor& params) {
  const scalar_t* lhs0 = inputs.lhs0.data_ptr<scalar_t>();
  const scalar_t* lhs1 = inputs.lhs1.data_ptr<scalar_t>();
  const scalar_t* rhs0 = inputs.rhs0.data_ptr<scalar_t>();
  const scalar_t* rhs1 = inputs.rhs1.data_ptr<scalar_t>();
  scalar_t* d0 = output.d0.data_ptr<scalar_t>();
  scalar_t* d1 = output.d1.data_ptr<scalar_t>();
  scalar_t* d2 = output.d2.data_ptr<scalar_t>();
  const scalar_t* parameters = params.data_ptr<scalar_t>();
  const int64_t batches = inputs.lhs0.size(0);
  const int64_t limbs = inputs.lhs0.size(1);
  const int64_t coefficients = inputs.lhs0.size(2);
  const int64_t parameter_row_stride = params.stride(0);
  const int64_t parameter_limb_stride = params.stride(1);
  const int64_t elements = batches * limbs * coefficients;

  at::parallel_for(
      0,
      elements,
      fhelium::cpu::adaptive_grain(elements),
      [&](const int64_t begin, const int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
          const int64_t batch_limb = index / coefficients;
          const int64_t limb = batch_limb % limbs;
          const int64_t batch = batch_limb / limbs;
          const int64_t coefficient = index - batch_limb * coefficients;
          const int64_t lhs0_offset = batch * inputs.lhs0.stride(0) +
                                      limb * inputs.lhs0.stride(1) +
                                      coefficient * inputs.lhs0.stride(2);
          const int64_t lhs1_offset = batch * inputs.lhs1.stride(0) +
                                      limb * inputs.lhs1.stride(1) +
                                      coefficient * inputs.lhs1.stride(2);
          const int64_t rhs0_offset = batch * inputs.rhs0.stride(0) +
                                      limb * inputs.rhs0.stride(1) +
                                      coefficient * inputs.rhs0.stride(2);
          const int64_t rhs1_offset = batch * inputs.rhs1.stride(0) +
                                      limb * inputs.rhs1.stride(1) +
                                      coefficient * inputs.rhs1.stride(2);
          const int64_t out_offset = batch * output.d0.stride(0) +
                                     limb * output.d0.stride(1) +
                                     coefficient * output.d0.stride(2);
          const auto constants = fhelium::cpu::load_constants(
              parameters, parameter_row_stride, parameter_limb_stride, limb);
          const scalar_t a0 = lhs0[lhs0_offset];
          const scalar_t a1 = lhs1[lhs1_offset];
          const scalar_t b0 = rhs0[rhs0_offset];
          const scalar_t b1 = rhs1[rhs1_offset];
          d0[out_offset] = fhelium::cpu::multiply(a0, b0, constants);
          const scalar_t cross01 = fhelium::cpu::multiply(a0, b1, constants);
          const scalar_t cross10 = fhelium::cpu::multiply(a1, b0, constants);
          d1[out_offset] =
              fhelium::cpu::add_lazy(cross01, cross10, constants.twice_modulus);
          d2[out_offset] = fhelium::cpu::multiply(a1, b1, constants);
        }
      });
}

torch::Tensor multiply_two_component_cpu(const torch::Tensor lhs,
                                         const torch::Tensor rhs,
                                         const torch::Tensor params) {
  TORCH_CHECK(lhs.device().is_cpu(), kOperation, " requires CPU operands");
  const auto inputs = fhelium::ckks::check_two_component_multiply_inputs(
      lhs, rhs, params, kOperation);
  const auto output = fhelium::ckks::allocate_three_component_output(lhs);
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), kOperation, [&] {
    multiply_two_component_loop<scalar_t>(inputs, output, params);
  });
  return output.packed;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  m.impl("multiply_two_component_ntt_montgomery", &multiply_two_component_cpu);
}
