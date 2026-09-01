#include <torch/library.h>
#include <torch/torch.h>

#include "rns_standard_arithmetic_cuda.h"

torch::Tensor add_standard(const torch::Tensor lhs,
                           const torch::Tensor rhs,
                           const torch::Tensor rns_params) {
  return rns_add_standard_cuda(lhs, rhs, rns_params);
}

void add_standard_(torch::Tensor lhs,
                   const torch::Tensor rhs,
                   const torch::Tensor rns_params) {
  rns_add_standard_inplace_cuda(lhs, rhs, rns_params);
}

torch::Tensor sub_standard(const torch::Tensor lhs,
                           const torch::Tensor rhs,
                           const torch::Tensor rns_params) {
  return rns_sub_standard_cuda(lhs, rhs, rns_params);
}

void sub_standard_(torch::Tensor lhs,
                   const torch::Tensor rhs,
                   const torch::Tensor rns_params) {
  rns_sub_standard_inplace_cuda(lhs, rhs, rns_params);
}

torch::Tensor montgomery_mul_row_scalars_standard(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_row_scalars_standard_cuda(
      residues, row_scalars, rns_params);
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CUDA, m) {
  m.impl("add_standard", &add_standard);
  m.impl("add_standard_", &add_standard_);
  m.impl("sub_standard", &sub_standard);
  m.impl("sub_standard_", &sub_standard_);
  m.impl("montgomery_mul_row_scalars_standard",
         &montgomery_mul_row_scalars_standard);
}
