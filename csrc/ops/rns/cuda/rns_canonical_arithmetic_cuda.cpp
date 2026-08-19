#include <torch/library.h>
#include <torch/torch.h>

#include "rns_canonical_arithmetic_cuda.h"

torch::Tensor add_canonical(const torch::Tensor lhs,
                            const torch::Tensor rhs,
                            const torch::Tensor rns_params) {
  return rns_add_canonical_cuda(lhs, rhs, rns_params);
}

void add_canonical_(torch::Tensor lhs,
                    const torch::Tensor rhs,
                    const torch::Tensor rns_params) {
  rns_add_canonical_inplace_cuda(lhs, rhs, rns_params);
}

torch::Tensor sub_canonical(const torch::Tensor lhs,
                            const torch::Tensor rhs,
                            const torch::Tensor rns_params) {
  return rns_sub_canonical_cuda(lhs, rhs, rns_params);
}

void sub_canonical_(torch::Tensor lhs,
                    const torch::Tensor rhs,
                    const torch::Tensor rns_params) {
  rns_sub_canonical_inplace_cuda(lhs, rhs, rns_params);
}

torch::Tensor montgomery_mul_row_scalars_canonical(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_row_scalars_canonical_cuda(
      residues, row_scalars, rns_params);
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CUDA, m) {
  m.impl("add_canonical", &add_canonical);
  m.impl("add_canonical_", &add_canonical_);
  m.impl("sub_canonical", &sub_canonical);
  m.impl("sub_canonical_", &sub_canonical_);
  m.impl("montgomery_mul_row_scalars_canonical",
         &montgomery_mul_row_scalars_canonical);
}
