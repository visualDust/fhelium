#include <torch/torch.h>

#include "../backends.h"
#include "rns_standard_arithmetic_hip.h"

namespace fhelium::rns::gpu {

at::Tensor add_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  return rns_add_standard_hip(lhs, rhs, rns_params);
}

void add_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  rns_add_standard_inplace_hip(lhs, rhs, rns_params);
}

at::Tensor sub_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  return rns_sub_standard_hip(lhs, rhs, rns_params);
}

void sub_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  rns_sub_standard_inplace_hip(lhs, rhs, rns_params);
}

at::Tensor montgomery_mul_row_scalars_standard(const at::Tensor& residues,
                                               const at::Tensor& row_scalars,
                                               const at::Tensor& rns_params) {
  return rns_montgomery_mul_row_scalars_standard_hip(
      residues, row_scalars, rns_params);
}

at::Tensor sum_standard_batch(const at::Tensor& source,
                              int64_t dim,
                              const at::Tensor& rns_params) {
  return rns_sum_standard_batch_hip(source, dim, rns_params);
}

}  // namespace fhelium::rns::gpu
