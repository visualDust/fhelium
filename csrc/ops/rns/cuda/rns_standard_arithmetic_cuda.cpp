#include <torch/torch.h>

#include "../backends.h"
#include "rns_standard_arithmetic_cuda.h"

namespace fhelium::rns::gpu {

void add_standard_out(const at::Tensor& lhs,
                      const at::Tensor& rhs,
                      const at::Tensor& rns_params,
                      at::Tensor& out) {
  rns_add_standard_out_cuda(lhs, rhs, rns_params, out);
}

void sub_standard_out(const at::Tensor& lhs,
                      const at::Tensor& rhs,
                      const at::Tensor& rns_params,
                      at::Tensor& out) {
  rns_sub_standard_out_cuda(lhs, rhs, rns_params, out);
}

void negate_standard_out(const at::Tensor& source,
                         const at::Tensor& rns_params,
                         at::Tensor& out) {
  rns_negate_standard_out_cuda(source, rns_params, out);
}

void montgomery_mul_row_scalars_standard_out(const at::Tensor& residues,
                                             const at::Tensor& row_scalars,
                                             const at::Tensor& rns_params,
                                             at::Tensor& out) {
  rns_montgomery_mul_row_scalars_standard_out_cuda(
      residues, row_scalars, rns_params, out);
}

void sum_standard_batch_out(const at::Tensor& source,
                            int64_t dim,
                            const at::Tensor& rns_params,
                            at::Tensor& out) {
  rns_sum_standard_batch_out_cuda(source, dim, rns_params, out);
}

}  // namespace fhelium::rns::gpu
