#pragma once

#include <ATen/MemoryOverlap.h>
#include <torch/torch.h>

// Public dense RNS operands use [..., limb, coefficient] layout. Native
// implementations flatten only the homogeneous batch prefix into a
// metadata-only [batch, limb, coefficient] view. Tensor::view rejects layouts
// that cannot be collapsed without copying; callers must make any required
// repacking visible to the caller.
inline torch::Tensor view_rns_batch_3d(const torch::Tensor& tensor,
                                       const char* name) {
  TORCH_CHECK(tensor.dim() >= 2,
              name,
              " must have [..., limb, coefficient] layout; got rank ",
              tensor.dim());
  TORCH_CHECK(tensor.size(-2) > 0, name, " must contain at least one RNS limb");
  TORCH_CHECK(
      tensor.size(-1) > 0, name, " must contain at least one coefficient");
  for (int64_t axis = 0; axis < tensor.dim() - 2; ++axis) {
    TORCH_CHECK(
        tensor.size(axis) > 0, name, " batch dimensions must be nonzero");
  }
  return tensor.view({-1, tensor.size(-2), tensor.size(-1)});
}

inline torch::Tensor view_coefficient_batch_2d(const torch::Tensor& tensor,
                                               const char* name) {
  TORCH_CHECK(tensor.dim() >= 1,
              name,
              " must have [..., coefficient] layout; got rank ",
              tensor.dim());
  TORCH_CHECK(
      tensor.size(-1) > 0, name, " must contain at least one coefficient");
  for (int64_t axis = 0; axis < tensor.dim() - 1; ++axis) {
    TORCH_CHECK(
        tensor.size(axis) > 0, name, " batch dimensions must be nonzero");
  }
  return tensor.view({-1, tensor.size(-1)});
}

inline void check_rns_binary_3d(const torch::Tensor& lhs,
                                const torch::Tensor& rhs,
                                const char* operation,
                                bool allow_rhs_singleton_batch) {
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3,
              operation,
              " requires canonical [batch, limb, coefficient] operands");
  TORCH_CHECK(lhs.size(1) == rhs.size(1),
              operation,
              " operand limb counts differ: ",
              lhs.size(1),
              " vs ",
              rhs.size(1));
  TORCH_CHECK(lhs.size(2) == rhs.size(2),
              operation,
              " operand coefficient counts differ: ",
              lhs.size(2),
              " vs ",
              rhs.size(2));
  TORCH_CHECK(rhs.size(0) == lhs.size(0) ||
                  (allow_rhs_singleton_batch && rhs.size(0) == 1),
              operation,
              " operand batch counts differ: ",
              lhs.size(0),
              " vs ",
              rhs.size(0));
}

inline void check_rns_parameter_rows(const torch::Tensor& operand,
                                     const torch::Tensor& rns_params,
                                     const char* operation) {
  TORCH_CHECK(rns_params.dim() == 2,
              operation,
              " requires [parameter, limb] RNS parameters");
  TORCH_CHECK(operand.dim() == 3,
              operation,
              " requires a canonical [batch, limb, coefficient] operand");
  TORCH_CHECK(operand.size(1) == rns_params.size(1),
              operation,
              " operand and parameter limb counts differ: ",
              operand.size(1),
              " vs ",
              rns_params.size(1));
}

inline void check_mutable_rns_output(const torch::Tensor& out,
                                     const torch::Tensor& read_only_operand,
                                     const torch::Tensor& rns_params) {
  // Parallel native kernels require every logical output element to own a
  // distinct storage location. The in-place schema permits out to alias its
  // lhs input, but read-only operands and parameter tables must remain
  // independent of the storage being mutated.
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, read_only_operand);
  at::assert_no_overlap(out, rns_params);
}

inline void check_rns_row_vector(const torch::Tensor& values,
                                 int64_t limb_count,
                                 const char* operation,
                                 const char* name) {
  TORCH_CHECK(values.dim() == 1,
              operation,
              " requires ",
              name,
              " to be a rank-one limb vector");
  TORCH_CHECK(values.size(0) == limb_count,
              operation,
              " ",
              name,
              " count must match limbs: ",
              values.size(0),
              " vs ",
              limb_count);
}
