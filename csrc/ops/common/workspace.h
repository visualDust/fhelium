#pragma once

// Workspace plans of native operations.
//
// An operation family that needs temporary Tensors declares them in one plan
// function: the plan takes the operation's operands (only their sizes, dtypes
// and devices are read, so Meta and fake Tensors qualify) and places every
// temporary at a byte offset of one workspace. The size query of the family
// returns the plan's byte count, and execution takes each temporary as a view
// of the caller's workspace, so the sizes are written once. Temporaries whose
// lifetimes end are released in stack order and their bytes are reused.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fhelium::detail {

// Byte alignment of the workspace base and of every temporary.
constexpr int64_t kWorkspaceAlignment = 256;

// One temporary of a plan: a contiguous Tensor at `offset` bytes.
struct WorkspaceTensor {
  int64_t offset = -1;
  std::vector<int64_t> sizes;
  c10::ScalarType dtype = c10::ScalarType::Undefined;

  bool defined() const { return offset >= 0; }
  // The temporary as a view of `workspace`.
  at::Tensor in(const at::Tensor& workspace) const {
    TORCH_INTERNAL_ASSERT(defined(), "The plan has no such temporary");
    int64_t elements = 1;
    for (const auto size : sizes) elements *= size;
    return workspace
        .narrow(
            0, offset, elements * static_cast<int64_t>(c10::elementSize(dtype)))
        .view(dtype)
        .view(sizes);
  }
};

// Places the temporaries of a plan.
class WorkspaceLayout {
 public:
  WorkspaceTensor take(std::vector<int64_t> sizes, c10::ScalarType dtype) {
    int64_t bytes = static_cast<int64_t>(c10::elementSize(dtype));
    for (const auto size : sizes) bytes *= size;
    WorkspaceTensor tensor{top_, std::move(sizes), dtype};
    top_ += (bytes + kWorkspaceAlignment - 1) / kWorkspaceAlignment *
            kWorkspaceAlignment;
    peak_ = std::max(peak_, top_);
    return tensor;
  }
  WorkspaceTensor like(const at::Tensor& value) {
    return take(value.sizes().vec(), value.scalar_type());
  }
  // Temporaries taken after `mark()` are released by `release(mark)`.
  int64_t mark() const { return top_; }
  void release(int64_t mark) { top_ = mark; }
  int64_t bytes() const { return peak_; }

 private:
  int64_t top_ = 0;
  int64_t peak_ = 0;
};

// Checks a caller's workspace for a plan of `bytes` on the device of
// `operand`.
inline void check_workspace(const at::Tensor& workspace,
                            int64_t bytes,
                            const at::Tensor& operand,
                            const char* operation) {
  if (bytes == 0) return;
  TORCH_CHECK_VALUE(workspace.defined() && workspace.dim() == 1 &&
                        workspace.scalar_type() == at::kByte &&
                        workspace.is_contiguous() &&
                        workspace.device() == operand.device() &&
                        workspace.numel() >= bytes &&
                        reinterpret_cast<uintptr_t>(workspace.data_ptr()) %
                                kWorkspaceAlignment ==
                            0,
                    operation,
                    " requires a contiguous uint8 workspace of at least ",
                    bytes,
                    " bytes on ",
                    operand.device(),
                    " with a ",
                    kWorkspaceAlignment,
                    "-byte aligned base");
}

// A workspace of `bytes` for a functional form.
inline at::Tensor allocate_workspace(int64_t bytes, const at::Tensor& operand) {
  return at::empty(
      {bytes}, at::TensorOptions().dtype(at::kByte).device(operand.device()));
}

}  // namespace fhelium::detail
