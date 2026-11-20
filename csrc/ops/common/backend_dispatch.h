#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <optional>

// Device routing for public libfhelium functions. Selection follows the Torch
// dispatcher's CPU/CUDA rule: any CUDA Tensor argument (HIP builds expose the
// CUDA device type) selects the GPU implementation; otherwise all-CPU
// arguments select the CPU implementation. Undefined Tensors do not
// participate.

namespace fhelium::detail {

enum class Backend { kCpu, kGpu, kUnsupported };

struct BackendSelection {
  bool gpu = false;
  bool unsupported = false;
};

inline void collect(BackendSelection& selection, const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return;
  }
  if (tensor.is_cuda()) {
    selection.gpu = true;
  } else if (!tensor.is_cpu()) {
    selection.unsupported = true;
  }
}

inline void collect(BackendSelection& selection,
                    const std::optional<at::Tensor>& tensor) {
  if (tensor.has_value()) {
    collect(selection, *tensor);
  }
}

inline void collect(BackendSelection& selection, at::TensorList tensors) {
  for (const auto& tensor : tensors) {
    collect(selection, tensor);
  }
}

template <typename T>
inline void collect(BackendSelection&, const T&) {}

template <typename... Args>
Backend select_backend(const Args&... args) {
  BackendSelection selection;
  (collect(selection, args), ...);
  if (selection.gpu) {
    return Backend::kGpu;
  }
  return selection.unsupported ? Backend::kUnsupported : Backend::kCpu;
}

[[noreturn]] inline void unsupported_backend(const char* operation,
                                             Backend backend) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              operation,
                              " has no ",
                              backend == Backend::kCpu   ? "CPU"
                              : backend == Backend::kGpu ? "GPU"
                                                         : "matching device",
                              " implementation in this libfhelium build");
}

}  // namespace fhelium::detail

#if defined(FHELIUM_WITH_CPU)
#define FHELIUM_DETAIL_CPU_CASE(op, ...) \
  case ::fhelium::detail::Backend::kCpu: \
    return cpu::op(__VA_ARGS__);
#else
#define FHELIUM_DETAIL_CPU_CASE(op, ...)
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
#define FHELIUM_DETAIL_GPU_CASE(op, ...) \
  case ::fhelium::detail::Backend::kGpu: \
    return gpu::op(__VA_ARGS__);
#else
#define FHELIUM_DETAIL_GPU_CASE(op, ...)
#endif

#if defined(FHELIUM_WITH_CUDA)
#define FHELIUM_DETAIL_CUDA_CASE(op, ...) \
  FHELIUM_DETAIL_GPU_CASE(op, __VA_ARGS__)
#else
#define FHELIUM_DETAIL_CUDA_CASE(op, ...)
#endif

// Route `op` to the implementations compiled for its availability class:
// CPU_GPU (CPU, CUDA, and HIP), GPU (CUDA and HIP), or CUDA (CUDA only).
#define FHELIUM_DISPATCH_IMPL(op, cpu_case, gpu_case, ...)                 \
  {                                                                        \
    const auto fhelium_backend =                                           \
        ::fhelium::detail::select_backend(__VA_ARGS__);                    \
    switch (fhelium_backend) {                                             \
      cpu_case(op, __VA_ARGS__) gpu_case(op, __VA_ARGS__) default : break; \
    }                                                                      \
    ::fhelium::detail::unsupported_backend(FHELIUM_DISPATCH_NAMESPACE #op, \
                                           fhelium_backend);               \
  }

#define FHELIUM_DETAIL_NO_CASE(op, ...)

#define FHELIUM_DISPATCH_CPU_GPU(op, ...) \
  FHELIUM_DISPATCH_IMPL(                  \
      op, FHELIUM_DETAIL_CPU_CASE, FHELIUM_DETAIL_GPU_CASE, __VA_ARGS__)
#define FHELIUM_DISPATCH_GPU(op, ...) \
  FHELIUM_DISPATCH_IMPL(              \
      op, FHELIUM_DETAIL_NO_CASE, FHELIUM_DETAIL_GPU_CASE, __VA_ARGS__)
#define FHELIUM_DISPATCH_CUDA(op, ...) \
  FHELIUM_DISPATCH_IMPL(               \
      op, FHELIUM_DETAIL_NO_CASE, FHELIUM_DETAIL_CUDA_CASE, __VA_ARGS__)
