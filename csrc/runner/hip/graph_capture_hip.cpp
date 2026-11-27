#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>

#include "../graph_capture.h"

namespace fhelium::exec::detail {

// HIP builds run segments without capture.
bool graph_capture_available() { return false; }

int64_t current_graph_device() {
  int device = 0;
  TORCH_CHECK(hipGetDevice(&device) == hipSuccess, "hipGetDevice failed");
  return device;
}

GraphPool new_graph_pool() { return {0, 0}; }

int64_t graph_pool_bytes(int64_t, GraphPool) { return 0; }

std::unique_ptr<CapturedGraph> capture_graph(int64_t,
                                             GraphPool,
                                             const std::function<void()>&) {
  TORCH_CHECK(false, "Graph capture is not available in HIP builds");
}

}  // namespace fhelium::exec::detail
