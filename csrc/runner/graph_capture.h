#pragma once

// Device graph capture for Runner segments. The CUDA implementation captures
// with the CUDA runtime stream-capture API and routes the captured allocations
// into a private pool of the c10 CUDA caching allocator; builds without a GPU
// backend report that capture is unavailable and run segments directly.

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace fhelium::exec::detail {

// A private allocator pool shared by the graphs of the Runners of one Arena
// and device, or of one Runner and device.
using GraphPool = std::pair<unsigned long long, unsigned long long>;

class CapturedGraph {
 public:
  virtual ~CapturedGraph() = default;
  // Replays on the current stream of the captured device.
  virtual void replay() = 0;
  // Facts recorded by capture_graph: graph nodes and host seconds spent in
  // stream capture and in instantiation.
  int64_t nodes = 0;
  double capture_seconds = 0;
  double instantiate_seconds = 0;
};

bool graph_capture_available();
// The current device index of the GPU backend.
int64_t current_graph_device();
GraphPool new_graph_pool();
// Device memory that the caching allocator holds for `pool` on `device`.
int64_t graph_pool_bytes(int64_t device, GraphPool pool);
// Captures the work that `body` enqueues on `device` into a graph whose
// allocations come from `pool`. Capture runs on the capture stream of
// `device`, one stream per device shared by all captures, which waits for the
// current stream; the current stream waits for the capture stream afterwards.
// Graphs that share a pool reuse the pool blocks freed by each other's
// captures, so a graph's temporaries may hold another graph's data between
// replays; only values whose lifetime ends within a replay may stay in pool
// memory.
std::unique_ptr<CapturedGraph> capture_graph(int64_t device,
                                             GraphPool pool,
                                             const std::function<void()>& body);

}  // namespace fhelium::exec::detail
