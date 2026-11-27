#include <c10/core/Event.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>

#include "../graph_capture.h"

namespace fhelium::exec::detail {

namespace {

namespace allocator = c10::cuda::CUDACachingAllocator;

// An instantiated CUDA graph that holds one reference to its allocator pool.
class CudaGraph final : public CapturedGraph {
 public:
  CudaGraph(c10::DeviceIndex device, GraphPool pool, cudaGraphExec_t exec)
      : device_(device), pool_(pool), exec_(exec) {}
  ~CudaGraph() override {
    c10::cuda::CUDAGuard guard(device_);
    allocator::releasePool(device_, pool_);
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(exec_));
  }
  void replay() override {
    c10::cuda::CUDAGuard guard(device_);
    C10_CUDA_CHECK(
        cudaGraphLaunch(exec_, c10::cuda::getCurrentCUDAStream(device_)));
  }

 private:
  c10::DeviceIndex device_;
  GraphPool pool_;
  cudaGraphExec_t exec_;
};

// One capture stream per device. The caching allocator reuses a freed pool
// block only for allocations on the stream that freed it, so graphs that
// share a pool reuse each other's temporaries only when they are captured on
// one stream.
c10::cuda::CUDAStream capture_stream(c10::DeviceIndex device) {
  static std::mutex mutex;
  static std::map<c10::DeviceIndex, c10::cuda::CUDAStream> streams;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = streams.find(device);
  if (found == streams.end())
    found = streams.emplace(device, c10::cuda::getStreamFromPool(false, device))
                .first;
  return found->second;
}

std::mutex& capture_mutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace

bool graph_capture_available() { return true; }

int64_t current_graph_device() { return c10::cuda::current_device(); }

GraphPool new_graph_pool() {
  // PyTorch numbers its graph pools with small counters; the top bit keeps
  // these identifiers out of that range.
  static std::atomic<unsigned long long> next{1};
  return {0, (1ULL << 63) | next.fetch_add(1)};
}

int64_t graph_pool_bytes(int64_t device, GraphPool pool) {
  int64_t total = 0;
  for (const auto& segment : allocator::snapshot(pool, false).segments)
    if (segment.device == device && segment.owner_private_pool_id == pool)
      total += static_cast<int64_t>(segment.total_size);
  return total;
}

std::unique_ptr<CapturedGraph> capture_graph(
    int64_t device, GraphPool pool, const std::function<void()>& body) {
  const auto index = static_cast<c10::DeviceIndex>(device);
  c10::cuda::CUDAGuard guard(index);
  const auto current = c10::cuda::getCurrentCUDAStream(index);
  // Captures share the stream of their device and run one at a time.
  std::lock_guard<std::mutex> capturing(capture_mutex());
  const auto side = capture_stream(index);
  c10::Event ready(c10::DeviceType::CUDA);
  ready.record(current);
  ready.block(side);

  c10::cuda::CUDAStreamGuard stream_guard(side);
  unsigned long long capture_id = 0;
  // Allocations of the capturing stream come from `pool`; other streams keep
  // the default pools.
  allocator::beginAllocateToPool(index, pool, [&capture_id](cudaStream_t s) {
    cudaStreamCaptureStatus status;
    unsigned long long id;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, &id));
    return status == cudaStreamCaptureStatusActive && id == capture_id;
  });
  cudaGraph_t graph = nullptr;
  using Clock = std::chrono::steady_clock;
  const auto seconds = [](Clock::time_point since) {
    return std::chrono::duration<double>(Clock::now() - since).count();
  };
  const auto capture_start = Clock::now();
  const auto end_capture = [&] {
    const auto status = cudaStreamEndCapture(side, &graph);
    allocator::markCaptureEnd(index);
    allocator::endAllocateToPool(index, pool);
    return status;
  };
  const auto begin = cudaStreamBeginCapture(side, cudaStreamCaptureModeGlobal);
  if (begin != cudaSuccess) {
    allocator::endAllocateToPool(index, pool);
    allocator::releasePool(index, pool);
    C10_CUDA_CHECK(begin);
  }
  allocator::markCaptureBegin(index);
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(side, &status, &capture_id));
  try {
    body();
  } catch (...) {
    // End the capture so that the stream leaves capture mode before the
    // error propagates.
    end_capture();
    if (graph != nullptr) cudaGraphDestroy(graph);
    allocator::releasePool(index, pool);
    (void)cudaGetLastError();
    throw;
  }
  const auto ended = end_capture();
  const auto capture_seconds = seconds(capture_start);
  cudaGraphExec_t exec = nullptr;
  auto instantiated = ended;
  size_t nodes = 0;
  const auto instantiate_start = Clock::now();
  if (ended == cudaSuccess) {
    C10_CUDA_CHECK_WARN(cudaGraphGetNodes(graph, nullptr, &nodes));
    instantiated = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph));
  }
  const auto instantiate_seconds = seconds(instantiate_start);
  if (instantiated != cudaSuccess) {
    allocator::releasePool(index, pool);
    C10_CUDA_CHECK(instantiated);
  }
  auto captured = std::make_unique<CudaGraph>(index, pool, exec);
  captured->nodes = static_cast<int64_t>(nodes);
  captured->capture_seconds = capture_seconds;
  captured->instantiate_seconds = instantiate_seconds;

  c10::Event done(c10::DeviceType::CUDA);
  done.record(side);
  done.block(current);
  return captured;
}

}  // namespace fhelium::exec::detail
