#include "hip_info.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static inline int getAttr(hipDeviceAttribute_t attr, int dev) {
  int v = 0;
  hipError_t st = hipDeviceGetAttribute(&v, attr, dev);
  if (st != hipSuccess) throw std::runtime_error(hipGetErrorString(st));
  return v;
}

static inline void hipCheckError() {
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(e));
  }
}

std::vector<DeviceInfo> query_all_devices() {
  int device_count = 0;
  hipGetDeviceCount(&device_count);

  std::vector<DeviceInfo> devices;
  devices.reserve(device_count);

  for (int i = 0; i < device_count; ++i) {
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, i);

    DeviceInfo info;

    // Basic
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.computeCapability = prop.gcnArchName;

    // Memory
    info.totalGlobalMem = prop.totalGlobalMem;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
    info.totalConstMem = prop.totalConstMem;
    info.l2CacheSize = prop.l2CacheSize;
    info.memoryBusWidth = prop.memoryBusWidth;
    info.memPitch = prop.memPitch;
    info.textureAlignment = prop.textureAlignment;
    info.texturePitchAlignment = prop.texturePitchAlignment;

    // Clocks
    info.memoryClockRate = getAttr(hipDeviceAttributeMemoryClockRate, i);
    info.clockRate = getAttr(hipDeviceAttributeClockRate, i);

    // Execution
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    info.regsPerBlock = prop.regsPerBlock;
    info.regsPerMultiprocessor = prop.regsPerMultiprocessor;
    info.warpSize = prop.warpSize;
    info.asyncEngineCount = prop.asyncEngineCount;

    // Launch config
    info.maxThreadsDim = {
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]};
    info.maxGridSize = {
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]};

    // Unified memory
    info.unifiedAddressing = prop.unifiedAddressing;
    info.managedMemory = prop.managedMemory;
    info.concurrentManagedAccess = prop.concurrentManagedAccess;
    info.canMapHostMemory = prop.canMapHostMemory;

    // Capabilities
    info.deviceOverlap = getAttr(hipDeviceAttributeDeviceOverlap, i);
    info.cooperativeLaunch = getAttr(hipDeviceAttributeCooperativeLaunch, i);
    info.cooperativeMultiDeviceLaunch =
        getAttr(hipDeviceAttributeCooperativeLaunch, i);
    info.isMultiGpuBoard = prop.isMultiGpuBoard;

    devices.push_back(std::move(info));
  }

  return devices;
}

// ========== P2P Info Querying and Bandwidth Testing ==========
// Note: reference implementation from NVIDIA CUDA samples
// https://github.com/NVIDIA/cuda-samples.git
// under Samples/5_Simulations/p2pBandwidthLatencyTest

P2PInfo query_p2p_connectivity(int numDevices) {
  P2PInfo p2pInfo;
  p2pInfo.numDevices = numDevices;
  int matrixSize = numDevices * numDevices;
  p2pInfo.canAccess.resize(matrixSize, false);
  p2pInfo.unidirectionalBandwidthNoP2P.resize(matrixSize, 0.0);
  p2pInfo.unidirectionalBandwidthP2P.resize(matrixSize, 0.0);
  p2pInfo.bidirectionalBandwidthNoP2P.resize(matrixSize, 0.0);
  p2pInfo.bidirectionalBandwidthP2P.resize(matrixSize, 0.0);

  for (int i = 0; i < numDevices; i++) {
    hipSetDevice(i);
    hipCheckError();

    for (int j = 0; j < numDevices; j++) {
      if (i == j) {
        p2pInfo.canAccess[i * numDevices + j] = true;
      } else {
        int access = 0;
        hipDeviceCanAccessPeer(&access, i, j);
        hipCheckError();
        p2pInfo.canAccess[i * numDevices + j] = (access != 0);
      }
    }
  }

  return p2pInfo;
}

// Helper function to perform P2P copy
static void performP2PCopy(int* dest,
                           int destDevice,
                           int* src,
                           int srcDevice,
                           int numElems,
                           int repeat,
                           bool p2paccess,
                           hipStream_t stream) {
  if (p2paccess) {
    for (int r = 0; r < repeat; r++) {
      hipMemcpyPeerAsync(
          dest, destDevice, src, srcDevice, sizeof(int) * numElems, stream);
    }
  } else {
    for (int r = 0; r < repeat; r++) {
      hipMemcpyPeerAsync(
          dest, destDevice, src, srcDevice, sizeof(int) * numElems, stream);
    }
  }
}

void test_p2p_bandwidth(P2PInfo& p2pInfo, int numElems) {
  int numGPUs = p2pInfo.numDevices;
  if (numGPUs < 2) {
    return;  // No P2P testing needed for single device
  }

  int repeat = 5;
  std::vector<int*> buffers(numGPUs);
  std::vector<int*> buffersD2D(numGPUs);
  std::vector<hipEvent_t> start(numGPUs);
  std::vector<hipEvent_t> stop(numGPUs);
  std::vector<hipStream_t> stream(numGPUs);
  std::vector<hipStream_t> stream0(numGPUs);
  std::vector<hipStream_t> stream1(numGPUs);

  // Allocate resources
  for (int d = 0; d < numGPUs; d++) {
    hipSetDevice(d);
    hipStreamCreateWithFlags(&stream[d], hipStreamNonBlocking);
    hipMalloc(&buffers[d], numElems * sizeof(int));
    hipMemset(buffers[d], 0, numElems * sizeof(int));
    hipMalloc(&buffersD2D[d], numElems * sizeof(int));
    hipMemset(buffersD2D[d], 0, numElems * sizeof(int));
    hipEventCreate(&start[d]);
    hipEventCreate(&stop[d]);
    hipCheckError();

    // For bidirectional
    hipStreamCreateWithFlags(&stream0[d], hipStreamNonBlocking);
    hipStreamCreateWithFlags(&stream1[d], hipStreamNonBlocking);
    hipCheckError();
  }

  // Test unidirectional bandwidth without P2P
  for (int i = 0; i < numGPUs; i++) {
    hipSetDevice(i);
    hipStreamSynchronize(stream[i]);
    hipCheckError();

    for (int j = 0; j < numGPUs; j++) {
      hipEventRecord(start[i], stream[i]);
      hipCheckError();

      if (i == j) {
        performP2PCopy(buffers[i],
                       i,
                       buffersD2D[i],
                       i,
                       numElems,
                       repeat,
                       false,
                       stream[i]);
      } else {
        performP2PCopy(
            buffers[j], j, buffers[i], i, numElems, repeat, false, stream[i]);
      }

      hipEventRecord(stop[i], stream[i]);
      hipCheckError();
      hipStreamSynchronize(stream[i]);
      hipCheckError();

      float time_ms;
      hipEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = numElems * sizeof(int) * repeat / 1e9;
      if (i == j) {
        gb *= 2;  // count both read and write
      }
      p2pInfo.unidirectionalBandwidthNoP2P[i * numGPUs + j] = gb / time_s;
    }
  }

  // Test unidirectional bandwidth with P2P
  for (int i = 0; i < numGPUs; i++) {
    hipSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      bool access = p2pInfo.canAccess[i * numGPUs + j];
      bool peerAccessEnabled = false;
      if (access && i != j) {
        hipSetDevice(i);
        hipError_t err = hipDeviceEnablePeerAccess(j, 0);
        if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
          hipGetLastError();  // Clear error if already enabled
          hipSetDevice(j);
          err = hipDeviceEnablePeerAccess(i, 0);
          if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
            hipGetLastError();  // Clear error if already enabled
            peerAccessEnabled = true;
          }
          hipSetDevice(i);
        } else {
          hipGetLastError();  // Clear error
        }
      }

      hipStreamSynchronize(stream[i]);
      hipCheckError();

      hipEventRecord(start[i], stream[i]);
      hipCheckError();

      if (i == j) {
        performP2PCopy(buffers[i],
                       i,
                       buffersD2D[i],
                       i,
                       numElems,
                       repeat,
                       access,
                       stream[i]);
      } else {
        performP2PCopy(
            buffers[j], j, buffers[i], i, numElems, repeat, access, stream[i]);
      }

      hipEventRecord(stop[i], stream[i]);
      hipCheckError();
      hipStreamSynchronize(stream[i]);
      hipCheckError();

      float time_ms;
      hipEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = numElems * sizeof(int) * repeat / 1e9;
      if (i == j) {
        gb *= 2;
      }
      p2pInfo.unidirectionalBandwidthP2P[i * numGPUs + j] = gb / time_s;

      if (peerAccessEnabled && i != j) {
        hipSetDevice(i);
        hipError_t err = hipDeviceDisablePeerAccess(j);
        if (err != hipSuccess && err != hipErrorPeerAccessNotEnabled) {
          hipCheckError();
        } else {
          hipGetLastError();  // Clear error
        }
        hipSetDevice(j);
        err = hipDeviceDisablePeerAccess(i);
        if (err != hipSuccess && err != hipErrorPeerAccessNotEnabled) {
          hipCheckError();
        } else {
          hipGetLastError();  // Clear error
        }
        hipSetDevice(i);
      }
    }
  }

  // Test bidirectional bandwidth without P2P
  for (int i = 0; i < numGPUs; i++) {
    hipSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      hipStreamSynchronize(stream0[i]);
      hipStreamSynchronize(stream1[j]);
      hipCheckError();

      hipEventRecord(start[i], stream0[i]);
      hipCheckError();

      if (i == j) {
        performP2PCopy(buffers[i],
                       i,
                       buffersD2D[i],
                       i,
                       numElems,
                       repeat,
                       false,
                       stream0[i]);
        performP2PCopy(buffersD2D[i],
                       i,
                       buffers[i],
                       i,
                       numElems,
                       repeat,
                       false,
                       stream1[i]);
      } else {
        performP2PCopy(
            buffers[i], i, buffers[j], j, numElems, repeat, false, stream1[j]);
        performP2PCopy(
            buffers[j], j, buffers[i], i, numElems, repeat, false, stream0[i]);
      }

      hipEventRecord(stop[j], stream1[j]);
      hipStreamWaitEvent(stream0[i], stop[j], 0);
      hipEventRecord(stop[i], stream0[i]);
      hipCheckError();

      hipStreamSynchronize(stream0[i]);
      hipStreamSynchronize(stream1[j]);
      hipCheckError();

      float time_ms;
      hipEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * repeat / 1e9;
      if (i == j) {
        gb *= 2;
      }
      p2pInfo.bidirectionalBandwidthNoP2P[i * numGPUs + j] = gb / time_s;
    }
  }

  // Test bidirectional bandwidth with P2P
  for (int i = 0; i < numGPUs; i++) {
    hipSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      bool access = p2pInfo.canAccess[i * numGPUs + j];
      bool peerAccessEnabled = false;
      if (access && i != j) {
        hipSetDevice(i);
        hipError_t err = hipDeviceEnablePeerAccess(j, 0);
        if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
          hipGetLastError();  // Clear error if already enabled
          hipSetDevice(j);
          err = hipDeviceEnablePeerAccess(i, 0);
          if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
            hipGetLastError();  // Clear error if already enabled
            peerAccessEnabled = true;
          }
          hipSetDevice(i);
        } else {
          hipGetLastError();  // Clear error
        }
      }

      hipStreamSynchronize(stream0[i]);
      hipStreamSynchronize(stream1[j]);
      hipCheckError();

      hipEventRecord(start[i], stream0[i]);
      hipCheckError();

      if (i == j) {
        performP2PCopy(buffers[i],
                       i,
                       buffersD2D[i],
                       i,
                       numElems,
                       repeat,
                       access,
                       stream0[i]);
        performP2PCopy(buffersD2D[i],
                       i,
                       buffers[i],
                       i,
                       numElems,
                       repeat,
                       access,
                       stream1[i]);
      } else {
        performP2PCopy(
            buffers[i], i, buffers[j], j, numElems, repeat, access, stream1[j]);
        performP2PCopy(
            buffers[j], j, buffers[i], i, numElems, repeat, access, stream0[i]);
      }

      hipEventRecord(stop[j], stream1[j]);
      hipStreamWaitEvent(stream0[i], stop[j], 0);
      hipEventRecord(stop[i], stream0[i]);
      hipCheckError();

      hipStreamSynchronize(stream0[i]);
      hipStreamSynchronize(stream1[j]);
      hipCheckError();

      float time_ms;
      hipEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * repeat / 1e9;
      if (i == j) {
        gb *= 2;
      }
      p2pInfo.bidirectionalBandwidthP2P[i * numGPUs + j] = gb / time_s;

      if (peerAccessEnabled && i != j) {
        hipSetDevice(i);
        hipError_t err = hipDeviceDisablePeerAccess(j);
        if (err != hipSuccess && err != hipErrorPeerAccessNotEnabled) {
          hipCheckError();
        } else {
          hipGetLastError();  // Clear error
        }
        hipSetDevice(j);
        err = hipDeviceDisablePeerAccess(i);
        if (err != hipSuccess && err != hipErrorPeerAccessNotEnabled) {
          hipCheckError();
        } else {
          hipGetLastError();  // Clear error
        }
        hipSetDevice(i);
      }
    }
  }

  // Cleanup
  for (int d = 0; d < numGPUs; d++) {
    hipSetDevice(d);
    hipFree(buffers[d]);
    hipFree(buffersD2D[d]);
    hipEventDestroy(start[d]);
    hipEventDestroy(stop[d]);
    hipStreamDestroy(stream[d]);
    hipStreamDestroy(stream0[d]);
    hipStreamDestroy(stream1[d]);
    hipCheckError();
  }
}

P2PInfo query_p2p_info(int numDevices, bool testBandwidth) {
  P2PInfo p2pInfo = query_p2p_connectivity(numDevices);
  if (testBandwidth && numDevices > 1) {
    test_p2p_bandwidth(p2pInfo);
  }
  return p2pInfo;
}

CudaSystemInfo query_cuda_system_info(bool testP2PBandwidth) {
  CudaSystemInfo sysInfo;
  sysInfo.devices = query_all_devices();
  int numDevices = static_cast<int>(sysInfo.devices.size());
  sysInfo.p2pInfo = query_p2p_info(numDevices, testP2PBandwidth);
  return sysInfo;
}

// Wrapper functions to convert structures to Python dictionaries

static py::dict DeviceInfo_to_dict(const DeviceInfo& d) {
  py::dict out;

  out["name"] = d.name;
  out["major"] = d.major;
  out["minor"] = d.minor;
  out["computeCapability"] = d.computeCapability;

  out["totalGlobalMem"] = d.totalGlobalMem;
  out["sharedMemPerBlock"] = d.sharedMemPerBlock;
  out["sharedMemPerMultiprocessor"] = d.sharedMemPerMultiprocessor;
  out["totalConstMem"] = d.totalConstMem;
  out["l2CacheSize"] = d.l2CacheSize;
  out["memoryBusWidth"] = d.memoryBusWidth;
  out["memPitch"] = d.memPitch;
  out["textureAlignment"] = d.textureAlignment;
  out["texturePitchAlignment"] = d.texturePitchAlignment;

  out["memoryClockRate"] = d.memoryClockRate;
  out["clockRate"] = d.clockRate;

  out["multiProcessorCount"] = d.multiProcessorCount;
  out["maxThreadsPerBlock"] = d.maxThreadsPerBlock;
  out["maxThreadsPerMultiProcessor"] = d.maxThreadsPerMultiProcessor;
  out["regsPerBlock"] = d.regsPerBlock;
  out["regsPerMultiprocessor"] = d.regsPerMultiprocessor;
  out["warpSize"] = d.warpSize;
  out["asyncEngineCount"] = d.asyncEngineCount;

  out["maxThreadsDim"] = d.maxThreadsDim;
  out["maxGridSize"] = d.maxGridSize;

  out["unifiedAddressing"] = d.unifiedAddressing;
  out["managedMemory"] = d.managedMemory;
  out["concurrentManagedAccess"] = d.concurrentManagedAccess;
  out["canMapHostMemory"] = d.canMapHostMemory;

  out["deviceOverlap"] = d.deviceOverlap;
  out["cooperativeLaunch"] = d.cooperativeLaunch;
  out["cooperativeMultiDeviceLaunch"] = d.cooperativeMultiDeviceLaunch;
  out["isMultiGpuBoard"] = d.isMultiGpuBoard;

  return out;
}

static py::dict P2PInfo_to_dict(const P2PInfo& p2p) {
  py::dict out;

  out["numDevices"] = p2p.numDevices;

  // Convert connectivity matrix to nested list
  py::list canAccessList;
  for (int i = 0; i < p2p.numDevices; i++) {
    py::list row;
    for (int j = 0; j < p2p.numDevices; j++) {
      row.append(p2p.canAccess[i * p2p.numDevices + j]);
    }
    canAccessList.append(row);
  }
  out["canAccess"] = canAccessList;

  // Convert bandwidth matrices to nested lists
  auto matrix_to_list = [&p2p](const std::vector<double>& matrix) {
    py::list result;
    for (int i = 0; i < p2p.numDevices; i++) {
      py::list row;
      for (int j = 0; j < p2p.numDevices; j++) {
        row.append(matrix[i * p2p.numDevices + j]);
      }
      result.append(row);
    }
    return result;
  };

  out["unidirectionalBandwidthNoP2P"] =
      matrix_to_list(p2p.unidirectionalBandwidthNoP2P);
  out["unidirectionalBandwidthP2P"] =
      matrix_to_list(p2p.unidirectionalBandwidthP2P);
  out["bidirectionalBandwidthNoP2P"] =
      matrix_to_list(p2p.bidirectionalBandwidthNoP2P);
  out["bidirectionalBandwidthP2P"] =
      matrix_to_list(p2p.bidirectionalBandwidthP2P);

  return out;
}

static py::dict CudaSystemInfo_to_dict(const CudaSystemInfo& sysInfo) {
  py::dict out;

  // Convert devices
  py::dict devicesDict;
  for (size_t i = 0; i < sysInfo.devices.size(); ++i) {
    devicesDict[py::int_(i)] = DeviceInfo_to_dict(sysInfo.devices[i]);
  }
  out["devices"] = devicesDict;

  // Convert P2P info
  out["p2p"] = P2PInfo_to_dict(sysInfo.p2pInfo);

  return out;
}

// Python interface functions

py::dict get_cuda_device_properties_py() {
  auto devices = query_all_devices();

  py::dict result;
  for (size_t i = 0; i < devices.size(); ++i) {
    result[py::int_(i)] = DeviceInfo_to_dict(devices[i]);
  }
  return result;
}

py::dict get_cuda_system_info_py(bool testP2PBandwidth = true) {
  CudaSystemInfo sysInfo = query_cuda_system_info(testP2PBandwidth);
  return CudaSystemInfo_to_dict(sysInfo);
}

PYBIND11_MODULE(cuda_info, m) {
  m.def("get_cuda_device_properties",
        &get_cuda_device_properties_py,
        "Return detailed CUDA device properties");

  m.def(
      "get_cuda_system_info",
      &get_cuda_system_info_py,
      "Return complete CUDA system information including devices and P2P info",
      py::arg("testP2PBandwidth") = true);
}
