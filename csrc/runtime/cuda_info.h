#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

struct DeviceInfo {
  std::string name;
  int major;
  int minor;
  std::string computeCapability;

  // Memory
  uint64_t totalGlobalMem;
  int sharedMemPerBlock;
  int sharedMemPerMultiprocessor;
  int totalConstMem;
  int l2CacheSize;
  int memoryBusWidth;
  size_t memPitch;
  size_t textureAlignment;
  size_t texturePitchAlignment;

  // Clocks
  int memoryClockRate;
  int clockRate;

  // Execution
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int regsPerBlock;
  int regsPerMultiprocessor;
  int warpSize;
  int asyncEngineCount;

  // Launch configuration
  std::vector<int> maxThreadsDim;
  std::vector<int> maxGridSize;

  // Unified memory
  bool unifiedAddressing;
  bool managedMemory;
  bool concurrentManagedAccess;
  bool canMapHostMemory;

  // Capabilities
  bool deviceOverlap;
  bool cooperativeLaunch;
  bool cooperativeMultiDeviceLaunch;
  bool isMultiGpuBoard;
};

// P2P connectivity and bandwidth information
struct P2PInfo {
  // Connectivity matrix: canAccess[i * numDevices + j] indicates if device i
  // can access device j
  std::vector<bool> canAccess;

  // Unidirectional bandwidth matrix (GB/s) with P2P disabled
  std::vector<double> unidirectionalBandwidthNoP2P;

  // Unidirectional bandwidth matrix (GB/s) with P2P enabled
  std::vector<double> unidirectionalBandwidthP2P;

  // Bidirectional bandwidth matrix (GB/s) with P2P disabled
  std::vector<double> bidirectionalBandwidthNoP2P;

  // Bidirectional bandwidth matrix (GB/s) with P2P enabled
  std::vector<double> bidirectionalBandwidthP2P;

  int numDevices;
};

// Complete CUDA system information
struct CudaSystemInfo {
  std::vector<DeviceInfo> devices;
  P2PInfo p2pInfo;
};

// Query all devices
std::vector<DeviceInfo> query_all_devices();

// Query P2P connectivity (without bandwidth testing)
P2PInfo query_p2p_connectivity(int numDevices);

// Test P2P bandwidth between devices
void test_p2p_bandwidth(P2PInfo& p2pInfo, int numElems = 40000000);

// Query complete P2P information (connectivity + bandwidth)
P2PInfo query_p2p_info(int numDevices, bool testBandwidth = true);

// Query complete CUDA system information
CudaSystemInfo query_cuda_system_info(bool testP2PBandwidth = true);
