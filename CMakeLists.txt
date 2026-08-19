cmake_minimum_required(VERSION 3.18...3.26)

project(fhelium-csrc LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FHEliumTorch)
include(FHEliumBackends)
fhelium_configure_torch_parallel()

if(FHELIUM_BUILD_CUDA)
  include(FHEliumCuda)
endif()

include(FHEliumNativeTargets)
include(FHEliumNativeInstall)
fhelium_prepare_native_install()

add_fhelium_torch_module(_ops "${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops" torchops)
if(FHELIUM_BUILD_CUDA)
  add_fhelium_cuda_info_module(
    cuda_info "${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime/cuda_info.cpp" cuda)
endif()

fhelium_finalize_native_install()
