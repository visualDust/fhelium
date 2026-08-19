# Configure the CUDA toolkit and Torch's CUDA CMake targets.
#
# This module runs in the including directory scope and defines the CUDA
# variables consumed by the native operator targets in CMakeLists.txt.

find_package(
  Python3
  COMPONENTS Development.Module
  REQUIRED)

# Preserve the requested architecture list before TorchConfig modifies CUDA
# flags in the current configure scope.
set(_FHELIUM_REQUESTED_CUDA_ARCHITECTURES "")
if(DEFINED ENV{CUDAARCHS})
  set(_FHELIUM_REQUESTED_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES
                                            STREQUAL "OFF")
  set(_FHELIUM_REQUESTED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Select nvcc through standard CMake inputs, then the conventional CUDA_HOME, a
# /usr/local toolkit matching the selected Torch build, Conda, and finally
# /usr/local/cuda. CMake owns compiler identity after the first configure.
set(_FHELIUM_VERSIONED_CUDA_ROOT
    "/usr/local/cuda-${FHELIUM_TORCH_CUDA_VERSION}")
if(NOT CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}")
  elseif(CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
  elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
  elseif(EXISTS "${_FHELIUM_VERSIONED_CUDA_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "${_FHELIUM_VERSIONED_CUDA_ROOT}/bin/nvcc")
    set(CUDAToolkit_ROOT "${_FHELIUM_VERSIONED_CUDA_ROOT}")
  elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}")
  elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CUDAToolkit_ROOT "/usr/local/cuda")
  endif()
endif()

if(CMAKE_CUDA_COMPILER)
  set(CUDA_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# TorchConfig still uses legacy FindCUDA. Reuse the modern discovery result so
# toolkits that store libcudart under targets/<triple>/lib need no extra build
# variables.
get_filename_component(_FHELIUM_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_CUDA_ROOT "${_FHELIUM_CUDA_BIN_DIR}" DIRECTORY)
get_target_property(_FHELIUM_CUDART CUDA::cudart IMPORTED_LOCATION)
set(CUDA_TOOLKIT_ROOT_DIR
    "${_CUDA_ROOT}"
    CACHE PATH "CUDA toolkit root for TorchConfig" FORCE)
set(CUDA_INCLUDE_DIRS
    "${CUDAToolkit_INCLUDE_DIRS}"
    CACHE STRING "CUDA includes for TorchConfig" FORCE)
set(CUDA_CUDART_LIBRARY
    "${_FHELIUM_CUDART}"
    CACHE FILEPATH "CUDA runtime for TorchConfig" FORCE)

string(REGEX MATCH "^[0-9]+" FHELIUM_TORCH_CUDA_MAJOR
             "${FHELIUM_TORCH_CUDA_VERSION}")
string(REGEX MATCH "^[0-9]+" FHELIUM_TOOLKIT_CUDA_MAJOR
             "${CUDAToolkit_VERSION}")
if(NOT FHELIUM_TORCH_CUDA_MAJOR STREQUAL FHELIUM_TOOLKIT_CUDA_MAJOR)
  message(
    FATAL_ERROR
      "Torch uses CUDA ${FHELIUM_TORCH_CUDA_VERSION}, but the selected "
      "nvcc/toolkit is ${CUDAToolkit_VERSION}. CUDA major versions must match.")
endif()
string(REGEX MATCH "^[0-9]+\\.[0-9]+" FHELIUM_TORCH_CUDA_MAJOR_MINOR
             "${FHELIUM_TORCH_CUDA_VERSION}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" FHELIUM_TOOLKIT_CUDA_MAJOR_MINOR
             "${CUDAToolkit_VERSION}")
if(NOT FHELIUM_TORCH_CUDA_MAJOR_MINOR STREQUAL FHELIUM_TOOLKIT_CUDA_MAJOR_MINOR)
  message(
    WARNING
      "Torch uses CUDA ${FHELIUM_TORCH_CUDA_VERSION}, while the selected "
      "nvcc/toolkit is ${CUDAToolkit_VERSION}. Minor-version compatibility "
      "is not guaranteed; prefer a matching toolkit.")
endif()

# CPU-only builds avoid TorchConfig because a CUDA-enabled Torch package makes
# it require CUDA even when only torch_cpu is consumed. This module is included
# only for CUDA-enabled builds.
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
set(Torch_DIR
    "${TORCH_CMAKE_PATH}/Torch"
    CACHE PATH "Torch CMake package selected by the build backend" FORCE)
find_package(Torch CONFIG REQUIRED)

# Torch injects its build-time architecture list into global CUDA flags. Keep
# its other flags and let FHElium's target CUDA_ARCHITECTURES property control
# code generation.
separate_arguments(_FHELIUM_TORCH_CUDA_FLAGS UNIX_COMMAND "${CMAKE_CUDA_FLAGS}")
set(_FHELIUM_FILTERED_CUDA_FLAGS "")
set(_FHELIUM_SKIP_CUDA_ARCH_VALUE FALSE)
foreach(_cuda_flag IN LISTS _FHELIUM_TORCH_CUDA_FLAGS)
  if(_FHELIUM_SKIP_CUDA_ARCH_VALUE)
    set(_FHELIUM_SKIP_CUDA_ARCH_VALUE FALSE)
  elseif(_cuda_flag STREQUAL "-gencode" OR _cuda_flag STREQUAL
                                           "--generate-code")
    set(_FHELIUM_SKIP_CUDA_ARCH_VALUE TRUE)
  elseif(_cuda_flag MATCHES "^(-gencode|--generate-code)=")
    # Architecture and value are encoded in this argument.
  else()
    list(APPEND _FHELIUM_FILTERED_CUDA_FLAGS "${_cuda_flag}")
  endif()
endforeach()
string(JOIN " " CMAKE_CUDA_FLAGS ${_FHELIUM_FILTERED_CUDA_FLAGS})

# Older TorchConfig releases reference CUDA::nvToolsExt even when modern CUDA
# toolkits no longer provide that target.
if(NOT TARGET CUDA::nvToolsExt)
  find_library(
    NVTOOLSEXT_LIBRARY
    NAMES nvToolsExt libnvToolsExt
    PATHS
      "${_CUDA_ROOT}/lib64"
      "${_CUDA_ROOT}/lib"
      "$ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/nvidia/nvtx/lib"
    NO_DEFAULT_PATH)
  if(NVTOOLSEXT_LIBRARY)
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION
                                                      "${NVTOOLSEXT_LIBRARY}")
  else()
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
endif()

if(_FHELIUM_REQUESTED_CUDA_ARCHITECTURES)
  set(_FHELIUM_CUDA_ARCHITECTURES "${_FHELIUM_REQUESTED_CUDA_ARCHITECTURES}")
else()
  set(_FHELIUM_CUDA_ARCHITECTURES "")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    list(APPEND _FHELIUM_CUDA_ARCHITECTURES 80)
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND _FHELIUM_CUDA_ARCHITECTURES 90)
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND _FHELIUM_CUDA_ARCHITECTURES 120)
  endif()
  if(NOT _FHELIUM_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "FHElium CUDA builds require CUDA 11.0 or newer, or "
                        "an explicit CMAKE_CUDA_ARCHITECTURES override")
  endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES
    "${_FHELIUM_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures selected by FHElium" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "${_FHELIUM_CUDA_ARCHITECTURES}")
message(STATUS "FHElium CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
