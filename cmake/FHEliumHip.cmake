# Configure the HIP toolkit and the selected Torch HIP libraries.

if(CMAKE_VERSION VERSION_LESS 3.21)
  message(FATAL_ERROR "FHElium HIP builds require CMake 3.21 or newer")
endif()
find_package(Python3 COMPONENTS Development.Module REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "from torch.utils.cpp_extension import ROCM_HOME; print(ROCM_HOME or '')"
  OUTPUT_VARIABLE FHELIUM_ROCM_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(CMAKE_HIP_COMPILER_ROCM_ROOT)
  set(FHELIUM_ROCM_ROOT "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
endif()
if(NOT FHELIUM_ROCM_ROOT)
  message(FATAL_ERROR "HIP builds require a ROCm SDK; set ROCM_HOME to its root")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${FHELIUM_ROCM_ROOT}")
if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX})
  find_program(FHELIUM_HIP_CLANG NAMES clang++
               PATHS "${FHELIUM_ROCM_ROOT}/llvm/bin"
                     "${FHELIUM_ROCM_ROOT}/lib/llvm/bin"
               NO_DEFAULT_PATH REQUIRED)
  set(CMAKE_HIP_COMPILER "${FHELIUM_HIP_CLANG}")
endif()
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${FHELIUM_ROCM_ROOT}")
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
  set(CMAKE_HIP_ARCHITECTURES "$ENV{PYTORCH_ROCM_ARCH}")
endif()
enable_language(HIP)
find_package(hip CONFIG REQUIRED)

foreach(_torch_library IN ITEMS torch torch_hip c10_hip)
  find_library(FHELIUM_TORCH_${_torch_library}_LIBRARY
               NAMES ${_torch_library}
               PATHS "${TORCH_LIB_PATH}"
               NO_DEFAULT_PATH REQUIRED)
endforeach()

message(STATUS "FHElium HIP SDK: ${FHELIUM_ROCM_ROOT}")
message(STATUS "Torch HIP runtime version: ${FHELIUM_TORCH_HIP_VERSION}")
message(STATUS "FHElium HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
