# Discover the target Python/Torch installation and its CPU build requirements.

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)

execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent)"
  RESULT_VARIABLE FHELIUM_TORCH_QUERY_RESULT
  OUTPUT_VARIABLE TORCH_PACKAGE_ROOT
  ERROR_VARIABLE FHELIUM_TORCH_QUERY_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT FHELIUM_TORCH_QUERY_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "FHElium must build against the Torch package in the target environment. "
      "Install the selected Torch build and scikit-build-core first, then run "
      "pip with --no-build-isolation --no-cache-dir. Python reported: "
      "${FHELIUM_TORCH_QUERY_ERROR}")
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch.version.cuda or '')"
  OUTPUT_VARIABLE FHELIUM_TORCH_CUDA_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

set(TORCH_CMAKE_PATH "${TORCH_PACKAGE_ROOT}/share/cmake")
set(TORCH_LIB_PATH "${TORCH_PACKAGE_ROOT}/lib")
set(TORCH_INCLUDE_DIRS "${TORCH_PACKAGE_ROOT}/include"
                       "${TORCH_PACKAGE_ROOT}/include/torch/csrc/api/include")
set(TORCH_CONFIG_PATH "${TORCH_CMAKE_PATH}/Torch/TorchConfig.cmake")
if(NOT EXISTS "${TORCH_CONFIG_PATH}")
  message(
    FATAL_ERROR "Selected Torch package has no config: ${TORCH_CONFIG_PATH}")
endif()
foreach(_torch_library IN ITEMS torch_cpu c10)
  find_library(
    FHELIUM_TORCH_${_torch_library}_LIBRARY
    NAMES ${_torch_library}
    PATHS "${TORCH_LIB_PATH}"
    NO_DEFAULT_PATH REQUIRED)
endforeach()

if(SKBUILD_STATE STREQUAL "editable")
  set(FHELIUM_TORCH_INSTALL_RPATH "${TORCH_LIB_PATH}")
elseif(APPLE)
  set(FHELIUM_TORCH_INSTALL_RPATH "@loader_path/../../../torch/lib")
elseif(UNIX)
  set(FHELIUM_TORCH_INSTALL_RPATH "$ORIGIN/../../../torch/lib")
else()
  set(FHELIUM_TORCH_INSTALL_RPATH "")
endif()
set(PYTHON_MODULE_SUFFIX ".${Python3_SOABI}")

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Selected Torch package: ${TORCH_PACKAGE_ROOT}")
message(STATUS "Detected Python SOABI: ${Python3_SOABI}")

macro(fhelium_configure_torch_parallel)
  # ATen's parallel_for is header-defined. Match the parallel backend compiled
  # into the selected Torch package without linking a second runtime.
  set(FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS "")
  set(FHELIUM_TORCH_PARALLEL_INCLUDE_DIRS "")
  set(_FHELIUM_TORCH_ATEN_CONFIG "${TORCH_PACKAGE_ROOT}/include/ATen/Config.h")
  if(FHELIUM_BUILD_CPU AND EXISTS "${_FHELIUM_TORCH_ATEN_CONFIG}")
    file(STRINGS "${_FHELIUM_TORCH_ATEN_CONFIG}" _FHELIUM_TORCH_OPENMP_CONFIG
         REGEX "^#define AT_PARALLEL_OPENMP 1$")
    if(_FHELIUM_TORCH_OPENMP_CONFIG)
      if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Torch's wheel supplies omp.h and libomp. Dynamic lookup binds the
        # MODULE target to that loaded runtime instead of a Homebrew copy.
        set(FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp")
      else()
        find_package(
          OpenMP
          COMPONENTS CXX
          REQUIRED)
        separate_arguments(_FHELIUM_OPENMP_CXX_FLAGS UNIX_COMMAND
                           "${OpenMP_CXX_FLAGS}")
        set(FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS
            "${_FHELIUM_OPENMP_CXX_FLAGS}")
        set(FHELIUM_TORCH_PARALLEL_INCLUDE_DIRS "${OpenMP_CXX_INCLUDE_DIRS}")
      endif()
      message(STATUS "Torch intra-op backend: OpenMP")
    else()
      message(STATUS "Torch intra-op backend: non-OpenMP")
    endif()
  endif()
endmacro()
