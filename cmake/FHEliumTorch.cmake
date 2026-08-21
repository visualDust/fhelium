# Discover the target Python/Torch installation and its CPU build requirements.

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
set(FHELIUM_PYTHON_SOABI "${Python3_SOABI}")

execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
  RESULT_VARIABLE FHELIUM_PYTHON_SUFFIX_QUERY_RESULT
  OUTPUT_VARIABLE FHELIUM_PYTHON_EXTENSION_SUFFIX
  ERROR_VARIABLE FHELIUM_PYTHON_SUFFIX_QUERY_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT FHELIUM_PYTHON_SUFFIX_QUERY_RESULT EQUAL 0
   OR NOT FHELIUM_PYTHON_EXTENSION_SUFFIX)
  message(
    FATAL_ERROR "FHElium could not determine Python's native extension suffix: "
                "${FHELIUM_PYTHON_SUFFIX_QUERY_ERROR}")
endif()

if(WIN32
   AND NOT FHELIUM_PYTHON_SOABI
   AND FHELIUM_PYTHON_EXTENSION_SUFFIX MATCHES
       "^\\.(cp[0-9]+-win_amd64)\\.pyd$")
  set(FHELIUM_PYTHON_SOABI "${CMAKE_MATCH_1}")
endif()
if(NOT FHELIUM_PYTHON_SOABI)
  message(FATAL_ERROR "FHElium could not determine Python's SOABI")
endif()

if(WIN32)
  if(NOT MSVC)
    message(FATAL_ERROR "FHElium Windows native builds require MSVC")
  endif()
  if(NOT MSVC_CXX_ARCHITECTURE_ID STREQUAL "x64")
    message(
      FATAL_ERROR
        "FHElium Windows native builds support only MSVC x64; detected "
        "'${MSVC_CXX_ARCHITECTURE_ID}'")
  endif()
  if(NOT FHELIUM_PYTHON_SOABI MATCHES "win_amd64$"
     OR NOT FHELIUM_PYTHON_EXTENSION_SUFFIX MATCHES "\\.pyd$")
    message(
      FATAL_ERROR
        "FHElium Windows native builds require a win_amd64 CPython target; "
        "detected SOABI '${FHELIUM_PYTHON_SOABI}' and extension suffix "
        "'${FHELIUM_PYTHON_EXTENSION_SUFFIX}'")
  endif()
endif()

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

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Selected Torch package: ${TORCH_PACKAGE_ROOT}")
message(STATUS "Detected Python SOABI: ${FHELIUM_PYTHON_SOABI}")
message(
  STATUS "Detected Python extension suffix: ${FHELIUM_PYTHON_EXTENSION_SUFFIX}")

function(_fhelium_configure_msvc_torch_openmp)
  find_package(
    OpenMP
    COMPONENTS CXX
    REQUIRED)
  separate_arguments(_FHELIUM_OPENMP_CXX_FLAGS NATIVE_COMMAND
                     "${OpenMP_CXX_FLAGS}")
  set(_FHELIUM_HAS_MSVC_OPENMP_FLAG FALSE)
  foreach(_FHELIUM_OPENMP_CXX_FLAG IN LISTS _FHELIUM_OPENMP_CXX_FLAGS)
    if(_FHELIUM_OPENMP_CXX_FLAG MATCHES "^[-/]openmp$")
      set(_FHELIUM_HAS_MSVC_OPENMP_FLAG TRUE)
    endif()
  endforeach()
  if(NOT _FHELIUM_HAS_MSVC_OPENMP_FLAG)
    message(
      FATAL_ERROR
        "FHElium requires MSVC OpenMP code generation for Torch's OpenMP "
        "parallel backend, but FindOpenMP returned '${OpenMP_CXX_FLAGS}'")
  endif()

  set(_FHELIUM_TORCH_OPENMP_DLL "${TORCH_LIB_PATH}/libiomp5md.dll")
  if(NOT EXISTS "${_FHELIUM_TORCH_OPENMP_DLL}")
    message(
      FATAL_ERROR
        "The selected Windows Torch package declares AT_PARALLEL_OPENMP=1 "
        "but does not provide ${_FHELIUM_TORCH_OPENMP_DLL}")
  endif()

  if(NOT CMAKE_LINKER OR NOT EXISTS "${CMAKE_LINKER}")
    message(
      FATAL_ERROR
        "FHElium requires the MSVC link.exe export-inspection tool; detected "
        "CMAKE_LINKER='${CMAKE_LINKER}'")
  endif()
  get_filename_component(_FHELIUM_MSVC_LINK_NAME "${CMAKE_LINKER}" NAME)
  string(TOLOWER "${_FHELIUM_MSVC_LINK_NAME}" _FHELIUM_MSVC_LINK_NAME)
  if(NOT _FHELIUM_MSVC_LINK_NAME STREQUAL "link.exe")
    message(
      FATAL_ERROR
        "FHElium Windows Torch OpenMP validation requires MSVC link.exe; "
        "detected '${CMAKE_LINKER}'")
  endif()

  if(NOT CMAKE_AR OR NOT EXISTS "${CMAKE_AR}")
    message(
      FATAL_ERROR
        "FHElium requires the MSVC lib.exe import-library tool; detected "
        "CMAKE_AR='${CMAKE_AR}'")
  endif()
  get_filename_component(_FHELIUM_MSVC_LIB_NAME "${CMAKE_AR}" NAME)
  string(TOLOWER "${_FHELIUM_MSVC_LIB_NAME}" _FHELIUM_MSVC_LIB_NAME)
  if(NOT _FHELIUM_MSVC_LIB_NAME STREQUAL "lib.exe")
    message(
      FATAL_ERROR
        "FHElium Windows Torch OpenMP import-library generation requires "
        "MSVC lib.exe; detected '${CMAKE_AR}'")
  endif()

  execute_process(
    COMMAND "${CMAKE_LINKER}" /dump /headers /exports
            "${_FHELIUM_TORCH_OPENMP_DLL}"
    RESULT_VARIABLE _FHELIUM_TORCH_OPENMP_DUMP_RESULT
    OUTPUT_VARIABLE _FHELIUM_TORCH_OPENMP_DUMP
    ERROR_VARIABLE _FHELIUM_TORCH_OPENMP_DUMP_ERROR)
  if(NOT _FHELIUM_TORCH_OPENMP_DUMP_RESULT EQUAL 0)
    message(
      FATAL_ERROR
        "Could not inspect ${_FHELIUM_TORCH_OPENMP_DLL} with MSVC link.exe: "
        "${_FHELIUM_TORCH_OPENMP_DUMP_ERROR}")
  endif()
  string(FIND "${_FHELIUM_TORCH_OPENMP_DUMP}" "8664 machine (x64)"
              _FHELIUM_TORCH_OPENMP_AMD64_OFFSET)
  if(_FHELIUM_TORCH_OPENMP_AMD64_OFFSET LESS 0)
    message(
      FATAL_ERROR "The selected Torch OpenMP runtime is not an AMD64 DLL: "
                  "${_FHELIUM_TORCH_OPENMP_DLL}")
  endif()

  # ATen/ParallelOpenMP.h calls the two omp_get functions. MSVC's current
  # parallel-region expansion also references _vcomp_fork and its runtime
  # selection marker. These are the complete OpenMP undefined-symbol set in
  # FHElium's current Windows CPU objects.
  set(_FHELIUM_TORCH_OPENMP_REQUIRED_EXPORTS
      _vcomp_fork omp_get_num_threads omp_get_thread_num
      _You_must_link_with_Microsoft_OpenMP_library)
  foreach(_FHELIUM_TORCH_OPENMP_EXPORT IN
          LISTS _FHELIUM_TORCH_OPENMP_REQUIRED_EXPORTS)
    string(FIND "${_FHELIUM_TORCH_OPENMP_DUMP}"
                " ${_FHELIUM_TORCH_OPENMP_EXPORT}\r\n"
                _FHELIUM_TORCH_OPENMP_EXPORT_OFFSET)
    if(_FHELIUM_TORCH_OPENMP_EXPORT_OFFSET LESS 0)
      string(FIND "${_FHELIUM_TORCH_OPENMP_DUMP}"
                  " ${_FHELIUM_TORCH_OPENMP_EXPORT}\n"
                  _FHELIUM_TORCH_OPENMP_EXPORT_OFFSET)
    endif()
    if(_FHELIUM_TORCH_OPENMP_EXPORT_OFFSET LESS 0)
      message(
        FATAL_ERROR
          "The selected Torch OpenMP runtime ${_FHELIUM_TORCH_OPENMP_DLL} "
          "does not export required compatibility symbol "
          "${_FHELIUM_TORCH_OPENMP_EXPORT}")
    endif()
  endforeach()

  set(_FHELIUM_TORCH_OPENMP_BUILD_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/fhelium_torch_openmp")
  set(_FHELIUM_TORCH_OPENMP_DEF
      "${_FHELIUM_TORCH_OPENMP_BUILD_DIR}/fhelium_torch_openmp.def")
  set(_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY
      "${_FHELIUM_TORCH_OPENMP_BUILD_DIR}/fhelium_torch_openmp.lib")
  file(MAKE_DIRECTORY "${_FHELIUM_TORCH_OPENMP_BUILD_DIR}")
  set(_FHELIUM_TORCH_OPENMP_DEF_CONTENT "LIBRARY libiomp5md.dll\nEXPORTS\n")
  foreach(_FHELIUM_TORCH_OPENMP_EXPORT IN
          LISTS _FHELIUM_TORCH_OPENMP_REQUIRED_EXPORTS)
    # The OpenMP selection marker is an unresolved link sentinel with no
    # relocation. It must remain an ordinary DEF entry: DATA would publish only
    # its __imp_ name and would not satisfy MSVC's direct symbol.
    string(APPEND _FHELIUM_TORCH_OPENMP_DEF_CONTENT
           "  ${_FHELIUM_TORCH_OPENMP_EXPORT}\n")
  endforeach()
  file(WRITE "${_FHELIUM_TORCH_OPENMP_DEF}"
       "${_FHELIUM_TORCH_OPENMP_DEF_CONTENT}")
  file(REMOVE "${_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY}")
  execute_process(
    COMMAND "${CMAKE_AR}" /nologo "/def:${_FHELIUM_TORCH_OPENMP_DEF}"
            /machine:x64 "/out:${_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY}"
    RESULT_VARIABLE _FHELIUM_TORCH_OPENMP_LIB_RESULT
    OUTPUT_VARIABLE _FHELIUM_TORCH_OPENMP_LIB_OUTPUT
    ERROR_VARIABLE _FHELIUM_TORCH_OPENMP_LIB_ERROR)
  if(NOT _FHELIUM_TORCH_OPENMP_LIB_RESULT EQUAL 0
     OR NOT EXISTS "${_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY}")
    message(
      FATAL_ERROR
        "MSVC lib.exe could not generate the private Torch OpenMP import "
        "library: ${_FHELIUM_TORCH_OPENMP_LIB_OUTPUT} "
        "${_FHELIUM_TORCH_OPENMP_LIB_ERROR}")
  endif()

  if(NOT TARGET FHElium::TorchOpenMP)
    add_library(FHElium::TorchOpenMP UNKNOWN IMPORTED GLOBAL)
  endif()
  # Model only the private build-tree import library. The Torch runtime DLL is
  # neither attached as an imported location nor installed by FHElium.
  set_target_properties(
    FHElium::TorchOpenMP PROPERTIES IMPORTED_LOCATION
                                    "${_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY}")

  set(FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS
      "${_FHELIUM_OPENMP_CXX_FLAGS}"
      PARENT_SCOPE)
  set(FHELIUM_TORCH_PARALLEL_INCLUDE_DIRS
      "${OpenMP_CXX_INCLUDE_DIRS}"
      PARENT_SCOPE)
  set(FHELIUM_TORCH_PARALLEL_LINK_OPTIONS
      "$<HOST_LINK:/NODEFAULTLIB:VCOMP>"
      PARENT_SCOPE)
  set(FHELIUM_TORCH_PARALLEL_LIBRARIES
      FHElium::TorchOpenMP
      PARENT_SCOPE)
  message(
    STATUS
      "Torch OpenMP runtime: ${_FHELIUM_TORCH_OPENMP_DLL} via private AMD64 "
      "import library ${_FHELIUM_TORCH_OPENMP_IMPORT_LIBRARY}")
endfunction()

macro(fhelium_configure_torch_parallel)
  # ATen's parallel_for is header-defined. Match the parallel backend compiled
  # into the selected Torch package without linking a second runtime.
  set(FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS "")
  set(FHELIUM_TORCH_PARALLEL_INCLUDE_DIRS "")
  set(FHELIUM_TORCH_PARALLEL_LINK_OPTIONS "")
  set(FHELIUM_TORCH_PARALLEL_LIBRARIES "")
  set(_FHELIUM_TORCH_ATEN_CONFIG "${TORCH_PACKAGE_ROOT}/include/ATen/Config.h")
  set(_FHELIUM_TORCH_OPENMP_CONFIG "")
  if(FHELIUM_BUILD_CPU)
    if(EXISTS "${_FHELIUM_TORCH_ATEN_CONFIG}")
      file(STRINGS "${_FHELIUM_TORCH_ATEN_CONFIG}" _FHELIUM_TORCH_OPENMP_CONFIG
           REGEX "^#define AT_PARALLEL_OPENMP 1$")
    elseif(WIN32 AND MSVC)
      message(
        FATAL_ERROR
          "The selected Windows Torch package has no ATen configuration: "
          "${_FHELIUM_TORCH_ATEN_CONFIG}")
    endif()

    if(WIN32
       AND MSVC
       AND NOT _FHELIUM_TORCH_OPENMP_CONFIG)
      message(
        FATAL_ERROR
          "FHElium Windows CPU builds require the selected Torch package to "
          "declare AT_PARALLEL_OPENMP=1 in ${_FHELIUM_TORCH_ATEN_CONFIG}")
    endif()

    if(_FHELIUM_TORCH_OPENMP_CONFIG)
      if(WIN32 AND MSVC)
        _fhelium_configure_msvc_torch_openmp()
      elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
    elseif(EXISTS "${_FHELIUM_TORCH_ATEN_CONFIG}")
      message(STATUS "Torch intra-op backend: non-OpenMP")
    endif()
  endif()
endmacro()
