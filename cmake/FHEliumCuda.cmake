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

# Select nvcc through standard CMake inputs. Windows additionally checks the
# versioned environment/layout used by NVIDIA's installer before the generic
# CUDA_PATH, while Unix retains the CUDA_HOME, /usr/local, and Conda order.
# CMake owns compiler identity after the first configure.
set(_FHELIUM_VERSIONED_CUDA_ROOT
    "/usr/local/cuda-${FHELIUM_TORCH_CUDA_VERSION}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT
             "${FHELIUM_TORCH_CUDA_VERSION}")
if(NOT CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}")
  elseif(WIN32)
    set(_FHELIUM_WINDOWS_CUDA_ROOT_HINTS "")
    if(CUDAToolkit_ROOT)
      list(APPEND _FHELIUM_WINDOWS_CUDA_ROOT_HINTS "${CUDAToolkit_ROOT}")
    endif()
    if(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
      list(APPEND _FHELIUM_WINDOWS_CUDA_ROOT_HINTS "$ENV{CUDA_HOME}")
    endif()

    if(_FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT)
      string(REPLACE "." "_" _FHELIUM_TORCH_CUDA_ENV_VERSION
                     "${_FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT}")
      set(_FHELIUM_TORCH_CUDA_ENV_NAME
          "CUDA_PATH_V${_FHELIUM_TORCH_CUDA_ENV_VERSION}")
      if(DEFINED ENV{${_FHELIUM_TORCH_CUDA_ENV_NAME}}
         AND NOT "$ENV{${_FHELIUM_TORCH_CUDA_ENV_NAME}}" STREQUAL "")
        list(APPEND _FHELIUM_WINDOWS_CUDA_ROOT_HINTS
             "$ENV{${_FHELIUM_TORCH_CUDA_ENV_NAME}}")
      endif()
      foreach(_FHELIUM_PROGRAM_FILES_ENV IN ITEMS ProgramW6432 ProgramFiles)
        if(DEFINED ENV{${_FHELIUM_PROGRAM_FILES_ENV}}
           AND NOT "$ENV{${_FHELIUM_PROGRAM_FILES_ENV}}" STREQUAL "")
          list(
            APPEND
            _FHELIUM_WINDOWS_CUDA_ROOT_HINTS
            "$ENV{${_FHELIUM_PROGRAM_FILES_ENV}}/NVIDIA GPU Computing Toolkit/CUDA/v${_FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT}"
          )
        endif()
      endforeach()
      if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
        get_filename_component(_FHELIUM_CUDA_PATH_PARENT "$ENV{CUDA_PATH}"
                               DIRECTORY)
        list(
          APPEND
          _FHELIUM_WINDOWS_CUDA_ROOT_HINTS
          "${_FHELIUM_CUDA_PATH_PARENT}/v${_FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT}"
        )
      endif()
    endif()

    if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
      list(APPEND _FHELIUM_WINDOWS_CUDA_ROOT_HINTS "$ENV{CUDA_PATH}")
    endif()
    if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
      list(APPEND _FHELIUM_WINDOWS_CUDA_ROOT_HINTS "$ENV{CONDA_PREFIX}")
    endif()
    list(REMOVE_DUPLICATES _FHELIUM_WINDOWS_CUDA_ROOT_HINTS)

    unset(_FHELIUM_WINDOWS_NVCC CACHE)
    unset(_FHELIUM_WINDOWS_NVCC)
    find_program(
      _FHELIUM_WINDOWS_NVCC
      NAMES nvcc.exe nvcc
      HINTS ${_FHELIUM_WINDOWS_CUDA_ROOT_HINTS}
      PATH_SUFFIXES bin Library/bin
      NO_DEFAULT_PATH)
    if(NOT _FHELIUM_WINDOWS_NVCC)
      find_program(_FHELIUM_WINDOWS_NVCC NAMES nvcc.exe nvcc)
    endif()
    if(_FHELIUM_WINDOWS_NVCC)
      set(CMAKE_CUDA_COMPILER "${_FHELIUM_WINDOWS_NVCC}")
      if(NOT CUDAToolkit_ROOT)
        get_filename_component(_FHELIUM_WINDOWS_CUDA_BIN
                               "${_FHELIUM_WINDOWS_NVCC}" DIRECTORY)
        get_filename_component(_FHELIUM_WINDOWS_CUDA_ROOT
                               "${_FHELIUM_WINDOWS_CUDA_BIN}" DIRECTORY)
        set(CUDAToolkit_ROOT "${_FHELIUM_WINDOWS_CUDA_ROOT}")
      endif()
    endif()
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
  if(WIN32 AND _FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT)
    # Visual Studio's CUDA .props files resolve their toolkit through a
    # versioned CUDA_PATH_V<major>_<minor> environment property. Populate it for
    # this configure process when discovery selected a versioned layout.
    get_filename_component(_FHELIUM_PRE_ENABLE_CUDA_COMPILER_NAME
                           "${CMAKE_CUDA_COMPILER}" NAME)
    if(IS_ABSOLUTE "${CMAKE_CUDA_COMPILER}"
       AND _FHELIUM_PRE_ENABLE_CUDA_COMPILER_NAME MATCHES "^nvcc(\\.exe)?$")
      get_filename_component(_FHELIUM_PRE_ENABLE_CUDA_BIN
                             "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_FHELIUM_PRE_ENABLE_CUDA_ROOT
                             "${_FHELIUM_PRE_ENABLE_CUDA_BIN}" DIRECTORY)
      string(REPLACE "." "_" _FHELIUM_TORCH_CUDA_ENV_VERSION
                     "${_FHELIUM_TORCH_CUDA_MAJOR_MINOR_HINT}")
      set(_FHELIUM_TORCH_CUDA_ENV_NAME
          "CUDA_PATH_V${_FHELIUM_TORCH_CUDA_ENV_VERSION}")
      set("ENV{${_FHELIUM_TORCH_CUDA_ENV_NAME}}"
          "${_FHELIUM_PRE_ENABLE_CUDA_ROOT}")
    endif()
  endif()
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

if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
  # Some Windows Torch packages export /EHsc and /bigobj inside one nested
  # CXX-only generator expression. Visual Studio's CUDA generation does not
  # preserve that nested list reliably. Flatten only this known expression into
  # equivalent per-option CXX guards; retain every option, definition, include,
  # and link requirement exported by Torch.
  set(_FHELIUM_TORCH_MALFORMED_MSVC_OPTIONS
      "$<$<COMPILE_LANGUAGE:CXX>:/permissive->;$<$<COMPILE_LANGUAGE:CXX>:;$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/Z7>;/EHsc;/bigobj>"
  )
  set(_FHELIUM_TORCH_FLAT_MSVC_OPTIONS
      "$<$<COMPILE_LANGUAGE:CXX>:/permissive->;$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:/Z7>;$<$<COMPILE_LANGUAGE:CXX>:/EHsc>;$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
  )
  set(_FHELIUM_REWRITTEN_TORCH_TARGETS "")
  foreach(_FHELIUM_TORCH_TARGET IN ITEMS c10 c10_cuda torch_cpu torch_cuda)
    if(TARGET ${_FHELIUM_TORCH_TARGET})
      get_target_property(_FHELIUM_TORCH_TARGET_OPTIONS
                          ${_FHELIUM_TORCH_TARGET} INTERFACE_COMPILE_OPTIONS)
      if(_FHELIUM_TORCH_TARGET_OPTIONS)
        set(_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN
            "${_FHELIUM_TORCH_TARGET_OPTIONS}")
        string(
          REPLACE "${_FHELIUM_TORCH_MALFORMED_MSVC_OPTIONS}"
                  "${_FHELIUM_TORCH_FLAT_MSVC_OPTIONS}"
                  _FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN
                  "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}")

        # Validate the complete result, including properties containing one
        # known expression plus an additional unknown nested form. These
        # fingerprints cannot occur in the flat guarded options above.
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}"
                    "$<$<COMPILE_LANGUAGE:CXX>:;"
                    _FHELIUM_TORCH_RESIDUAL_CXX_LIST_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}" "/EHsc;"
                    _FHELIUM_TORCH_RESIDUAL_EHSC_HEAD_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}" ";/EHsc"
                    _FHELIUM_TORCH_RESIDUAL_EHSC_TAIL_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}" "/bigobj;"
                    _FHELIUM_TORCH_RESIDUAL_BIGOBJ_HEAD_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}" ";/bigobj"
                    _FHELIUM_TORCH_RESIDUAL_BIGOBJ_TAIL_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}"
                    "/EHsc;/bigobj>" _FHELIUM_TORCH_RESIDUAL_EHSC_BIGOBJ_OFFSET)
        string(FIND "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}"
                    "/bigobj;/EHsc>" _FHELIUM_TORCH_RESIDUAL_BIGOBJ_EHSC_OFFSET)
        if(_FHELIUM_TORCH_RESIDUAL_CXX_LIST_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_EHSC_HEAD_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_EHSC_TAIL_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_BIGOBJ_HEAD_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_BIGOBJ_TAIL_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_EHSC_BIGOBJ_OFFSET GREATER_EQUAL 0
           OR _FHELIUM_TORCH_RESIDUAL_BIGOBJ_EHSC_OFFSET GREATER_EQUAL 0)
          message(
            FATAL_ERROR
              "Torch target ${_FHELIUM_TORCH_TARGET} exports a residual "
              "unsupported nested MSVC compile-option expression after the "
              "known safe rewrite; refusing to discard or broadly suppress "
              "Torch usage requirements. Original: "
              "${_FHELIUM_TORCH_TARGET_OPTIONS} Rewritten: "
              "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}")
        endif()

        if(NOT _FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN STREQUAL
           _FHELIUM_TORCH_TARGET_OPTIONS)
          set_property(
            TARGET ${_FHELIUM_TORCH_TARGET}
            PROPERTY INTERFACE_COMPILE_OPTIONS
                     "${_FHELIUM_TORCH_TARGET_OPTIONS_REWRITTEN}")
          list(APPEND _FHELIUM_REWRITTEN_TORCH_TARGETS ${_FHELIUM_TORCH_TARGET})
        endif()
      endif()
    endif()
  endforeach()
  if(_FHELIUM_REWRITTEN_TORCH_TARGETS)
    message(STATUS "Flattened nested MSVC CXX options on Torch targets: "
                   "${_FHELIUM_REWRITTEN_TORCH_TARGETS}")
  endif()
endif()

# Torch injects its build-time architecture list into global CUDA flags. Keep
# its other flags and let FHElium's target CUDA_ARCHITECTURES property control
# code generation.
if(MSVC)
  # Do not tokenize TorchConfig's `-Xcompiler=" /EHsc"`: doing so turns the host
  # option into a bare nvcc input. Normalize that required host option, then
  # remove only the two supported forms of Torch's architecture option from the
  # raw flag string.
  string(REPLACE "-Xcompiler=\" /EHsc\"" "-Xcompiler=/EHsc" CMAKE_CUDA_FLAGS
                 "${CMAKE_CUDA_FLAGS}")
  string(REGEX REPLACE "(^|[ \t])(-gencode|--generate-code)(=|[ \t]+)[^ \t]+"
                       "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
else()
  separate_arguments(_FHELIUM_TORCH_CUDA_FLAGS UNIX_COMMAND
                     "${CMAKE_CUDA_FLAGS}")
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
endif()

# Older TorchConfig releases reference CUDA::nvToolsExt even when modern CUDA
# toolkits no longer provide that target.
if(NOT TARGET CUDA::nvToolsExt)
  find_library(
    NVTOOLSEXT_LIBRARY
    NAMES nvToolsExt libnvToolsExt
    PATHS
      "${_CUDA_ROOT}/lib/x64"
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
