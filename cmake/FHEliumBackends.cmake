# Resolve the requested native execution backends against the selected Torch.

set(FHELIUM_NATIVE_BACKENDS
    "AUTO"
    CACHE STRING "Native execution backends: AUTO, CPU, CUDA, or CPU+CUDA")
set_property(CACHE FHELIUM_NATIVE_BACKENDS PROPERTY STRINGS AUTO CPU CUDA
                                                    "CPU;CUDA")

function(fhelium_normalize_backend_list input output)
  string(TOUPPER "${input}" _requested)
  string(REPLACE "," ";" _requested "${_requested}")
  string(REPLACE "+" ";" _requested "${_requested}")
  string(REPLACE " " ";" _requested "${_requested}")
  list(FILTER _requested EXCLUDE REGEX "^$")
  list(REMOVE_DUPLICATES _requested)
  foreach(_backend IN LISTS _requested)
    if(NOT _backend STREQUAL "AUTO"
       AND NOT _backend STREQUAL "CPU"
       AND NOT _backend STREQUAL "CUDA")
      message(
        FATAL_ERROR
          "Unknown FHELIUM_NATIVE_BACKENDS entry '${_backend}'; expected AUTO, CPU, CUDA, or CPU+CUDA"
      )
    endif()
  endforeach()
  if("AUTO" IN_LIST _requested AND NOT _requested STREQUAL "AUTO")
    message(
      FATAL_ERROR
        "FHELIUM_NATIVE_BACKENDS=AUTO cannot be combined with other backends")
  endif()
  set(${output}
      "${_requested}"
      PARENT_SCOPE)
endfunction()

fhelium_normalize_backend_list("${FHELIUM_NATIVE_BACKENDS}"
                               _FHELIUM_REQUESTED_BACKENDS)
if(_FHELIUM_REQUESTED_BACKENDS STREQUAL "AUTO")
  set(_FHELIUM_ENABLED_BACKENDS CPU)
  if(FHELIUM_TORCH_CUDA_VERSION)
    list(APPEND _FHELIUM_ENABLED_BACKENDS CUDA)
  endif()
else()
  set(_FHELIUM_ENABLED_BACKENDS ${_FHELIUM_REQUESTED_BACKENDS})
endif()
if(NOT _FHELIUM_ENABLED_BACKENDS)
  message(FATAL_ERROR "FHELIUM_NATIVE_BACKENDS selected no native backend")
endif()

set(FHELIUM_BUILD_CPU OFF)
set(FHELIUM_BUILD_CUDA OFF)
if("CPU" IN_LIST _FHELIUM_ENABLED_BACKENDS)
  set(FHELIUM_BUILD_CPU ON)
endif()
if("CUDA" IN_LIST _FHELIUM_ENABLED_BACKENDS)
  set(FHELIUM_BUILD_CUDA ON)
endif()
if(FHELIUM_BUILD_CUDA AND NOT FHELIUM_TORCH_CUDA_VERSION)
  message(
    FATAL_ERROR
      "FHELIUM_NATIVE_BACKENDS requests CUDA, but the selected target Torch "
      "package is CPU-only")
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

message(STATUS "FHElium native backends: ${_FHELIUM_ENABLED_BACKENDS}")
