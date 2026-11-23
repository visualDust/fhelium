# Define libfhelium, its Torch operator registration module, and the backend
# runtime-information module.

function(_fhelium_map_msvc_build_paths target)
  if(NOT MSVC)
    return()
  endif()

  # __FILE__ appears in Torch assertions and operator registration diagnostics.
  # Map builder-local roots to stable labels before those strings enter a wheel.
  target_compile_options(
    ${target}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/experimental:deterministic>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/experimental:deterministic>"
  )
  target_link_options(${target} PRIVATE "$<$<LINK_LANGUAGE:CXX>:/Brepro>"
                      "$<$<LINK_LANGUAGE:CUDA>:-Xlinker=/Brepro>")
  set(_path_mappings
      "${CMAKE_SOURCE_DIR}|fhelium/source" "${CMAKE_BINARY_DIR}|fhelium/build"
      "${TORCH_PACKAGE_ROOT}|torch")
  if(FHELIUM_BUILD_CUDA)
    list(APPEND _path_mappings "${_CUDA_ROOT}|cuda-toolkit")
  endif()
  foreach(_path_mapping IN LISTS _path_mappings)
    string(REPLACE "|" ";" _path_mapping_parts "${_path_mapping}")
    list(GET _path_mapping_parts 0 _path_mapping_source)
    list(GET _path_mapping_parts 1 _path_mapping_destination)
    file(TO_NATIVE_PATH "${_path_mapping_source}" _path_mapping_source)
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/pathmap:${_path_mapping_source}=${_path_mapping_destination}>"
    )
    # nvcc forwards -Xcompiler text to cl unquoted, so a root containing a
    # space, such as the default CUDA installation under Program Files, would
    # split the option. CUDA objects keep those roots unmapped.
    if(NOT _path_mapping_source MATCHES " ")
      target_compile_options(
        ${target}
        PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/pathmap:${_path_mapping_source}=${_path_mapping_destination}>"
      )
    endif()
  endforeach()
endfunction()

function(_fhelium_torch_cxx_standard output)
  set(_standard 17)
  file(STRINGS "${TORCH_PACKAGE_ROOT}/include/ATen/ATen.h"
       _FHELIUM_TORCH_REQUIRES_CXX20 REGEX "__cplusplus < 202002L")
  # Torch's bit-field initializers require C++20 even when ATen.h has no
  # corresponding version guard.
  file(STRINGS "${TORCH_PACKAGE_ROOT}/include/c10/core/AutogradState.h"
       _FHELIUM_TORCH_BITFIELD_INITIALIZERS REGEX ": 1 =")
  if(MSVC
     OR _FHELIUM_TORCH_REQUIRES_CXX20
     OR _FHELIUM_TORCH_BITFIELD_INITIALIZERS)
    set(_standard 20)
  endif()
  set(${output}
      ${_standard}
      PARENT_SCOPE)
endfunction()

# Compile definitions naming the backends compiled into libfhelium. Public
# functions route to these backends, and the Torch registration module registers
# kernels only for them.
set(FHELIUM_BACKEND_DEFINITIONS "")
if(FHELIUM_BUILD_CPU)
  list(APPEND FHELIUM_BACKEND_DEFINITIONS FHELIUM_WITH_CPU)
endif()
if(FHELIUM_BUILD_CUDA)
  list(APPEND FHELIUM_BACKEND_DEFINITIONS FHELIUM_WITH_CUDA)
endif()
if(FHELIUM_BUILD_HIP)
  list(APPEND FHELIUM_BACKEND_DEFINITIONS FHELIUM_WITH_HIP)
endif()

# The C++ SDK (libfhelium, public headers, and the FHElium CMake package) is
# installed under one root. Wheels place it in the package's native/ directory.
# Editable installs keep it in the ABI-specific build directory so that source
# checkouts can hold builds for several Python ABIs.
if(SKBUILD_STATE STREQUAL "editable")
  set(FHELIUM_SDK_ROOT "${CMAKE_BINARY_DIR}/sdk")
  set(FHELIUM_SDK_LIBRARY_RPATH "${FHELIUM_SDK_ROOT}/lib")
else()
  set(FHELIUM_SDK_ROOT "native")
  if(APPLE)
    set(FHELIUM_SDK_LIBRARY_RPATH "@loader_path/../lib")
  else()
    set(FHELIUM_SDK_LIBRARY_RPATH "$ORIGIN/../lib")
  endif()
endif()

if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(FHELIUM_VERSION_FULL "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['project']['version'])"
      "${CMAKE_SOURCE_DIR}/pyproject.toml"
    OUTPUT_VARIABLE FHELIUM_VERSION_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" FHELIUM_VERSION
             "${FHELIUM_VERSION_FULL}")

# Builds library `name` from the sources below every directory that follows
# `include_dir`. Sources under `cpu/`, `cuda/`, and `hip/` directories are
# compiled only for the enabled backends.
function(add_fhelium_library name include_dir)
  _fhelium_torch_cxx_standard(_FHELIUM_CXX_STANDARD)

  set(CPP_SOURCES "")
  set(CU_SOURCES "")
  set(HIP_SOURCES "")
  foreach(src_dir IN LISTS ARGN)
    file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS "${src_dir}/*.cpp")
    list(FILTER COMMON_SOURCES EXCLUDE REGEX "/(cpu|cuda|hip)/")
    list(APPEND CPP_SOURCES ${COMMON_SOURCES})
    if(FHELIUM_BUILD_CPU)
      file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS "${src_dir}/*/cpu/*.cpp")
      list(APPEND CPP_SOURCES ${CPU_SOURCES})
    endif()
    if(FHELIUM_BUILD_CUDA)
      file(GLOB_RECURSE CUDA_CPP_SOURCES CONFIGURE_DEPENDS
           "${src_dir}/*/cuda/*.cpp")
      file(GLOB_RECURSE DIRECTORY_CU_SOURCES CONFIGURE_DEPENDS
           "${src_dir}/*/cuda/*.cu")
      list(APPEND CPP_SOURCES ${CUDA_CPP_SOURCES})
      list(APPEND CU_SOURCES ${DIRECTORY_CU_SOURCES})
    endif()
    if(FHELIUM_BUILD_HIP)
      file(GLOB_RECURSE HIP_CPP_SOURCES CONFIGURE_DEPENDS
           "${src_dir}/*/hip/*.cpp")
      file(GLOB_RECURSE DIRECTORY_HIP_SOURCES CONFIGURE_DEPENDS
           "${src_dir}/*/hip/*.hip")
      list(APPEND CPP_SOURCES ${HIP_CPP_SOURCES})
      list(APPEND HIP_SOURCES ${DIRECTORY_HIP_SOURCES})
    endif()
  endforeach()

  list(SORT CPP_SOURCES)
  list(SORT CU_SOURCES)
  list(SORT HIP_SOURCES)
  # Stabilize generated symbol names with nvcc's CUDA 12.9+ option.
  if(FHELIUM_BUILD_CUDA
     AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
     AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
    foreach(_FHELIUM_CUDA_SOURCE IN LISTS CU_SOURCES)
      file(RELATIVE_PATH _FHELIUM_CUDA_SOURCE_RELATIVE "${CMAKE_SOURCE_DIR}"
           "${_FHELIUM_CUDA_SOURCE}")
      string(SHA256 _FHELIUM_CUDA_SOURCE_HASH
                    "fhelium/${_FHELIUM_CUDA_SOURCE_RELATIVE}")
      string(SUBSTRING "${_FHELIUM_CUDA_SOURCE_HASH}" 0 16
                       _FHELIUM_CUDA_SOURCE_SEED)
      set_property(
        SOURCE "${_FHELIUM_CUDA_SOURCE}"
        APPEND
        PROPERTY COMPILE_OPTIONS
                 "--frandom-seed=0x${_FHELIUM_CUDA_SOURCE_SEED}")
    endforeach()
  endif()
  message(STATUS "Library ${name} C++ sources: ${CPP_SOURCES}")
  if(FHELIUM_BUILD_CUDA)
    message(STATUS "Library ${name} CUDA sources: ${CU_SOURCES}")
  endif()

  if(FHELIUM_BUILD_HIP)
    message(STATUS "Library ${name} HIP sources: ${HIP_SOURCES}")
  endif()

  add_library(${name} SHARED ${CPP_SOURCES} ${CU_SOURCES} ${HIP_SOURCES})
  target_compile_definitions(${name} PRIVATE FHELIUM_BUILDING_LIBRARY
                                             ${FHELIUM_BACKEND_DEFINITIONS})
  if(NOT MSVC)
    target_compile_definitions(
      ${name} PRIVATE _GLIBCXX_USE_CXX11_ABI=${FHELIUM_TORCH_CXX11_ABI})
  endif()
  target_include_directories(${name} PUBLIC "$<BUILD_INTERFACE:${include_dir}>")
  _fhelium_map_msvc_build_paths(${name})
  if(MSVC
     AND FHELIUM_BUILD_CUDA
     AND CMAKE_GENERATOR MATCHES "^Visual Studio"
     AND NOT CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR)
    # Persist the discovered root in the generated project. NVIDIA's .props
    # otherwise consult a versioned process environment variable again when a
    # later cmake --build invocation launches MSBuild.
    set_target_properties(${name} PROPERTIES VS_GLOBAL_CudaToolkitCustomDir
                                             "${_CUDA_ROOT}/")
  endif()
  target_link_libraries(${name} PRIVATE "${FHELIUM_TORCH_torch_cpu_LIBRARY}"
                                        "${FHELIUM_TORCH_c10_LIBRARY}")
  # OpenMP runtime symbols bind to the runtime loaded with Torch, as for the
  # Torch operator module.
  if(APPLE AND _FHELIUM_TORCH_OPENMP_CONFIG)
    target_link_options(${name} PRIVATE "LINKER:-undefined,dynamic_lookup")
  endif()
  target_link_directories(${name} PRIVATE "${TORCH_LIB_PATH}")
  target_include_directories(
    ${name} PRIVATE ${TORCH_INCLUDE_DIRS}
                    ${FHELIUM_TORCH_PARALLEL_INCLUDE_DIRS})
  foreach(_parallel_option IN LISTS FHELIUM_TORCH_PARALLEL_COMPILE_OPTIONS)
    target_compile_options(
      ${name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${_parallel_option}>")
  endforeach()
  if(FHELIUM_TORCH_PARALLEL_LINK_OPTIONS)
    target_link_options(${name} PRIVATE ${FHELIUM_TORCH_PARALLEL_LINK_OPTIONS})
  endif()
  if(FHELIUM_TORCH_PARALLEL_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${FHELIUM_TORCH_PARALLEL_LIBRARIES})
  endif()

  if(FHELIUM_BUILD_CUDA)
    target_link_libraries(${name} PRIVATE torch torch_cuda c10_cuda)
    foreach(_cuda_library_root IN
            ITEMS "${_CUDA_ROOT}/lib" "${_CUDA_ROOT}/lib64"
                  "${_CUDA_ROOT}/lib/x64")
      if(EXISTS "${_cuda_library_root}")
        target_link_directories(${name} PRIVATE "${_cuda_library_root}")
      endif()
    endforeach()
    target_include_directories(${name} SYSTEM
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_CUDA_ROOT}/include/cccl")
      target_include_directories(${name} SYSTEM
                                 PRIVATE "${_CUDA_ROOT}/include/cccl")
    endif()
    # Every CUDA translation unit is self-contained. Non-RDC objects retain the
    # configured virtual PTX image in the installed library on both Linux and
    # Windows.
    set_target_properties(
      ${name}
      PROPERTIES CUDA_STANDARD ${_FHELIUM_CXX_STANDARD}
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_EXTENSIONS OFF
                 CUDA_SEPARABLE_COMPILATION OFF
                 CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                 CUDA_VISIBILITY_PRESET hidden)
  endif()

  if(FHELIUM_BUILD_HIP)
    target_link_libraries(
      ${name}
      PRIVATE "${FHELIUM_TORCH_torch_LIBRARY}"
              "${FHELIUM_TORCH_torch_hip_LIBRARY}"
              "${FHELIUM_TORCH_c10_hip_LIBRARY}" hip::host)
    # Runtime lookup is defined by INSTALL_RPATH, not the build SDK location.
    target_link_options(${name} PRIVATE
                        "$<$<LINK_LANGUAGE:HIP>:-fno-rtlib-add-rpath>")
    target_compile_definitions(
      ${name}
      PRIVATE USE_ROCM=1
              HIPBLAS_V2
              "$<$<COMPILE_LANGUAGE:HIP>:CUDA_HAS_FP16=1>"
              "$<$<COMPILE_LANGUAGE:HIP>:__HIP_NO_HALF_OPERATORS__=1>"
              "$<$<COMPILE_LANGUAGE:HIP>:__HIP_NO_HALF_CONVERSIONS__=1>"
              "$<$<COMPILE_LANGUAGE:HIP>:HIP_ENABLE_WARP_SYNC_BUILTINS=1>")
    set_target_properties(
      ${name}
      PROPERTIES HIP_STANDARD ${_FHELIUM_CXX_STANDARD}
                 HIP_STANDARD_REQUIRED ON
                 HIP_EXTENSIONS OFF
                 HIP_VISIBILITY_PRESET hidden)
  endif()

  set_target_properties(
    ${name}
    PROPERTIES CXX_STANDARD ${_FHELIUM_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               CXX_VISIBILITY_PRESET hidden
               BUILD_RPATH "${TORCH_LIB_PATH}"
               INSTALL_RPATH "${FHELIUM_TORCH_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH FALSE
               INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
               OUTPUT_NAME fhelium)

  install(
    TARGETS ${name}
    LIBRARY DESTINATION ${FHELIUM_SDK_ROOT}/lib
    RUNTIME DESTINATION ${FHELIUM_SDK_ROOT}/lib
    ARCHIVE DESTINATION ${FHELIUM_SDK_ROOT}/lib)
  install(DIRECTORY "${include_dir}/" DESTINATION ${FHELIUM_SDK_ROOT}/include)

  set(FHELIUM_LIBRARY_FILE
      "${CMAKE_SHARED_LIBRARY_PREFIX}fhelium${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(FHELIUM_IMPORT_LIBRARY_FILE "")
  if(WIN32)
    set(FHELIUM_IMPORT_LIBRARY_FILE
        "${CMAKE_IMPORT_LIBRARY_PREFIX}fhelium${CMAKE_IMPORT_LIBRARY_SUFFIX}")
  endif()
  string(TOLOWER "${_FHELIUM_ENABLED_BACKENDS}" FHELIUM_PACKAGE_BACKENDS)
  set(FHELIUM_PACKAGE_CXX_STANDARD ${_FHELIUM_CXX_STANDARD})
  set(_package_dir "${CMAKE_CURRENT_BINARY_DIR}/FHEliumPackage")
  configure_file("${CMAKE_SOURCE_DIR}/cmake/FHEliumConfig.cmake.in"
                 "${_package_dir}/FHEliumConfig.cmake" @ONLY)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${_package_dir}/FHEliumConfigVersion.cmake"
    VERSION ${FHELIUM_VERSION}
    COMPATIBILITY ExactVersion)
  install(FILES "${_package_dir}/FHEliumConfig.cmake"
                "${_package_dir}/FHEliumConfigVersion.cmake"
          DESTINATION ${FHELIUM_SDK_ROOT}/share/cmake/FHElium)
endfunction()

function(add_fhelium_torch_module name src_dir library out_subdir)
  _fhelium_torch_cxx_standard(_FHELIUM_CXX_STANDARD)
  file(GLOB REGISTRATION_SOURCES CONFIGURE_DEPENDS "${src_dir}/*.cpp")
  list(SORT REGISTRATION_SOURCES)
  message(STATUS "Torch module ${name} C++ sources: ${REGISTRATION_SOURCES}")

  add_library(${name} MODULE ${REGISTRATION_SOURCES})
  target_compile_definitions(${name} PRIVATE ${FHELIUM_BACKEND_DEFINITIONS})
  if(NOT MSVC)
    target_compile_definitions(
      ${name} PRIVATE _GLIBCXX_USE_CXX11_ABI=${FHELIUM_TORCH_CXX11_ABI})
  endif()
  _fhelium_map_msvc_build_paths(${name})
  target_link_libraries(
    ${name} PRIVATE ${library} "${FHELIUM_TORCH_torch_cpu_LIBRARY}"
                    "${FHELIUM_TORCH_c10_LIBRARY}")
  target_link_directories(${name} PRIVATE "${TORCH_LIB_PATH}")
  target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS})
  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD ${_FHELIUM_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               BUILD_RPATH "$<TARGET_FILE_DIR:${library}>;${TORCH_LIB_PATH}"
               INSTALL_RPATH
               "${FHELIUM_SDK_LIBRARY_RPATH};${FHELIUM_TORCH_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH FALSE
               INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
               OUTPUT_NAME "${name}"
               SUFFIX "${FHELIUM_PYTHON_EXTENSION_SUFFIX}")

  if(SKBUILD_STATE STREQUAL "editable")
    install(
      TARGETS ${name}
      LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir}
      RUNTIME
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir})
  else()
    install(
      TARGETS ${name}
      LIBRARY DESTINATION native/${out_subdir}
      RUNTIME DESTINATION native/${out_subdir})
  endif()
endfunction()

function(add_fhelium_runtime_info_module name cpp_path out_subdir)
  python3_add_library(${name} MODULE WITH_SOABI ${cpp_path})
  _fhelium_map_msvc_build_paths(${name})
  if(MSVC
     AND FHELIUM_BUILD_CUDA
     AND CMAKE_GENERATOR MATCHES "^Visual Studio"
     AND NOT CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR)
    set_target_properties(${name} PROPERTIES VS_GLOBAL_CudaToolkitCustomDir
                                             "${_CUDA_ROOT}/")
  endif()
  if(FHELIUM_BUILD_HIP)
    target_link_libraries(${name} PRIVATE hip::host)
  else()
    target_link_libraries(${name} PRIVATE CUDA::cudart)
    target_include_directories(${name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS})
  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               INSTALL_RPATH ""
               INSTALL_RPATH_USE_LINK_PATH FALSE
               INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
               OUTPUT_NAME "${name}"
               SUFFIX "${FHELIUM_PYTHON_EXTENSION_SUFFIX}")
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      TARGETS ${name}
      LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir}
      RUNTIME
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir})
  else()
    install(
      TARGETS ${name}
      LIBRARY DESTINATION native/${out_subdir}
      RUNTIME DESTINATION native/${out_subdir})
  endif()
endfunction()
