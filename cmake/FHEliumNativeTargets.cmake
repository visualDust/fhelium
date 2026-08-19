# Define the native operator and CUDA runtime-information targets.

function(add_fhelium_torch_module name src_dir out_subdir)
  file(GLOB_RECURSE SCHEMA_SOURCES CONFIGURE_DEPENDS "${src_dir}/*.cpp")
  list(FILTER SCHEMA_SOURCES EXCLUDE REGEX "/(cpu|cuda)/")

  set(CPP_SOURCES ${SCHEMA_SOURCES})
  set(CU_SOURCES "")
  if(FHELIUM_BUILD_CPU)
    file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS "${src_dir}/*/cpu/*.cpp")
    list(APPEND CPP_SOURCES ${CPU_SOURCES})
  endif()
  if(FHELIUM_BUILD_CUDA)
    file(GLOB_RECURSE CUDA_CPP_SOURCES CONFIGURE_DEPENDS
         "${src_dir}/*/cuda/*.cpp")
    file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS "${src_dir}/*/cuda/*.cu")
    list(APPEND CPP_SOURCES ${CUDA_CPP_SOURCES})
  endif()

  list(SORT CPP_SOURCES)
  list(SORT CU_SOURCES)
  message(STATUS "Torch module ${name} C++ sources: ${CPP_SOURCES}")
  if(FHELIUM_BUILD_CUDA)
    message(STATUS "Torch module ${name} CUDA sources: ${CU_SOURCES}")
  endif()

  add_library(${name} MODULE ${CPP_SOURCES} ${CU_SOURCES})
  target_link_libraries(${name} PRIVATE "${FHELIUM_TORCH_torch_cpu_LIBRARY}"
                                        "${FHELIUM_TORCH_c10_LIBRARY}")
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

  if(FHELIUM_BUILD_CUDA)
    target_link_libraries(${name} PRIVATE torch torch_cuda c10_cuda)
    foreach(_cuda_library_root IN ITEMS "${_CUDA_ROOT}/lib"
                                        "${_CUDA_ROOT}/lib64")
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
    set_target_properties(
      ${name}
      PROPERTIES CUDA_STANDARD 17
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_EXTENSIONS OFF
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               BUILD_RPATH "${TORCH_LIB_PATH}"
               INSTALL_RPATH "${FHELIUM_TORCH_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH FALSE
               INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
               OUTPUT_NAME "${name}${PYTHON_MODULE_SUFFIX}")

  if(SKBUILD_STATE STREQUAL "editable")
    install(
      TARGETS ${name}
      LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir})
  else()
    install(TARGETS ${name} LIBRARY DESTINATION native/${out_subdir})
  endif()
endfunction()

function(add_fhelium_cuda_info_module name cpp_path out_subdir)
  python3_add_library(${name} MODULE WITH_SOABI ${cpp_path})
  target_link_libraries(${name} PRIVATE CUDA::cudart)
  target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS}
                                             ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               INSTALL_RPATH ""
               INSTALL_RPATH_USE_LINK_PATH FALSE
               INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      TARGETS ${name}
      LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/${out_subdir})
  else()
    install(TARGETS ${name} LIBRARY DESTINATION native/${out_subdir})
  endif()
endfunction()
