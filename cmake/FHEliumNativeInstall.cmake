# Register native artifact cleanup, wrapper generation, and ABI finalization.

macro(fhelium_prepare_native_install)
  # Remove current-ABI artifacts before installing replacements. This prevents a
  # failed partial install from reporting a valid manifest or retaining
  # cuda_info after a CPU-only rebuild.
  if(SKBUILD_STATE STREQUAL "editable")
    set(FHELIUM_INSTALL_PACKAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fhelium")
  else()
    set(FHELIUM_INSTALL_PACKAGE_ROOT "\${CMAKE_INSTALL_PREFIX}")
  endif()
  install(
    CODE "
        # Earlier builds installed the operators as a Python-named module.
        file(GLOB _FHELIUM_STALE_OPS LIST_DIRECTORIES FALSE
          \"${FHELIUM_INSTALL_PACKAGE_ROOT}/native/torchops/_ops*\")
        file(GLOB _FHELIUM_STALE_CUDA_INFO LIST_DIRECTORIES FALSE
          \"${FHELIUM_INSTALL_PACKAGE_ROOT}/native/cuda/cuda_info*${FHELIUM_PYTHON_EXTENSION_SUFFIX}\")
        if(WIN32)
          # Python3_add_library previously emitted an ABI-unqualified .pyd.
          file(GLOB _FHELIUM_STALE_LEGACY_CUDA_INFO LIST_DIRECTORIES FALSE
            \"${FHELIUM_INSTALL_PACKAGE_ROOT}/native/cuda/cuda_info.pyd\")
        endif()
        file(GLOB _FHELIUM_STALE_MANIFESTS LIST_DIRECTORIES FALSE
          \"${FHELIUM_INSTALL_PACKAGE_ROOT}/native/torchops/_build_manifest*${FHELIUM_PYTHON_SOABI}*.json\")
        foreach(_FHELIUM_STALE_ARTIFACT IN LISTS _FHELIUM_STALE_OPS
                                                _FHELIUM_STALE_CUDA_INFO
                                                _FHELIUM_STALE_LEGACY_CUDA_INFO
                                                _FHELIUM_STALE_MANIFESTS)
          file(REMOVE \${_FHELIUM_STALE_ARTIFACT})
        endforeach()
      ")
endmacro()

macro(fhelium_finalize_native_install)
  # Generated wrappers describe backend-neutral schemas and therefore remain
  # identical across CPU, CUDA, HIP, and combined builds.
  set(WrapperGenScript ${CMAKE_SOURCE_DIR}/scripts/generate_native_wrappers.py)
  add_custom_target(
    native_wrappers_check
    COMMAND
      ${Python3_EXECUTABLE} ${WrapperGenScript} --library-dir
      "$<TARGET_FILE_DIR:fhelium_torch>" --output
      "${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/wrapper" --check
    DEPENDS fhelium_torch "${WrapperGenScript}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT
      "Checking generated wrappers against the freshly built fhelium_torch target"
    VERBATIM)

  if(SKBUILD_STATE STREQUAL "editable")
    set(FHELIUM_WRAPPER_OUTPUT_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/fhelium/native/wrapper")
    set(FHELIUM_INSTALLED_SDK_ROOT "${FHELIUM_SDK_ROOT}")
  else()
    set(FHELIUM_WRAPPER_OUTPUT_DIR "\${CMAKE_INSTALL_PREFIX}/native/wrapper")
    set(FHELIUM_INSTALLED_SDK_ROOT "\${CMAKE_INSTALL_PREFIX}/native")
  endif()
  install(
    CODE "
    execute_process(
        COMMAND \"${Python3_EXECUTABLE}\" \"${WrapperGenScript}\"
                --library-dir \"${FHELIUM_INSTALLED_SDK_ROOT}/lib\"
                --output \"${FHELIUM_WRAPPER_OUTPUT_DIR}\"
                --no-format
        WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR \"Wrapper generation failed with exit code: \${result}\")
    endif()
  ")

  # Wheels place the SDK at the package-relative native/ root, which the
  # manifest leaves implicit. Editable manifests record the build-tree SDK.
  if(SKBUILD_STATE STREQUAL "editable")
    set(FHELIUM_FINAL_PACKAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fhelium")
    set(FHELIUM_FINAL_SDK_ROOT "${FHELIUM_SDK_ROOT}")
  else()
    set(FHELIUM_FINAL_PACKAGE_ROOT "\${CMAKE_INSTALL_PREFIX}")
    set(FHELIUM_FINAL_SDK_ROOT "")
  endif()
  set(FHELIUM_FINALIZE_SCRIPT
      "${CMAKE_SOURCE_DIR}/scripts/finalize_native_build.py")
  string(JOIN "," FHELIUM_FINAL_BACKENDS ${_FHELIUM_ENABLED_BACKENDS})
  install(
    CODE "
    execute_process(
        COMMAND \"${Python3_EXECUTABLE}\" \"${FHELIUM_FINALIZE_SCRIPT}\"
                --source-root \"${CMAKE_SOURCE_DIR}\"
                --package-root \"${FHELIUM_FINAL_PACKAGE_ROOT}\"
                --backends \"${FHELIUM_FINAL_BACKENDS}\"
                --sdk-root \"${FHELIUM_FINAL_SDK_ROOT}\"
                --hip-architectures \"${CMAKE_HIP_ARCHITECTURES}\"
                --hip-compiler-id \"${CMAKE_HIP_COMPILER_ID}\"
                --hip-compiler-version \"${CMAKE_HIP_COMPILER_VERSION}\"
                --hip-version \"${hip_VERSION}\"
        WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
        RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR \"Native build finalization failed with exit code: \${result}\")
    endif()
  ")
endmacro()
