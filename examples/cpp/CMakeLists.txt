cmake_minimum_required(VERSION 3.18)

project(fhelium_cpp_example LANGUAGES CXX)

# CMAKE_PREFIX_PATH lists fhelium.native.cmake_prefix_path and
# torch.utils.cmake_prefix_path of one Python environment; see examples/README.md.
find_package(FHElium CONFIG REQUIRED)

add_executable(ckks_matrix_vector ckks_matrix_vector.cpp)
target_link_libraries(ckks_matrix_vector PRIVATE FHElium::fhelium)
