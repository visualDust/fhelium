cmake_minimum_required(VERSION 3.18)

project(fhelium_cpp_example LANGUAGES CXX)

# CMAKE_PREFIX_PATH lists fhelium.native.cmake_prefix_path and
# torch.utils.cmake_prefix_path of one Python environment; see examples/README.md.
find_package(FHElium CONFIG REQUIRED)

add_executable(rns_ckks_example rns_ckks_example.cpp)
target_link_libraries(rns_ckks_example PRIVATE FHElium::fhelium)
