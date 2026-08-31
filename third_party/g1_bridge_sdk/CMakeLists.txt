cmake_minimum_required(VERSION 3.14)
project(g1_bridge_sdk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# unitree_sdk2
# The Teleopit bridge only links the SDK library.  Disabling the SDK examples
# avoids their optional Boost/YAML dependencies during the local bridge build.
set(BUILD_EXAMPLES OFF CACHE BOOL "Build unitree_sdk2 examples" FORCE)
add_subdirectory(thirdparty/unitree_sdk2)

pybind11_add_module(g1_bridge_sdk MODULE src/g1_bridge.cpp)
target_link_libraries(g1_bridge_sdk PRIVATE pybind11::module unitree_sdk2)
# Unitree's prebuilt x86_64 SDK is built with libstdc++'s C++11 ABI.  Keep the
# extension ABI aligned when using a non-system compiler such as Zig/Clang.
target_compile_definitions(g1_bridge_sdk PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(g1_bridge_sdk PRIVATE -O3)
