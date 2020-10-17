cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deepfilter)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(deepfilter deepfilter.cpp)
target_link_libraries(deepfilter "${TORCH_LIBRARIES}")
set_property(TARGET deepfilter PROPERTY CXX_STANDARD 14)
