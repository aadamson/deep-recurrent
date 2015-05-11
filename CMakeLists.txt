cmake_minimum_required(VERSION 3.0)
include_directories(Eigen)
include_directories(data_utils)

project(DeepRecurrent)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

add_executable(DeepRecurrent drnt.cpp)
add_executable(Baseline softmax_reg.cpp)

install (TARGETS Baseline DESTINATION bin)
install (TARGETS DeepRecurrent DESTINATION bin)
