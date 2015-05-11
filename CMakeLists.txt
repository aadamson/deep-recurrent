cmake_minimum_required(VERSION 3.0)
include_directories(Eigen)
include_directories(data_utils)

project(DeepRecurrent)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(DeepRecurrent drnt.cpp)
