cmake_minimum_required(VERSION 2.8)
include_directories(Eigen)
include_directories(data_utils)

project(DeepRecurrent)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

add_executable(DeepRecurrent drnt_run.cpp)
add_executable(DeepRecurrentSearch drnt_gridsearch.cpp)
add_executable(DeepRecurrentSentence drnt_forward.cpp)

add_executable(GruDeepRecurrent gru_drnt_run.cpp)
add_executable(GruDeepRecurrentSearch gru_drnt_gridsearch.cpp)
add_executable(GruDeepRecurrentSentence gru_drnt_forward.cpp)
add_executable(GruDeepRecurrentNorms gru_drnt_norms.cpp)

add_executable(Baseline softmax_reg.cpp)

install (TARGETS Baseline DESTINATION bin)

install (TARGETS DeepRecurrent DESTINATION bin)
install (TARGETS DeepRecurrentSearch DESTINATION bin)
install (TARGETS DeepRecurrentSentence DESTINATION bin)

install (TARGETS GruDeepRecurrent DESTINATION bin)
install (TARGETS GruDeepRecurrentSearch DESTINATION bin)
install (TARGETS GruDeepRecurrentSentence DESTINATION bin)
install (TARGETS GruDeepRecurrentNorms DESTINATION bin)
