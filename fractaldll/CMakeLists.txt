cmake_minimum_required(VERSION 3.29)
project(fractaldll)

set(CMAKE_CXX_STANDARD 17)

#habilitar OpemMp
set(CMAKE_CXX_FLAGS "${CXX_CMAKE_FLAGS} -fopenmp")


add_library(fractaldll SHARED library.cpp)
