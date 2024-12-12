cmake_minimum_required(VERSION 3.29)
project(mpi)

include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

set(CMAKE_CXX_STANDARD 17)

add_executable(mpi main.cpp)
target_link_libraries(mpi msmpi)






