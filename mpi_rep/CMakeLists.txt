cmake_minimum_required(VERSION 3.29)
project(mpi_rep)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI)

include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi_rep main.cpp)

target_link_libraries(mpi_rep ${MPI_LIBRARIES})