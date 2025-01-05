cmake_minimum_required(VERSION 3.29)
project(mpi_rep)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(fmt)

if(MPI_FOUND)
    message(STATUS "MPI FOUD")
    message(STATUS "MPI version: ${MPI_CXX_VERSION}")
    message(STATUS "MPI Include path: ${MPI_INCLUDE_PATH}")
    message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")
    message(STATUS "MPI Include path: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI Link Flags: ${MPI_lINK_FLAGS}")
else ()
    message(FATAL_ERROR "MPI Not FOUND")
endif ()

include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi mpi.cpp)
target_link_libraries(mpi ${MPI_LIBRARIES})

add_executable(matrices_multi matrices_multi.cpp)
target_link_libraries(matrices_multi ${MPI_LIBRARIES} fmt::fmt)


