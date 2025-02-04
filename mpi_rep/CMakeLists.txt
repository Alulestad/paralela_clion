cmake_minimum_required(VERSION 3.29)
project(mpi_rep)

set(CMAKE_CXX_STANDARD 17)

#habilitar OpemMp
#set(CMAKE_CXX_FLAGS "${CXX_CMAKE_FLAGS} -fopenmp")
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
if(OpenMP_CXX_FOUND)
    message(STATUS "MPI Link Flags: ${MPI_LINK_FLAGS}")
endif()

find_package(MPI)
find_package(fmt)

find_package(SFML)
find_package(glfw3)

if (MPI_FOUND)
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

#MPI
add_executable(mpi_correccion  mpi_correccion.cpp)
target_link_libraries(mpi_correccion ${MPI_LIBRARIES})

add_executable(mpi_correccion_profe mpi_correccion_profe.cpp)
target_link_libraries(mpi_correccion_profe ${MPI_LIBRARIES})

add_executable(mpi_matrices_multi mpi_matrices_multi.cpp)
target_link_libraries(mpi_matrices_multi ${MPI_LIBRARIES} fmt::fmt)

add_executable(mpi_promedio mpi_promedio.cpp)
target_link_libraries(mpi_promedio ${MPI_LIBRARIES})

add_executable(mpi_suma mpi_suma.cpp)
target_link_libraries(mpi_suma ${MPI_LIBRARIES})



#OMP
add_executable(omp_fractal omp_fractal.cpp omp_fps_counter.cpp omp_fps_counter.h )
target_link_libraries(omp_fractal fmt::fmt sfml::sfml glfw OpenMP::OpenMP_CXX)

add_executable(omp_paleta omp_paleta.cpp)
target_link_libraries(omp_paleta fmt::fmt sfml::sfml glfw OpenMP::OpenMP_CXX)

add_executable(omp_promedio omp_promedio.cpp)
target_link_libraries(omp_promedio fmt::fmt OpenMP::OpenMP_CXX)

add_executable(omp_suma omp_suma.cpp )
target_link_libraries(omp_suma OpenMP::OpenMP_CXX)


