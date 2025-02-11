cmake_minimum_required(VERSION 3.29)
project(Grupal_PPC_G1)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if (MPI_FOUND)
message(STATUS "MPI FOUND")
message(STATUS "MPI C++ Version: ${MPI_CXX_VERSION}")
message(STATUS "MPI Include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")
message(STATUS "MPI C++ Compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI Link Flags: ${MPI_LINK_FLAGS}")
else()
message(FATAL_ERROR "MPI Not FOUND")
endif()
include_directories(${MPI_INCLUDE_PATH})

find_package(glfw3 REQUIRED)


# -------------------------------------------------------------------------

# Buscar OpenGL (para visualización)
find_package(OpenGL REQUIRED)

# Agregar el ejecutable (código en main.cpp)
add_executable(Grupal_PPC_G1 main.cpp
main.cpp
fps_counter.cpp
fps_counter.h)
find_package(fmt)

# Enlazar las librerías: MPI, glfw3 y OpenGL.
target_link_libraries(Grupal_PPC_G1 ${MPI_LIBRARIES} glfw ${OPENGL_LIBRARIES} fmt::fmt)

