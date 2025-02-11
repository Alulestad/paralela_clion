cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

add_executable(cuda
        kernel.cu
        main.cpp
)

target_include_directories(cuda PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#este directorio es donde estan los .h de cuda

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#--------------------------------------------------------------
#[[find_package(OpenCl)
add_executable(cl_001 main_cl.cpp)

target_link_libraries(cl_001 ${OpenCL_LIBRARIES})

target_include_directories(cl_001 PUBLIC
        ${OpenCl_INCLUDE_DIRS}
)]]

#CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES

find_package(OpenCL)
add_executable(cl_01 main_cl.cpp)
target_link_libraries(cl_01 ${OpenCL_LIBRARIES})
target_include_directories(cl_01 PUBLIC ${OpenCL_INCLUDE_DIRS})

