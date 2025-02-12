cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)
find_package(SFML)

add_executable(cuda
        kernel.cu
        main.cpp
)

target_include_directories(cuda PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#este directorio es donde estan los .h de cuda

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda fmt::fmt sfml::sfml ${CUDA_LIBRARIES} cudart)

#CUDA -- reudccion vector
add_executable(cuda_reduccion_vector
        main_reduccion_vector.cu
)

target_include_directories(cuda_reduccion_vector PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(cuda_reduccion_vector PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_reduccion_vector fmt::fmt sfml::sfml ${CUDA_LIBRARIES} cudart)

#--------------------------------------------------------------
#[[find_package(OpenCl)
add_executable(cl_001 main_cl.cpp)

target_link_libraries(cl_001 ${OpenCL_LIBRARIES})

target_include_directories(cl_001 PUBLIC
        ${OpenCl_INCLUDE_DIRS}
)]]

#CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES

#---------------------------------------------------------

add_executable(suma_vectores_cuda
        kernel_suma_vectores.cu
        main_suma_vectores.cpp
)

target_include_directories(suma_vectores_cuda PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#este directorio es donde estan los .h de cuda

set_target_properties(suma_vectores_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#------frantal con cuda

add_executable(cuda_fractal
        kernel_w.cu
        main_w.cpp
)

target_include_directories(cuda_fractal PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#este directorio es donde estan los .h de cuda

set_target_properties(cuda_fractal PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_fractal fmt::fmt sfml::sfml)



#---Open CL
find_package(OpenCL)
add_executable(suma_vectores_cl main_cl_suma_vectores.cpp)
target_link_libraries(suma_vectores_cl ${OpenCL_LIBRARIES})
target_include_directories(suma_vectores_cl PUBLIC ${OpenCL_INCLUDE_DIRS})




#-- OMP reduccion de vectores



