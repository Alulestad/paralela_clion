cmake_minimum_required(VERSION 3.29)
project(ejemplo02_fractal2)

set(CMAKE_CXX_STANDARD 17)


#habilitar OpemMp
set(CMAKE_CXX_FLAGS "${CXX_CMAKE_FLAGS} -fopenmp")

find_package(fmt)
find_package(SFML)
find_package(glfw3)


add_executable(ejemplo02_fractal2 main.cpp
        fps_counter.cpp
        fps_counter.h)
target_link_libraries(ejemplo02_fractal2 fmt::fmt)
target_link_libraries(ejemplo02_fractal2 sfml::sfml)
target_link_libraries(ejemplo02_fractal2 glfw)