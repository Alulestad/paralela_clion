cmake_minimum_required(VERSION 3.29)
project(ejemplo01)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "C:/Users/Dami/CLionProjects/ninja-win/ninja.exe" CACHE FILEPATH "Ninja make program")

find_package(glfw3)
find_package(fmt)

add_executable(ejemplo01 main.cpp)
target_link_libraries(ejemplo01 fmt::fmt)
target_link_libraries(ejemplo01 glfw)
