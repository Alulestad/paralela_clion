message(STATUS "Conan: Using CMakeDeps conandeps_legacy.cmake aggregator via include()")
message(STATUS "Conan: It is recommended to use explicit find_package() per dependency instead")

find_package(glfw3)
find_package(SFML)
find_package(fmt)

set(CONANDEPS_LEGACY  glfw  sfml::sfml  fmt::fmt )