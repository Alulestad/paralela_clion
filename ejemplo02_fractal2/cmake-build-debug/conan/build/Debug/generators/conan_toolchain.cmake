# Conan automatically generated toolchain file
# DO NOT EDIT MANUALLY, it will be overwritten

# Avoid including toolchain file several times (bad if appending to variables like
#   CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/323
include_guard()
message(STATUS "Using Conan toolchain: ${CMAKE_CURRENT_LIST_FILE}")
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeToolchain' generator only works with CMake >= 3.15")
endif()

########## 'user_toolchain' block #############
# Include one or more CMake user toolchain from tools.cmake.cmaketoolchain:user_toolchain



########## 'generic_system' block #############
# Definition of system, platform and toolset





########## 'compilers' block #############

set(CMAKE_C_COMPILER "C:/Program Files/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/mingw64/bin/c++.exe")
set(CMAKE_RC_COMPILER "C:/Program Files/mingw64/bin/windres.exe")


########## 'arch_flags' block #############
# Define C++ flags, C flags and linker flags from 'settings.arch'

message(STATUS "Conan toolchain: Defining architecture flag: -m64")
string(APPEND CONAN_CXX_FLAGS " -m64")
string(APPEND CONAN_C_FLAGS " -m64")
string(APPEND CONAN_SHARED_LINKER_FLAGS " -m64")
string(APPEND CONAN_EXE_LINKER_FLAGS " -m64")


########## 'libcxx' block #############
# Definition of libcxx from 'compiler.libcxx' setting, defining the
# right CXX_FLAGS for that libcxx



########## 'cppstd' block #############
# Define the C++ and C standards from 'compiler.cppstd' and 'compiler.cstd'

function(conan_modify_std_watch variable access value current_list_file stack)
    set(conan_watched_std_variable "17")
    if (${variable} STREQUAL "CMAKE_C_STANDARD")
        set(conan_watched_std_variable "")
    endif()
    if ("${access}" STREQUAL "MODIFIED_ACCESS" AND NOT "${value}" STREQUAL "${conan_watched_std_variable}")
        message(STATUS "Warning: Standard ${variable} value defined in conan_toolchain.cmake to ${conan_watched_std_variable} has been modified to ${value} by ${current_list_file}")
    endif()
    unset(conan_watched_std_variable)
endfunction()

message(STATUS "Conan toolchain: C++ Standard 17 with extensions OFF")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
variable_watch(CMAKE_CXX_STANDARD conan_modify_std_watch)


########## 'extra_flags' block #############
# Include extra C++, C and linker flags from configuration tools.build:<type>flags
# and from CMakeToolchain.extra_<type>_flags

# Conan conf flags start: 
# Conan conf flags end


########## 'cmake_flags_init' block #############
# Define CMAKE_<XXX>_FLAGS from CONAN_<XXX>_FLAGS

foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${config} config)
    if(DEFINED CONAN_CXX_FLAGS_${config})
      string(APPEND CMAKE_CXX_FLAGS_${config}_INIT " ${CONAN_CXX_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_C_FLAGS_${config})
      string(APPEND CMAKE_C_FLAGS_${config}_INIT " ${CONAN_C_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_SHARED_LINKER_FLAGS_${config})
      string(APPEND CMAKE_SHARED_LINKER_FLAGS_${config}_INIT " ${CONAN_SHARED_LINKER_FLAGS_${config}}")
    endif()
    if(DEFINED CONAN_EXE_LINKER_FLAGS_${config})
      string(APPEND CMAKE_EXE_LINKER_FLAGS_${config}_INIT " ${CONAN_EXE_LINKER_FLAGS_${config}}")
    endif()
endforeach()

if(DEFINED CONAN_CXX_FLAGS)
  string(APPEND CMAKE_CXX_FLAGS_INIT " ${CONAN_CXX_FLAGS}")
endif()
if(DEFINED CONAN_C_FLAGS)
  string(APPEND CMAKE_C_FLAGS_INIT " ${CONAN_C_FLAGS}")
endif()
if(DEFINED CONAN_SHARED_LINKER_FLAGS)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " ${CONAN_SHARED_LINKER_FLAGS}")
endif()
if(DEFINED CONAN_EXE_LINKER_FLAGS)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " ${CONAN_EXE_LINKER_FLAGS}")
endif()


########## 'extra_variables' block #############
# Definition of extra CMake variables from tools.cmake.cmaketoolchain:extra_variables



########## 'try_compile' block #############
# Blocks after this one will not be added when running CMake try/checks

get_property( _CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if(_CMAKE_IN_TRY_COMPILE)
    message(STATUS "Running toolchain IN_TRY_COMPILE")
    return()
endif()


########## 'find_paths' block #############
# Define paths to find packages, programs, libraries, etc.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/conan_cmakedeps_paths.cmake")
  message(STATUS "Conan toolchain: Including CMakeDeps generated conan_find_paths.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/conan_cmakedeps_paths.cmake")
else()

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Definition of CMAKE_MODULE_PATH
# the generators folder (where conan generates files, like this toolchain)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# Definition of CMAKE_PREFIX_PATH, CMAKE_XXXXX_PATH
# The Conan local "generators" folder, where this toolchain is saved.
list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} )
list(PREPEND CMAKE_LIBRARY_PATH "C:/Users/Dami/.conan2/p/b/glfw2e1fe0c49e955/p/lib" "C:/Users/Dami/.conan2/p/b/sfml5a18b07a47e98/p/lib" "C:/Users/Dami/.conan2/p/b/freet8597dfb4ff8e0/p/lib" "C:/Users/Dami/.conan2/p/b/libpnf133f1ebd9860/p/lib" "C:/Users/Dami/.conan2/p/b/zlibbd0ecee327164/p/lib" "C:/Users/Dami/.conan2/p/b/bzip242361b636bf65/p/lib" "C:/Users/Dami/.conan2/p/b/brotl4558b90cd1118/p/lib" "lib" "C:/Users/Dami/.conan2/p/b/flacfc77c48f84cb6/p/lib" "C:/Users/Dami/.conan2/p/b/opena88bbcecde8374/p/lib" "C:/Users/Dami/.conan2/p/b/vorbi409010e116cec/p/lib" "C:/Users/Dami/.conan2/p/b/ogga3482cd742c8d/p/lib" "lib" "C:/Users/Dami/.conan2/p/b/fmt7fba7dc0c9fe1/p/lib")
list(PREPEND CMAKE_INCLUDE_PATH "C:/Users/Dami/.conan2/p/b/glfw2e1fe0c49e955/p/include" "C:/Users/Dami/.conan2/p/b/sfml5a18b07a47e98/p/include" "C:/Users/Dami/.conan2/p/b/freet8597dfb4ff8e0/p/include" "C:/Users/Dami/.conan2/p/b/freet8597dfb4ff8e0/p/include/freetype2" "C:/Users/Dami/.conan2/p/b/libpnf133f1ebd9860/p/include" "C:/Users/Dami/.conan2/p/b/zlibbd0ecee327164/p/include" "C:/Users/Dami/.conan2/p/b/bzip242361b636bf65/p/include" "C:/Users/Dami/.conan2/p/b/brotl4558b90cd1118/p/include" "C:/Users/Dami/.conan2/p/b/brotl4558b90cd1118/p/include/brotli" "include" "C:/Users/Dami/.conan2/p/b/flacfc77c48f84cb6/p/include" "C:/Users/Dami/.conan2/p/b/opena88bbcecde8374/p/include" "C:/Users/Dami/.conan2/p/b/opena88bbcecde8374/p/include/AL" "C:/Users/Dami/.conan2/p/b/vorbi409010e116cec/p/include" "C:/Users/Dami/.conan2/p/b/ogga3482cd742c8d/p/include" "include" "C:/Users/Dami/.conan2/p/b/fmt7fba7dc0c9fe1/p/include")
set(CONAN_RUNTIME_LIB_DIRS "C:/Users/Dami/.conan2/p/b/glfw2e1fe0c49e955/p/bin" "C:/Users/Dami/.conan2/p/b/sfml5a18b07a47e98/p/bin" "C:/Users/Dami/.conan2/p/b/freet8597dfb4ff8e0/p/bin" "C:/Users/Dami/.conan2/p/b/libpnf133f1ebd9860/p/bin" "C:/Users/Dami/.conan2/p/b/zlibbd0ecee327164/p/bin" "C:/Users/Dami/.conan2/p/b/bzip242361b636bf65/p/bin" "C:/Users/Dami/.conan2/p/b/brotl4558b90cd1118/p/bin" "bin" "C:/Users/Dami/.conan2/p/b/flacfc77c48f84cb6/p/bin" "C:/Users/Dami/.conan2/p/b/opena88bbcecde8374/p/bin" "C:/Users/Dami/.conan2/p/b/vorbi409010e116cec/p/bin" "C:/Users/Dami/.conan2/p/b/ogga3482cd742c8d/p/bin" "bin" "C:/Users/Dami/.conan2/p/b/fmt7fba7dc0c9fe1/p/bin" )

endif()


########## 'pkg_config' block #############
# Define pkg-config from 'tools.gnu:pkg_config' executable and paths

if (DEFINED ENV{PKG_CONFIG_PATH})
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR};$ENV{PKG_CONFIG_PATH}")
else()
set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR};")
endif()


########## 'rpath' block #############
# Defining CMAKE_SKIP_RPATH



########## 'output_dirs' block #############
# Definition of CMAKE_INSTALL_XXX folders

set(CMAKE_INSTALL_BINDIR "bin")
set(CMAKE_INSTALL_SBINDIR "bin")
set(CMAKE_INSTALL_LIBEXECDIR "bin")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_OLDINCLUDEDIR "include")


########## 'variables' block #############
# Definition of CMake variables from CMakeToolchain.variables values

# Variables
# Variables  per configuration



########## 'preprocessor' block #############
# Preprocessor definitions from CMakeToolchain.preprocessor_definitions values

# Preprocessor definitions per configuration



if(CMAKE_POLICY_DEFAULT_CMP0091)  # Avoid unused and not-initialized warnings
endif()
