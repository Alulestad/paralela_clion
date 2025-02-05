########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND flac_COMPONENT_NAMES FLAC::FLAC FLAC::FLAC++)
list(REMOVE_DUPLICATES flac_COMPONENT_NAMES)
if(DEFINED flac_FIND_DEPENDENCY_NAMES)
  list(APPEND flac_FIND_DEPENDENCY_NAMES Ogg)
  list(REMOVE_DUPLICATES flac_FIND_DEPENDENCY_NAMES)
else()
  set(flac_FIND_DEPENDENCY_NAMES Ogg)
endif()
set(Ogg_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(flac_PACKAGE_FOLDER_RELEASE "C:/Users/fing.labcom/.conan2/p/flacd1f93feeba5ca/p")
set(flac_BUILD_MODULES_PATHS_RELEASE )


set(flac_INCLUDE_DIRS_RELEASE )
set(flac_RES_DIRS_RELEASE )
set(flac_DEFINITIONS_RELEASE "-DFLAC__NO_DLL")
set(flac_SHARED_LINK_FLAGS_RELEASE )
set(flac_EXE_LINK_FLAGS_RELEASE )
set(flac_OBJECTS_RELEASE )
set(flac_COMPILE_DEFINITIONS_RELEASE "FLAC__NO_DLL")
set(flac_COMPILE_OPTIONS_C_RELEASE )
set(flac_COMPILE_OPTIONS_CXX_RELEASE )
set(flac_LIB_DIRS_RELEASE "${flac_PACKAGE_FOLDER_RELEASE}/lib")
set(flac_BIN_DIRS_RELEASE )
set(flac_LIBRARY_TYPE_RELEASE STATIC)
set(flac_IS_HOST_WINDOWS_RELEASE 1)
set(flac_LIBS_RELEASE FLAC++ FLAC)
set(flac_SYSTEM_LIBS_RELEASE )
set(flac_FRAMEWORK_DIRS_RELEASE )
set(flac_FRAMEWORKS_RELEASE )
set(flac_BUILD_DIRS_RELEASE )
set(flac_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(flac_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${flac_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${flac_COMPILE_OPTIONS_C_RELEASE}>")
set(flac_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${flac_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${flac_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${flac_EXE_LINK_FLAGS_RELEASE}>")


set(flac_COMPONENTS_RELEASE FLAC::FLAC FLAC::FLAC++)
########### COMPONENT FLAC::FLAC++ VARIABLES ############################################

set(flac_FLAC_FLAC++_INCLUDE_DIRS_RELEASE )
set(flac_FLAC_FLAC++_LIB_DIRS_RELEASE "${flac_PACKAGE_FOLDER_RELEASE}/lib")
set(flac_FLAC_FLAC++_BIN_DIRS_RELEASE )
set(flac_FLAC_FLAC++_LIBRARY_TYPE_RELEASE STATIC)
set(flac_FLAC_FLAC++_IS_HOST_WINDOWS_RELEASE 1)
set(flac_FLAC_FLAC++_RES_DIRS_RELEASE )
set(flac_FLAC_FLAC++_DEFINITIONS_RELEASE )
set(flac_FLAC_FLAC++_OBJECTS_RELEASE )
set(flac_FLAC_FLAC++_COMPILE_DEFINITIONS_RELEASE )
set(flac_FLAC_FLAC++_COMPILE_OPTIONS_C_RELEASE "")
set(flac_FLAC_FLAC++_COMPILE_OPTIONS_CXX_RELEASE "")
set(flac_FLAC_FLAC++_LIBS_RELEASE FLAC++)
set(flac_FLAC_FLAC++_SYSTEM_LIBS_RELEASE )
set(flac_FLAC_FLAC++_FRAMEWORK_DIRS_RELEASE )
set(flac_FLAC_FLAC++_FRAMEWORKS_RELEASE )
set(flac_FLAC_FLAC++_DEPENDENCIES_RELEASE FLAC::FLAC)
set(flac_FLAC_FLAC++_SHARED_LINK_FLAGS_RELEASE )
set(flac_FLAC_FLAC++_EXE_LINK_FLAGS_RELEASE )
set(flac_FLAC_FLAC++_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(flac_FLAC_FLAC++_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${flac_FLAC_FLAC++_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${flac_FLAC_FLAC++_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${flac_FLAC_FLAC++_EXE_LINK_FLAGS_RELEASE}>
)
set(flac_FLAC_FLAC++_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${flac_FLAC_FLAC++_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${flac_FLAC_FLAC++_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT FLAC::FLAC VARIABLES ############################################

set(flac_FLAC_FLAC_INCLUDE_DIRS_RELEASE )
set(flac_FLAC_FLAC_LIB_DIRS_RELEASE "${flac_PACKAGE_FOLDER_RELEASE}/lib")
set(flac_FLAC_FLAC_BIN_DIRS_RELEASE )
set(flac_FLAC_FLAC_LIBRARY_TYPE_RELEASE STATIC)
set(flac_FLAC_FLAC_IS_HOST_WINDOWS_RELEASE 1)
set(flac_FLAC_FLAC_RES_DIRS_RELEASE )
set(flac_FLAC_FLAC_DEFINITIONS_RELEASE "-DFLAC__NO_DLL")
set(flac_FLAC_FLAC_OBJECTS_RELEASE )
set(flac_FLAC_FLAC_COMPILE_DEFINITIONS_RELEASE "FLAC__NO_DLL")
set(flac_FLAC_FLAC_COMPILE_OPTIONS_C_RELEASE "")
set(flac_FLAC_FLAC_COMPILE_OPTIONS_CXX_RELEASE "")
set(flac_FLAC_FLAC_LIBS_RELEASE FLAC)
set(flac_FLAC_FLAC_SYSTEM_LIBS_RELEASE )
set(flac_FLAC_FLAC_FRAMEWORK_DIRS_RELEASE )
set(flac_FLAC_FLAC_FRAMEWORKS_RELEASE )
set(flac_FLAC_FLAC_DEPENDENCIES_RELEASE Ogg::ogg)
set(flac_FLAC_FLAC_SHARED_LINK_FLAGS_RELEASE )
set(flac_FLAC_FLAC_EXE_LINK_FLAGS_RELEASE )
set(flac_FLAC_FLAC_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(flac_FLAC_FLAC_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${flac_FLAC_FLAC_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${flac_FLAC_FLAC_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${flac_FLAC_FLAC_EXE_LINK_FLAGS_RELEASE}>
)
set(flac_FLAC_FLAC_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${flac_FLAC_FLAC_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${flac_FLAC_FLAC_COMPILE_OPTIONS_C_RELEASE}>")