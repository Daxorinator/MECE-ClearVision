# lz4Config.cmake — wraps the system lz4 library as an imported CMake target.
#
# Ubuntu 18.04 (bionic) ships liblz4-dev without a CMake config file, so
# depthai-core's Hunter build cannot find lz4 via find_package(lz4 CONFIG).
# This file provides the lz4::lz4 target that Hunter expects.

if(TARGET lz4::lz4)
    return()
endif()

find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)
find_path(LZ4_INCLUDE_DIR NAMES lz4.h REQUIRED)

add_library(lz4::lz4 SHARED IMPORTED)
set_target_properties(lz4::lz4 PROPERTIES
    IMPORTED_LOCATION "${LZ4_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
)

set(lz4_FOUND TRUE)
