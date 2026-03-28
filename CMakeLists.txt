cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0071)
    cmake_policy(SET CMP0071 NEW)
endif()

# HunterGate must be called before any project() command — including ours.
# These values must match exactly what extern/depthai-core/CMakeLists.txt uses.
set(HUNTER_CONFIGURATION_TYPES "Release" CACHE STRING "Hunter dependencies list of build configurations")

# depthai-core build options — disable everything we don't use to reduce build time.
# These must be set before add_subdirectory(extern/depthai-core).
set(DEPTHAI_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(DEPTHAI_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(DEPTHAI_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
set(DEPTHAI_OPENCV_SUPPORT  OFF CACHE BOOL "" FORCE)   # we use getData() directly
set(DEPTHAI_ENABLE_BACKWARD OFF CACHE BOOL "" FORCE)   # backward-cpp stack traces
include("${CMAKE_CURRENT_SOURCE_DIR}/extern/depthai-core/cmake/HunterGate.cmake")
HunterGate(
    URL  "https://github.com/cpp-pm/hunter/archive/v0.23.322.tar.gz"
    SHA1 "cb0ea1f74f4a2c49a807de34885743495fccccbe"
    LOCAL
)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
endif()

project(stereo_calibration CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-march=armv8-a -mcpu=cortex-a57 -ffast-math)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# ---------------------------------------------------------------------------
# Camera backend — no longer used by view_synthesis (OAK-D via USB).
# Retained for depth_pipeline which still uses CSI/V4L2 cameras.
#   ON  (default) — nvarguscamerasrc GStreamer pipeline  (Jetson CSI cameras)
#   OFF           — V4L2 device index                    (USB cameras)
# ---------------------------------------------------------------------------
option(CAMERA_BACKEND_CSI
    "Use nvarguscamerasrc GStreamer pipeline for Jetson CSI cameras" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui ximgproc objdetect dnn video videoio
                                           cudaarithm cudaimgproc cudawarping cudastereo)
find_package(Qt5 REQUIRED COMPONENTS Widgets OpenGL)
find_package(Threads REQUIRED)

# ---- depthai-core 2.x (OAK-D Lite) ----
# Built from source as a git submodule, pinned to v2.26.0 (matches pip package).
# cmake/lz4Config.cmake wraps system liblz4-dev, which on Ubuntu 18.04 does not
# ship a CMake config file — this unblocks Hunter's dependency resolution.
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")
add_subdirectory(extern/depthai-core)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_executable(depth_pipeline
    src/depth_pipeline.cpp
    src/oak_receiver.cpp
)
target_link_libraries(depth_pipeline
    ${OpenCV_LIBS}
    depthai::core
    Qt5::Widgets
    Threads::Threads
)

add_executable(view_synthesis
    src/view_synthesis.cpp
    src/face_tracker.cpp
    src/oak_receiver.cpp
)
target_link_libraries(view_synthesis
    ${OpenCV_LIBS}
    depthai::core
    Qt5::Widgets
    Qt5::OpenGL
    GLESv2
    EGL
    Threads::Threads
)
target_precompile_headers(view_synthesis PRIVATE
    <opencv2/core.hpp>
    <opencv2/imgproc.hpp>
    <opencv2/cudaarithm.hpp>
    <opencv2/cudaimgproc.hpp>
    <opencv2/cudawarping.hpp>
    <QApplication>
    <QOpenGLWidget>
    <QOpenGLFunctions>
    <QTimer>
    <QKeyEvent>
)

add_executable(oak_test src/oak_test.cpp)
target_link_libraries(oak_test depthai::core Threads::Threads)

add_executable(yunet_test demos/yunet_test.cpp)
target_link_libraries(yunet_test ${OpenCV_LIBS} Threads::Threads)

# Copy the YuNet model next to yunet_test
set(YUNET_MODEL "${CMAKE_SOURCE_DIR}/src/models/face_detection_yunet_2021sep.onnx")
if(EXISTS "${YUNET_MODEL}")
    add_custom_command(TARGET yunet_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${YUNET_MODEL}"
            "$<TARGET_FILE_DIR:yunet_test>/face_detection_yunet_2021sep.onnx"
        COMMENT "Copying YuNet model next to yunet_test"
    )
endif()

# ---------------------------------------------------------------------------
# Propagate backend choice to depth_pipeline
# ---------------------------------------------------------------------------
if(CAMERA_BACKEND_CSI)
    message(STATUS "Camera backend: CSI (nvarguscamerasrc)")
    target_compile_definitions(depth_pipeline PRIVATE CAMERA_BACKEND_CSI)
else()
    message(STATUS "Camera backend: V4L2")
endif()
