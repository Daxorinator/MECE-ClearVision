cmake_minimum_required(VERSION 3.10)
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
# Must be 2.x — depthai 3.x is not supported on JetPack 4.x / L4T R32.
# Added as a git submodule (same pattern as libSGM) to stay compatible with CMake 3.10.
# Pinned to v2.32.0. To initialise after cloning: git submodule update --init --recursive extern/depthai-core
add_subdirectory(extern/depthai-core)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_executable(depth_pipeline src/depth_pipeline.cpp)
target_link_libraries(depth_pipeline
    ${OpenCV_LIBS}
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
