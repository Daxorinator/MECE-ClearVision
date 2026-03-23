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
# Camera backend
#   ON  (default) — nvarguscamerasrc GStreamer pipeline  (Jetson CSI cameras)
#   OFF           — V4L2 device index                    (USB cameras)
# ---------------------------------------------------------------------------
option(CAMERA_BACKEND_CSI
    "Use nvarguscamerasrc GStreamer pipeline for Jetson CSI cameras" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui ximgproc objdetect dnn video videoio
                                           cudaarithm cudaimgproc cudawarping cudastereo)
find_package(Qt5 REQUIRED COMPONENTS Widgets OpenGL)
find_package(Threads REQUIRED)

# ---- libSGM (CUDA semi-global matching) ----
set(LIBSGM_SHARED        OFF CACHE BOOL "" FORCE)
set(ENABLE_SAMPLES       OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS         OFF CACHE BOOL "" FORCE)
set(BUILD_OPENCV_WRAPPER OFF CACHE BOOL "" FORCE)
# AUTO_DETECT_ARCH (ON by default in libSGM) will detect sm_53 on Jetson Nano
add_subdirectory(extern/libSGM)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration
    ${OpenCV_LIBS}
    Qt5::Widgets
    Threads::Threads
)

add_executable(depth_pipeline src/depth_pipeline.cpp)
target_link_libraries(depth_pipeline
    ${OpenCV_LIBS}
    Qt5::Widgets
    Threads::Threads
)

add_executable(view_synthesis src/view_synthesis.cpp src/face_tracker.cpp)
target_include_directories(view_synthesis PRIVATE ${CMAKE_SOURCE_DIR}/extern/libSGM/include)
target_link_libraries(view_synthesis
    ${OpenCV_LIBS}
    sgm
    Qt5::Widgets
    Qt5::OpenGL
    GLESv2
    EGL
    Threads::Threads
)

add_executable(yunet_test demos/yunet_test.cpp)
target_link_libraries(yunet_test ${OpenCV_LIBS} Threads::Threads)

# Copy the YuNet model next to yunet_test (demo only — view_synthesis uses Haar + LK).
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
# Propagate backend choice to all targets
# ---------------------------------------------------------------------------
if(CAMERA_BACKEND_CSI)
    message(STATUS "Camera backend: CSI (nvarguscamerasrc)")
    target_compile_definitions(calibration   PRIVATE CAMERA_BACKEND_CSI)
    target_compile_definitions(depth_pipeline PRIVATE CAMERA_BACKEND_CSI)
    target_compile_definitions(view_synthesis PRIVATE CAMERA_BACKEND_CSI)
else()
    message(STATUS "Camera backend: V4L2")
endif()
