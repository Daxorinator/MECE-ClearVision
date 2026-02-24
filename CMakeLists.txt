cmake_minimum_required(VERSION 3.10)
project(stereo_calibration CXX)

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

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d ximgproc objdetect dnn videoio
                                           cudaarithm cudaimgproc cudawarping cudastereo)
find_package(Qt5 REQUIRED COMPONENTS Widgets OpenGL)
find_package(Threads REQUIRED)

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
target_link_libraries(view_synthesis
    ${OpenCV_LIBS}
    Qt5::Widgets
    Qt5::OpenGL
    GLESv2
    EGL
    Threads::Threads
)

# Copy the YuNet face detection model next to the binary after each build
# so the app can find it regardless of working directory.
# 2022mar is required — the 2023mar model crashes OpenCV 4.5.x (Jetson L4T).
set(YUNET_MODEL "${CMAKE_SOURCE_DIR}/src/models/face_detection_yunet_2022mar.onnx")
if(EXISTS "${YUNET_MODEL}")
    add_custom_command(TARGET view_synthesis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${YUNET_MODEL}"
            "$<TARGET_FILE_DIR:view_synthesis>/face_detection_yunet_2022mar.onnx"
        COMMENT "Copying YuNet model to build directory"
    )
else()
    message(WARNING "YuNet model not found at ${YUNET_MODEL} — face tracking will be disabled")
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
