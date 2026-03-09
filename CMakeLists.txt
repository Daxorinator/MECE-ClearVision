cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
project(stereo_calibration CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a> $<$<COMPILE_LANGUAGE:CXX>:-mcpu=cortex-a57> $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
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

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui ximgproc objdetect dnn videoio
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

add_library(disp_invert_cuda STATIC src/disp_invert.cu)
set_target_properties(disp_invert_cuda PROPERTIES POSITION_INDEPENDENT_CODE OFF)
target_include_directories(disp_invert_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(view_synthesis src/view_synthesis.cpp src/face_tracker.cpp)
target_include_directories(view_synthesis PRIVATE ${CMAKE_SOURCE_DIR}/extern/libSGM/include)
target_link_libraries(view_synthesis
    ${OpenCV_LIBS}
    sgm
    disp_invert_cuda
    Qt5::Widgets
    Qt5::OpenGL
    GLESv2
    EGL
    Threads::Threads
)

add_executable(yunet_test demos/yunet_test.cpp)
target_link_libraries(yunet_test ${OpenCV_LIBS} Threads::Threads)

# Copy the YuNet face detection model next to the binary after each build
# so the app can find it regardless of working directory.
# 2021sep is the original model that shipped with FaceDetectorYN (OpenCV 4.5.4).
# 2022mar/2023mar use newer ONNX opsets that fail to load on OpenCV 4.5.x (Jetson L4T).
set(YUNET_MODEL "${CMAKE_SOURCE_DIR}/src/models/face_detection_yunet_2021sep.onnx")
if(EXISTS "${YUNET_MODEL}")
    foreach(_tgt view_synthesis yunet_test)
        add_custom_command(TARGET ${_tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${YUNET_MODEL}"
                "$<TARGET_FILE_DIR:${_tgt}>/face_detection_yunet_2021sep.onnx"
            COMMENT "Copying YuNet model next to ${_tgt}"
        )
    endforeach()
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
