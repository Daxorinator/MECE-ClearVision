cmake_minimum_required(VERSION 3.16)
project(stereo_calibration CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)

add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration
    ${OpenCV_LIBS}
    Qt5::Widgets
    PkgConfig::LIBCAMERA
)

add_executable(depth_pipeline src/depth_pipeline.cpp)
target_link_libraries(depth_pipeline
    ${OpenCV_LIBS}
    Qt5::Widgets
    PkgConfig::LIBCAMERA
)
