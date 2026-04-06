#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#define FT_SMOOTH  0.25f   // EMA alpha for head position smoothing

/*
 * Head position in the face-tracker camera's coordinate frame (metres).
 * x: lateral  (positive = camera's right = viewer's left)
 * y: vertical (positive = down)
 * z: depth    (positive = away from camera = toward screen)
 * valid: false until first successful iris detection
 *
 * x/y are relative to calibration reference.  z is absolute.
 * When mapping to OAK-D camera frame: negate x (cameras face opposite directions).
 */
struct HeadPos {
    float x{0.f}, y{0.f}, z{0.6f};
    bool valid{false};
};

/*
 * FaceTracker — background thread that opens a USB webcam and runs a MediaPipe
 * FaceMesh ONNX model (478 landmarks, iris indices 468–477) via TensorRT to
 * estimate the viewer's 3-D head position from iris diameter.
 *
 * On the first frame (or after losing the face) FaceMesh is run on the full
 * frame.  After each successful inference the crop is updated from the iris
 * position so subsequent frames use a tighter, face-centred input.
 *
 * Iris depth formula:  Z = IRIS_DIAM_M * focal_px / iris_diam_px
 * where IRIS_DIAM_M = 11.7 mm (average human iris).
 *
 * Call calibrate() once the viewer is looking straight ahead and at a
 * comfortable distance; this zeros the x/y reference.
 *
 * TRT engine is built from ONNX on first run (~2–5 min on Nano) and cached
 * next to the ONNX file with a .trt extension for fast subsequent loads.
 */
class FaceTracker {
public:
    FaceTracker()  = default;
    ~FaceTracker() { stop(); }

    // facemesh_onnx : path to face_landmark_with_attention.onnx (478 landmarks)
    // yunet_onnx    : path to face_detection_yunet_2021sep.onnx (face detector)
    bool start(int camera_index, const std::string &facemesh_onnx,
               const std::string &yunet_onnx);
    void stop();

    // Store current iris position as the "looking straight ahead" reference.
    void calibrate();

    HeadPos headPos() const;

    // True once the TRT engine is loaded and the first detection has fired.
    bool isActive() const;

private:
    void threadLoop();

    std::string facemesh_onnx_;
    std::string yunet_onnx_;

    std::thread        worker_;
    std::atomic<bool>  running_{false};
    std::atomic<bool>  active_{false};
    mutable std::mutex mtx_;

    HeadPos head_pos_;
    float   ref_iris_x_{0.f}, ref_iris_y_{0.f};
    bool    calibrated_{false};

    int camera_index_{0};
};
