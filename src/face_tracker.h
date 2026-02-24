#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

/* ---- Tuning constants ---- */
#define FT_SMOOTH      0.35f   // EMA alpha — higher = faster but noisier
#define FT_SENSITIVITY 0.8f    // radians of yaw → u_shift offset
// Valid range for u_shift: restrict to the bilateral overlap of both cameras.
// u_shift=0 → pure left-camera view (no parallax), u_shift=1 → full disparity
// shift.  Values outside [0.2, 0.8] produce extreme one-sided views with many
// unfillable holes.
#define FT_SHIFT_MIN   0.2f
#define FT_SHIFT_MAX   0.8f
#define FT_BACKEND     cv::dnn::DNN_BACKEND_DEFAULT
#define FT_TARGET      cv::dnn::DNN_TARGET_CPU

/*
 * FaceTracker — background thread that opens a USB webcam, runs YuNet
 * face detection + solvePnP head pose estimation, and exposes a
 * mutex-protected u_shift value in [0, 1] to the render thread.
 *
 * u_shift = 0.5 + (yaw - ref_yaw) * FT_SENSITIVITY
 *
 * Call calibrate() when the user is looking straight ahead to zero
 * the reference yaw.
 */
class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker();

    // Start background thread.  yunet_model = path to *.onnx detector.
    bool start(int camera_index, const std::string &yunet_model);

    void stop();

    // Store current EMA yaw as the "looking straight ahead" reference.
    void calibrate();

    // Thread-safe; returns current u_shift [0, 1].
    float shift() const;

    bool isActive() const;

private:
    void threadLoop();

    // YuNet — bounding-box + 5 keypoints in one pass, no extra model needed.
    cv::Ptr<cv::FaceDetectorYN>  detector;

    // Canonical 3-D face model (mm), origin at nose tip.
    std::vector<cv::Point3f>     model_points;

    // Estimated camera intrinsics for solvePnP.
    cv::Mat                      cam_matrix, dist_coeffs;

    std::thread                  worker;
    std::atomic<bool>            running{false};
    mutable std::mutex           mtx;

    // Protected by mtx:
    float  u_shift_val{0.5f};
    float  ema_yaw{0.0f};
    float  ref_yaw{0.0f};
    bool   first_detection{true};

    int         camera_index_{0};
    std::string yunet_model_;
};
