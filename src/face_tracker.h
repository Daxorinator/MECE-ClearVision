#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>   // CascadeClassifier

/* ---- Tuning constants ---- */
#define FT_SMOOTH        0.08f   // EMA alpha — lower = smoother, more lag
#define FT_SENSITIVITY   0.8f    // normalised X displacement → u_shift scale
#define FT_SHIFT_MIN     0.2f
#define FT_SHIFT_MAX     0.8f
#define FT_DETECT_EVERY  10      // run Haar every N frames
#define FT_MIN_POINTS    4       // force re-detect if fewer LK points survive
#define FT_MAX_POINTS    12      // goodFeaturesToTrack max corners

/*
 * FaceTracker — background thread that opens a USB webcam, runs a Haar
 * cascade detector every N frames and Lucas-Kanade sparse optical-flow
 * tracking in between.  Exposes a mutex-protected u_shift in [0, 1].
 *
 * Face lateral position is mapped to u_shift via:
 *   rel_x   = (face_centre_x − half_width) / half_width   ∈ [−1, 1]
 *   u_shift = 0.5 + (ema(rel_x) − ref_x) * FT_SENSITIVITY
 *
 * Call calibrate() when the user is looking straight ahead to zero ref_x.
 */
class FaceTracker {
public:
    FaceTracker()  = default;
    ~FaceTracker() { stop(); }

    // Start background thread.  cascade_path = path to Haar *.xml file.
    bool start(int camera_index, const std::string &cascade_path);
    void stop();

    // Store current EMA position as the "looking straight ahead" reference.
    void calibrate();

    // Thread-safe; returns current u_shift [0, 1].
    float shift()    const;
    bool  isActive() const;

private:
    void threadLoop();

    cv::CascadeClassifier cascade_;

    std::thread       worker_;
    std::atomic<bool> running_{false};
    mutable std::mutex mtx_;

    // Protected by mtx_:
    float u_shift_val_{0.5f};
    float ema_x_{0.0f};
    float ref_x_{0.0f};
    bool  first_detection_{true};

    int         camera_index_{0};
    std::string cascade_path_;
};
