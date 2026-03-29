#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <opencv2/core.hpp>

// Forward declaration — keeps <depthai/depthai.hpp> out of every file that
// includes this header. The full definition is only needed in oak_receiver.cpp.
namespace dai { class Device; }

// Frames received from the OAK-D Lite over USB.
// disparity:   CV_16U, subpixel 5-bit encoding — divide by 32.0 for pixel disparity.
//              When want_color=true:  aligned to RGB camera, 1280×720.
//              When want_color=false: aligned to left mono camera, 640×480.
// color:       CV_8UC1, NV12 layout (want_color=true only).
// left_rect:   CV_8UC1 grayscale rectified left  image, 640×480 (want_left_rect=true).
// right_rect:  CV_8UC1 grayscale rectified right image, 640×480 (want_right_rect=true).
//              In want_color=false mode left_rect/right_rect share the same coordinate
//              frame as the disparity — use these for DIBR colour + disocclusion fill.
// confidence:  CV_8U, 0 = maximum confidence (optional, may be empty)
struct OAKFrame {
    cv::Mat disparity;
    cv::Mat color;
    cv::Mat left_rect;
    cv::Mat right_rect;
    cv::Mat confidence;
    bool valid{false};
};

// Background thread that keeps the OAK-D Lite pipeline running and exposes
// the latest disparity + colour frame pair. Mirrors the face_tracker pattern:
// start() launches the thread, getFrame() is safe to call from any thread.
class OAKReceiver {
public:
    OAKReceiver()  = default;
    ~OAKReceiver() { stop(); }

    bool start();
    void stop();

    // Returns true and moves the latest frame out if a new one has arrived since
    // the last call. Returns false (and leaves out unchanged) if no new frame.
    bool getFrame(OAKFrame &out);

    // Update stereo config at runtime — safe to call from any thread after start().
    // Changes take effect within one frame.
    void setStereoConfig(bool medianOn, int confidenceThreshold);

    // Configuration — set before calling start().
    // want_color=false: skips RGB camera, color XLink stream, and depth alignment.
    //   Disparity comes back at native mono resolution (640×480 for 480P) — 6.5× less USB bandwidth.
    // want_right_rect=true: streams the rectified right grayscale image (640×480).
    //   Adds ~18 MB/s USB bandwidth.  Used for right-eye disocclusion fill.
    // want_confidence=false: skips the confidence map stream.
    bool want_color{true};
    bool want_left_rect{false};
    bool want_right_rect{false};
    bool want_confidence{false};

    // Camera intrinsics read from device EEPROM after start().
    // Valid only after start() returns true.
    float fx{0}, fy{0}, cx{0}, cy{0};
    float baseline_m{0.075f};  // default OAK-D Lite baseline

private:
    void threadLoop(std::shared_ptr<dai::Device> device);

    std::thread       worker_;
    std::atomic<bool> running_{false};
    std::mutex        mtx_;
    OAKFrame          latest_;
    bool              new_frame_{false};

    // Pending stereo config — written by setStereoConfig(), applied by threadLoop().
    std::mutex cfg_mtx_;
    bool       cfg_dirty_{false};
    bool       cfg_median_on_{true};
    int        cfg_confidence_{200};
};
