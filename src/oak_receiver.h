#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <opencv2/core.hpp>
#include <depthai/depthai.hpp>

// Frames received from the OAK-D Lite over USB.
// disparity: CV_16U, subpixel 5-bit encoding — divide by 32.0 for pixel disparity
// color:     CV_8UC3 BGR, RGB-aligned to disparity resolution
// confidence: CV_8U, 0 = maximum confidence (optional, may be empty)
struct OAKFrame {
    cv::Mat disparity;
    cv::Mat color;
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

    // Returns true and copies the latest frame if a new one has arrived since
    // the last call. Returns false (and leaves out unchanged) if no new frame.
    bool getFrame(OAKFrame &out);

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
};
