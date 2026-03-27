#include "oak_receiver.h"
#include <cstdio>
#include <depthai/depthai.hpp>

bool OAKReceiver::start()
{
    dai::Pipeline pipeline;

    // --- Mono cameras ---
    auto camLeft  = pipeline.create<dai::node::MonoCamera>();
    auto camRight = pipeline.create<dai::node::MonoCamera>();
    camLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    camRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    camLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    camRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

    // --- Color camera ---
    auto camRGB = pipeline.create<dai::node::ColorCamera>();
    camRGB->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRGB->setInterleaved(false);
    camRGB->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    // --- Stereo depth ---
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    stereo->setLeftRightCheck(true);
    stereo->setSubpixel(true);  // 5-bit subpixel, RAW16 range 0-3040
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);
    camLeft->out.link(stereo->left);
    camRight->out.link(stereo->right);

    // --- XLink outputs ---
    auto xoutDisp  = pipeline.create<dai::node::XLinkOut>();
    auto xoutColor = pipeline.create<dai::node::XLinkOut>();
    auto xoutConf  = pipeline.create<dai::node::XLinkOut>();
    xoutDisp->setStreamName("disparity");
    xoutColor->setStreamName("color");
    xoutConf->setStreamName("confidence");
    stereo->disparity.link(xoutDisp->input);
    camRGB->isp.link(xoutColor->input);
    stereo->confidenceMap.link(xoutConf->input);

    try {
        auto device = std::make_shared<dai::Device>(pipeline);

        // Read intrinsics from EEPROM
        auto calib = device->readCalibration();
        auto intr  = calib.getCameraIntrinsics(dai::CameraBoardSocket::RGB, 1920, 1080);
        fx = intr[0][0];
        fy = intr[1][1];
        cx = intr[0][2];
        cy = intr[1][2];
        baseline_m = calib.getBaselineDistance(
            dai::CameraBoardSocket::LEFT, dai::CameraBoardSocket::RIGHT) / 100.0f;

        printf("OAKReceiver: fx=%.1f fy=%.1f cx=%.1f cy=%.1f baseline=%.1f mm\n",
               fx, fy, cx, cy, baseline_m * 1000.0f);

        running_ = true;
        worker_ = std::thread([this, device]() {
            threadLoop(device);
        });
    } catch (const std::exception &e) {
        fprintf(stderr, "OAKReceiver: failed to open device: %s\n", e.what());
        return false;
    }

    return true;
}

void OAKReceiver::stop()
{
    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

bool OAKReceiver::getFrame(OAKFrame &out)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!new_frame_) return false;
    out = latest_;
    new_frame_ = false;
    return true;
}

void OAKReceiver::threadLoop(std::shared_ptr<dai::Device> device)
{
    auto dispQueue  = device->getOutputQueue("disparity",  1, false);
    auto colorQueue = device->getOutputQueue("color",      1, false);
    auto confQueue  = device->getOutputQueue("confidence", 1, false);

    while (running_) {
        auto dispFrame  = dispQueue->get<dai::ImgFrame>();
        auto colorFrame = colorQueue->get<dai::ImgFrame>();
        auto confFrame  = confQueue->get<dai::ImgFrame>();

        if (!dispFrame || !colorFrame) continue;

        // Construct cv::Mat from raw frame data — avoids requiring depthai-core
        // to be built with OpenCV support (DEPTHAI_OPENCV_SUPPORT).
        OAKFrame f;
        {
            auto &d = dispFrame->getData();
            cv::Mat tmp(dispFrame->getHeight(), dispFrame->getWidth(),
                        CV_16UC1, const_cast<uint8_t*>(d.data()));
            f.disparity = tmp.clone();
        }
        {
            // ColorCamera::isp always outputs NV12 (Y plane + interleaved UV plane).
            // setColorOrder/setInterleaved only affect the video encoder output.
            auto &d = colorFrame->getData();
            int w = (int)colorFrame->getWidth();
            int h = (int)colorFrame->getHeight();
            cv::Mat nv12(h * 3 / 2, w, CV_8UC1, const_cast<uint8_t*>(d.data()));
            cv::cvtColor(nv12, f.color, cv::COLOR_YUV2BGR_NV12);
        }
        if (confFrame) {
            auto &d = confFrame->getData();
            cv::Mat tmp(confFrame->getHeight(), confFrame->getWidth(),
                        CV_8UC1, const_cast<uint8_t*>(d.data()));
            f.confidence = tmp.clone();
        }
        f.valid = true;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            latest_    = std::move(f);
            new_frame_ = true;
        }
    }
}
