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

    // --- Stereo depth (common config) ---
    auto stereoDepth = pipeline.create<dai::node::StereoDepth>();
    stereoDepth->setLeftRightCheck(true);
    stereoDepth->setSubpixel(true);
    stereoDepth->setMedianFilter(dai::StereoDepthProperties::MedianFilter::KERNEL_7x7);
    stereoDepth->setConfidenceThreshold(cfg_confidence_);
    camLeft->out.link(stereoDepth->left);
    camRight->out.link(stereoDepth->right);

    // Runtime config channel — allows setStereoConfig() to update median/confidence
    // on the fly without restarting the pipeline.
    auto xinConfig = pipeline.create<dai::node::XLinkIn>();
    xinConfig->setStreamName("stereoConfig");
    xinConfig->out.link(stereoDepth->inputConfig);

    auto xoutDisp = pipeline.create<dai::node::XLinkOut>();
    xoutDisp->setStreamName("disparity");
    stereoDepth->disparity.link(xoutDisp->input);

    if (want_left_rect) {
        auto xoutRectL = pipeline.create<dai::node::XLinkOut>();
        xoutRectL->setStreamName("rectifiedLeft");
        stereoDepth->rectifiedLeft.link(xoutRectL->input);
    }
    if (want_right_rect) {
        auto xoutRectR = pipeline.create<dai::node::XLinkOut>();
        xoutRectR->setStreamName("rectifiedRight");
        stereoDepth->rectifiedRight.link(xoutRectR->input);
    }

    // --- Color camera + depth alignment (only when color stream is wanted) ---
    if (want_color) {
        auto camRGB = pipeline.create<dai::node::ColorCamera>();
        camRGB->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
        camRGB->setIspScale(2, 3);  // 1920×1080 → 1280×720
        camRGB->setInterleaved(false);
        camRGB->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
        stereoDepth->setDepthAlign(dai::CameraBoardSocket::RGB);

        auto xoutColor = pipeline.create<dai::node::XLinkOut>();
        xoutColor->setStreamName("color");
        camRGB->isp.link(xoutColor->input);

        if (want_confidence) {
            auto xoutConf = pipeline.create<dai::node::XLinkOut>();
            xoutConf->setStreamName("confidence");
            stereoDepth->confidenceMap.link(xoutConf->input);
        }
    }

    try {
        auto device = std::make_shared<dai::Device>(pipeline);

        // Read intrinsics from EEPROM
        auto calib = device->readCalibration();
        if (want_color) {
            auto intr = calib.getCameraIntrinsics(dai::CameraBoardSocket::RGB, 1280, 720);
            fx = intr[0][0];  fy = intr[1][1];
            cx = intr[0][2];  cy = intr[1][2];
        } else {
            auto intr = calib.getCameraIntrinsics(dai::CameraBoardSocket::LEFT, 640, 480);
            fx = intr[0][0];  fy = intr[1][1];
            cx = intr[0][2];  cy = intr[1][2];
        }
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
    out = std::move(latest_);
    new_frame_ = false;
    return true;
}

void OAKReceiver::setStereoConfig(bool medianOn, int confidence)
{
    std::lock_guard<std::mutex> lock(cfg_mtx_);
    cfg_median_on_  = medianOn;
    cfg_confidence_ = confidence;
    cfg_dirty_      = true;
}

void OAKReceiver::threadLoop(std::shared_ptr<dai::Device> device)
{
    auto dispQueue   = device->getOutputQueue("disparity", 1, false);
    auto configQueue = device->getInputQueue("stereoConfig");
    std::shared_ptr<dai::DataOutputQueue> colorQueue, confQueue, rectLQueue, rectRQueue;
    if (want_color)      colorQueue = device->getOutputQueue("color",          1, false);
    if (want_left_rect)  rectLQueue = device->getOutputQueue("rectifiedLeft",  1, false);
    if (want_right_rect) rectRQueue = device->getOutputQueue("rectifiedRight", 1, false);
    if (want_confidence) confQueue  = device->getOutputQueue("confidence",     1, false);

    std::shared_ptr<dai::ImgFrame> lastColor;

    while (running_) {
        // Apply any pending stereo config change before grabbing the next frame.
        {
            std::lock_guard<std::mutex> lock(cfg_mtx_);
            if (cfg_dirty_) {
                dai::StereoDepthConfig cfg;
                cfg.setLeftRightCheck(true);
                cfg.setSubpixel(true);
                cfg.setMedianFilter(cfg_median_on_
                    ? dai::StereoDepthProperties::MedianFilter::KERNEL_7x7
                    : dai::StereoDepthProperties::MedianFilter::MEDIAN_OFF);
                cfg.setConfidenceThreshold(cfg_confidence_);
                configQueue->send(cfg);
                cfg_dirty_ = false;
            }
        }

        // Block only on disparity — it drives the frame rate.
        auto dispFrame = dispQueue->get<dai::ImgFrame>();
        if (!dispFrame) continue;

        if (colorQueue) {
            auto f = colorQueue->tryGet<dai::ImgFrame>();
            if (f) lastColor = f;
        }
        std::shared_ptr<dai::ImgFrame> confFrame, rectLFrame, rectRFrame;
        if (confQueue)  confFrame  = confQueue->tryGet<dai::ImgFrame>();
        if (rectLQueue) rectLFrame = rectLQueue->tryGet<dai::ImgFrame>();
        if (rectRQueue) rectRFrame = rectRQueue->tryGet<dai::ImgFrame>();

        // Construct cv::Mat from raw frame data — avoids requiring DEPTHAI_OPENCV_SUPPORT.
        OAKFrame f;
        {
            auto &d = dispFrame->getData();
            cv::Mat tmp(dispFrame->getHeight(), dispFrame->getWidth(),
                        CV_16UC1, const_cast<uint8_t*>(d.data()));
            f.disparity = tmp.clone();
        }
        if (lastColor) {
            // Store NV12 as-is — consumers convert to their required format.
            auto &d = lastColor->getData();
            int w = (int)lastColor->getWidth();
            int h = (int)lastColor->getHeight();
            cv::Mat nv12(h * 3 / 2, w, CV_8UC1, const_cast<uint8_t*>(d.data()));
            f.color = nv12.clone();
        }
        if (rectLFrame) {
            auto &d = rectLFrame->getData();
            cv::Mat tmp(rectLFrame->getHeight(), rectLFrame->getWidth(),
                        CV_8UC1, const_cast<uint8_t*>(d.data()));
            f.left_rect = tmp.clone();
        }
        if (rectRFrame) {
            auto &d = rectRFrame->getData();
            cv::Mat tmp(rectRFrame->getHeight(), rectRFrame->getWidth(),
                        CV_8UC1, const_cast<uint8_t*>(d.data()));
            f.right_rect = tmp.clone();
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
